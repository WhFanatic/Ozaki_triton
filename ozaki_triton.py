"""
Ozaki Scheme 原型实现 (Triton + PyTorch) - 多精度版本

支持的精度组合:
- 输入 dtype: FP64 / FP32 / FP16 / BF16
- 分片 dtype: FP32 / FP16 / BF16 / INT8
- 输出 dtype: 同输入

精度层次 (三个独立的累加位置):
1. tl.dot 内部 K 维归约: 硬件 Tensor Core 累加器 (FP16/BF16->FP32, INT8->INT32),
   决定单个 slice pair 的 alpha 上界, 不可改.
2. 跨 slice pair 的外层累加 'out': 软件累加器, FP64 输入用 FP64, 其余用 FP32.
   决定 num_splits 个 slice pair 求和的精度上界.
3. 输出 cast: store 时转回用户的 input_dtype.

两种典型用法:
A. "低精模拟高精": 输入 FP32/FP64, 分片 FP16/INT8 -> 用低精 Tensor Core 算高精 GEMM.
   足够大的 num_splits 下可以达到 FP64 精度 (需要 FP64 外层累加).
B. "低精输入高精中间": 输入 FP16/BF16, 分片 FP32 -> 减少直接低精 GEMM 损失.
   通常 num_splits=1 就够 (单 FP32 分片可无损包含 FP16 输入).
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================
# 0. dtype 元信息
# ============================================================

# 分片 dtype -> (有效尾数位, 是否整数)
_SLICE_DTYPE_INFO = {
    torch.float32:  (23, False),
    torch.float16:  (10, False),
    torch.bfloat16: (7,  False),
    torch.int8:     (7,  True),
}

_INPUT_DTYPES = {
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
}

# torch dtype -> triton dtype
_TORCH_TO_TL = {
    torch.float64:  tl.float64,
    torch.float32:  tl.float32,
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.int8:     tl.int8,
    torch.int32:    tl.int32,
}


def _select_dot_accum_dtype(slice_dtype: torch.dtype) -> torch.dtype:
    """tl.dot 的硬件累加器 dtype. 决定 alpha 上界."""
    if slice_dtype == torch.int8:
        return torch.int32
    return torch.float32


def _select_out_accum_dtype(input_dtype: torch.dtype) -> torch.dtype:
    """kernel 内 out 外层累加器 dtype. 决定跨slice pair求和的精度."""
    if input_dtype == torch.float64:
        return torch.float64
    return torch.float32


def compute_split_bits(
    n: int,
    slice_dtype: torch.dtype = torch.float16,
    dot_accum_dtype: torch.dtype = torch.float32,
) -> int:
    """
    计算每个分片可安全使用的尾数位数 alpha.

    tl.dot 内部硬件累加器限制: 两个分片操作数的有效尾数位之和 + log2(n)须 <= 累加器位数.
    alpha = floor((acc_bits - log2(n)) / 2), 且不超过分片 dtype 自身的有效位数.
    """
    if dot_accum_dtype == torch.int32:
        acc_bits = 31  # 留 1 位给符号
    elif dot_accum_dtype == torch.float32:
        acc_bits = 23
    else:
        raise ValueError(f"unsupported dot_accum_dtype: {dot_accum_dtype}")

    slice_bits, _ = _SLICE_DTYPE_INFO[slice_dtype]
    alpha = int((acc_bits - math.log2(n)) / 2)
    alpha = min(alpha, slice_bits)
    assert alpha >= 1, f"n={n} 太大, 无法安全分片 (alpha={alpha})"
    return alpha


# ============================================================
# 1. 矩阵分片 (Splitting)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 4},  num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 8},  num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 8},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16}, num_stages=2, num_warps=4),
    ],
    key=['M', 'K', 'NUM_SPLITS', 'IS_INT_SLICE', 'IS_FP64_RESIDUAL'],
)
@triton.jit
def split_matrix_kernel(
    A_ptr, slices_ptr, scales_ptr,
    M, K,
    stride_ab, stride_am, stride_ak,             # A 的 stride (batch, M, K)
    stride_sb, stride_sl, stride_sm, stride_sk,  # slices 的 stride (batch, num_splits, M, K)
    stride_cb, stride_cl, stride_cm,             # scales 的 stride (batch, num_splits, M)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,            # 必须 >= K，且是 2 的幂
    NUM_SPLITS: tl.constexpr,
    ALPHA: tl.constexpr,
    SLICE_DTYPE: tl.constexpr,
    RESIDUAL_DTYPE: tl.constexpr,
    IS_INT_SLICE: tl.constexpr,
    IS_FP64_RESIDUAL: tl.constexpr,  # 仅用于 autotune key 区分
    INT_CLAMP: tl.constexpr,
):
    """
    通用 split kernel: 将矩阵分解为 NUM_SPLITS 个分片.
    约束: BLOCK_K >= K, 整行一次性装入寄存器才能支持跨 split 迭代时 residual 始终驻留 SRAM.
    """
    pid_b = tl.program_id(0) # batch 索引
    pid_m = tl.program_id(1) # 行块索引

    # 按 batch 偏移指针
    A_ptr      = A_ptr      + pid_b * stride_ab
    slices_ptr = slices_ptr + pid_b * stride_sb
    scales_ptr = scales_ptr + pid_b * stride_cb

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    # 一次性加载整行 A 到寄存器
    a_offs = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    residual = tl.load(A_ptr + a_offs, mask=mask, other=0.0).to(RESIDUAL_DTYPE) # residual 在 RESIDUAL_DTYPE 下迭代, 分片写出时转 SLICE_DTYPE

    for l in tl.static_range(NUM_SPLITS):
        # row_max 行最大绝对值, 加小量保护后续 log2 运算
        row_max = tl.maximum(tl.max(tl.abs(residual), axis=-1), 1e-38)

        # scale = 2^(ceil(log2(row_max)) - ALPHA)
        scale = tl.exp2(tl.ceil(tl.log2(row_max)) - ALPHA)
        scale_offs = l * stride_cl + offs_m * stride_cm
        tl.store(scales_ptr + scale_offs, scale, mask=mask_m)

        # 提取分片并写回
        scaled = residual / scale[:, None]
        if IS_INT_SLICE:
            # round-half-away-from-zero + clamp
            scaled_round = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5)
            slice = tl.clamp(scaled_round, -INT_CLAMP - 1, INT_CLAMP).to(SLICE_DTYPE)
        else:
            slice = scaled.to(SLICE_DTYPE)

        slice_offs = l * stride_sl + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk # 注意用 slices 自己的 stride 计算 offs 而不是复用 a_offs
        tl.store(slices_ptr + slice_offs, slice, mask=mask)

        # SRAM 内更新 residual
        residual = residual - slice.to(RESIDUAL_DTYPE) * scale[:, None]


def split_matrix(
    A: torch.Tensor,
    num_splits: int,
    alpha: int,
    slice_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        A: (B, M, K), 任意支持的浮点 dtype
        num_splits: 分片数量
        alpha: 每个分片的尾数位数
    Returns:
        slices: (B, num_splits, M, K), dtype=slice_dtype, 分片张量
        scales: (B, num_splits, M), dtype=float32, 分片缩放系数
    """
    assert A.dim() == 3
    B, M, K = A.shape

    BLOCK_K = triton.next_power_of_2(K)
    assert BLOCK_K <= 8192, f"K={K} too large to fit in SRAM"

    residual_dtype = _select_out_accum_dtype(A.dtype)

    slices = torch.empty((B, num_splits, M, K), dtype=slice_dtype, device=A.device)
    scales = torch.empty((B, num_splits, M), dtype=torch.float32, device=A.device)

    grid = lambda META: (B, triton.cdiv(M, META['BLOCK_M']))
    split_matrix_kernel[grid](
        A, slices, scales,
        M, K,
        A.stride(0), A.stride(1), A.stride(2),
        slices.stride(0), slices.stride(1), slices.stride(2), slices.stride(3),
        scales.stride(0), scales.stride(1), scales.stride(2),
        BLOCK_K=BLOCK_K,
        NUM_SPLITS=num_splits,
        ALPHA=alpha,
        SLICE_DTYPE=_TORCH_TO_TL[slice_dtype],
        RESIDUAL_DTYPE=_TORCH_TO_TL[residual_dtype],
        IS_INT_SLICE=(slice_dtype == torch.int8),
        IS_FP64_RESIDUAL=(residual_dtype == torch.float64),
        INT_CLAMP=127,
    )

    return slices, scales


# ============================================================
# 2. Triton 矩阵乘法 kernel
# ============================================================

def prune_configs(configs, named_args, **kwargs):
    """根据 SMEM 预算和外层累加器精度过滤配置."""
    SMEM_LIMIT = 166912
    BYTES_PER_ELEM = 2  # 分片字节数, 取保守值
    pruned = []
    for cfg in configs:
        bm = cfg.kwargs['BLOCK_M']
        bn = cfg.kwargs['BLOCK_N']
        bk = cfg.kwargs['BLOCK_K']
        ns = cfg.num_stages
        smem = ns * (bm * bk + bk * bn) * BYTES_PER_ELEM
        if smem <= SMEM_LIMIT:
            pruned.append(cfg)
    return pruned


@triton.autotune(
    configs=[
        # 小 tile, FP64 外层累加器和 num_splits 大时的主力
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        # 中 tile
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        # 大 tile, 仅 FP32 外层 + num_splits 小时跑得动
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K', 'NUM_SPLITS', 'IS_INT_SLICE', 'IS_FP64_OUT'],
    prune_configs_by={'early_config_prune': prune_configs},
)
@triton.jit
def matmul_kernel(
    A_slices_ptr, B_slices_ptr, C_ptr,
    A_scales_ptr, B_scales_ptr,
    M, N, K,
    stride_ab, stride_al, stride_am, stride_ak,    # A_slices: (B, L, M, K)
    stride_bb, stride_bl, stride_bk, stride_bn,    # B_slices: (B, L, K, N)
    stride_cb, stride_cm, stride_cn,               # C: (B, M, N)
    stride_sab, stride_sal, stride_sam,            # A_scales: (B, L, M)
    stride_sbb, stride_sbl, stride_sbn,            # B_scales: (B, L, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SLICE_DTYPE: tl.constexpr,
    OUT_ACCUM_DTYPE: tl.constexpr,  # FP32 或 FP64, kernel 内 out 累加器
    OUT_DTYPE: tl.constexpr,        # 输出 C 的元素类型
    IS_INT_SLICE: tl.constexpr,
    IS_FP64_OUT: tl.constexpr,      # 仅用于 autotune key 区分
):
    """
    Ozaki 融合 GEMM.
    单次 K 循环内完成所有 splits x splits 个分片对的乘法并累加.
    内层 (硬件): tl.dot 用 FP32 (FP/BF 路径) 或 INT32 (INT8 路径) 累加.
    外层 (软件): out 用 OUT_ACCUM_DTYPE (FP32 或 FP64) 累加slice pair.
    输出: store 时转 OUT_DTYPE.
    """
    pid_b = tl.program_id(0) # batch 索引
    pid_m = tl.program_id(1) # 行块索引
    pid_n = tl.program_id(2) # 列块索引

    # 按 batch 偏移指针
    A_slices_ptr = A_slices_ptr + pid_b * stride_ab
    B_slices_ptr = B_slices_ptr + pid_b * stride_bb
    C_ptr        = C_ptr        + pid_b * stride_cb
    A_scales_ptr = A_scales_ptr + pid_b * stride_sab
    B_scales_ptr = B_scales_ptr + pid_b * stride_sbb

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # 行块范围
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # 列块范围
    mask_m = offs_m < M
    mask_n = offs_n < N

    # 外层累加器, FP64 输入时为 FP64, 其余 FP32
    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=OUT_ACCUM_DTYPE)

    # scales 始终是 FP32, 在 K 循环外预先加载
    sa_all = [tl.zeros((BLOCK_M,), dtype=tl.float32)] * NUM_SPLITS
    sb_all = [tl.zeros((BLOCK_N,), dtype=tl.float32)] * NUM_SPLITS

    for l in tl.static_range(NUM_SPLITS):
        sa_all[l] = tl.load(A_scales_ptr + l * stride_sal + offs_m * stride_sam, mask=mask_m, other=0.0) # (num_splits, BLOCK_M)
        sb_all[l] = tl.load(B_scales_ptr + l * stride_sbl + offs_n * stride_sbn, mask=mask_n, other=0.0) # (num_splits, BLOCK_N)

    # 沿 k 方向分块累加
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]
        offs_a = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        offs_b = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_tiles = [tl.zeros((BLOCK_M, BLOCK_K), dtype=SLICE_DTYPE)] * NUM_SPLITS
        for i in tl.static_range(NUM_SPLITS):
            a_ptrs = A_slices_ptr + i * stride_al + offs_a
            a_tiles[i] = tl.load(a_ptrs, mask=mask_a, other=0).to(SLICE_DTYPE)

        b_tiles = [tl.zeros((BLOCK_K, BLOCK_N), dtype=SLICE_DTYPE)] * NUM_SPLITS
        for j in tl.static_range(NUM_SPLITS):
            b_ptrs = B_slices_ptr + j * stride_bl + offs_b
            b_tiles[j] = tl.load(b_ptrs, mask=mask_b, other=0).to(SLICE_DTYPE)

        # num_splits^2 次交叉乘积
        for i in tl.static_range(NUM_SPLITS):
            a = a_tiles[i]
            sa = sa_all[i]
            for j in tl.static_range(NUM_SPLITS):
                b = b_tiles[j]
                sb = sb_all[j]
                # 硬件累加 (FP32 或 INT32) -> 转 FP32 -> 乘 FP32 scale -> 转外层 dtype 累加
                if IS_INT_SLICE:
                    dot_out = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
                else:
                    dot_out = tl.dot(a, b, out_dtype=tl.float32)
                scaled = dot_out * sa[:, None] * sb[None, :]
                out += scaled.to(OUT_ACCUM_DTYPE)

    offs_c = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]

    tl.store(C_ptr + offs_c, out.to(OUT_DTYPE), mask=mask_c)


# ============================================================
# 3. Ozaki Scheme 主流程
# ============================================================

def _canonicalize_batch(A, B):
    """
    把 A: (..., M, K) 和 B: (..., K, N) 的 batch 维 broadcast 到一致, 拍平成 3D.

    Returns:
        A3: (BN, M, K) FP32, contiguous
        B3: (BN, K, N) FP32, contiguous
        batch_shape: 原始 broadcast 后的 batch shape (用于最后 view 回去)
    """
    assert A.dim() >= 2 and B.dim() >= 2
    M, K  = A.shape[-2:]
    K2, N = B.shape[-2:]
    assert K == K2, f"contraction dim mismatch: A K={K}, B K={K2}"

    A_batch = A.shape[:-2]
    B_batch = B.shape[:-2]
    batch_shape = torch.broadcast_shapes(A_batch, B_batch)

    # expand 是零拷贝(只设置 stride=0); reshape 在 expand 后必要时会触发 contiguous 复制
    A_exp = A.expand(*batch_shape, M, K)
    B_exp = B.expand(*batch_shape, K, N)

    BN = math.prod(batch_shape)

    A3 = A_exp.reshape(BN, M, K).contiguous()
    B3 = B_exp.reshape(BN, K, N).contiguous()

    return A3, B3, batch_shape


def ozaki_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    num_splits: int = 3,
    slice_dtype: torch.dtype = torch.float16,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Ozaki Scheme 矩阵乘法: C = A @ B

    流程:
    1. Split: 将 A, B 分为 num_splits 个分片
    2. Compute & Sum: 所有 num_splits^2 个交叉乘积的计算和累加

    支持 broadcastable 的前置 batch 维, 语义同 torch.matmul.

    Args:
        A: (..., M, K), FP64/FP32/FP16/BF16
        B: (..., K, N), dtype 同 A
        num_splits: 分片数量. FP64 输入达到 FP64 精度通常需要 num_splits >= 5~6.
        slice_dtype: 分片 dtype, FP32/FP16/BF16/INT8.
        verbose: 调试信息

    Returns:
        C: (broadcast(...), M, N), dtype 同 A
    """
    assert A.dtype == B.dtype, f"A/B dtype mismatch: {A.dtype} vs {B.dtype}"
    assert A.dtype in _INPUT_DTYPES, f"unsupported input dtype: {A.dtype}"
    assert slice_dtype in _SLICE_DTYPE_INFO, f"unsupported slice dtype: {slice_dtype}"

    input_dtype = A.dtype
    dot_accum_dtype = _select_dot_accum_dtype(slice_dtype)
    out_accum_dtype = _select_out_accum_dtype(input_dtype)

    M, K = A.shape[-2:]
    N    = B.shape[-1]

    # 1. broadcast batch 维, 拍平成 3D
    A3, B3, batch_shape = _canonicalize_batch(A, B)
    BN = A3.shape[0]
    out_shape = (*batch_shape, M, N)

    if BN == 0:
        return torch.empty(out_shape, dtype=input_dtype, device=A.device)

    alpha = compute_split_bits(K, slice_dtype=slice_dtype, dot_accum_dtype=dot_accum_dtype)
    if verbose:
        print(f"[Ozaki] batch={batch_shape}, M={M}, K={K}, N={N}, "
              f"input={input_dtype}, slice={slice_dtype}, "
              f"dot_accum={dot_accum_dtype}, out_accum={out_accum_dtype}, "
              f"splits={num_splits}, alpha={alpha}")

    # 2. Split
    A_slices, A_scales = split_matrix(A3, num_splits, alpha, slice_dtype)  # A_slices: (BN, num_splits, M, K), A_scales: (BN, num_splits, M)

    B_T = B3.transpose(-1, -2)  # (BN, N, K), 零拷贝 view
    B_slices_T, B_scales = split_matrix(B_T.contiguous(), num_splits, alpha, slice_dtype)  # B_slices_T: (BN, num_splits, N, K), B_scales: (BN, num_splits, N)
    B_slices = B_slices_T.transpose(-1, -2)  # (BN, num_splits, K, N), 零拷贝 view

    # 3. Compute all cross-products and accumulate
    C = torch.empty((BN, M, N), dtype=input_dtype, device=A.device)
    grid = lambda META: (
        BN,
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    matmul_kernel[grid](
        A_slices, B_slices, C,
        A_scales, B_scales,
        M, N, K,
        A_slices.stride(0), A_slices.stride(1), A_slices.stride(2), A_slices.stride(3),
        B_slices.stride(0), B_slices.stride(1), B_slices.stride(2), B_slices.stride(3),
        C.stride(0), C.stride(1), C.stride(2),
        A_scales.stride(0), A_scales.stride(1), A_scales.stride(2),
        B_scales.stride(0), B_scales.stride(1), B_scales.stride(2),
        NUM_SPLITS=num_splits,
        SLICE_DTYPE=_TORCH_TO_TL[slice_dtype],
        OUT_ACCUM_DTYPE=_TORCH_TO_TL[out_accum_dtype],
        OUT_DTYPE=_TORCH_TO_TL[input_dtype],
        IS_INT_SLICE=(slice_dtype == torch.int8),
        IS_FP64_OUT=(out_accum_dtype == torch.float64),
    )

    return C.view(out_shape)
