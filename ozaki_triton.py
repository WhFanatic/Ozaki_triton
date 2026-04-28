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

from mytuner import optuna_autotune


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

def prune_split(configs, named_args, **kwargs):
    pruned = []
    for cfg in configs:
        bm = cfg.kwargs['BLOCK_M']
        bk = cfg.kwargs['BLOCK_K']
        nw = cfg.num_warps

        bmk2 = triton.next_power_of_2(bm * bk)

        if bm * bk <= 2048:
            if bmk2 // 512 <= nw <= bmk2 // 128:
                pruned.append(cfg)

    # print(f'pruned retained {len(pruned)} / {len(configs)} configs')
    return pruned if pruned else configs


# @triton.autotune(
#     configs=_expand_configs({
@optuna_autotune(
    param_space={
        'BLOCK_M':    [2**i for i in range(6)],
        'BLOCK_K':    [2**i for i in range(6, 12)],
        'num_warps':  [2**i for i in range(5)],
        'num_stages': [i for i in range(1, 7)],
    },
    # ),
    key=['M', 'K', 'NUM_SPLITS', 'IS_INT_SLICE', 'IS_FP64_RESIDUAL'],
    prune_configs_by={'early_config_prune': prune_split},
    n_trials=40,
)
@triton.jit
def split_matrix_kernel(
    A_ptr, slices_ptr, scales_ptr, residu_ptr ,
    M, K,
    stride_ab, stride_am, stride_ak,             # A 的 stride (batch, M, K)
    stride_sb, stride_sl, stride_sm, stride_sk,  # slices 的 stride (batch, num_splits, M, K)
    stride_cb, stride_cl, stride_cm,             # scales 的 stride (batch, num_splits, M)
    stride_rb, stride_rm, stride_rk,             # residual 的 stride (batch, M, K)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    """
    pid_b = tl.program_id(0) # batch 索引
    pid_m = tl.program_id(1) # 行块索引

    # 按 batch 偏移指针
    A_ptr      = A_ptr      + pid_b * stride_ab
    slices_ptr = slices_ptr + pid_b * stride_sb
    scales_ptr = scales_ptr + pid_b * stride_cb
    residu_ptr = residu_ptr + pid_b * stride_rb

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    for l in tl.static_range(NUM_SPLITS):
        for sweep in tl.static_range(2):
            if sweep == 0:
                row_max = tl.full([BLOCK_M], value=1e-38, dtype=tl.float32)
            else:
                scale = tl.exp2(tl.ceil(tl.log2(row_max)) - ALPHA)
                scale_offs = l * stride_cl + offs_m * stride_cm
                tl.store(scales_ptr + scale_offs, scale, mask=mask_m)

            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K
                mask = mask_m[:, None] & mask_k[None, :]

                r_ptrs = residu_ptr  + offs_m[:, None] * stride_rm + offs_k[None, :] * stride_rk
                if l == 0:
                    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                    tile = tl.load(a_ptrs, mask=mask, other=0.0).to(RESIDUAL_DTYPE) # 第一个 split 从 A 读
                else:
                    tile = tl.load(r_ptrs, mask=mask, other=0.0).to(RESIDUAL_DTYPE) # 后续 split 从 residual buffer 读

                if sweep == 0:
                # Pass 1: 扫描所有 K-tile 求 row_max (行最大绝对值), 加小量保护后续 log2 运算
                    row_max = tl.maximum(row_max, tl.max(tl.abs(tile), axis=-1).to(tl.float32))
                else:
                # Pass 2: 提取分片, 写 slice, 更新 residual
                    # 提取分片
                    scaled = tile / scale[:, None]
                    if IS_INT_SLICE:
                        # round-half-away-from-zero + clamp
                        scaled_round = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5)
                        slice = tl.clamp(scaled_round, -INT_CLAMP - 1, INT_CLAMP).to(SLICE_DTYPE)
                    else:
                        slice = scaled.to(SLICE_DTYPE)

                    # 写 slice
                    slice_offs = l * stride_sl + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk
                    tl.store(slices_ptr + slice_offs, slice, mask=mask)

                    # 更新 residual 并写回 buffer (最后一个 split 不需要写)
                    if l < NUM_SPLITS - 1:
                        tile -= slice.to(RESIDUAL_DTYPE) * scale[:, None]
                        tl.store(r_ptrs, tile, mask=mask)


def split_matrix(A, num_splits, alpha, slice_dtype):
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

    residual_dtype = _select_out_accum_dtype(A.dtype)

    slices = torch.empty((B, num_splits, M, K), dtype=slice_dtype, device=A.device)
    scales = torch.empty((B, num_splits, M), dtype=torch.float32, device=A.device)

    if num_splits > 1:
        residu = torch.empty((B, M, K), dtype=torch.float32, device=A.device) # 只在 num_splits > 1 时需要 residual buffer
    else:
        residu = A  # 占位, 不会被读

    grid = lambda META: (B, triton.cdiv(M, META['BLOCK_M']))
    split_matrix_kernel[grid](
        A, slices, scales, residu,
        M, K,
        A.stride(0), A.stride(1), A.stride(2),
        slices.stride(0), slices.stride(1), slices.stride(2), slices.stride(3),
        scales.stride(0), scales.stride(1), scales.stride(2),
        residu.stride(0) if num_splits > 1 else 0,
        residu.stride(1) if num_splits > 1 else 0,
        residu.stride(2) if num_splits > 1 else 0,
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

def prune_gemm(configs, named_args, **kwargs):
    """根据 SMEM 预算和外层累加器精度过滤配置."""
    SMEM_LIMIT = 192 * 1024
    REG_LIMIT_BYTES = 256 * 1024  # A100 寄存器 256KB/SM
    BYTES_PER_ELEM = 1 if kwargs['SLICE_DTYPE'] == torch.int8 else 2
    num_splits = int(kwargs['NUM_SPLITS'])
    pruned = []
    for cfg in configs:
        bm = cfg.kwargs['BLOCK_M']
        bn = cfg.kwargs['BLOCK_N']
        bk = cfg.kwargs['BLOCK_K']
        # SMEM: num_stages 个 K-block buffer
        smem = cfg.num_stages * (bm * bk + bk * bn) * BYTES_PER_ELEM
        # 寄存器: num_splits^2 个 INT32/FP32 累加器
        reg  = num_splits * num_splits * bm * bn * 4
        if smem <= SMEM_LIMIT and reg <= REG_LIMIT_BYTES:
            pruned.append(cfg)
    # print(f'pruned retained {len(pruned)} / {len(configs)} configs')
    return pruned if pruned else configs


@optuna_autotune(
    param_space={
        'BLOCK_M':    [2**i for i in range(4, 9)],
        'BLOCK_N':    [2**i for i in range(4, 9)],
        'BLOCK_K':    [2**i for i in range(5, 9)],
        'num_warps':  [2**i for i in range(5)],
        'num_stages': [i for i in range(1, 6)],
        'maxnreg':    [2**i for i in range(6, 9)],
    },
    key=['M', 'N', 'K', 'NUM_SPLITS', 'IS_INT_SLICE', 'IS_FP64_OUT'],
    prune_configs_by={'early_config_prune': prune_gemm},
    n_trials=100,
)
@triton.jit
def cross_gemm_kernel(
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
    pid   = tl.program_id(1) # swizzle 索引

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    GROUP_M: tl.constexpr = 8
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)  # 行块索引
    pid_n = (pid % num_pid_in_group) // group_size_m  # 列块索引

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

    # 内层累加器, FP32 or INT32 硬件累加. 预先开辟 num_splits^2 个以分离 matmul 和 scale, 保证 dot 流水通畅以及硬件累加
    # triton 限制不能构造二维 tensor 列表, 且即使一维列表在 acc = dot(a, b, acc) 这种调用方式下也有问题, 所以手动展开
    # 由于 NUM_SPLITS 为编译期常量, 所以判断分支为 False 的会自动消除
    ACC_DTYPE: tl.constexpr = tl.int32 if IS_INT_SLICE else tl.float32
    if NUM_SPLITS > 0:
        acc00 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    if NUM_SPLITS > 1:
        acc01 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc10 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc11 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    if NUM_SPLITS > 2:
        acc02 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc12 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc20 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc21 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc22 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    if NUM_SPLITS > 3:
        acc03 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc13 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc23 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc30 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc31 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        acc33 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    # 沿 k 方向分块累加
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]
        offs_a = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        offs_b = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # 预先加载所有 B_slices 避免重复加载
        b_tiles = []
        for j in tl.static_range(NUM_SPLITS):
            b_tiles += [tl.load(B_slices_ptr + j * stride_bl + offs_b, mask=mask_b, other=0).to(SLICE_DTYPE)]

        # num_splits^2 次交叉乘积
        for i in tl.static_range(NUM_SPLITS):
            a_ptrs = A_slices_ptr + i * stride_al + offs_a
            a = tl.load(a_ptrs, mask=mask_a, other=0).to(SLICE_DTYPE)
            for j in tl.static_range(NUM_SPLITS):
                b = b_tiles[j]
                # i, j 为编译期常量, 所有循环和分支判断会在编译期展开
                if i == 0 and j == 0: acc00 = tl.dot(a, b, acc00, out_dtype=ACC_DTYPE)
                if i == 0 and j == 1: acc01 = tl.dot(a, b, acc01, out_dtype=ACC_DTYPE)
                if i == 0 and j == 2: acc02 = tl.dot(a, b, acc02, out_dtype=ACC_DTYPE)
                if i == 0 and j == 3: acc03 = tl.dot(a, b, acc03, out_dtype=ACC_DTYPE)
                if i == 1 and j == 0: acc10 = tl.dot(a, b, acc10, out_dtype=ACC_DTYPE)
                if i == 1 and j == 1: acc11 = tl.dot(a, b, acc11, out_dtype=ACC_DTYPE)
                if i == 1 and j == 2: acc12 = tl.dot(a, b, acc12, out_dtype=ACC_DTYPE)
                if i == 1 and j == 3: acc13 = tl.dot(a, b, acc13, out_dtype=ACC_DTYPE)
                if i == 2 and j == 0: acc20 = tl.dot(a, b, acc20, out_dtype=ACC_DTYPE)
                if i == 2 and j == 1: acc21 = tl.dot(a, b, acc21, out_dtype=ACC_DTYPE)
                if i == 2 and j == 2: acc22 = tl.dot(a, b, acc22, out_dtype=ACC_DTYPE)
                if i == 2 and j == 3: acc23 = tl.dot(a, b, acc23, out_dtype=ACC_DTYPE)
                if i == 3 and j == 0: acc30 = tl.dot(a, b, acc30, out_dtype=ACC_DTYPE)
                if i == 3 and j == 1: acc31 = tl.dot(a, b, acc31, out_dtype=ACC_DTYPE)
                if i == 3 and j == 2: acc32 = tl.dot(a, b, acc32, out_dtype=ACC_DTYPE)
                if i == 3 and j == 3: acc33 = tl.dot(a, b, acc33, out_dtype=ACC_DTYPE)

    # 预先加载所有 B_scales 避免重复加载
    sb_all = [] # scales 始终是 FP32
    for l in tl.static_range(NUM_SPLITS):
        sb_all += [tl.load(B_scales_ptr + l * stride_sbl + offs_n * stride_sbn, mask=mask_n, other=0.0)] # (num_splits, BLOCK_N)

    # 外层累加器, FP64 or FP32, 不低于输入与分片精度
    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=OUT_ACCUM_DTYPE)
    for i in tl.static_range(NUM_SPLITS):
        sa = tl.load(A_scales_ptr + i * stride_sal + offs_m * stride_sam, mask=mask_m, other=0.0) # (num_splits, BLOCK_M)
        for j in tl.static_range(NUM_SPLITS):
            sb = sb_all[j]
            if i == 0 and j == 0: scaled = acc00.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 0 and j == 1: scaled = acc01.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 0 and j == 2: scaled = acc02.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 0 and j == 3: scaled = acc03.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 1 and j == 0: scaled = acc10.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 1 and j == 1: scaled = acc11.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 1 and j == 2: scaled = acc12.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 1 and j == 3: scaled = acc13.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 2 and j == 0: scaled = acc20.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 2 and j == 1: scaled = acc21.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 2 and j == 2: scaled = acc22.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 2 and j == 3: scaled = acc23.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 3 and j == 0: scaled = acc30.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 3 and j == 1: scaled = acc31.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 3 and j == 2: scaled = acc32.to(tl.float32) * sa[:, None] * sb[None, :]
            if i == 3 and j == 3: scaled = acc33.to(tl.float32) * sa[:, None] * sb[None, :]
            out += scaled.to(OUT_ACCUM_DTYPE)

    offs_c = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]

    tl.store(C_ptr + offs_c, out.to(OUT_DTYPE), mask=mask_c)


def cross_gemm(A_slices, B_slices, A_scales, B_scales, num_splits, output_dtype):
    """
    融合 Cross GEMM: 计算 C = sum_{i,j} scale_A_i * scale_B_j * (A_slice_i @ B_slice_j)
    Args:
        A_slices: (B, num_splits, M, K), dtype 为分片 dtype
        B_slices: (B, num_splits, K, N), dtype 同 A_slices
        A_scales: (B, num_splits, M), dtype=float32
        B_scales: (B, num_splits, N), dtype=float32
        num_splits: 分片数量
        output_dtype: 输出 dtype (同原始输入)
    Returns:
        C: (B, M, N), dtype=output_dtype
    """
    assert A_slices.dim() == 4 and B_slices.dim() == 4
    BN, _, M, K = A_slices.shape
    _, _, K2, N = B_slices.shape
    assert K == K2
    slice_dtype = A_slices.dtype

    out_accum_dtype = _select_out_accum_dtype(output_dtype)

    C = torch.empty((BN, M, N), dtype=output_dtype, device=A_slices.device)
    grid = lambda META: (
        BN,
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    cross_gemm_kernel[grid](
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
        OUT_DTYPE=_TORCH_TO_TL[output_dtype],
        IS_INT_SLICE=(slice_dtype == torch.int8),
        IS_FP64_OUT=(out_accum_dtype == torch.float64),
    )

    return C

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
    assert num_splits <= 4, f"num_splits must be <= 4, got {num_splits}"

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
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    main_stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream_b): # 多流并行加速, 因为 B 耗时较长所以先启动
        B_T = B3.transpose(-1, -2)  # (BN, N, K), 零拷贝 view
        B_slices_T, B_scales = split_matrix(B_T, num_splits, alpha, slice_dtype)  # B_slices_T: (BN, num_splits, N, K), B_scales: (BN, num_splits, N)
        B_slices = B_slices_T.transpose(-1, -2)  # (BN, num_splits, K, N), 零拷贝 view

    with torch.cuda.stream(stream_a):
        A_slices, A_scales = split_matrix(A3, num_splits, alpha, slice_dtype)  # A_slices: (BN, num_splits, M, K), A_scales: (BN, num_splits, M)

    main_stream.wait_stream(stream_a)
    main_stream.wait_stream(stream_b)

    # 3. Compute all cross-products and accumulate
    C = cross_gemm(A_slices, B_slices, A_scales, B_scales, num_splits, input_dtype)

    return C.view(out_shape)
