"""
Ozaki Scheme 原型实现 (Triton + PyTorch)

核心思想：将高精度矩阵分解为多个低精度分片 (slice)，利用低精度矩阵乘法
计算所有分片的交叉乘积，再用高精度累加还原结果。

本实现演示：FP32 → FP16 分片 → FP16 matmul → FP32 累加
即用多次 FP16 GEMM 模拟一次高精度 FP32 GEMM。

参考：Ozaki et al., "Error-free transformations of matrix multiplication by
using fast routines of matrix multiplication and its applications", 2012.
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================
# 1. 矩阵分片 (Splitting)
# ============================================================

def compute_split_bits(n: int, acc_bits: int = 23, mantissa_bits: int = 10) -> int:
    """
    计算每个分片可安全使用的尾数位数 alpha。
    
    为保证 FP16 乘法无舍入误差，两个操作数的有效尾数位之和不得超过
    累加器的尾数位数（FP32 为 23 位）。同时累加 n 个乘积需要额外
    log2(n) 位来避免溢出。
    
    alpha = floor((acc_bits - log2(n)) / 2)
    且不超过 FP16 尾数位数 (10)。
    """
    alpha = int((acc_bits - math.log2(n)) / 2)
    alpha = min(alpha, mantissa_bits)
    assert alpha >= 1, f"n={n} 太大，无法安全分片 (alpha={alpha})"
    return alpha


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 4},  num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 8},  num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 8},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16}, num_stages=2, num_warps=4),
    ],
    key=['M', 'K', 'NUM_SPLITS'],
)
@triton.jit
def split_matrix_kernel(
    A_ptr, slices_ptr, scales_ptr,
    M, K,
    stride_am, stride_ak,             # A 的 stride
    stride_sl, stride_sm, stride_sk,  # slices 的 stride (num_splits, M, K)
    stride_cl, stride_cm,             # scales 的 stride (num_splits, M)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,            # 必须 >= K，且是 2 的幂
    NUM_SPLITS: tl.constexpr,
    ALPHA: tl.constexpr,
):
    """
    融合版 split_matrix kernel: 将 FP32 矩阵分解为 NUM_SPLITS 个 FP16 分片.
    约束: BLOCK_K >= K, 整行一次性装入寄存器才能支持跨 split 迭代时 residual 始终驻留 SRAM.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    # 一次性加载整行 A 到寄存器
    a_offs = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    residual = tl.load(A_ptr + a_offs, mask=mask, other=0.0).to(tl.float32)

    for l in tl.static_range(NUM_SPLITS):
        # row_max 行最大绝对值, 加小量保护后续 log2 运算
        row_max = tl.maximum(tl.max(tl.abs(residual), axis=-1), 1e-38)

        # scale = 2^(ceil(log2(row_max)) - ALPHA)
        scale = tl.exp2(tl.ceil(tl.log2(row_max)) - ALPHA)
        scale_offs = l * stride_cl + offs_m * stride_cm
        tl.store(scales_ptr + scale_offs, scale, mask=mask_m)

        # 提取分片并写回
        slice = (residual / scale[:, None]).to(tl.float16)
        slice_offs = l * stride_sl + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk # 注意用 slices 自己的 stride
        tl.store(slices_ptr + slice_offs, slice, mask=mask)

        # SRAM 内更新 residual
        residual = residual - slice.to(tl.float32) * scale[:, None] # 要用 fp16 的 slice 转回 fp32 再计算残差, 才能保持精度


def split_matrix(
    A: torch.Tensor,
    num_splits: int,
    alpha: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    使用 Triton kernel 实现 split_matrix.

    Args:
        A: (M, K) FP32 矩阵
        num_splits: 分片数量
        alpha: 每个分片的尾数位数

    Returns:
        slices: (num_splits, M, K) FP16 分片张量
        scales: (num_splits, M) FP32 缩放系数张量
    """
    assert A.dtype == torch.float32
    assert A.dim() == 2
    M, K = A.shape

    BLOCK_K = triton.next_power_of_2(K)
    assert BLOCK_K <= 8192, f"K={K} too large to fit in SRAM"

    slices = torch.empty((num_splits, M, K), dtype=torch.float16, device=A.device)
    scales = torch.empty((num_splits, M), dtype=torch.float32, device=A.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    split_matrix_kernel[grid](
        A, slices, scales,
        M, K,
        A.stride(-2), A.stride(-1),
        slices.stride(0), slices.stride(1), slices.stride(2),
        scales.stride(0), scales.stride(1),
        BLOCK_K=BLOCK_K,
        NUM_SPLITS=num_splits,
        ALPHA=alpha,
    )

    return slices, scales


# ============================================================
# 2. Triton 矩阵乘法 kernel
# ============================================================

def prune_configs(configs, named_args, **kwargs):
    SMEM_LIMIT = 166912
    BYTES_PER_ELEM = 2
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
        # 小 tile, 适合 num_splits>=3 或小矩阵
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),

        # 中 tile, num_splits=2 时的主力
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),

        # 大 tile, 只有 num_splits=2 + 大矩阵能跑得动, 寄存器可能 spill, 留着让 autotune 自己判断
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K', 'NUM_SPLITS'],
    prune_configs_by={'early_config_prune': prune_configs},
)
@triton.jit
def matmul_kernel(
    A_slices_ptr, B_slices_ptr, C_ptr,
    A_scales_ptr, B_scales_ptr,
    M, N, K,
    stride_al, stride_am, stride_ak,
    stride_bl, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_sal, stride_sam,
    stride_sbl, stride_sbn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """
    Ozaki 融合 GEMM: 单次 K 循环内完成所有 splits x splits 个分片对的乘法并累加.
    A/B 读取次数从 splits^2 降到 2*splits, C 只写一次.
    """
    pid_m = tl.program_id(0) # 当前线程块负责的矩阵行块索引
    pid_n = tl.program_id(1) # 当前线程块负责的矩阵列块索引

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # 当前线程块负责的矩阵行索引列表
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # 当前线程块负责的矩阵列索引列表
    mask_m = offs_m < M
    mask_n = offs_n < N

    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 预先加载所有分片的缩放系数 scales
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

        a_tiles = [tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float16)] * NUM_SPLITS
        for i in tl.static_range(NUM_SPLITS):
            a_ptrs = A_slices_ptr + i * stride_al + offs_a
            a_tiles[i] = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float16) # (num_splits, BLOCK_M, BLOCK_K)

        b_tiles = [tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)] * NUM_SPLITS
        for j in tl.static_range(NUM_SPLITS):
            b_ptrs = B_slices_ptr + j * stride_bl + offs_b
            b_tiles[j] = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float16) # (num_splits, BLOCK_K, BLOCK_N)

        # num_splits^2 次交叉乘积
        for i in tl.static_range(NUM_SPLITS):
            a = a_tiles[i]
            sa = sa_all[i]
            for j in tl.static_range(NUM_SPLITS):
                b = b_tiles[j]
                sb = sb_all[j]
                out += tl.dot(a, b).to(tl.float32) * sa[:, None] * sb[None, :]

    offs_c = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]

    tl.store(C_ptr + offs_c, out, mask=mask_c)


# ============================================================
# 3. Ozaki Scheme 主流程
# ============================================================

def ozaki_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    num_splits: int = 3,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Ozaki Scheme 矩阵乘法：C = A @ B

    流程:
    1. Split: 将 A 分为 num_splits 个分片，B 分为 num_splits 个分片
    2. Compute & Sum: 所有 num_splits^2 个交叉乘积的计算和累加

    Args:
        A: (M, K) FP32 矩阵
        B: (K, N) FP32 矩阵
        num_splits: 分片数量
        verbose: 是否打印调试信息

    Returns:
        C: (M, N) FP32 结果
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    alpha = compute_split_bits(K)
    if verbose:
        print(f"[Ozaki] M={M}, K={K}, N={N}, splits={num_splits}, alpha={alpha}")

    # Step 1: Split
    A_slices, A_scales = split_matrix(A, num_splits, alpha)  # A_slices: (num_splits, M, K), A_scales: (num_splits, M)
    B_slices, B_scales = split_matrix(B.T, num_splits, alpha)  # 对 B 按列分片 = 对 B^T 按行分片
    B_slices = B_slices.transpose(-1, -2)

    # Step 2: Compute all cross-products and accumulate
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    matmul_kernel[grid](
        A_slices, B_slices, C,
        A_scales, B_scales,
        M, N, K,
        A_slices.stride(0), A_slices.stride(1), A_slices.stride(2),
        B_slices.stride(0), B_slices.stride(1), B_slices.stride(2),
        C.stride(0), C.stride(1),
        A_scales.stride(0), A_scales.stride(1),
        B_scales.stride(0), B_scales.stride(1),
        NUM_SPLITS=num_splits,
    )

    return C
