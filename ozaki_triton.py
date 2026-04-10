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
        triton.Config({'BLOCK_M': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64}, num_stages=2, num_warps=8),
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
        slices: [(scale_a, A_i_fp16), ...]，长度为 num_splits
    """
    assert A.dtype == torch.float32
    assert A.dim() == 2
    M, K = A.shape

    BLOCK_K = triton.next_power_of_2(K)
    assert BLOCK_K <= 8192, f"K={K} too large to fit in SRAM"

    slices_tensor = torch.empty((num_splits, M, K), dtype=torch.float16, device=A.device)
    scales_tensor = torch.empty((num_splits, M), dtype=torch.float32, device=A.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    split_matrix_kernel[grid](
        A, slices_tensor, scales_tensor,
        M, K,
        A.stride(-2), A.stride(-1),
        slices_tensor.stride(0), slices_tensor.stride(1), slices_tensor.stride(2),
        scales_tensor.stride(0), scales_tensor.stride(1),
        BLOCK_K=BLOCK_K,
        NUM_SPLITS=num_splits,
        ALPHA=alpha,
    )

    return list(zip(scales_tensor, slices_tensor))


# ============================================================
# 2. Triton 矩阵乘法 kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FP16 输入、FP32 累加的分块矩阵乘法。"""
    pid_m = tl.program_id(0) # 当前线程块负责的矩阵行块索引
    pid_n = tl.program_id(1) # 当前线程块负责的矩阵列块索引

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # 当前线程块负责的矩阵行索引列表
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # 当前线程块负责的矩阵列索引列表

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) # FP32 累加器, 累加分块矩阵乘积

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # 加载 A 的 tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float16)

        # 加载 B 的 tile [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float16)

        # FP16 乘、FP32 累加 (关键: 必须 FP32 累加, 否则会溢出)
        acc += tl.dot(a, b).to(tl.float32)

    # 写回
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul_fp16(A_fp16: torch.Tensor, B_fp16: torch.Tensor) -> torch.Tensor:
    """调用 Triton kernel 执行 FP16 matmul，返回 FP32 结果。"""
    assert A_fp16.dtype == torch.float16 and B_fp16.dtype == torch.float16
    M, K = A_fp16.shape
    K2, N = B_fp16.shape
    assert K == K2

    C = torch.empty((M, N), dtype=torch.float32, device=A_fp16.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    matmul_kernel[grid](
        A_fp16, B_fp16, C,
        M, N, K,
        A_fp16.stride(0), A_fp16.stride(1),
        B_fp16.stride(0), B_fp16.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


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
    
    流程：
    1. Split: 将 A 分为 sA 个分片，B 分为 sB 个分片
    2. Compute: 计算所有 sA × sB 个 FP16 交叉乘积
    3. Sum: 用 FP32 累加所有乘积（乘以对应的缩放因子）

    Args:
        A: (M, K) FP32 矩阵
        B: (K, N) FP32 矩阵
        num_splits: 分片数量，越多精度越高但计算量越大
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
    A_slices = split_matrix(A, num_splits, alpha)  # [(scale_a, A_i_fp16), ...]
    B_slices = split_matrix(B.T, num_splits, alpha)  # 对 B 按列分片 = 对 B^T 按行分片

    # Step 2 & 3: Compute all cross-products and accumulate
    C = torch.zeros((M, N), dtype=torch.float32, device=A.device)

    for i, (scale_a, A_i) in enumerate(A_slices):
        for j, (scale_b, B_j) in enumerate(B_slices):
            # A_i: (M, K) fp16, B_j: (N, K) fp16 (因为是 B^T 的分片)

            C_ij = triton_matmul_fp16(A_i, B_j.T)
            C_ij = C_ij * scale_a[:, None] * scale_b[None, :]

            C += C_ij

    return C
