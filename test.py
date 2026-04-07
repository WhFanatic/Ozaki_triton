"""Ozaki Scheme 测试脚本"""

import torch
from ozaki_triton import ozaki_matmul


def test_ozaki():
    """对比 Ozaki Scheme 与 PyTorch FP32 matmul 和 naive FP16 matmul 的精度。"""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    M, K, N = 256, 256, 256
    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)

    # 参考结果：FP32 matmul
    C_ref = A @ B

    # Naive FP16 matmul（直接截断，精度最差）
    C_naive_fp16 = (A.half() @ B.half()).float()

    # Ozaki Scheme（不同分片数）
    print("=" * 60)
    print("Ozaki Scheme 精度测试")
    print("=" * 60)

    results = {}
    for num_splits in [2, 3, 4]:
        C_ozaki = ozaki_matmul(A, B, num_splits=num_splits)

        # 相对误差
        rel_err = (C_ozaki - C_ref).norm() / C_ref.norm()
        max_err = (C_ozaki - C_ref).abs().max()
        results[num_splits] = (rel_err.item(), max_err.item())

    # Naive FP16 误差
    naive_rel = (C_naive_fp16 - C_ref).norm() / C_ref.norm()
    naive_max = (C_naive_fp16 - C_ref).abs().max()

    print("\n" + "-" * 60)
    print(f"{'方法':<25} {'相对误差':>15} {'最大绝对误差':>15}")
    print("-" * 60)
    print(f"{'Naive FP16':<25} {naive_rel.item():>15.6e} {naive_max.item():>15.6e}")
    for ns, (rel, mx) in results.items():
        print(f"{'Ozaki (splits=' + str(ns) + ')':<25} {rel:>15.6e} {mx:>15.6e}")
    print("-" * 60)
    print(f"{'FP32 (参考)':<25} {'0':>15} {'0':>15}")
    print()


if __name__ == "__main__":
    test_ozaki()
