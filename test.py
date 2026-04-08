"""Ozaki Scheme 测试脚本"""

import torch
import time
from ozaki_triton import ozaki_matmul


def benchmark(func, *args, warmup=10, repeats=20, **kwargs):
    """CUDA 计时工具"""
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def test_accuracy():
    """精度测试"""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    M, K, N = 256, 256, 256
    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)
    C_ref = A @ B

    print("=" * 50)
    print("精度测试 (256x256x256)")
    print("=" * 50)

    methods = [
        ("Naive FP16", lambda: (A.half() @ B.half()).float()),
        ("Ozaki (2)", lambda: ozaki_matmul(A, B, 2)),
        ("Ozaki (3)", lambda: ozaki_matmul(A, B, 3)),
        ("Ozaki (4)", lambda: ozaki_matmul(A, B, 4)),
    ]

    print(f"{'方法':<15} {'相对误差':>12} {'最大误差':>12}")
    print("-" * 50)

    for name, fn in methods:
        C = fn()
        rel = (C - C_ref).norm() / C_ref.norm()
        mx = (C - C_ref).abs().max()
        print(f"{name:<15} {rel.item():>12.6e} {mx.item():>12.6e}")
    print()


def test_performance():
    """性能测试"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("性能测试需要 GPU，跳过")
        return

    print("=" * 60)
    print("性能测试")
    print("=" * 60)
    print(f"{'尺寸':<10} {'FP32':>10} {'FP16':>10} {'Ozaki(3)':>10} {'slowdown':>10}")
    print("-" * 60)

    for size in [256, 512, 1024]:
        A = torch.randn(size, size, dtype=torch.float32, device=device)
        B = torch.randn(size, size, dtype=torch.float32, device=device)
        A_h, B_h = A.half(), B.half()

        t_fp32 = benchmark(lambda a, b: a @ b, A, B)
        t_fp16 = benchmark(lambda a, b: a @ b, A_h, B_h)
        t_ozaki = benchmark(ozaki_matmul, A, B, num_splits=3)

        print(f"{size}x{size:<6} {t_fp32:>10.2f} {t_fp16:>10.2f} {t_ozaki:>10.2f} {t_ozaki/t_fp32:>9.1f}x")
    print()


if __name__ == "__main__":
    test_accuracy()
    test_performance()
