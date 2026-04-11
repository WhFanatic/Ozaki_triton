"""Ozaki Scheme 测试脚本"""

import torch
from triton.testing import do_bench
from ozaki_triton import ozaki_matmul


def test_accuracy():
    """精度测试"""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A_shape = (2, 256, 512)
    B_shape = (2, 1, 512, 256)
    A = torch.randn(A_shape, dtype=torch.float32, device=device)
    B = torch.randn(B_shape, dtype=torch.float32, device=device)
    C_ref = A @ B

    print("=" * 50)
    print(f"精度测试 (A {A_shape}, B {B_shape})")
    print("=" * 50)

    methods = [
        ("Naive FP16", lambda: (A.half() @ B.half()).float()),
    ]
    for s in [2, 3, 4]:
        methods.append((f"Ozaki(split{s})", lambda s=s: ozaki_matmul(A, B, s)))

    print(f"{'方法':<18} {'最大误差':>14} {'相对误差':>14}")
    print("-" * 50)

    for name, fn in methods:
        C = fn()
        rel = (C - C_ref).norm() / C_ref.norm()
        mx = (C - C_ref).abs().max()
        print(f"{name:<18} {mx.item():>14.6e} {rel.item():>14.6e}")
    print()


def test_performance():
    """性能测试"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("性能测试需要 GPU，跳过")
        return

    print("=" * 80)
    print("性能测试")
    print("=" * 80)

    splits = [2, 3, 4]

    # 表头第一行
    header1 = f"{'尺寸':<8} {'FP32':>8} {'FP16':>8}"
    for s in splits:
        header1 += f"  Ozaki(split{s})"
    # 表头第二行
    header2 = f"{'':<8} {'':>8} {'':>8}"
    for s in splits:
        header2 += f" {'time / speedup':>19}"
    print(header1)
    print(header2)
    print("-" * 80)

    for size in [512, 1024, 2048]:
        A_shape = (4, size, size)
        B_shape = (4, size, size)
        A = torch.randn(A_shape, dtype=torch.float32, device=device)
        B = torch.randn(B_shape, dtype=torch.float32, device=device)
        A_h, B_h = A.half(), B.half()

        t_fp32 = do_bench(lambda: A @ B)
        t_fp16 = do_bench(lambda: A_h @ B_h)

        row = f"{size}x{size:<6} {t_fp32:>8.2f} {t_fp16:>8.2f}"
        for s in splits:
            t_ozaki = do_bench(lambda s=s: ozaki_matmul(A, B, num_splits=s))
            speedup = t_fp32 / t_ozaki
            row += f" {t_ozaki:>8.2f} / {speedup:>6.2f}x"
        print(row)
    print()


if __name__ == "__main__":
    test_accuracy()
    test_performance()
