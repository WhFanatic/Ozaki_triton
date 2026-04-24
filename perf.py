"""Ozaki FP64 -> INT8 性能测试脚本 (NCU 专用)"""
import torch
from ozaki_triton import ozaki_matmul


def run_bench(
    shape=(8, 1024, 2048, 2048),
    num_splits=2,
    input_dtype=torch.float64,
    slice_dtype=torch.int8,
):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    device = torch.device("cuda")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Pytorch compiled with CUDA: {torch.version.cuda}")

    B, M, N, K = shape
    a = torch.randn((B, M, K), dtype=input_dtype, device=device)
    b = torch.randn((B, K, N), dtype=input_dtype, device=device)

    # warmup
    for _ in range(10):
        c = ozaki_matmul(a, b, num_splits=num_splits, slice_dtype=slice_dtype)
    torch.cuda.synchronize()

    # run
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("Ozaki_Triton_GEMM")

    with torch.no_grad():
        c = ozaki_matmul(a, b, num_splits=num_splits, slice_dtype=slice_dtype)

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print('shape', c.shape)
    print('error', (torch.norm(c - a @ b) / torch.norm(a @ b)).item())

    return c


if __name__ == "__main__":
    torch.manual_seed(42)
    run_bench()
