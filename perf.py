"""Ozaki FP64 -> INT8 性能测试脚本"""
import torch
import torch.profiler
import json
from ozaki_triton import ozaki_matmul


def filter_json(input_file='profile_trace.json', output_file='profile_trace_filtered.json'):
    with open(input_file, 'r') as f:
        data = json.load(f)
    # filter out events with duration < 200 us
    data['traceEvents'] = [e for e in data['traceEvents'] if 'dur' not in e or e['dur'] >= 200]
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


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

    # run ncu perf
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("Ozaki_Triton_GEMM")

    with torch.no_grad():
        c = ozaki_matmul(a, b, num_splits=num_splits, slice_dtype=slice_dtype)

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    # run torch.profiler
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(skip_first=2, wait=2, warmup=2, active=3, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_modules=False,
        ) as prof:
            for _ in range(10):
                c = ozaki_matmul(a, b, num_splits=num_splits, slice_dtype=slice_dtype)
                prof.step()

    prof.export_chrome_trace("profile_trace.json")
    filter_json("profile_trace.json", "profile_trace_filtered.json")

    print('shape', c.shape)
    print('error', (torch.norm(c - a @ b) / torch.norm(a @ b)).item())

    return c


if __name__ == "__main__":
    torch.manual_seed(42)
    run_bench()
