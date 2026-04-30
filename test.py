"""Ozaki Scheme 测试脚本"""
import os
import random
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from triton.testing import do_bench
from concurrent.futures import ProcessPoolExecutor
import filelock

from ozaki_triton import ozaki_matmul


def set_context(gpu_id, use_tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    return device


def test(num_shapes=4, output_csv="test_results.csv"):
    """运行精度与性能的统一基准测试。
    测试流程：
        1. 生成随机形状 (B, M, N, K)，满足显存约束 (4*128^3 < numel < 32*2048^3)。
        2. 遍历预定义的 4 种数据类型场景 (如 FP32->FP16, FP64->INT8 等)。
        3. 对每种场景，计算参考结果 (Ref) 并对比 5 种方法：
            - Naive FP16 (Baseline)
            - Ozaki 分块 (Splits=1, 2, 3, 4)
        4. 记录时间、加速比、最大误差 (Max Error) 和相对误差 (Rel Error)。
    Args:
        num_shapes (int): 随机生成的测试矩阵形状数量。默认为 4。
        output_csv (str): 输出结果保存的 CSV 文件路径。默认为 "test_results.csv"。
    Returns:
        list: 包含所有测试结果字典的列表。如果无可用 GPU 则返回空列表。
    """
    if not torch.cuda.is_available():
        print("Test requires GPU, skipping")
        return []

    num_gpus = torch.cuda.device_count()

    scenarios = [
        ("FP32 -> FP16", torch.float32, torch.float16),
        ("FP32 -> INT8", torch.float32, torch.int8),
        ("FP64 -> FP16", torch.float64, torch.float16),
        ("FP64 -> INT8", torch.float64, torch.int8),
        # ("FP16 -> FP32", torch.float16, torch.float32),
    ]
    splits = [0, 1, 2, 3, 4]

    shapes = [(B, M, N, K) for B in 2**np.arange(7)
                           for M in 2**np.arange(7, 12)
                           for N in 2**np.arange(7, 12)
                           for K in 2**np.arange(8, 13)
              if 4 * 128**3 < B * M * N * K < 32 * 2048**3]
    shapes = random.choices(shapes, k=num_shapes)

    # 清空旧文件
    if os.path.exists(output_csv):
        os.remove(output_csv)

    total = len(shapes) * len(scenarios)
    print(f"\nStarting test: {len(shapes)} shapes x {len(scenarios)} scenarios = {total} cases using {num_gpus} GPUs")

    args_list = [(shape, scenarios, splits, i % num_gpus, output_csv)
                 for i, shape in enumerate(shapes)]

    # 将 shapes 分发给单/多 GPU 运行测试
    if num_gpus > 1:
        with ProcessPoolExecutor(max_workers=num_gpus) as pool:
            all_results = list(pool.map(_test_one_shape, args_list))
        results = [r for batch in all_results for r in batch]
    else:
        results = []
        for args in args_list:
            results.extend(_test_one_shape(args))

    print(f"Test completed. Results saved to {output_csv}")
    return results


def _test_one_shape(args):
    """单个 shape 的测试，运行在指定 GPU 上。"""
    shape, scenarios, splits, gpu_id, output_csv = args

    device = set_context(gpu_id)

    B, M, N, K = shape
    results = []

    for name, input_dtype, slice_dtype in scenarios:
        print(f"[GPU {gpu_id}] shape={shape}, scenario={name}")

        # 构造输入 (统一 FP64)
        row_scales = 10 ** ((torch.rand(M, device=device) * 2 - 1) * 2)
        col_scales = 10 ** ((torch.rand(N, device=device) * 2 - 1) * 2)
        a_fp64 = torch.randn((B, M, K), dtype=torch.float64, device=device) * row_scales[:, None]
        b_fp64 = torch.randn((B, K, N), dtype=torch.float64, device=device) * col_scales
        a = a_fp64.to(input_dtype)
        b = b_fp64.to(input_dtype)

        # Reference
        C_ref = torch.nan_to_num(a @ b).double()
        t_ref = do_bench(lambda: a @ b)

        for s in splits:
            # Naive (默认都用 FP16)
            if s == 0:
                C = torch.nan_to_num(a.half() @ b.half()).double()
                t = do_bench(lambda: (a.half() @ b.half()).to(input_dtype))
            # Ozaki 各 splits
            else:
                ozaki_fn = lambda s=s: ozaki_matmul(a, b, num_splits=s, slice_dtype=slice_dtype)
                C = ozaki_fn().double()
                t = do_bench(ozaki_fn)

            max_err = (C - C_ref).abs().max().item()
            rel_err = ((C - C_ref).norm() / C_ref.norm()).item()

            results.append({
                "scenario":    name,
                "input_dtype": str(input_dtype).split(".")[-1],
                "slice_dtype": str(slice_dtype).split(".")[-1],
                "shape":       f"{shape}",
                "splits":      s,
                "ref_time":    t_ref,
                "ozaki_time":  t,
                "speedup":     t_ref / t,
                "max_error":   max_err,
                "rel_error":   rel_err,
            })

    # 带文件锁追加写入 CSV
    fieldnames = ["scenario", "input_dtype", "slice_dtype", "shape", "splits",
                  "ref_time", "ozaki_time", "speedup", "max_error", "rel_error"]
    with filelock.FileLock(output_csv + ".lock"):
        write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(results)

    return results


def show_summary(results_csv="test_results.csv", output_txt="test_summary.txt"):
    """从统一的 results CSV 生成摘要报告."""
    print(f"\nGenerating summary from {results_csv}...")

    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    lines = []
    lines.append("=" * 90)
    lines.append("Ozaki Test Summary")
    lines.append("=" * 90)

    # 按 scenario 分组，保持出现顺序
    scenarios = []
    seen = set()
    for row in data:
        if row["scenario"] not in seen:
            scenarios.append(row["scenario"])
            seen.add(row["scenario"])

    for scenario in scenarios:
        scenario_rows = [r for r in data if r["scenario"] == scenario]
        lines.append(f"\nScenario: {scenario}")
        lines.append("-" * 90)
        header = (f"{'Method':<18} "
                  f"{'Speedup(min~max)':>24} "
                  f"{'MaxErr(min~max)':>28} "
                  f"{'RelErr(min~max)':>28}")
        lines.append(header)
        lines.append("-" * 90)

        # 按 method 分组，保持出现顺序
        methods_seen = []
        methods_set = set()
        for r in scenario_rows:
            splits = int(r["splits"])
            method = f"Ozaki(split{splits})" if splits > 0 else "Naive FP16"
            if method not in methods_set:
                methods_seen.append((method, splits))
                methods_set.add(method)

        for method, splits in methods_seen:
            method_rows = [
                r for r in scenario_rows
                if int(r["splits"]) == splits
            ]
            speedups  = [float(r["speedup"])   for r in method_rows]
            max_errs  = [float(r["max_error"])  for r in method_rows]
            rel_errs  = [float(r["rel_error"])  for r in method_rows]

            speedup_str  = f"{min(speedups):.3f} ~ {max(speedups):.3f}"
            max_err_str  = f"{min(max_errs):.3e} ~ {max(max_errs):.3e}"
            rel_err_str  = f"{min(rel_errs):.3e} ~ {max(rel_errs):.3e}"

            lines.append(
                f"{method:<18} "
                f"{speedup_str:>24} "
                f"{max_err_str:>28} "
                f"{rel_err_str:>28}"
            )

        lines.append("")  # scenario 间空行

    summary = "\n".join(lines)
    with open(output_txt, "w") as f:
        f.write(summary)

    print(f"Summary saved to {output_txt}")
    print(summary)
    return summary


def plot_summary(results_csv="test_results.csv", output_png="test_summary.png"):
    """从统一 CSV 绘制散点折线图."""
    print(f"\nGenerating plots from {results_csv}...")

    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # 保持顺序提取 scenarios 和 shapes
    scenarios, seen = [], set()
    for r in data:
        if r["scenario"] not in seen:
            scenarios.append(r["scenario"])
            seen.add(r["scenario"])

    shapes_per_scenario = {}
    for s in scenarios:
        shapes, seen_s = [], set()
        for r in data:
            if r["scenario"] == s and r["shape"] not in seen_s:
                shapes.append(r["shape"])
                seen_s.add(r["shape"])
        shapes_per_scenario[s] = shapes

    # methods 顺序: Naive, splits=1, splits=2, splits=3, splits=4
    method_keys = ["0", "1", "2", "3", "4"]
    method_labels = ["Naive FP16", "split=1", "split=2", "split=3", "split=4"]
    method_markers = ["s", "o", "^", "v", "D"]

    n = len(scenarios)
    fig, axes = plt.subplots(1, n, constrained_layout=True, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    # 给每个 shape 分配固定颜色
    all_shapes = sorted({sh for shs in shapes_per_scenario.values() for sh in shs})
    cmap = plt.get_cmap("tab10")
    shape_colors = {sh: cmap(i % 10) for i, sh in enumerate(all_shapes)}

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]

        for shape in shapes_per_scenario[scenario]:
            xs, ys = [], []
            for mk in method_keys:
                row = next(
                    (r for r in data
                     if r["scenario"] == scenario
                     and r["shape"] == shape
                     and r["splits"] == mk),
                    None,
                )
                if row is None:
                    xs.append(None)
                    ys.append(None)
                else:
                    xs.append(float(row["rel_error"]))
                    ys.append(float(row["speedup"]))

            color = shape_colors[shape]
            # 折线 (用实数对的部分)
            valid_xs = [x for x in xs if x is not None]
            valid_ys = [y for y in ys if y is not None]
            ax.plot(valid_xs, valid_ys, '-', color=color, linewidth=0.2,
                    alpha=0.6, label=shape)
            # 各 method 的 marker
            for x, y, marker, mlabel in zip(xs, ys, method_markers, method_labels):
                if x is None or y is None:
                    continue
                ax.scatter(x, y, marker=marker, color=color, s=20, linewidths=0, zorder=3)

        # speedup = 1 baseline
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2.5,
                   label='speedup = 1', zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Relative Error")
        ax.set_ylabel("Speedup")
        ax.set_title(scenario)
        ax.grid(True, which="both", alpha=0.3)

        marker_handles = [
            Line2D([0], [0], marker=m, color='gray', markerfacecolor='gray',
                    markeredgecolor='black', linestyle='', markersize=10, label=l)
            for m, l in zip(method_markers, method_labels)
        ]
        shape_handles = [
            Line2D([0], [0], color=shape_colors[sh], linewidth=2, label=sh)
            for sh in shapes_per_scenario[scenario]
        ]
        baseline_handle = Line2D([0], [0], color='black', linewidth=2.5, label='speedup = 1')
        if idx == 0:
            ax.legend(handles=marker_handles, loc='best', fontsize=8)

    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Plot saved to {output_png}")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    test()
    show_summary()
    plot_summary()