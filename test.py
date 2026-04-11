"""Ozaki Scheme 测试脚本"""

import torch
import csv
import matplotlib.pyplot as plt
from triton.testing import do_bench
from ozaki_triton import ozaki_matmul


def test_accuracy(output_csv="test_accuracy.csv"):
    """精度测试 - 覆盖 5 种精度组合场景"""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A_shape = (2, 256, 512)
    B_shape = (2, 1, 512, 256)

    scenarios = [
        ("FP32 -> FP16", torch.float32, torch.float16),
        ("FP32 -> INT8", torch.float32, torch.int8),
        ("FP64 -> FP16", torch.float64, torch.float16),
        ("FP64 -> INT8", torch.float64, torch.int8),
        ("FP16 -> FP32", torch.float16, torch.float32),
    ]

    results = []
    total = len(scenarios)

    print(f"\nStarting accuracy test, {total} scenarios...")

    for i, (name, input_dtype, slice_dtype) in enumerate(scenarios, 1):
        print(f"[{i}/{total}] Running scenario: {name}")

        A = torch.randn(A_shape, dtype=input_dtype, device=device)
        B = torch.randn(B_shape, dtype=input_dtype, device=device)
        C_ref = (A.double() @ B.double()).double()

        if slice_dtype == torch.int8:
            naive_name = "Naive FP16"
            naive_fn = lambda: (A.half() @ B.half()).to(input_dtype)
        else:
            naive_name = f"Naive {slice_dtype}"
            naive_fn = lambda: (A.to(slice_dtype) @ B.to(slice_dtype)).to(input_dtype)

        methods = [(naive_name, naive_fn)]
        for s in [2, 3, 4]:
            methods.append((f"Ozaki(split{s})", lambda s=s: ozaki_matmul(A, B, num_splits=s, slice_dtype=slice_dtype)))

        for name_method, fn in methods:
            C = fn()
            rel = (C - C_ref).norm() / C_ref.norm()
            mx = (C - C_ref).abs().max()
            results.append({
                "scenario": name,
                "input_dtype": str(input_dtype).split(".")[-1],
                "slice_dtype": str(slice_dtype).split(".")[-1],
                "method": name_method,
                "max_error": mx.item(),
                "rel_error": rel.item(),
            })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "input_dtype", "slice_dtype", "method", "max_error", "rel_error"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Accuracy test completed. Results saved to {output_csv}")
    return results


def test_performance(output_csv="test_performance.csv"):
    """性能测试 - 覆盖 5 种精度组合场景"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Performance test requires GPU, skipping")
        return []

    splits = [2, 3, 4]
    scenarios = [
        ("FP32 -> FP16", torch.float32, torch.float16),
        ("FP32 -> INT8", torch.float32, torch.int8),
        ("FP64 -> FP16", torch.float64, torch.float16),
        ("FP64 -> INT8", torch.float64, torch.int8),
        ("FP16 -> FP32", torch.float16, torch.float32),
    ]

    results = []
    total = len(scenarios) * 3  # 5 scenarios * 3 sizes

    print(f"\nStarting performance test, {total} runs...")

    count = 0
    for name, input_dtype, slice_dtype in scenarios:
        for size in [512, 1024, 2048]:
            count += 1
            print(f"[{count}/{total}] Running: {name}, size={size}x{size}")

            A_shape = (4, size, size)
            B_shape = (4, size, size)

            if input_dtype == torch.float64:
                A = torch.randn(A_shape, dtype=torch.float64, device=device)
                B = torch.randn(B_shape, dtype=torch.float64, device=device)
            else:
                A = torch.randn(A_shape, dtype=input_dtype, device=device)
                B = torch.randn(B_shape, dtype=input_dtype, device=device)

            t_ref = do_bench(lambda: A @ B)
            results.append({
                "scenario": name,
                "input_dtype": str(input_dtype).split(".")[-1],
                "slice_dtype": str(slice_dtype).split(".")[-1],
                "size": size,
                "ref_time": t_ref,
            })

            for s in splits:
                t_ozaki = do_bench(lambda s=s: ozaki_matmul(A, B, num_splits=s, slice_dtype=slice_dtype))
                speedup = t_ref / t_ozaki
                results.append({
                    "scenario": name,
                    "input_dtype": str(input_dtype).split(".")[-1],
                    "slice_dtype": str(slice_dtype).split(".")[-1],
                    "size": size,
                    "splits": s,
                    "ozaki_time": t_ozaki,
                    "speedup": speedup,
                })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "input_dtype", "slice_dtype", "size", "splits", "ref_time", "ozaki_time", "speedup"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Performance test completed. Results saved to {output_csv}")
    return results


def show_summary(accuracy_csv="test_accuracy.csv", performance_csv="test_performance.csv", output_txt="test_summary.txt"):
    """从 CSV 读取数据并生成摘要报告"""
    import csv

    print(f"\nGenerating summary from {accuracy_csv} and {performance_csv}...")

    lines = []

    # ========== 精度测试摘要 ==========
    lines.append("=" * 80)
    lines.append("Accuracy Test Summary")
    lines.append("=" * 80)

    with open(accuracy_csv, "r") as f:
        reader = csv.DictReader(f)
        accuracy_data = list(reader)

    scenarios = set(row["scenario"] for row in accuracy_data)
    for scenario in scenarios:
        scenario_rows = [r for r in accuracy_data if r["scenario"] == scenario]
        lines.append(f"\nScenario: {scenario}")
        lines.append("-" * 50)
        lines.append(f"{'Method':<18} {'Max Error':>14} {'Rel Error':>14}")
        lines.append("-" * 50)
        for row in scenario_rows:
            lines.append(f"{row['method']:<18} {float(row['max_error']):>14.6e} {float(row['rel_error']):>14.6e}")

    lines.append("")

    # ========== 性能测试摘要 ==========
    lines.append("=" * 80)
    lines.append("Performance Test Summary")
    lines.append("=" * 80)

    with open(performance_csv, "r") as f:
        reader = csv.DictReader(f)
        perf_data = list(reader)

    scenarios = set(row["scenario"] for row in perf_data)
    splits = [2, 3, 4]

    for scenario in scenarios:
        scenario_rows = [r for r in perf_data if r["scenario"] == scenario]
        lines.append(f"\nScenario: {scenario}")
        lines.append("-" * 80)

        header1 = f"{'Size':<8} {'Ref':>8}"
        for s in splits:
            header1 += f"  Ozaki(split{s})"
        header2 = f"{'':<8} {'':>8}"
        for s in splits:
            header2 += f" {'time / speedup':>19}"
        lines.append(header1)
        lines.append(header2)
        lines.append("-" * 80)

        for size in [512, 1024, 2048]:
            ref_rows = [r for r in scenario_rows if r.get("size") == str(size) and not r.get("splits")]
            if ref_rows:
                t_ref = float(ref_rows[0]["ref_time"])
                row = f"{size}x{size:<6} {t_ref:>8.2f}"
                for s in splits:
                    ozaki_rows = [r for r in scenario_rows if r.get("size") == str(size) and r.get("splits") == str(s)]
                    if ozaki_rows:
                        t_ozaki = float(ozaki_rows[0]["ozaki_time"])
                        speedup = float(ozaki_rows[0]["speedup"])
                        row += f" {t_ozaki:>8.2f} / {speedup:>6.2f}x"
                lines.append(row)

    summary = "\n".join(lines)

    with open(output_txt, "w") as f:
        f.write(summary)

    print(f"Summary saved to {output_txt}")
    print(summary)

    return summary


def plot_summary(accuracy_csv="test_accuracy.csv", performance_csv="test_performance.csv", output_png="test_summary.png"):
    """从 CSV 读取数据并绘制图表"""
    import csv

    print(f"\nGenerating plots from {accuracy_csv} and {performance_csv}...")

    # 读取 accuracy 数据
    with open(accuracy_csv, "r") as f:
        reader = csv.DictReader(f)
        accuracy_data = list(reader)

    # 读取 performance 数据
    with open(performance_csv, "r") as f:
        reader = csv.DictReader(f)
        perf_data = list(reader)

    scenarios = sorted(set(row["scenario"] for row in accuracy_data))
    splits = [2, 3, 4]
    sizes = [512, 1024, 2048]

    # 整理 accuracy 数据：每个 scenario 的 naive 精度和 Ozaki 精度
    acc_naive = {}  # scenario -> rel_error
    acc_ozaki = {s: {} for s in splits}  # split -> {scenario -> rel_error}
    for row in accuracy_data:
        scenario = row["scenario"]
        method = row["method"]
        rel_error = float(row["rel_error"])
        if method.startswith("Naive"):
            acc_naive[scenario] = rel_error
        elif method.startswith("Ozaki"):
            s = int(method.split("split")[1].rstrip(")"))
            acc_ozaki[s][scenario] = rel_error

    # 整理 performance 数据：计算 speedup
    perf_speedup = {size: {s: {} for s in splits} for size in sizes}  # size -> split -> {scenario -> speedup}
    for row in perf_data:
        scenario = row["scenario"]
        size = int(row["size"])
        splits_val = row.get("splits")
        if splits_val:
            s = int(splits_val)
            perf_speedup[size][s][scenario] = float(row["speedup"])

    # 创建图形：4 个子图按行排列
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ===== 子图 1: Relative Error =====
    ax = axes[0]
    x = splits
    # 先画 Naive 基准线（黑色虚线，只画 FP16）
    fp16_naive = None
    for scenario in scenarios:
        if "FP16" in scenario and scenario in acc_naive:
            fp16_naive = acc_naive[scenario]
            break
    if fp16_naive is not None:
        ax.axhline(y=fp16_naive, color='black', linestyle='--', linewidth=2, label='Naive FP16')
    # 再画 Ozaki 数据线
    for i, scenario in enumerate(scenarios):
        y = [acc_ozaki[s].get(scenario, None) for s in splits]
        ax.plot(x, y, '.-', label=scenario, linewidth=2, markersize=8)

    ax.set_xticks(splits)
    ax.set_xlabel("Number of Splits")
    ax.set_ylabel("Relative Error")
    ax.set_title("Accuracy Comparison")
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # ===== 子图 2-4: Performance (512/1024/2048) =====
    for idx, size in enumerate(sizes, start=1):
        ax = axes[idx]
        # 先画 speedup=1 基准线（黑色虚线，加粗）
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (speedup=1)')
        # 再画 Ozaki 数据线
        for i, scenario in enumerate(scenarios):
            y = [perf_speedup[size][s].get(scenario, None) for s in splits]
            ax.plot(x, y, '.-', label=scenario, linewidth=2, markersize=8)
        ax.set_xticks(splits)
        ax.set_xlabel("Number of Splits")
        ax.set_ylabel("Speedup")
        ax.set_title(f"Performance ({size}x{size})")
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

    print(f"Plot saved to {output_png}")


if __name__ == "__main__":
    test_accuracy()
    test_performance()
    show_summary()
    plot_summary()
