#!/usr/bin/env python3
import json
from pathlib import Path

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def main():
    repo_root = Path(__file__).resolve().parents[2]
    baseline = load_json(repo_root / "results" / "layout_phase1" / "baseline_metrics.json")
    benchmark = load_json(repo_root / "results" / "layout_phase1" / "bitmap_cache_benchmark.json")

    python_mean = baseline["python_mean_ms"]
    dotnet_before = baseline["dotnet_mean_ms_before"]
    dotnet_after = dotnet_before - benchmark["DeltaMs"]

    delta_before = dotnet_before - python_mean
    delta_after = dotnet_after - python_mean
    ratio_before = dotnet_before / python_mean
    ratio_after = dotnet_after / python_mean

    summary = {
        "python_mean_ms": python_mean,
        "dotnet_mean_ms_before": dotnet_before,
        "dotnet_mean_ms_after": dotnet_after,
        "delta_ms_before": delta_before,
        "delta_ms_after": delta_after,
        "ratio_before": ratio_before,
        "ratio_after": ratio_after,
        "bitmap_decode_savings_ms": benchmark["DeltaMs"],
        "benchmark_iterations": benchmark["Iterations"],
        "benchmark_warmup": benchmark["Warmup"],
        "benchmark_baseline_mean_ms": benchmark["BaselineMeanMs"],
        "benchmark_optimized_mean_ms": benchmark["OptimizedMeanMs"],
        "image_path": benchmark["ImagePath"],
        "image_width": benchmark["Width"],
        "image_height": benchmark["Height"],
    }

    output_path = repo_root / "results" / "layout_phase1" / "phase1_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
