import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from pipeline_utils import box_iou


def load_predictions(folder: Path) -> Dict[str, List[dict]]:
    preds = {}
    for p in folder.glob("*.json"):
        with open(p) as f:
            preds[p.stem] = json.load(f)
    return preds


def load_timings(csv_path: Path) -> List[float]:
    times = []
    with open(csv_path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                times.append(float(parts[1]))
    return times


def compare_variant(baseline: Dict[str, List[dict]], variant: Dict[str, List[dict]]):
    ious = []
    label_eq = []
    score_diffs = []
    delta_boxes = []
    for img, base_preds in baseline.items():
        var_preds = variant.get(img, [])
        n = min(len(base_preds), len(var_preds))
        for i in range(n):
            ious.append(box_iou(base_preds[i]["bbox"], var_preds[i]["bbox"]))
            label_eq.append(base_preds[i]["label"] == var_preds[i]["label"])
            score_diffs.append(abs(base_preds[i]["score"] - var_preds[i]["score"]))
        delta_boxes.append(len(var_preds) - len(base_preds))
    return {
        "mean_iou": float(np.mean(ious)) if ious else 1.0,
        "iou@0.5": float(np.mean([i >= 0.5 for i in ious])) if ious else 1.0,
        "iou@0.75": float(np.mean([i >= 0.75 for i in ious])) if ious else 1.0,
        "label_agreement": float(np.mean(label_eq)) if label_eq else 1.0,
        "score_delta": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "delta_boxes": float(np.mean(delta_boxes)) if delta_boxes else 0.0,
        "ious": ious,
    }


def report_variant(name: str, variant_dir: Path, metrics: dict, times: List[float]) -> None:
    md = variant_dir / "REPORT.md"
    with open(md, "w") as f:
        f.write(f"# {name}\n\n")
        f.write("|metric|value|\n|---|---:|\n")
        f.write(f"|mean IoU|{metrics['mean_iou']:.3f}|\n")
        f.write(f"|IoU@0.5|{metrics['iou@0.5']:.3f}|\n")
        f.write(f"|IoU@0.75|{metrics['iou@0.75']:.3f}|\n")
        f.write(f"|label agreement|{metrics['label_agreement']:.3f}|\n")
        f.write(f"|Δ boxes|{metrics['delta_boxes']:.3f}|\n")
        f.write(f"|Δ score|{metrics['score_delta']:.3f}|\n")
        if times:
            f.write(f"|mean time (ms)|{np.mean(times):.2f}|\n")
            f.write(f"|p95 time (ms)|{np.percentile(times,95):.2f}|\n")
        size_info = variant_dir / "model_info.json"
        if size_info.exists():
            with open(size_info) as sf:
                size = json.load(sf)["model_size_bytes"] / 1024 ** 2
                f.write(f"|model size (MB)|{size:.1f}|\n")
        overlay_files = sorted((variant_dir / "overlays").glob("*.png"))[:3]
        for ov in overlay_files:
            f.write(f"\n![{ov.stem}](overlays/{ov.name})\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    res = Path(args.results)
    baseline_preds = load_predictions(res / "baseline" / "predictions")
    baseline_times = load_timings(res / "baseline" / "timings.csv")

    variants = {
        "baseline": (baseline_preds, baseline_times),
    }
    for d in res.iterdir():
        if d.is_dir() and d.name != "baseline" and (d / "predictions").exists():
            preds = load_predictions(d / "predictions")
            times = load_timings(d / "timings.csv")
            variants[d.name] = (preds, times)

    comp_table = []
    iou_plots = {}

    for name, (preds, times) in variants.items():
        if name == "baseline":
            metrics = {
                "mean_iou": 1.0,
                "iou@0.5": 1.0,
                "iou@0.75": 1.0,
                "label_agreement": 1.0,
                "delta_boxes": 0.0,
                "score_delta": 0.0,
                "ious": [1.0] * sum(len(v) for v in baseline_preds.values()),
            }
        else:
            metrics = compare_variant(baseline_preds, preds)
        report_variant(name, res / name, metrics, times)
        size_mb = 0.0
        info = res / name / "model_info.json"
        if info.exists():
            with open(info) as f:
                size_mb = json.load(f)["model_size_bytes"] / 1024 ** 2
        comp_table.append(
            [
                name,
                metrics["mean_iou"],
                np.mean(times) if times else 0.0,
                np.percentile(times, 95) if times else 0.0,
                size_mb,
            ]
        )
        iou_plots[name] = metrics["ious"]

    with open(res / "COMPARE.md", "w") as f:
        f.write("# Comparison\n\n")
        f.write("|variant|mean IoU|mean time (ms)|p95 time (ms)|model size (MB)|\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in comp_table:
            f.write(
                f"|{row[0]}|{row[1]:.3f}|{row[2]:.2f}|{row[3]:.2f}|{row[4]:.1f}|\n"
            )

    plt.figure()
    for name, (_, times) in variants.items():
        plt.hist(times, bins=10, alpha=0.5, label=name)
    plt.legend()
    plt.xlabel("ms")
    plt.ylabel("count")
    plt.title("Inference time distribution")
    plt.savefig(res / "times.png")
    plt.close()

    plt.figure()
    for name, ious in iou_plots.items():
        if name == "baseline":
            continue
        plt.hist(ious, bins=10, alpha=0.5, label=name)
    plt.legend()
    plt.xlabel("IoU")
    plt.ylabel("count")
    plt.title("IoU vs baseline")
    plt.savefig(res / "iou.png")
    plt.close()

    run_dir = res / f"run-{np.datetime64('now', 's').astype(str).replace('-', '').replace(':', '').replace('T', '-') }"
    run_dir.mkdir(exist_ok=True)
    for name in variants.keys():
        os.symlink(os.path.abspath(res / name), run_dir / name)
    import shutil

    shutil.copy(res / "COMPARE.md", run_dir / "COMPARE.md")
    for name in variants.keys():
        rpt = res / name / "REPORT.md"
        if rpt.exists():
            shutil.copy(rpt, run_dir / f"{name}-REPORT.md")


if __name__ == "__main__":
    main()
