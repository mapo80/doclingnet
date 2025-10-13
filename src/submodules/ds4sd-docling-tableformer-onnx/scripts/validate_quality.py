#!/usr/bin/env python3
"""Validate quality of INT8 model against baseline and produce KPIs."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import List, Tuple

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoImageProcessor, AutoConfig

from pipeline_utils import (
    iter_images,
    ensure_dir,
    save_json,
    seed_everything,
    print_env_info,
    box_iou,
    summarize_times,
)


def run_model(model_path: Path, dataset: str, out_dir: Path, processor, id2label) -> Tuple[List[Tuple[str, float]], Path]:
    """Run ORT model and save predictions and timings."""
    ensure_dir(out_dir / "predictions")
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    timings = []
    for path, img in iter_images(dataset):
        inputs = processor(images=img, return_tensors="np")
        t0 = time.time()
        out = sess.run(None, {"pixel_values": inputs["pixel_values"]})
        dt = (time.time() - t0) * 1000
        timings.append((path.name, dt))
        struct = type(
            "OrtOut",
            (),
            {"logits": torch.from_numpy(out[0]), "pred_boxes": torch.from_numpy(out[1])},
        )()
        res = processor.post_process_object_detection(
            struct, threshold=0.25, target_sizes=[img.size[::-1]]
        )[0]
        order = np.argsort(-res["scores"])
        preds = []
        for i, idx in enumerate(order):
            box = res["boxes"][idx].tolist()
            preds.append(
                {
                    "id": i,
                    "bbox": [float(v) for v in box],
                    "label": id2label[int(res["labels"][idx])],
                    "score": float(res["scores"][idx]),
                }
            )
        save_json(out_dir / "predictions" / f"{path.stem}.json", preds)
    with open(out_dir / "timings.csv", "w") as f:
        for name, ms in timings:
            f.write(f"{name},{ms:.3f}\n")
    info = {"model_size_bytes": model_path.stat().st_size}
    save_json(out_dir / "model_info.json", info)
    return timings, out_dir


def load_preds(folder: Path, name: str):
    with open(folder / f"{name}.json") as f:
        return json.load(f)
def match_iou(base: List[dict], var: List[dict]) -> List[float]:
    ious = []
    for label in {b["label"] for b in base} | {v["label"] for v in var}:
        b_boxes = [b for b in base if b["label"] == label]
        v_boxes = [v for v in var if v["label"] == label]
        used = [False] * len(v_boxes)
        for b in b_boxes:
            best_iou = 0.0
            best_j = -1
            for j, v in enumerate(v_boxes):
                if used[j]:
                    continue
                iou = box_iou(b["bbox"], v["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= 0.1 and best_j >= 0:
                used[best_j] = True
                ious.append(best_iou)
    return ious

def quality_metrics(baseline_dir: Path, var_dir: Path) -> Tuple[float, float, float]:
    ious_all = []
    delta_boxes = []
    for pred_file in baseline_dir.glob("*.json"):
        name = pred_file.name
        base = load_preds(baseline_dir, name[:-5])
        var = load_preds(var_dir, name[:-5]) if (var_dir / name).exists() else []
        ious = match_iou(base, var)
        ious_all.extend(ious)
        delta = (len(var) - len(base)) / max(1, len(base)) * 100.0
        delta_boxes.append(delta)
    mean_iou = float(np.mean(ious_all)) if ious_all else 1.0
    iou50 = float(np.mean([i >= 0.5 for i in ious_all])) if ious_all else 1.0
    delta_box = float(np.mean(delta_boxes)) if delta_boxes else 0.0
    return mean_iou, iou50, delta_box

def parse_timings(path: Path) -> Tuple[float, float]:
    vals = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2 or parts[1] == "ms":
                continue
            vals.append(float(parts[1]))
    stats = summarize_times(vals)
    return stats["mean"], stats["p95"]


def update_compare(rows: List[dict], out_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_markdown(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate INT8 model quality")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    seed_everything()
    print_env_info()

    processor = AutoImageProcessor.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    config = AutoConfig.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    id2label = config.id2label

    base_dir = Path("results/baseline/predictions")
    opt_dir = Path("results/onnx-optimized/predictions")
    int8_dir = Path("results/onnx-int8-dynamic")

    # run int8 model
    timings_int8, _ = run_model(
        Path("models/heron-int8-dynamic.onnx"), args.dataset, int8_dir, processor, id2label
    )

    mean_iou_opt, iou50_opt, dbox_opt = quality_metrics(base_dir, opt_dir)
    mean_iou_int8, iou50_int8, dbox_int8 = quality_metrics(base_dir, int8_dir / "predictions")

    mean_opt, p95_opt = parse_timings(Path("results/onnx-optimized/timings.csv"))
    mean_int8, p95_int8 = parse_timings(int8_dir / "timings.csv")

    size_opt = Path("results/onnx-optimized/model_info.json").stat().st_size
    size_int8 = (int8_dir / "model_info.json").stat().st_size
    with open("results/onnx-optimized/model_info.json") as f:
        size_opt = json.load(f)["model_size_bytes"] / 1024 ** 2
    with open(int8_dir / "model_info.json") as f:
        size_int8 = json.load(f)["model_size_bytes"] / 1024 ** 2
    with open("results/baseline/model_info.json") as f:
        size_base = json.load(f)["model_size_bytes"] / 1024 ** 2
    mean_base, p95_base = parse_timings(Path("results/baseline/timings.csv"))

    rows = [
        {
            "variant": "baseline",
            "size(MB)": f"{size_base:.1f}",
            "mean ms": f"{mean_base:.2f}",
            "p95 ms": f"{p95_base:.2f}",
            "IoU mean": "1.00",
            "IoU@0.5": "1.00",
            "Δ box %": "0.0",
        },
        {
            "variant": "onnx-optimized",
            "size(MB)": f"{size_opt:.1f}",
            "mean ms": f"{mean_opt:.2f}",
            "p95 ms": f"{p95_opt:.2f}",
            "IoU mean": f"{mean_iou_opt:.3f}",
            "IoU@0.5": f"{iou50_opt:.3f}",
            "Δ box %": f"{dbox_opt:.2f}",
        },
        {
            "variant": "onnx-int8-dynamic",
            "size(MB)": f"{size_int8:.1f}",
            "mean ms": f"{mean_int8:.2f}",
            "p95 ms": f"{p95_int8:.2f}",
            "IoU mean": f"{mean_iou_int8:.3f}",
            "IoU@0.5": f"{iou50_int8:.3f}",
            "Δ box %": f"{dbox_int8:.2f}",
        },
    ]
    update_compare(rows, Path("results/COMPARE.md"))

    # acceptance check
    if not (
        iou50_int8 >= 0.995
        and mean_iou_int8 >= 0.99
        and abs(dbox_int8) <= 1.0
        and (
            (mean_int8 <= mean_opt * 0.9 and p95_int8 <= p95_opt * 0.9)
            or size_int8 <= 50.0
        )
    ):
        raise RuntimeError("Quality or performance thresholds not met")

    print(
        f"INT8 model metrics: meanIoU {mean_iou_int8:.3f}, IoU@0.5 {iou50_int8:.3f}, Δbox {dbox_int8:.2f}%"
    )


if __name__ == "__main__":
    main()
