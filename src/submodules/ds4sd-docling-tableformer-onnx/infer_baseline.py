import argparse
import csv
import os
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor

from pipeline_utils import (
    seed_everything,
    print_env_info,
    iter_images,
    ensure_dir,
    draw_boxes,
    save_json,
    summarize_times,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with HF model")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out", default="results/baseline")
    args = parser.parse_args()

    seed_everything()
    print_env_info()

    out_root = Path(args.out)
    pred_dir = out_root / "predictions"
    overlay_dir = out_root / "overlays"
    ensure_dir(pred_dir)
    ensure_dir(overlay_dir)

    processor = AutoImageProcessor.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    model.eval()

    timings = []
    counts = []

    for path, img in iter_images(args.dataset):
        inputs = processor(images=img, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = (time.perf_counter() - start) * 1000
        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=0.25, target_sizes=target_sizes
        )[0]
        boxes = results["boxes"].tolist()
        labels = [model.config.id2label[int(i)] for i in results["labels"]]
        scores = results["scores"].tolist()
        order = np.argsort(-np.array(scores))
        boxes = [boxes[i] for i in order]
        labels = [labels[i] for i in order]
        scores = [scores[i] for i in order]
        preds = [
            {"id": idx, "bbox": box, "label": lab, "score": float(scr)}
            for idx, (box, lab, scr) in enumerate(zip(boxes, labels, scores))
        ]
        save_json(pred_dir / f"{path.stem}.json", preds)
        draw_boxes(img, boxes, labels, scores, overlay_dir / f"{path.stem}.png")
        timings.append({"filename": path.name, "ms": elapsed})
        counts.append(len(preds))

    with open(out_root / "timings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ms"])
        writer.writeheader()
        writer.writerows(timings)

    times = [t["ms"] for t in timings]
    metrics = {
        "time_ms": summarize_times(times),
        "num_images": len(timings),
        "total_boxes": int(sum(counts)),
    }
    save_json(out_root / "metrics.json", metrics)

    model_size = 0
    cache_dir = Path("models/hf-cache")
    if cache_dir.exists():
        for p in cache_dir.rglob("*"):
            if p.is_file():
                model_size += p.stat().st_size
    save_json(out_root / "model_info.json", {"model_size_bytes": model_size})


if __name__ == "__main__":
    main()
