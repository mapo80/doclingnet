import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoImageProcessor, AutoConfig

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
    parser = argparse.ArgumentParser(description="Inference with ONNX model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True, help="name for results subdir")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out-root", default="results")
    args = parser.parse_args()

    seed_everything()
    print_env_info()

    out_root = Path(args.out_root) / args.variant
    pred_dir = out_root / "predictions"
    overlay_dir = out_root / "overlays"
    ensure_dir(pred_dir)
    ensure_dir(overlay_dir)

    processor = AutoImageProcessor.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    config = AutoConfig.from_pretrained("ds4sd/docling-layout-heron", cache_dir="models/hf-cache")
    id2label = config.id2label
    session = ort.InferenceSession(
        args.model, providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in session.get_outputs()[:2]]

    timings = []
    counts = []

    for path, img in iter_images(args.dataset):
        inputs = processor(images=img, return_tensors="pt")
        start = time.perf_counter()
        logits, boxes = session.run(output_names, {"pixel_values": inputs["pixel_values"].numpy()})
        elapsed = (time.perf_counter() - start) * 1000
        ort_struct = type(
            "OrtOut",
            (),
            {
                "logits": torch.from_numpy(logits),
                "pred_boxes": torch.from_numpy(boxes),
            },
        )()
        results = processor.post_process_object_detection(
            ort_struct, threshold=0.25, target_sizes=[img.size[::-1]]
        )[0]
        boxes_l = results["boxes"].tolist()
        labels = [id2label[int(i)] for i in results["labels"]]
        scores = results["scores"].tolist()
        order = np.argsort(-np.array(scores))
        boxes_l = [boxes_l[i] for i in order]
        labels = [labels[i] for i in order]
        scores = [scores[i] for i in order]
        preds = [
            {"id": idx, "bbox": box, "label": lab, "score": float(scr)}
            for idx, (box, lab, scr) in enumerate(zip(boxes_l, labels, scores))
        ]
        save_json(pred_dir / f"{path.stem}.json", preds)
        draw_boxes(img, boxes_l, labels, scores, overlay_dir / f"{path.stem}.png")
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

    model_size = Path(args.model).stat().st_size
    save_json(out_root / "model_info.json", {"model_size_bytes": model_size})


if __name__ == "__main__":
    main()
