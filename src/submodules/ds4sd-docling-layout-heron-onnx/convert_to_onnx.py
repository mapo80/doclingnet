import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from transformers import AutoModelForObjectDetection, AutoImageProcessor

from pipeline_utils import seed_everything, print_env_info, iter_images, box_iou


def export(model, onnx_path: Path) -> None:
    dummy = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path.as_posix(),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=17,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
    )
    size_mb = onnx_path.stat().st_size / 1024 ** 2
    print(f"Saved ONNX model to {onnx_path} ({size_mb:.1f} MB)")


def parity_check(model, session: ort.InferenceSession, processor, images) -> None:
    diffs = []
    ious = []
    label_ok = True
    score_ok = True
    for path, img in images:
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            pt_out = model(**inputs)
        ort_out = session.run(None, {"pixel_values": inputs["pixel_values"].numpy()})
        pt_res = processor.post_process_object_detection(
            pt_out, threshold=0.25, target_sizes=[img.size[::-1]]
        )[0]
        ort_struct = type(
            "OrtOut",
            (),
            {
                "logits": torch.from_numpy(ort_out[0]),
                "pred_boxes": torch.from_numpy(ort_out[1]),
            },
        )()
        ort_res = processor.post_process_object_detection(
            ort_struct, threshold=0.25, target_sizes=[img.size[::-1]]
        )[0]
        # sort by score
        order_pt = np.argsort(-pt_res["scores"].numpy())
        order_ort = np.argsort(-ort_res["scores"].numpy())
        pt_boxes = pt_res["boxes"].numpy()[order_pt]
        ort_boxes = ort_res["boxes"].numpy()[order_ort]
        pt_labels = pt_res["labels"].numpy()[order_pt]
        ort_labels = ort_res["labels"].numpy()[order_ort]
        pt_scores = pt_res["scores"].numpy()[order_pt]
        ort_scores = ort_res["scores"].numpy()[order_ort]
        n = min(len(pt_boxes), len(ort_boxes))
        for i in range(n):
            diffs.append(np.max(np.abs(pt_boxes[i] - ort_boxes[i])))
            ious.append(box_iou(pt_boxes[i].tolist(), ort_boxes[i].tolist()))
            if pt_labels[i] != ort_labels[i]:
                label_ok = False
            if abs(float(pt_scores[i]) - float(ort_scores[i])) > 0.01:
                score_ok = False
    med_diff = float(np.median(diffs)) if diffs else 0.0
    iou50 = float(np.mean([i >= 0.5 for i in ious])) if ious else 1.0
    if med_diff >= 2 or iou50 <= 0.98 or not label_ok or not score_ok:
        raise RuntimeError(
            f"Parity check failed: med diff {med_diff:.2f}, IoU@0.5 {iou50:.3f}, labels {label_ok}, scores {score_ok}"
        )
    print(
        f"Parity check ok: median bbox diff {med_diff:.2f}px, IoU@0.5 {iou50:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Heron model to ONNX")
    parser.add_argument("--output", default="models/heron-converted.onnx")
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()

    seed_everything()
    print_env_info()

    processor = AutoImageProcessor.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export(model, out_path)

    sess = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
    images = list(iter_images(args.dataset))[:1]
    parity_check(model, sess, processor, images)


if __name__ == "__main__":
    main()
