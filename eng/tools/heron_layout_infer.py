#!/usr/bin/env python3
"""Run the Heron layout ONNX model and emit bounding boxes in page coordinates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_INPUT = 640
SCORE_THRESHOLD = 0.20


def _letterbox(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    original_width, original_height = image.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Image has invalid dimensions")

    if original_width == MODEL_INPUT and original_height == MODEL_INPUT:
        metadata = {
            "originalWidth": original_width,
            "originalHeight": original_height,
            "scaledWidth": original_width,
            "scaledHeight": original_height,
            "scale": 1.0,
            "offsetX": 0.0,
            "offsetY": 0.0,
        }
        return image.convert("RGB"), metadata

    scale = min(MODEL_INPUT / original_width, MODEL_INPUT / original_height)
    if scale <= 0:
        raise ValueError("Computed scale is not positive")

    scaled_width = max(1, min(MODEL_INPUT, int(round(original_width * scale))))
    scaled_height = max(1, min(MODEL_INPUT, int(round(original_height * scale))))
    resized = image.resize((scaled_width, scaled_height), Image.BICUBIC)

    offset_x = (MODEL_INPUT - scaled_width) / 2.0
    offset_y = (MODEL_INPUT - scaled_height) / 2.0

    pad_left = int(math.floor(offset_x))
    pad_top = int(math.floor(offset_y))
    pad_right = MODEL_INPUT - pad_left - scaled_width
    pad_bottom = MODEL_INPUT - pad_top - scaled_height

    letterboxed = Image.new("RGB", (MODEL_INPUT, MODEL_INPUT), color=(0, 0, 0))
    letterboxed.paste(resized, (pad_left, pad_top))

    metadata = {
        "originalWidth": original_width,
        "originalHeight": original_height,
        "scaledWidth": scaled_width,
        "scaledHeight": scaled_height,
        "scale": scale,
        "offsetX": float(pad_left),
        "offsetY": float(pad_top),
    }
    return letterboxed, metadata


def _prepare_tensor(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def _decode(logits: np.ndarray, boxes: np.ndarray) -> List[Dict[str, Any]]:
    if logits.ndim != 3 or boxes.ndim != 3:
        return []

    queries = logits.shape[1]
    classes = logits.shape[2]
    detections: List[Dict[str, Any]] = []

    for i in range(queries):
        max_score = float(np.max(logits[0, i, 1:]))
        exp_scores = np.exp(logits[0, i, :] - max_score)
        sum_scores = float(np.sum(exp_scores))
        probabilities = exp_scores / (sum_scores if sum_scores > 0 else 1.0)

        best_class = int(np.argmax(probabilities[1:])) + 1
        best_prob = float(probabilities[best_class])
        if best_prob < SCORE_THRESHOLD:
            continue

        cx = float(boxes[0, i, 0]) * MODEL_INPUT
        cy = float(boxes[0, i, 1]) * MODEL_INPUT
        width = float(boxes[0, i, 2]) * MODEL_INPUT
        height = float(boxes[0, i, 3]) * MODEL_INPUT

        left = max(0.0, cx - width / 2.0)
        top = max(0.0, cy - height / 2.0)
        right = min(MODEL_INPUT, left + width)
        bottom = min(MODEL_INPUT, top + height)
        clamped_width = max(0.0, right - left)
        clamped_height = max(0.0, bottom - top)
        if clamped_width <= 1.0 or clamped_height <= 1.0:
            continue

        detections.append(
            {
                "classIndex": best_class,
                "confidence": best_prob,
                "letterboxed": {
                    "left": left,
                    "top": top,
                    "width": clamped_width,
                    "height": clamped_height,
                },
            }
        )

    return detections


def _reproject(detections: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    scale = float(metadata["scale"])
    offset_x = float(metadata["offsetX"])
    offset_y = float(metadata["offsetY"])
    scaled_width = float(metadata["scaledWidth"])
    scaled_height = float(metadata["scaledHeight"])
    original_width = float(metadata["originalWidth"])
    original_height = float(metadata["originalHeight"])

    content_max_x = offset_x + scaled_width
    content_max_y = offset_y + scaled_height

    for det in detections:
        lb = det["letterboxed"]
        left = min(max(lb["left"], offset_x), content_max_x) - offset_x
        top = min(max(lb["top"], offset_y), content_max_y) - offset_y
        right = min(max(lb["left"] + lb["width"], offset_x), content_max_x) - offset_x
        bottom = min(max(lb["top"] + lb["height"], offset_y), content_max_y) - offset_y

        orig_left = max(0.0, min(original_width, left / scale))
        orig_top = max(0.0, min(original_height, top / scale))
        orig_right = max(0.0, min(original_width, right / scale))
        orig_bottom = max(0.0, min(original_height, bottom / scale))

        det["page"] = {
            "left": orig_left,
            "top": orig_top,
            "width": max(0.0, orig_right - orig_left),
            "height": max(0.0, orig_bottom - orig_top),
        }
        det["label"] = "Text"


def run_inference(image_path: Path, model_path: Path) -> Dict[str, Any]:
    image = Image.open(image_path)
    letterboxed, metadata = _letterbox(image)
    tensor = _prepare_tensor(letterboxed)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = {"pixel_values": tensor}
    outputs = session.run(None, inputs)

    logits = None
    boxes = None
    for name, value in zip(session.get_outputs(), outputs):
        if name.name == "logits":
            logits = value
        elif name.name == "pred_boxes":
            boxes = value

    if logits is None or boxes is None:
        raise RuntimeError("Model did not produce logits/pred_boxes outputs")

    detections = _decode(np.asarray(logits), np.asarray(boxes))
    _reproject(detections, metadata)

    return {
        "image": str(image_path),
        "model": str(model_path),
        "metadata": metadata,
        "detections": detections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the source image")
    parser.add_argument("model", type=Path, help="Path to the Heron ONNX model")
    parser.add_argument("output", type=Path, help="Destination JSON file")
    parser.add_argument("--letterbox", type=Path, help="Optional path to save the normalised 640x640 PNG")
    args = parser.parse_args()

    result = run_inference(args.image, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    if args.letterbox:
        letterboxed, _ = _letterbox(Image.open(args.image))
        args.letterbox.parent.mkdir(parents=True, exist_ok=True)
        letterboxed.save(args.letterbox)


if __name__ == "__main__":
    main()
