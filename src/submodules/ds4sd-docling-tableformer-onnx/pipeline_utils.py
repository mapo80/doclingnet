import json
import os
import platform
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
except Exception:  # pragma: no cover - torch may not be installed when importing
    torch = None  # type: ignore

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore


SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def print_env_info() -> None:
    """Print basic environment information."""
    python_version = platform.python_version()
    torch_version = torch.__version__ if torch is not None else "not installed"
    ort_version = ort.__version__ if ort is not None else "not installed"
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    os_name = platform.platform()
    print(
        f"Python {python_version}\n"
        f"torch {torch_version}\n"
        f"onnxruntime {ort_version}\n"
        f"device {device}\n"
        f"OS {os_name}"
    )


def iter_images(folder: str) -> Iterable[Tuple[Path, Image.Image]]:
    """Yield (path, image) pairs from a folder for common formats."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in exts:
            yield p, Image.open(p).convert("RGB")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def draw_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    path: Path,
) -> None:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for box, label, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        txt = f"{label}:{score:.2f}"
        draw.text((x0, y0), txt, fill="red", font=font)
    img.save(path)


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU of two boxes [x0, y0, x1, y1]."""
    ax0, ay0, ax1, ay1 = box1
    bx0, by0, bx1, by1 = box2
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def summarize_times(times_ms: List[float]) -> dict:
    arr = np.array(times_ms)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }
