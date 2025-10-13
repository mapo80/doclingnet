#!/usr/bin/env python3
"""Benchmark ONNX Runtime OpenVINO model on a set of images or synthetic data."""
import argparse
import csv
import json
import logging
import hashlib
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import onnxruntime as ort


def load_image(path: Path, target_h: int, target_w: int, dtype=np.float32) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    scale = min(target_w / img.width, target_h / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img = img.resize((new_w, new_h))
    canvas = Image.new("RGB", (target_w, target_h))
    canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    arr = np.asarray(canvas).transpose(2, 0, 1)
    if dtype == np.float32:
        arr = arr.astype("float32") / 255.0
    else:
        arr = arr.astype("int8")
    return arr[np.newaxis, :]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--target-h", type=int)
    parser.add_argument("--target-w", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs-per-image", type=int, default=1)
    parser.add_argument("--threads-intra", type=int, default=0)
    parser.add_argument("--threads-inter", type=int, default=1)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--device", default="CPU_FP32")
    args = parser.parse_args()

    if not args.images and not args.synthetic:
        parser.error("either --images or --synthetic required")

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    variant = args.variant_name + "-ov"
    ep = "OpenVINO"
    device = args.device
    providers = [("OpenVINOExecutionProvider", {"device_type": args.device})]
    run_dir = Path(args.output) / variant / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = logging.getLogger("bench")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir / "logs.txt")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.info("starting benchmark")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL if args.sequential else ort.ExecutionMode.ORT_PARALLEL
    if args.threads_intra > 0:
        so.intra_op_num_threads = args.threads_intra
    if args.threads_inter > 0:
        so.inter_op_num_threads = args.threads_inter

    session = ort.InferenceSession(args.model, sess_options=so, providers=providers)
    input_meta = session.get_inputs()[0]
    shape = list(input_meta.shape)
    h = shape[2] if len(shape) >= 4 and shape[2] not in (None, -1) else args.target_h
    w = shape[3] if len(shape) >= 4 and shape[3] not in (None, -1) else args.target_w
    if h is None or w is None:
        raise ValueError("target-h and target-w required when model has dynamic spatial dimensions")
    input_name = input_meta.name
    dtype = np.float32 if input_meta.type == "tensor(float)" else np.int8

    timings: List[float] = []
    csv_path = run_dir / "timings.csv"
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "ms"])
        if args.synthetic:
            images = ["synthetic"]
        else:
            img_dir = Path(args.images)
            images = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
                images.extend(sorted(img_dir.glob(ext)))
        for img in images:
            if args.synthetic:
                tensor = np.random.rand(1, 3, h, w).astype("float32") if dtype == np.float32 else np.random.randint(-128, 128, size=(1,3,h,w), dtype=np.int8)
                fname = "synthetic"
            else:
                tensor = load_image(Path(img), h, w, dtype)
                fname = Path(img).name
            for _ in range(args.warmup):
                session.run(None, {input_name: tensor})
            start = time.perf_counter()
            for _ in range(args.runs_per_image):
                session.run(None, {input_name: tensor})
            dt = (time.perf_counter() - start) * 1000 / max(1, args.runs_per_image)
            writer.writerow([fname, f"{dt:.3f}"])
            timings.append(dt)
            logger.info("%s %.3f ms", fname, dt)

    summary = {
        "count": len(timings),
        "mean_ms": statistics.mean(timings) if timings else 0.0,
        "median_ms": statistics.median(timings) if timings else 0.0,
        "p95_ms": (np.percentile(timings, 95).item() if timings else 0.0),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    precision = "int8" if "int8" in Path(args.model).name.lower() or "int8" in args.variant_name.lower() else "fp32"
    model_info = {
        "model_path": str(Path(args.model).resolve()),
        "model_size_bytes": Path(args.model).stat().st_size,
        "ep": ep,
        "device": device,
        "precision": precision,
    }
    (run_dir / "model_info.json").write_text(json.dumps(model_info, indent=2))

    env = {
        "python": sys.version.split()[0],
        "dotnet": None,
        "ort": ort.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    logging.shutdown()
    manifest = {}
    for p in sorted(run_dir.glob("*")):
        if p.name == "manifest.json":
            continue
        manifest[p.name] = sha256_file(p)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
