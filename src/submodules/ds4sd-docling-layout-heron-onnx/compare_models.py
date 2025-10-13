import os
import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForObjectDetection


def generate_input(batch_size: int = 1) -> torch.Tensor:
    """Generate a random image tensor in the expected format."""
    return torch.randn(batch_size, 3, 640, 640)


def run_torch(model, x: torch.Tensor, runs: int = 5):
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(runs):
            out = model(x)[0:2]
        elapsed = (time.perf_counter() - start) / runs
    return out, elapsed


def run_onnx(session: ort.InferenceSession, x: torch.Tensor, runs: int = 5):
    start = time.perf_counter()
    for _ in range(runs):
        out = session.run(None, {"pixel_values": x.numpy()})
    elapsed = (time.perf_counter() - start) / runs
    return out, elapsed


def main():
    x = generate_input()
    model = AutoModelForObjectDetection.from_pretrained("ds4sd/docling-layout-heron")
    model.eval()
    model.config.return_dict = False
    pt_out, pt_time = run_torch(model, x)
    print(f"PyTorch average time: {pt_time*1000:.2f} ms")

    # Baseline ONNX
    sess = ort.InferenceSession("docling_layout_heron.onnx", providers=["CPUExecutionProvider"])
    onnx_out, onnx_time = run_onnx(sess, x)
    base_size = os.path.getsize("docling_layout_heron.onnx") / 1024**2
    print(f"ONNX size: {base_size:.1f} MB, time: {onnx_time*1000:.2f} ms")
    print("Max abs diff logits vs PyTorch:", np.max(np.abs(pt_out[0].numpy() - onnx_out[0])))
    print("Max abs diff boxes vs PyTorch:", np.max(np.abs(pt_out[1].numpy() - onnx_out[1])))

    def report_variant(name: str, path: str):
        if not os.path.exists(path):
            print(f"{name} model not found: {path}")
            return
        try:
            sess_v = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"{name} load failed: {e}")
            return
        out_v, t_v = run_onnx(sess_v, x)
        size_v = os.path.getsize(path) / 1024**2
        rel_logits = np.max(np.abs(onnx_out[0] - out_v[0])) / np.maximum(
            np.max(np.abs(onnx_out[0])), 1e-9
        )
        rel_boxes = np.max(np.abs(onnx_out[1] - out_v[1])) / np.maximum(
            np.max(np.abs(onnx_out[1])), 1e-9
        )
        print(
            f"{name} size: {size_v:.1f} MB, time: {t_v*1000:.2f} ms, "
            f"max rel diff logits: {rel_logits*100:.2f}%, boxes: {rel_boxes*100:.2f}%"
        )

    report_variant("Dynamic Quantized ONNX", "docling_layout_heron_quant.onnx")
    report_variant("Float16 ONNX", "docling_layout_heron_fp16.onnx")
    report_variant("Static Quantized ONNX", "docling_layout_heron_static.onnx")


if __name__ == "__main__":
    main()
