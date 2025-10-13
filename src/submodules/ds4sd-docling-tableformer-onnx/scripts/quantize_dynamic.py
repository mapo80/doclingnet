#!/usr/bin/env python3
"""Apply targeted dynamic INT8 quantization to the optimized ONNX model."""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import time

from onnx import TensorProto
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from pipeline_utils import seed_everything, print_env_info


def quick_run(model_path: Path) -> None:
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    # warmup
    sess.run(None, {"pixel_values": dummy})
    # one measured run
    t0 = time.time()
    sess.run(None, {"pixel_values": dummy})
    dt = (time.time() - t0) * 1000
    print(f"Dummy inference ok ({dt:.1f} ms)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization")
    parser.add_argument("--input", default="models/heron-optimized.onnx")
    parser.add_argument("--output", default="models/heron-int8-dynamic.onnx")
    args = parser.parse_args()

    seed_everything()
    quantize_dynamic(
        args.input,
        args.output,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"DefaultTensorType": TensorProto.FLOAT},
    )
    size_mb = Path(args.output).stat().st_size / 1024 ** 2
    print(f"Saved quantized model to {args.output} ({size_mb:.1f} MB)")

    quick_run(Path(args.output))


if __name__ == "__main__":
    main()
