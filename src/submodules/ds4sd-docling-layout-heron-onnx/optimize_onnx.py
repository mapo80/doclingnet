import argparse
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic
from transformers import AutoModelForObjectDetection, AutoImageProcessor

from pipeline_utils import seed_everything, print_env_info, iter_images
from convert_to_onnx import parity_check


def optimize(input_path: Path, output_path: Path) -> None:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path.as_posix()
    ort.InferenceSession(input_path.as_posix(), sess_options, providers=["CPUExecutionProvider"])
    size_mb = output_path.stat().st_size / 1024 ** 2
    print(f"Saved optimized model to {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize ONNX model")
    parser.add_argument("--input", default="models/heron-converted.onnx")
    parser.add_argument("--output", default="models/heron-optimized.onnx")
    parser.add_argument("--quant-output", default=None)
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()

    seed_everything()
    print_env_info()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimize(input_path, output_path)

    if args.quant_output:
        q_path = Path(args.quant_output)
        quantize_dynamic(output_path.as_posix(), q_path.as_posix())
        q_size = q_path.stat().st_size / 1024 ** 2
        print(f"Saved quantized model to {q_path} ({q_size:.1f} MB)")

    processor = AutoImageProcessor.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "ds4sd/docling-layout-heron", cache_dir="models/hf-cache"
    )
    model.eval()
    sess = ort.InferenceSession(output_path.as_posix(), providers=["CPUExecutionProvider"])
    images = list(iter_images(args.dataset))[:1]
    parity_check(model, sess, processor, images)


if __name__ == "__main__":
    main()
