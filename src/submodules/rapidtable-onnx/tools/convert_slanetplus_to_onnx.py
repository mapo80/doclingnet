#!/usr/bin/env python3
"""
Convert SLANetPlus PaddlePaddle model to ONNX format
"""
import paddle2onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

# Paths
MODEL_DIR = Path("/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models")
PDMODEL_PATH = MODEL_DIR / "inference.pdmodel"
PDIPARAMS_PATH = MODEL_DIR / "inference.pdiparams"
ONNX_OUTPUT_PATH = MODEL_DIR / "slanetplus.onnx"

print("="*70)
print("SLANETPLUS PADDLEPADDLE → ONNX CONVERSION")
print("="*70)
print(f"\nInput files:")
print(f"  Model:  {PDMODEL_PATH}")
print(f"  Params: {PDIPARAMS_PATH}")
print(f"\nOutput:")
print(f"  ONNX:   {ONNX_OUTPUT_PATH}")
print()

# Convert PaddlePaddle model to ONNX
print("-"*70)
print("Converting model...")
print("-"*70)

try:
    onnx_model = paddle2onnx.command.c_paddle_to_onnx(
        model_file=str(PDMODEL_PATH),
        params_file=str(PDIPARAMS_PATH),
        opset_version=13,
        enable_onnx_checker=True,
        enable_auto_update_opset=True,
        verbose=True
    )

    # Save ONNX model
    with open(ONNX_OUTPUT_PATH, "wb") as f:
        f.write(onnx_model)

    print(f"\n✅ Conversion successful!")

    # Get file size
    size_mb = ONNX_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"   Output file: {ONNX_OUTPUT_PATH}")
    print(f"   File size: {size_mb:.2f} MB")

except Exception as e:
    print(f"\n❌ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Validate ONNX model
print("\n" + "-"*70)
print("Validating ONNX model...")
print("-"*70)

try:
    # Load ONNX model
    session = ort.InferenceSession(str(ONNX_OUTPUT_PATH), providers=["CPUExecutionProvider"])

    # Get input/output info
    print(f"\n📊 Model Info:")
    print(f"   Inputs:")
    for input_meta in session.get_inputs():
        print(f"     - {input_meta.name}: {input_meta.shape} ({input_meta.type})")

    print(f"   Outputs:")
    for output_meta in session.get_outputs():
        print(f"     - {output_meta.name}: {output_meta.shape} ({output_meta.type})")

    # Test with dummy input
    print("\n" + "-"*70)
    print("Testing with dummy input...")
    print("-"*70)

    # Create dummy input based on typical SLANetPlus input size
    # SLANetPlus typically expects [batch, channels, height, width]
    # Common sizes are 488x488 or 640x640
    dummy_input = np.random.randn(1, 3, 488, 488).astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input})

    print(f"\n✅ ONNX inference successful!")
    print(f"   Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"   Output {i}: shape={output.shape}, dtype={output.dtype}")

except Exception as e:
    print(f"\n⚠️  ONNX validation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nNote: Validation failed but ONNX file was created.")
    print("      This may be due to incorrect input shape guessing.")
    print("      The model might still work with correct input dimensions.")

print("\n" + "="*70)
print("CONVERSION COMPLETE")
print("="*70)
