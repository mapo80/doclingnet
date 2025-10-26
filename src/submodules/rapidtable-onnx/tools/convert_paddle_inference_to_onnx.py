#!/usr/bin/env python3
"""
Convert PaddlePaddle inference model to ONNX
Using Paddle2ONNX with proper Python API approach
"""
import sys
import os
from pathlib import Path

MODEL_DIR = Path("/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models")
PDMODEL = str(MODEL_DIR / "inference.pdmodel")
PDIPARAMS = str(MODEL_DIR / "inference.pdiparams")
OUTPUT_ONNX = str(MODEL_DIR / "slanetplus_converted.onnx")

print("="*70)
print("PADDLEPADDLE INFERENCE MODEL → ONNX CONVERSION")
print("="*70)
print(f"\nInput files:")
print(f"  Model:  {PDMODEL}")
print(f"  Params: {PDIPARAMS}")
print(f"\nOutput:")
print(f"  ONNX:   {OUTPUT_ONNX}")
print()

# Method 1: Try using subprocess to call paddle2onnx directly
print("-"*70)
print("Attempting conversion using paddle2onnx...")
print("-"*70)

try:
    import subprocess

    # Install paddle2onnx if not present
    print("Installing paddle2onnx...")
    install_cmd = [sys.executable, "-m", "pip", "install", "paddle2onnx==1.3.1", "-q"]
    subprocess.run(install_cmd, check=True, timeout=120)

    print("✅ paddle2onnx installed")

    # Now try to convert
    print("\nConverting model...")

    # Use Python API instead of CLI
    import paddle2onnx

    # Read model files
    with open(PDMODEL, 'rb') as f:
        model_content = f.read()

    with open(PDIPARAMS, 'rb') as f:
        params_content = f.read()

    print(f"  Model size: {len(model_content) / 1024:.1f} KB")
    print(f"  Params size: {len(params_content) / (1024*1024):.1f} MB")

    # Convert using Python API
    print("\nRunning paddle2onnx.export...")
    onnx_model = paddle2onnx.export(
        model_content=model_content,
        params_content=params_content,
        opset_version=13,
        enable_onnx_checker=True,
        enable_auto_update_opset=True,
        deploy_backend="onnxruntime"
    )

    # Save ONNX model
    with open(OUTPUT_ONNX, 'wb') as f:
        f.write(onnx_model)

    print(f"✅ Conversion successful!")

    size_mb = Path(OUTPUT_ONNX).stat().st_size / (1024 * 1024)
    print(f"\nOutput file: {OUTPUT_ONNX}")
    print(f"File size: {size_mb:.2f} MB")

    # Validate with ONNX Runtime
    print("\n" + "-"*70)
    print("Validating ONNX model...")
    print("-"*70)

    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(OUTPUT_ONNX, providers=["CPUExecutionProvider"])

    print(f"✅ ONNX model loaded successfully")
    print(f"\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")

    print(f"\nModel outputs:")
    for out in session.get_outputs():
        print(f"  - {out.name}: {out.shape} ({out.type})")

    # Try a test inference
    print("\n" + "-"*70)
    print("Testing inference with dummy input...")
    print("-"*70)

    # Create dummy input (assuming 488x488 based on SLANetPlus)
    dummy_input = np.random.randn(1, 3, 488, 488).astype(np.float32)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: dummy_input})

    print(f"✅ Inference successful!")
    print(f"   Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"   Output {i}: shape={output.shape}, dtype={output.dtype}, range=[{output.min():.3f}, {output.max():.3f}]")

    print("\n" + "="*70)
    print("✅ CONVERSION AND VALIDATION SUCCESSFUL!")
    print("="*70)

    sys.exit(0)

except subprocess.CalledProcessError as e:
    print(f"❌ Installation failed: {e}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   paddle2onnx may not be compatible with this system")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("❌ CONVERSION FAILED")
print("="*70)
print("\nTroubleshooting:")
print("1. Ensure paddle2onnx is compatible with your Python version")
print("2. Try on Linux system (better compatibility)")
print("3. Check if model files are valid PaddlePaddle inference models")
print("4. Verify PaddlePaddle version compatibility")

sys.exit(1)
