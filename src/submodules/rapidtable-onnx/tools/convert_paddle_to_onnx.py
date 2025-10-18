#!/usr/bin/env python3
"""
Convert SLANetPlus PaddlePaddle model to ONNX using custom approach
We'll use PaddlePaddle's built-in export functionality
"""
import sys
import os
os.environ['PADDLE_SKIP_INIT_CHECK'] = '1'

# Try direct conversion using subprocess and paddle tools
import subprocess
from pathlib import Path

MODEL_DIR = Path("/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models")
PDMODEL = MODEL_DIR / "inference.pdmodel"
PDIPARAMS = MODEL_DIR / "inference.pdiparams"
OUTPUT_ONNX = MODEL_DIR / "slanetplus_converted.onnx"

print("="*70)
print("SLANETPLUS PADDLEPADDLE → ONNX CONVERSION")
print("="*70)
print(f"\nInput files:")
print(f"  Model:  {PDMODEL}")
print(f"  Params: {PDIPARAMS}")
print(f"\nOutput:")
print(f"  ONNX:   {OUTPUT_ONNX}")
print()

# Method 1: Try using paddle2onnx via subprocess (command line tool)
print("-"*70)
print("Method 1: Trying paddle2onnx command line tool...")
print("-"*70)

try:
    cmd = [
        "paddle2onnx",
        "--model_dir", str(MODEL_DIR),
        "--model_filename", "inference.pdmodel",
        "--params_filename", "inference.pdiparams",
        "--save_file", str(OUTPUT_ONNX),
        "--opset_version", "13",
        "--enable_onnx_checker", "True"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode == 0 and OUTPUT_ONNX.exists():
        print("✅ paddle2onnx CLI conversion successful!")
        size_mb = OUTPUT_ONNX.stat().st_size / (1024 * 1024)
        print(f"   Output: {OUTPUT_ONNX}")
        print(f"   Size: {size_mb:.2f} MB")
        sys.exit(0)
    else:
        print(f"❌ paddle2onnx CLI failed:")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")

except FileNotFoundError:
    print("⚠️  paddle2onnx command not found in PATH")
except Exception as e:
    print(f"❌ Error: {e}")

# Method 2: Try X2Paddle
print("\n" + "-"*70)
print("Method 2: Trying X2Paddle...")
print("-"*70)

try:
    # Check if x2paddle is installed
    result = subprocess.run(["pip3", "show", "x2paddle"], capture_output=True, text=True)

    if result.returncode != 0:
        print("⚠️  x2paddle not installed, installing...")
        install_result = subprocess.run(
            ["pip3", "install", "x2paddle"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if install_result.returncode != 0:
            print(f"❌ Failed to install x2paddle: {install_result.stderr}")
            raise Exception("x2paddle installation failed")

    # Use x2paddle
    cmd = [
        "x2paddle",
        "--framework=paddle",
        "--model", str(PDMODEL),
        "--params", str(PDIPARAMS),
        "--save_dir", str(MODEL_DIR / "x2paddle_out"),
        "--convert_to_onnx=True"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    if result.returncode == 0:
        # Check if ONNX file was created
        onnx_out = MODEL_DIR / "x2paddle_out" / "model.onnx"
        if onnx_out.exists():
            # Move to our desired location
            import shutil
            shutil.copy(onnx_out, OUTPUT_ONNX)
            print("✅ X2Paddle conversion successful!")
            size_mb = OUTPUT_ONNX.stat().st_size / (1024 * 1024)
            print(f"   Output: {OUTPUT_ONNX}")
            print(f"   Size: {size_mb:.2f} MB")
            sys.exit(0)

    print(f"❌ X2Paddle failed:")
    print(f"   stdout: {result.stdout}")
    print(f"   stderr: {result.stderr}")

except Exception as e:
    print(f"❌ X2Paddle error: {e}")

# Method 3: Try using PaddlePaddle inference API to export
print("\n" + "-"*70)
print("Method 3: Direct PaddlePaddle export...")
print("-"*70)

try:
    import paddle
    from paddle.static import InputSpec
    from paddle.jit import to_static

    print("⚠️  This method requires the original model code, which we don't have")
    print("   Skipping direct export method")

except ImportError:
    print("⚠️  PaddlePaddle not properly installed")

# All methods failed
print("\n" + "="*70)
print("❌ ALL CONVERSION METHODS FAILED")
print("="*70)
print("\nReasons:")
print("1. paddle2onnx has compatibility issues on this system")
print("2. X2Paddle may not support this model architecture")
print("3. Direct export requires original model definition code")
print("\nSuggestions:")
print("- Use the pre-converted ONNX from HuggingFace (already downloaded)")
print("- Try conversion on Linux system where paddle2onnx works better")
print("- Contact model authors for official ONNX version")

sys.exit(1)
