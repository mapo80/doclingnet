#!/usr/bin/env python3
"""
Validate SLANetPlus ONNX model against Python baseline
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time

# Paths
ONNX_PATH = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models/slanet-plus.onnx"
IMAGE_PATH = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/images/HAL.2009.page_77.pdf_125051.png"

print("="*70)
print("SLANETPLUS ONNX MODEL VALIDATION")
print("="*70)
print(f"\nONNX Model: {ONNX_PATH}")
print(f"Test Image: {IMAGE_PATH}")
print()

# Load ONNX model
print("-"*70)
print("Loading ONNX model...")
print("-"*70)

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# Get model info
print(f"\n📊 Model Info:")
print(f"   Inputs:")
for input_meta in session.get_inputs():
    print(f"     - {input_meta.name}: {input_meta.shape} ({input_meta.type})")

print(f"   Outputs:")
for output_meta in session.get_outputs():
    print(f"     - {output_meta.name}: {output_meta.shape} ({output_meta.type})")

# Prepare image based on typical SLANetPlus preprocessing
print("\n" + "-"*70)
print("Preprocessing image...")
print("-"*70)

img = cv2.imread(IMAGE_PATH)
print(f"Original image shape: {img.shape}")

# SLANetPlus typically uses 488x488 input
IMG_SIZE = 488
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
print(f"Resized image shape: {img_resized.shape}")

# Normalize to [0, 1] and transpose to CHW format
img_float = img_resized.astype(np.float32) / 255.0
img_transposed = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
img_batch = np.expand_dims(img_transposed, axis=0)  # Add batch dimension

print(f"Input tensor shape: {img_batch.shape}")
print(f"Input tensor dtype: {img_batch.dtype}")
print(f"Input tensor range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")

# Run inference
print("\n" + "-"*70)
print("Running ONNX inference...")
print("-"*70)

try:
    input_name = session.get_inputs()[0].name

    start_time = time.time()
    outputs = session.run(None, {input_name: img_batch})
    elapsed = time.time() - start_time

    print(f"\n✅ ONNX inference successful!")
    print(f"   Inference time: {elapsed:.3f}s")
    print(f"   Number of outputs: {len(outputs)}")

    for i, output in enumerate(outputs):
        print(f"   Output {i}: shape={output.shape}, dtype={output.dtype}")
        if len(output.shape) > 0:
            print(f"            range=[{output.min():.3f}, {output.max():.3f}]")

    # Try to interpret outputs
    print("\n" + "-"*70)
    print("Interpreting outputs...")
    print("-"*70)

    # SLANetPlus typically outputs structure tokens and cell coordinates
    # The exact format depends on the model, let's inspect what we got

    if len(outputs) >= 2:
        structure_probs = outputs[0]
        bbox_preds = outputs[1] if len(outputs) > 1 else None

        print(f"\nStructure predictions:")
        print(f"  Shape: {structure_probs.shape}")

        if bbox_preds is not None:
            print(f"\nBBox predictions:")
            print(f"  Shape: {bbox_preds.shape}")

            # Try to count non-zero bboxes
            if len(bbox_preds.shape) == 3:  # [batch, num_boxes, coords]
                non_zero_boxes = np.count_nonzero(np.any(bbox_preds[0] != 0, axis=1))
                print(f"  Non-zero boxes: {non_zero_boxes}")

except Exception as e:
    print(f"\n❌ ONNX inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Compare with Python baseline
print("\n" + "="*70)
print("COMPARING WITH PYTHON BASELINE")
print("="*70)

try:
    from slanet_plus_table import SLANetPlus

    print("\nRunning Python SLANetPlus...")
    model_py = SLANetPlus()

    start_time = time.time()
    html, cells, elapsed_py = model_py(img)
    total_py = time.time() - start_time

    print(f"✅ Python inference successful!")
    print(f"   Reported time: {elapsed_py:.3f}s")
    print(f"   Total time: {total_py:.3f}s")
    print(f"   Number of cells: {cells.shape[0]}")
    print(f"   Cell format: {cells.shape}")
    print(f"   First 5 cells:")
    for i in range(min(5, cells.shape[0])):
        print(f"     {i}: {cells[i]}")

    print("\n" + "-"*70)
    print("COMPARISON SUMMARY")
    print("-"*70)
    print(f"Python cells detected: {cells.shape[0]}")
    print(f"ONNX inference time: {elapsed:.3f}s")
    print(f"Python inference time: {elapsed_py:.3f}s")
    print(f"Speedup: {elapsed_py/elapsed:.2f}x")

except ImportError:
    print("\n⚠️  slanet-plus-table not installed, skipping Python baseline comparison")
except Exception as e:
    print(f"\n⚠️  Python baseline comparison failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
