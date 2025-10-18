#!/usr/bin/env python3
"""
Generate visualizations using the ORIGINAL Python slanet-plus-table package
For comparison with ONNX converted model results
"""
import cv2
import numpy as np
from pathlib import Path
import json

print("="*70)
print("PYTHON BASELINE VISUALIZATION")
print("Using original slanet-plus-table package")
print("="*70)

# Try to import slanet-plus-table
try:
    from slanet_plus_table import SLANetPlus
    print("✅ slanet-plus-table imported successfully")
except ImportError as e:
    print(f"❌ Failed to import slanet-plus-table: {e}")
    print("\nTrying to install...")
    import subprocess
    subprocess.run(["pip3", "install", "slanet-plus-table"], check=True)
    from slanet_plus_table import SLANetPlus
    print("✅ Installed and imported slanet-plus-table")

# Initialize model
print("\nInitializing SLANetPlus model...")
model = SLANetPlus()
print("✅ Model initialized")

# Test images
IMAGE_DIR = Path("/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/images")
OUTPUT_DIR = Path("/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/temp/python_baseline_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

test_images = [
    "HAL.2009.page_77.pdf_125051.png",
    "HAL.2015.page_43.pdf_125177.png",
    "HAL.2017.page_79.pdf_125247.png"
]

results = []

for img_name in test_images:
    print("\n" + "-"*70)
    print(f"Processing: {img_name}")
    print("-"*70)

    img_path = IMAGE_DIR / img_name
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        continue

    orig_h, orig_w = img.shape[:2]
    print(f"Image size: {orig_w}x{orig_h}")

    # Run inference with OCR
    try:
        # Import RapidOCR
        from rapidocr_onnxruntime import RapidOCR
        ocr_engine = RapidOCR()

        print("Running OCR...")
        ocr_result, _ = ocr_engine(img)

        if ocr_result is None:
            print("⚠️ No OCR results, using empty list")
            ocr_result = []
        else:
            print(f"✅ OCR detected {len(ocr_result)} text regions")

        print("Running table structure recognition...")
        pred_html, pred_bboxes, elapse = model(img, ocr_result)

        print(f"✅ Inference complete in {elapse:.3f}s")
        print(f"  Detected cells: {len(pred_bboxes)}")

        # Save results
        result = {
            "image": img_name,
            "cells_detected": len(pred_bboxes),
            "inference_time": elapse,
            "html": pred_html
        }
        results.append(result)

        # Create visualization
        print("Creating visualization...")
        vis_img = img.copy()
        scale_factor = 2
        vis_img = cv2.resize(vis_img, (orig_w * scale_factor, orig_h * scale_factor))

        for i, bbox in enumerate(pred_bboxes):
            # bbox format from slanet-plus-table: [x1, y1, x2, y2, x3, y3, x4, y4]
            x_coords = bbox[0::2] * scale_factor
            y_coords = bbox[1::2] * scale_factor

            points = np.array([
                [x_coords[0], y_coords[0]],
                [x_coords[1], y_coords[1]],
                [x_coords[2], y_coords[2]],
                [x_coords[3], y_coords[3]]
            ], dtype=np.int32)

            # Draw polygon in BLUE (to differentiate from ONNX green)
            cv2.polylines(vis_img, [points], isClosed=True, color=(255, 0, 0), thickness=2)

            # Draw cell number
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            cv2.putText(vis_img, str(i), (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        output_path = OUTPUT_DIR / f"{Path(img_name).stem}_python_baseline.png"
        cv2.imwrite(str(output_path), vis_img)

        print(f"✅ Visualization saved: {output_path}")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing rapidocr-onnxruntime...")
        import subprocess
        subprocess.run(["pip3", "install", "rapidocr-onnxruntime"], check=True)
        print("⚠️ Please re-run the script after installation")
        break
    except Exception as e:
        print(f"❌ Error processing {img_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Save summary
if results:
    summary_path = OUTPUT_DIR / "python_baseline_results.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for result in results:
        print(f"\n{result['image']}:")
        print(f"  Cells detected: {result['cells_detected']}")
        print(f"  Inference time: {result['inference_time']:.3f}s")

    avg_time = sum(r['inference_time'] for r in results) / len(results)
    print(f"\nAverage inference time: {avg_time:.3f}s")
    print(f"\n✅ Results saved to: {summary_path}")
    print(f"✅ Visualizations saved to: {OUTPUT_DIR}")

    print("\n" + "="*70)
    print("COMPARISON NOTES")
    print("="*70)
    print("Python baseline visualizations have BLUE bounding boxes")
    print("ONNX model visualizations have GREEN bounding boxes")
    print("Compare the two sets of images to identify differences")

else:
    print("\n❌ No results generated")
