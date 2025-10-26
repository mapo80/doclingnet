#!/usr/bin/env python3
"""
Visualize SLANetPlus ONNX detections with bounding boxes
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Paths
ONNX_PATH = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models/slanet-plus.onnx"
IMAGES_DIR = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/images"
OUTPUT_DIR = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/temp/visualizations"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*70)
print("SLANETPLUS ONNX DETECTION VISUALIZATION")
print("="*70)

# Load ONNX model
print("\n✅ Loading ONNX model...")
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# Get test images
images_path = Path(IMAGES_DIR)
test_images = [
    "HAL.2009.page_77.pdf_125051.png",
    "HAL.2015.page_43.pdf_125177.png",
    "HAL.2017.page_79.pdf_125247.png"
]

IMG_SIZE = 488

for img_name in test_images:
    img_path = images_path / img_name
    if not img_path.exists():
        print(f"\n⚠️  {img_name} not found, skipping")
        continue

    print(f"\n" + "-"*70)
    print(f"Processing: {img_name}")
    print("-"*70)

    # Load and preprocess image
    img = cv2.imread(str(img_path))
    orig_h, orig_w = img.shape[:2]
    print(f"Original size: {orig_w}x{orig_h}")

    # Prepare input for ONNX
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_float = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_float, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)

    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_batch})

    bbox_coords = outputs[0]  # shape: (1, N, 8)
    structure_probs = outputs[1]  # shape: (1, N, 50)

    print(f"ONNX outputs:")
    print(f"  BBox coords: {bbox_coords.shape}")
    print(f"  Structure probs: {structure_probs.shape}")

    # Create visualization image (scaled up for better visibility)
    vis_scale = 2
    vis_img = cv2.resize(img, (orig_w * vis_scale, orig_h * vis_scale))

    # Draw bounding boxes
    # The coordinates seem to be normalized [0, 1] based on the resized image
    num_boxes = bbox_coords.shape[1]
    drawn_boxes = 0

    for i in range(num_boxes):
        bbox = bbox_coords[0, i]  # 8 coordinates: [x1, y1, x2, y2, x3, y3, x4, y4]

        # Check if bbox is non-zero
        if np.all(bbox == 0):
            continue

        # Extract 8-point coordinates (normalized 0-1)
        x_coords = bbox[0::2]  # x1, x2, x3, x4
        y_coords = bbox[1::2]  # y1, y2, y3, y4

        # Convert to pixel coordinates on original image
        # The normalization is relative to the 488x488 input
        # We need to scale back to original image size
        x_coords_px = x_coords * IMG_SIZE * (orig_w / IMG_SIZE)
        y_coords_px = y_coords * IMG_SIZE * (orig_h / IMG_SIZE)

        # Scale for visualization
        x_coords_vis = x_coords_px * vis_scale
        y_coords_vis = y_coords_px * vis_scale

        # Draw 4-point polygon
        points = np.array([
            [x_coords_vis[0], y_coords_vis[0]],
            [x_coords_vis[1], y_coords_vis[1]],
            [x_coords_vis[2], y_coords_vis[2]],
            [x_coords_vis[3], y_coords_vis[3]]
        ], dtype=np.int32)

        # Draw polygon in blue
        cv2.polylines(vis_img, [points], isClosed=True, color=(255, 0, 0), thickness=2)

        # Draw index number
        center_x = int(np.mean(x_coords_vis))
        center_y = int(np.mean(y_coords_vis))
        cv2.putText(vis_img, str(i), (center_x, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        drawn_boxes += 1

    print(f"Drawn boxes: {drawn_boxes}/{num_boxes}")

    # Save visualization
    output_path = Path(OUTPUT_DIR) / f"{img_name.replace('.png', '_onnx_viz.png')}"
    cv2.imwrite(str(output_path), vis_img)
    print(f"✅ Saved: {output_path}")

    # Also create a version with raw coordinates (no scaling assumption)
    vis_img_raw = cv2.resize(img, (orig_w * vis_scale, orig_h * vis_scale))
    drawn_raw = 0

    for i in range(num_boxes):
        bbox = bbox_coords[0, i]

        if np.all(bbox == 0):
            continue

        # Try interpreting as normalized [0,1] directly on original image
        x_coords = bbox[0::2]
        y_coords = bbox[1::2]

        # Direct scaling to original image
        x_coords_px = x_coords * orig_w * vis_scale
        y_coords_px = y_coords * orig_h * vis_scale

        points = np.array([
            [x_coords_px[0], y_coords_px[0]],
            [x_coords_px[1], y_coords_px[1]],
            [x_coords_px[2], y_coords_px[2]],
            [x_coords_px[3], y_coords_px[3]]
        ], dtype=np.int32)

        cv2.polylines(vis_img_raw, [points], isClosed=True, color=(0, 0, 255), thickness=2)

        center_x = int(np.mean(x_coords_px))
        center_y = int(np.mean(y_coords_px))
        cv2.putText(vis_img_raw, str(i), (center_x, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        drawn_raw += 1

    output_path_raw = Path(OUTPUT_DIR) / f"{img_name.replace('.png', '_onnx_raw.png')}"
    cv2.imwrite(str(output_path_raw), vis_img_raw)
    print(f"✅ Saved (raw coords): {output_path_raw}")
    print(f"Drawn raw boxes: {drawn_raw}/{num_boxes}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated images:")
print("  *_onnx_viz.png  - Scaled via 488x488 assumption")
print("  *_onnx_raw.png  - Direct normalized [0,1] coordinates")
