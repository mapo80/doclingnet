#!/usr/bin/env python3
"""
Test ONNX SLANetPlus model with CORRECT postprocessing
Based on slanet-plus-table source code analysis
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Paths
ONNX_PATH = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models/slanetplus_converted.onnx"
IMAGE_PATH = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/images/HAL.2009.page_77.pdf_125051.png"
OUTPUT_DIR = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/temp/correct_postprocessing"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*70)
print("SLANETPLUS ONNX WITH CORRECT POSTPROCESSING")
print("="*70)

# Character dictionary from slanet-plus-table source
character = ['sos', '<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>', '<td', '>', '</td>',
             ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"', ' colspan="6"', ' colspan="7"',
             ' colspan="8"', ' colspan="9"', ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"',
             ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"', ' colspan="18"', ' colspan="19"',
             ' colspan="20"', ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"', ' rowspan="6"',
             ' rowspan="7"', ' rowspan="8"', ' rowspan="9"', ' rowspan="10"', ' rowspan="11"', ' rowspan="12"',
             ' rowspan="13"', ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"', ' rowspan="18"',
             ' rowspan="19"', ' rowspan="20"', '<td></td>', 'eos']

td_tokens = ['<td>', '<td', '<td></td>']
beg_str = "sos"
end_str = "eos"

# Build dict
char_dict = {char: i for i, char in enumerate(character)}
beg_idx = char_dict[beg_str]
end_idx = char_dict[end_str]
ignored_tokens = [beg_idx, end_idx]

# Load image
print(f"\nLoading image: {IMAGE_PATH}")
img = cv2.imread(IMAGE_PATH)
orig_h, orig_w = img.shape[:2]
print(f"Original size: {orig_w}x{orig_h}")

# Preprocess (same as slanet-plus-table)
IMG_SIZE = 488
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize keeping aspect ratio
scale = IMG_SIZE / max(orig_h, orig_w)
h_resize = round(orig_h * scale)
w_resize = round(orig_w * scale)
img_resized = cv2.resize(img_rgb, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)
print(f"Resized to: {w_resize}x{h_resize}")

# Pad to 488x488
pad_h = IMG_SIZE - h_resize
pad_w = IMG_SIZE - w_resize
img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
print(f"Padded to: {IMG_SIZE}x{IMG_SIZE}")

# Normalize
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
img_float = img_padded.astype(np.float32) / 255.0
img_normalized = (img_float - mean) / std

# Transpose and add batch dimension
img_tensor = np.transpose(img_normalized, (2, 0, 1))
img_batch = np.expand_dims(img_tensor, axis=0)

print(f"Input tensor shape: {img_batch.shape}")

# Run ONNX inference
print("\n" + "-"*70)
print("Running ONNX inference...")
print("-"*70)

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img_batch})

bbox_preds = outputs[0]  # [batch, N, 8]
structure_probs = outputs[1]  # [batch, N, 50]

print(f"✅ Inference complete")
print(f"  BBox preds shape: {bbox_preds.shape}")
print(f"  Structure probs shape: {structure_probs.shape}")

# CORRECT POSTPROCESSING (from slanet-plus-table source)
print("\n" + "-"*70)
print("Applying CORRECT postprocessing...")
print("-"*70)

# Decode structure
structure_idx = structure_probs.argmax(axis=2)
structure_max_probs = structure_probs.max(axis=2)

structure_list = []
bbox_list = []
score_list = []

for idx in range(structure_idx.shape[1]):
    char_idx = int(structure_idx[0, idx])

    # Stop at end token
    if idx > 0 and char_idx == end_idx:
        break

    # Skip special tokens
    if char_idx in ignored_tokens:
        continue

    text = character[char_idx]

    # Extract bbox for td tokens
    if text in td_tokens:
        bbox = bbox_preds[0, idx].copy()  # 8 coordinates

        # CRITICAL: Decode bbox coordinates
        # padding_shape = [IMG_SIZE, IMG_SIZE] (the padded image size)
        # ori_shape = [orig_w, orig_h] (original image size)

        pad_w, pad_h = IMG_SIZE, IMG_SIZE
        w, h = orig_w, orig_h
        ratio_w = pad_w / w
        ratio_h = pad_h / h
        ratio = min(ratio_w, ratio_h)

        # Scale from normalized [0,1] to padded image coordinates
        bbox[0::2] *= pad_w  # x coordinates
        bbox[1::2] *= pad_h  # y coordinates

        # Scale to original image coordinates
        bbox[0::2] /= ratio
        bbox[1::2] /= ratio

        bbox_list.append(bbox.astype(int))

    structure_list.append(text)
    score_list.append(structure_max_probs[0, idx])

structure_score = np.mean(score_list) if score_list else 0.0

print(f"\n✅ Postprocessing complete!")
print(f"  Structure length: {len(structure_list)}")
print(f"  Cell bboxes: {len(bbox_list)}")
print(f"  Structure score: {structure_score:.3f}")

print(f"\nFirst 10 structure tokens:")
for i, token in enumerate(structure_list[:10]):
    print(f"  {i}: {token}")

print(f"\nFirst 5 cell bboxes:")
for i, bbox in enumerate(bbox_list[:5]):
    print(f"  Cell {i}: {bbox}")

# Visualize
print("\n" + "-"*70)
print("Creating visualization...")
print("-"*70)

vis_img = img.copy()
scale_factor = 2
vis_img = cv2.resize(vis_img, (orig_w * scale_factor, orig_h * scale_factor))

for i, bbox in enumerate(bbox_list):
    # bbox is 8-point: [x1, y1, x2, y2, x3, y3, x4, y4]
    x_coords = bbox[0::2] * scale_factor
    y_coords = bbox[1::2] * scale_factor

    points = np.array([
        [x_coords[0], y_coords[0]],
        [x_coords[1], y_coords[1]],
        [x_coords[2], y_coords[2]],
        [x_coords[3], y_coords[3]]
    ], dtype=np.int32)

    # Draw polygon in green
    cv2.polylines(vis_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw cell number
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    cv2.putText(vis_img, str(i), (center_x, center_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

output_path = Path(OUTPUT_DIR) / f"{Path(IMAGE_PATH).stem}_correct_postprocessing.png"
cv2.imwrite(str(output_path), vis_img)

print(f"✅ Visualization saved: {output_path}")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"✅ Detected {len(bbox_list)} table cells with correct postprocessing")
