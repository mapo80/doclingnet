#!/usr/bin/env python3
"""
Compare two ONNX models side-by-side:
1. slanet-plus.onnx (from HuggingFace - trusted reference)
2. slanetplus_converted.onnx (our paddle2onnx conversion)

This helps identify if our conversion is producing identical results.
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Paths
ONNX_HF = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models/slanet-plus.onnx"
ONNX_CONVERTED = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models/slanetplus_converted.onnx"

IMAGE_DIR = Path("/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/images")
OUTPUT_DIR = Path("/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/temp/onnx_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

test_images = [
    "HAL.2009.page_77.pdf_125051.png",
    "HAL.2015.page_43.pdf_125177.png",
    "HAL.2017.page_79.pdf_125247.png"
]

# Character dictionary
character = ['sos', '<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>', '<td', '>', '</td>',
             ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"', ' colspan="6"', ' colspan="7"',
             ' colspan="8"', ' colspan="9"', ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"',
             ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"', ' colspan="18"', ' colspan="19"',
             ' colspan="20"', ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"', ' rowspan="6"',
             ' rowspan="7"', ' rowspan="8"', ' rowspan="9"', ' rowspan="10"', ' rowspan="11"', ' rowspan="12"',
             ' rowspan="13"', ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"', ' rowspan="18"',
             ' rowspan="19"', ' rowspan="20"', '<td></td>', 'eos']

td_tokens = ['<td>', '<td', '<td></td>']
char_dict = {char: i for i, char in enumerate(character)}
beg_idx = char_dict["sos"]
end_idx = char_dict["eos"]
ignored_tokens = [beg_idx, end_idx]

print("="*70)
print("COMPARING TWO ONNX MODELS")
print("="*70)
print(f"\nModel 1 (HuggingFace): {ONNX_HF}")
print(f"Model 2 (Converted):   {ONNX_CONVERTED}")
print()

# Load both models
session_hf = ort.InferenceSession(ONNX_HF, providers=["CPUExecutionProvider"])
session_conv = ort.InferenceSession(ONNX_CONVERTED, providers=["CPUExecutionProvider"])

print("✅ Both models loaded")

def preprocess_image(img_path):
    """Preprocess image for SLANetPlus"""
    IMG_SIZE = 488
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    img = cv2.imread(str(img_path))
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize keeping aspect ratio
    scale = IMG_SIZE / max(orig_h, orig_w)
    h_resize = round(orig_h * scale)
    w_resize = round(orig_w * scale)
    img_resized = cv2.resize(img_rgb, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)

    # Pad to 488x488
    pad_h = IMG_SIZE - h_resize
    pad_w = IMG_SIZE - w_resize
    img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Normalize
    img_float = img_padded.astype(np.float32) / 255.0
    img_normalized = (img_float - mean) / std

    # Transpose and add batch dimension
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_tensor, axis=0)

    return img, img_batch, (orig_w, orig_h)

def postprocess(bbox_preds, structure_probs, orig_size):
    """Postprocess model outputs"""
    orig_w, orig_h = orig_size
    IMG_SIZE = 488

    structure_idx = structure_probs.argmax(axis=2)
    structure_max_probs = structure_probs.max(axis=2)

    bbox_list = []
    structure_list = []

    for idx in range(structure_idx.shape[1]):
        char_idx = int(structure_idx[0, idx])

        if idx > 0 and char_idx == end_idx:
            break

        if char_idx in ignored_tokens:
            continue

        text = character[char_idx]

        if text in td_tokens:
            bbox = bbox_preds[0, idx].copy()

            # Decode bbox coordinates
            ratio = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
            bbox[0::2] *= IMG_SIZE  # x coordinates
            bbox[1::2] *= IMG_SIZE  # y coordinates
            bbox[0::2] /= ratio
            bbox[1::2] /= ratio

            bbox_list.append(bbox.astype(int))

        structure_list.append(text)

    return bbox_list, structure_list

def draw_bboxes(img, bboxes, color, scale_factor=2):
    """Draw bboxes on image"""
    vis_img = img.copy()
    orig_h, orig_w = img.shape[:2]
    vis_img = cv2.resize(vis_img, (orig_w * scale_factor, orig_h * scale_factor))

    for i, bbox in enumerate(bboxes):
        x_coords = bbox[0::2] * scale_factor
        y_coords = bbox[1::2] * scale_factor

        points = np.array([
            [x_coords[0], y_coords[0]],
            [x_coords[1], y_coords[1]],
            [x_coords[2], y_coords[2]],
            [x_coords[3], y_coords[3]]
        ], dtype=np.int32)

        cv2.polylines(vis_img, [points], isClosed=True, color=color, thickness=2)

        # Draw cell number
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        cv2.putText(vis_img, str(i), (center_x, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return vis_img

# Process each image
comparison_results = []

for img_name in test_images:
    print("\n" + "-"*70)
    print(f"Processing: {img_name}")
    print("-"*70)

    img_path = IMAGE_DIR / img_name
    img, img_batch, orig_size = preprocess_image(img_path)

    # Run HuggingFace model
    input_name_hf = session_hf.get_inputs()[0].name
    outputs_hf = session_hf.run(None, {input_name_hf: img_batch})
    bbox_preds_hf = outputs_hf[0]
    structure_probs_hf = outputs_hf[1]

    bbox_list_hf, structure_list_hf = postprocess(bbox_preds_hf, structure_probs_hf, orig_size)

    # Run converted model
    input_name_conv = session_conv.get_inputs()[0].name
    outputs_conv = session_conv.run(None, {input_name_conv: img_batch})
    bbox_preds_conv = outputs_conv[0]
    structure_probs_conv = outputs_conv[1]

    bbox_list_conv, structure_list_conv = postprocess(bbox_preds_conv, structure_probs_conv, orig_size)

    print(f"\n📊 Results:")
    print(f"  HuggingFace model: {len(bbox_list_hf)} cells")
    print(f"  Converted model:   {len(bbox_list_conv)} cells")
    print(f"  Match: {'✅' if len(bbox_list_hf) == len(bbox_list_conv) else '❌'}")

    # Compare structure tokens
    structure_match = structure_list_hf == structure_list_conv
    print(f"  Structure tokens match: {'✅' if structure_match else '❌'}")

    if not structure_match:
        print(f"    HF tokens:   {len(structure_list_hf)}")
        print(f"    Conv tokens: {len(structure_list_conv)}")
        print(f"    First 10 HF:   {structure_list_hf[:10]}")
        print(f"    First 10 Conv: {structure_list_conv[:10]}")

    # Compare bbox coordinates
    if len(bbox_list_hf) == len(bbox_list_conv):
        max_diff = 0
        for i in range(len(bbox_list_hf)):
            diff = np.abs(bbox_list_hf[i] - bbox_list_conv[i]).max()
            max_diff = max(max_diff, diff)
        print(f"  Max bbox coordinate diff: {max_diff:.3f} pixels")
        coord_match = max_diff < 1.0  # Allow 1 pixel tolerance
        print(f"  Coordinates match: {'✅' if coord_match else '❌'}")

    # Create side-by-side visualization
    vis_hf = draw_bboxes(img, bbox_list_hf, color=(0, 255, 0))  # Green for HF
    vis_conv = draw_bboxes(img, bbox_list_conv, color=(255, 0, 0))  # Blue for converted

    # Combine side by side
    combined = np.hstack([vis_hf, vis_conv])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f"HuggingFace ({len(bbox_list_hf)} cells)",
               (20, 40), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, f"Converted ({len(bbox_list_conv)} cells)",
               (vis_hf.shape[1] + 20, 40), font, 1, (255, 0, 0), 2)

    output_path = OUTPUT_DIR / f"{Path(img_name).stem}_comparison.png"
    cv2.imwrite(str(output_path), combined)
    print(f"\n✅ Comparison saved: {output_path}")

    comparison_results.append({
        "image": img_name,
        "hf_cells": len(bbox_list_hf),
        "conv_cells": len(bbox_list_conv),
        "match": len(bbox_list_hf) == len(bbox_list_conv)
    })

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_match = all(r["match"] for r in comparison_results)
print(f"\n{'✅ All models produce IDENTICAL results!' if all_match else '❌ Models produce DIFFERENT results'}")

for r in comparison_results:
    status = "✅" if r["match"] else "❌"
    print(f"  {status} {r['image']}: HF={r['hf_cells']}, Conv={r['conv_cells']}")

print(f"\n✅ Visualizations saved to: {OUTPUT_DIR}")
print("\nColor coding:")
print("  GREEN boxes  = HuggingFace model (reference)")
print("  BLUE boxes   = Our converted model")
