#!/usr/bin/env python3
"""
Final evaluation of ONNX SLANetPlus with CORRECT postprocessing on FinTabNet
"""
import cv2
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path
import time

# Paths
ONNX_PATH = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/models/slanetplus_converted.onnx"
GT_JSON = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/sample_annotations.json"
IMAGES_DIR = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/images"
OUTPUT_DIR = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/rapidtable-onnx/temp/final_evaluation"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*70)
print("FINAL SLANETPLUS ONNX EVALUATION (CORRECT POSTPROCESSING)")
print("="*70)

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
beg_idx, end_idx = char_dict["sos"], char_dict["eos"]
ignored_tokens = [beg_idx, end_idx]

# Load GT
with open(GT_JSON) as f:
    gt_data = json.load(f)
gt_by_filename = {item['filename']: item for item in gt_data}
print(f"\n✅ Loaded GT for {len(gt_by_filename)} images")

# Load ONNX
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print(f"✅ ONNX model loaded")

# Process images
images_path = Path(IMAGES_DIR)
test_images = [
    "HAL.2009.page_77.pdf_125051.png",
    "HAL.2015.page_43.pdf_125177.png",
    "HAL.2017.page_79.pdf_125247.png"
]

IMG_SIZE = 488
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def calculate_iou(box1, box2):
    """box1: [x1,y1,x2,y2,x3,y3,x4,y4], box2: [x,y,w,h]"""
    x_coords = [box1[i] for i in range(0, 8, 2)]
    y_coords = [box1[i] for i in range(1, 8, 2)]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    box1_xywh = [x1, y1, x2-x1, y2-y1]
    
    inter_x1 = max(box1_xywh[0], box2[0])
    inter_y1 = max(box1_xywh[1], box2[1])
    inter_x2 = min(box1_xywh[0]+box1_xywh[2], box2[0]+box2[2])
    inter_y2 = min(box1_xywh[1]+box1_xywh[3], box2[1]+box2[3])
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = box1_xywh[2] * box1_xywh[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

results = []
total_tp, total_fp, total_fn = 0, 0, 0
total_time = 0

print("\n" + "="*70)
for img_name in test_images:
    img_path = images_path / img_name
    if img_name not in gt_by_filename:
        continue
    
    print(f"\nProcessing: {img_name}")
    
    # Load image
    img = cv2.imread(str(img_path))
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize keeping aspect
    scale = IMG_SIZE / max(orig_h, orig_w)
    h_resize = round(orig_h * scale)
    w_resize = round(orig_w * scale)
    img_resized = cv2.resize(img_rgb, (w_resize, h_resize))
    
    # Pad
    pad_h, pad_w = IMG_SIZE - h_resize, IMG_SIZE - w_resize
    img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    # Normalize
    img_float = img_padded.astype(np.float32) / 255.0
    img_normalized = (img_float - mean) / std
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_tensor, axis=0)
    
    # Inference
    start = time.time()
    outputs = session.run(None, {session.get_inputs()[0].name: img_batch})
    elapsed = time.time() - start
    total_time += elapsed
    
    bbox_preds, structure_probs = outputs[0], outputs[1]
    
    # Postprocess
    structure_idx = structure_probs.argmax(axis=2)
    structure_max_probs = structure_probs.max(axis=2)
    
    bbox_list = []
    for idx in range(structure_idx.shape[1]):
        char_idx = int(structure_idx[0, idx])
        if idx > 0 and char_idx == end_idx:
            break
        if char_idx in ignored_tokens:
            continue
        text = character[char_idx]
        if text in td_tokens:
            bbox = bbox_preds[0, idx].copy()
            pad_w_coord, pad_h_coord = IMG_SIZE, IMG_SIZE
            w, h = orig_w, orig_h
            ratio = min(pad_w_coord / w, pad_h_coord / h)
            bbox[0::2] *= pad_w_coord
            bbox[1::2] *= pad_h_coord
            bbox[0::2] /= ratio
            bbox[1::2] /= ratio
            bbox_list.append(bbox.astype(int))
    
    # Compare with GT
    gt_regions = gt_by_filename[img_name]['regions']
    gt_boxes = [[r['x'], r['y'], r['width'], r['height']] for r in gt_regions]
    
    # Match predictions to GT
    matched_gt = set()
    matched_pred = set()
    
    for pred_idx, pred_box in enumerate(bbox_list):
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou >= 0.5:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                break
    
    tp = len(matched_gt)
    fp = len(bbox_list) - tp
    fn = len(gt_boxes) - tp
    
    total_tp += tp
    total_fp += fp
    total_fn += fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Pred: {len(bbox_list)}, GT: {len(gt_boxes)}")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    print(f"  P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    print(f"  Time: {elapsed:.3f}s")
    
    results.append({
        'filename': img_name,
        'num_predictions': len(bbox_list),
        'num_gt': len(gt_boxes),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'inference_time': elapsed
    })

# Summary
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\nOverall Metrics:")
print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1 Score: {f1:.3f}")
print(f"  Avg inference: {total_time/len(results):.3f}s")

# Save results
output_file = Path(OUTPUT_DIR) / "final_evaluation_results.json"
with open(output_file, 'w') as f:
    json.dump({
        'model': 'SLANetPlus ONNX (Converted with Correct Postprocessing)',
        'total_images': len(results),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'overall_precision': precision,
        'overall_recall': recall,
        'overall_f1': f1,
        'avg_time': total_time / len(results),
        'per_image_results': results
    }, f, indent=2)

print(f"\n✅ Results saved: {output_file}")
print("="*70)
