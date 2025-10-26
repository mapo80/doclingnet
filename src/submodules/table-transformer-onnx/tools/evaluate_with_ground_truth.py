#!/usr/bin/env python3
"""
Evaluate Table Transformer on FinTabNet dataset with ground truth comparison
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models"
ONNX_MODEL_PATH = MODEL_DIR / "onnx" / "model_quantized.onnx"
CONFIG_PATH = MODEL_DIR / "config.json"
PREPROCESSOR_CONFIG_PATH = MODEL_DIR / "preprocessor_config.json"
DATASET_DIR = Path(__file__).parent.parent.parent / "ds4sd-docling-tableformer-onnx" / "dataset" / "FinTabNet" / "images"
ANNOTATIONS_PATH = Path(__file__).parent.parent.parent / "ds4sd-docling-tableformer-onnx" / "dataset" / "FinTabNet" / "sample_annotations.json"
OUTPUT_DIR = Path(__file__).parent.parent / "temp"
RESULTS_PATH = OUTPUT_DIR / "evaluation_results.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_configs():
    """Load model and preprocessor configuration"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    with open(PREPROCESSOR_CONFIG_PATH) as f:
        preprocessor_config = json.load(f)
    return config, preprocessor_config


def load_ground_truth():
    """Load ground truth annotations"""
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    # Convert to dict indexed by filename
    gt_dict = {}
    for ann in annotations:
        filename = ann['filename']
        regions = ann['regions']
        gt_dict[filename] = regions

    return gt_dict


def preprocess_image(image_path, preprocessor_config):
    """Preprocess image for Table Transformer model"""
    size = preprocessor_config.get("size", {})
    if isinstance(size, dict):
        target_size = (size.get("height", 800), size.get("width", 800))
    else:
        target_size = (size, size) if isinstance(size, int) else (800, 800)

    image_mean = preprocessor_config.get("image_mean", [0.485, 0.456, 0.406])
    image_std = preprocessor_config.get("image_std", [0.229, 0.224, 0.225])

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_float = img_resized.astype(np.float32) / 255.0

    mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
    img_normalized = (img_float - mean) / std

    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)

    return img_tensor, img.shape[:2]  # Return original image size


def run_inference(session, image_tensor):
    """Run ONNX inference"""
    input_name = session.get_inputs()[0].name
    start = time.time()
    outputs = session.run(None, {input_name: image_tensor})
    elapsed = time.time() - start
    return outputs, elapsed


def decode_outputs(outputs, config):
    """Decode model outputs with softmax and filtering"""
    logits = outputs[0]
    boxes = outputs[1]

    # Apply softmax
    logits_stable = logits[0] - np.max(logits[0], axis=-1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    pred_classes = np.argmax(probs, axis=-1)
    pred_scores = np.max(probs, axis=-1)

    num_classes = probs.shape[-1]
    no_object_class = num_classes - 1

    threshold = 0.7
    mask = (pred_scores > threshold) & (pred_classes != no_object_class)

    filtered_boxes = boxes[0][mask]
    filtered_classes = pred_classes[mask]
    filtered_scores = pred_scores[mask]

    return {
        'boxes': filtered_boxes,
        'classes': filtered_classes,
        'scores': filtered_scores,
        'num_detections': len(filtered_boxes)
    }


def convert_bbox_format(boxes, orig_h, orig_w):
    """
    Convert from normalized [cx, cy, w, h] to pixel [x, y, w, h]
    boxes: [N, 4] in normalized [0,1] format
    Returns: [N, 4] in pixel coordinates
    """
    pixel_boxes = []
    for box in boxes:
        cx, cy, bw, bh = box
        x = int((cx - bw/2) * orig_w)
        y = int((cy - bh/2) * orig_h)
        w = int(bw * orig_w)
        h = int(bh * orig_h)
        pixel_boxes.append([x, y, w, h])

    return np.array(pixel_boxes)


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    box format: [x, y, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_predictions_to_gt(pred_boxes, pred_classes, gt_regions, iou_threshold=0.5):
    """
    Match predicted boxes to ground truth regions
    Returns: (true_positives, false_positives, false_negatives, matches)
    """
    # Convert GT regions to box format [x, y, w, h]
    gt_boxes = [[r['x'], r['y'], r['width'], r['height']] for r in gt_regions]
    gt_labels = [r['label'] for r in gt_regions]

    matched_gt = set()
    matches = []
    true_positives = 0
    false_positives = 0

    # For each prediction, find best matching GT
    for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            gt_label = gt_labels[best_gt_idx]
            class_match = (int(pred_class) == gt_label)
            matches.append({
                'pred_idx': int(pred_idx),
                'gt_idx': int(best_gt_idx),
                'iou': float(best_iou),
                'pred_class': int(pred_class),
                'gt_label': int(gt_label),
                'class_match': class_match
            })
            if class_match:
                true_positives += 1
            else:
                false_positives += 1
        else:
            false_positives += 1

    false_negatives = len(gt_boxes) - len(matched_gt)

    return true_positives, false_positives, false_negatives, matches


def draw_comparison(image_path, pred_boxes, pred_classes, pred_scores, gt_regions, matches, output_path):
    """Draw predictions and ground truth for comparison"""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    # Draw GT in green
    for region in gt_regions:
        x, y, w, h = int(region['x']), int(region['y']), int(region['width']), int(region['height'])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"GT:L{region['label']}"
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw predictions in red/blue (matched/unmatched)
    matched_pred_indices = {m['pred_idx'] for m in matches}

    for i, (box, cls, score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
        x, y, w, h = box
        color = (255, 0, 0) if i in matched_pred_indices else (0, 0, 255)  # Blue=matched, Red=unmatched
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        label = f"P:C{int(cls)}:{score:.2f}"
        cv2.putText(img, label, (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(str(output_path), img)


def main():
    print("="*70)
    print("TABLE TRANSFORMER EVALUATION WITH GROUND TRUTH")
    print("="*70)

    # Load config and GT
    config, preprocessor_config = load_configs()
    gt_dict = load_ground_truth()
    print(f"\n✅ Loaded ground truth for {len(gt_dict)} images")

    # Load ONNX model
    print(f"✅ Loading ONNX model...")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])

    # Get all images
    image_files = sorted(DATASET_DIR.glob("*.png"))
    print(f"✅ Found {len(image_files)} images in dataset\n")

    # Run evaluation
    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0.0

    for i, image_path in enumerate(image_files, 1):
        filename = image_path.name

        print(f"{'-'*70}")
        print(f"[{i}/{len(image_files)}] {filename}")

        # Check if GT exists
        if filename not in gt_dict:
            print(f"  ⚠️  No ground truth annotations")
            continue

        gt_regions = gt_dict[filename]

        # Preprocess and run inference
        img_tensor, (orig_h, orig_w) = preprocess_image(image_path, preprocessor_config)
        outputs, elapsed = run_inference(session, img_tensor)
        total_time += elapsed

        # Decode
        detections = decode_outputs(outputs, config)
        pred_boxes_pixel = convert_bbox_format(detections['boxes'], orig_h, orig_w)

        # Match with GT
        tp, fp, fn, matches = match_predictions_to_gt(
            pred_boxes_pixel,
            detections['classes'],
            gt_regions,
            iou_threshold=0.5
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculate metrics for this image
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Detections: {detections['num_detections']} | GT: {len(gt_regions)}")
        print(f"  TP={tp} FP={fp} FN={fn}")
        print(f"  Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")
        print(f"  Inference: {elapsed:.3f}s")

        # Draw comparison
        vis_path = OUTPUT_DIR / f"{image_path.stem}_comparison.png"
        draw_comparison(image_path, pred_boxes_pixel, detections['classes'],
                       detections['scores'], gt_regions, matches, vis_path)
        print(f"  Saved: {vis_path.name}")

        # Store result
        results.append({
            'filename': filename,
            'num_predictions': int(detections['num_detections']),
            'num_gt': len(gt_regions),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'inference_time': float(elapsed),
            'matches': matches
        })

    # Overall metrics
    print("\n" + "="*70)
    print("OVERALL EVALUATION")
    print("="*70)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print(f"\nTotal images evaluated: {len(results)}")
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"\nOverall Precision: {overall_precision:.3f}")
    print(f"Overall Recall: {overall_recall:.3f}")
    print(f"Overall F1 Score: {overall_f1:.3f}")
    print(f"\nTotal inference time: {total_time:.2f}s")
    print(f"Average inference time: {total_time/len(results):.3f}s")

    # Save results
    summary = {
        'total_images': len(results),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'overall_precision': float(overall_precision),
        'overall_recall': float(overall_recall),
        'overall_f1': float(overall_f1),
        'total_time': float(total_time),
        'avg_time': float(total_time / len(results)),
        'per_image_results': results
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to {RESULTS_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()
