#!/usr/bin/env python3
"""
RapidTable inference script for FinTabNet dataset
Tests cell detection and compares with ground truth annotations
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from rapid_table import RapidTable, RapidTableInput, ModelType

# Paths
DATASET_DIR = Path(__file__).parent.parent.parent / "ds4sd-docling-tableformer-onnx" / "dataset" / "FinTabNet" / "images"
ANNOTATIONS_PATH = Path(__file__).parent.parent.parent / "ds4sd-docling-tableformer-onnx" / "dataset" / "FinTabNet" / "sample_annotations.json"
OUTPUT_DIR = Path(__file__).parent.parent / "temp"
RESULTS_PATH = OUTPUT_DIR / "rapidtable_evaluation_results.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth():
    """Load ground truth annotations"""
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    gt_dict = {}
    for ann in annotations:
        filename = ann['filename']
        regions = ann['regions']
        gt_dict[filename] = regions

    return gt_dict


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


def match_predictions_to_gt(pred_cells, gt_regions, iou_threshold=0.5):
    """
    Match predicted cells to ground truth regions
    Returns: (true_positives, false_positives, false_negatives, matches)
    """
    gt_boxes = [[r['x'], r['y'], r['width'], r['height']] for r in gt_regions]
    gt_labels = [r['label'] for r in gt_regions]

    matched_gt = set()
    matches = []
    true_positives = 0
    false_positives = 0

    # For each prediction, find best matching GT
    for pred_idx, pred_cell in enumerate(pred_cells):
        pred_box = pred_cell['bbox']  # [x1, y1, x2, y2]
        # Convert to [x, y, w, h]
        pred_box_xywh = [pred_box[0], pred_box[1], pred_box[2]-pred_box[0], pred_box[3]-pred_box[1]]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box_xywh, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            gt_label = gt_labels[best_gt_idx]
            matches.append({
                'pred_idx': int(pred_idx),
                'gt_idx': int(best_gt_idx),
                'iou': float(best_iou),
                'gt_label': int(gt_label)
            })
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(gt_boxes) - len(matched_gt)

    return true_positives, false_positives, false_negatives, matches


def draw_comparison(image_path, cells, gt_regions, matches, output_path):
    """Draw predictions and ground truth for comparison"""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w = img.shape[:2]
    scale_factor = w / 1000.0
    font_scale = 0.35 * scale_factor
    font_thickness = max(1, int(1 * scale_factor))
    box_thickness = max(1, int(2 * scale_factor))

    # Draw GT in green
    for region in gt_regions:
        x, y, rw, rh = int(region['x']), int(region['y']), int(region['width']), int(region['height'])
        cv2.rectangle(img, (x, y), (x+rw, y+rh), (0, 255, 0), box_thickness)

    # Draw predictions in blue (matched) or red (unmatched)
    matched_pred_indices = {m['pred_idx'] for m in matches}

    for i, cell in enumerate(cells):
        bbox = cell['bbox']
        x1, y1, x2, y2 = bbox
        color = (255, 0, 0) if i in matched_pred_indices else (0, 0, 255)  # Blue=matched, Red=unmatched
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        # Add small label
        label = f"P{i}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img, (x1, y1-text_height-2), (x1+text_width, y1), color, -1)
        cv2.putText(img, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    cv2.imwrite(str(output_path), img)


def main():
    print("="*70)
    print("RAPIDTABLE EVALUATION WITH GROUND TRUTH")
    print("="*70)

    # Load GT
    gt_dict = load_ground_truth()
    print(f"\n✅ Loaded ground truth for {len(gt_dict)} images")

    # Initialize RapidTable with SLANet Plus model (best for cell detection)
    print(f"✅ Initializing RapidTable (SLANetPlus model)...")
    input_args = RapidTableInput(model_type=ModelType.SLANETPLUS)
    table_engine = RapidTable(input_args)
    print(f"✅ RapidTable initialized")

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
            print(f"  ⚠️  No ground truth annotations - running inference only")

            # Run inference to test the model
            start = time.time()
            try:
                result = table_engine(str(image_path))
                elapsed = time.time() - start

                if result and hasattr(result, 'cell_bboxes'):
                    cells = result.cell_bboxes
                    print(f"  Detections: {len(cells)}")
                    print(f"  Inference: {elapsed:.3f}s")
                else:
                    print(f"  No cells detected")
            except Exception as e:
                print(f"  ❌ Error: {e}")

            continue

        gt_regions = gt_dict[filename]

        # Run inference
        start = time.time()
        try:
            result = table_engine(str(image_path))
            elapsed = time.time() - start
            total_time += elapsed
        except Exception as e:
            print(f"  ❌ Error during inference: {e}")
            continue

        # Extract cells
        if result and hasattr(result, 'cell_bboxes'):
            cells = result.cell_bboxes
        else:
            print(f"  ⚠️  No cells detected")
            cells = []

        # Match with GT
        tp, fp, fn, matches = match_predictions_to_gt(cells, gt_regions, iou_threshold=0.5)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Detections: {len(cells)} | GT: {len(gt_regions)}")
        print(f"  TP={tp} FP={fp} FN={fn}")
        print(f"  Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")
        print(f"  Inference: {elapsed:.3f}s")

        # Draw comparison
        vis_path = OUTPUT_DIR / f"{image_path.stem}_rapidtable_comparison.png"
        draw_comparison(image_path, cells, gt_regions, matches, vis_path)
        print(f"  Saved: {vis_path.name}")

        # Store result
        results.append({
            'filename': filename,
            'num_predictions': len(cells),
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
    if len(results) > 0:
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
            'model': 'RapidTable-SLANetPlus',
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
    else:
        print("\n⚠️  No images with ground truth found")

    print("="*70)


if __name__ == "__main__":
    main()
