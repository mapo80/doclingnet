#!/usr/bin/env python3
"""
Table Transformer ONNX Inference Script
Uses Xenova/table-transformer-structure-recognition-v1.1-all quantized ONNX model
Tests on FinTabNet benchmark dataset
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models"
ONNX_MODEL_PATH = MODEL_DIR / "onnx" / "model_quantized.onnx"
CONFIG_PATH = MODEL_DIR / "config.json"
PREPROCESSOR_CONFIG_PATH = MODEL_DIR / "preprocessor_config.json"
DATASET_DIR = Path(__file__).parent.parent.parent / "ds4sd-docling-tableformer-onnx" / "dataset" / "FinTabNet" / "benchmark"
OUTPUT_DIR = Path("/tmp/table_transformer_results")
VISUALIZATIONS_DIR = Path(__file__).parent.parent / "temp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_configs():
    """Load model and preprocessor configuration"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    with open(PREPROCESSOR_CONFIG_PATH) as f:
        preprocessor_config = json.load(f)
    return config, preprocessor_config


def preprocess_image(image_path, preprocessor_config):
    """
    Preprocess image for Table Transformer model
    Following HuggingFace preprocessing pipeline
    """
    # Get preprocessing params
    size = preprocessor_config.get("size", {})
    if isinstance(size, dict):
        target_size = (size.get("height", 800), size.get("width", 800))
    else:
        target_size = (size, size) if isinstance(size, int) else (800, 800)

    image_mean = preprocessor_config.get("image_mean", [0.485, 0.456, 0.406])
    image_std = preprocessor_config.get("image_std", [0.229, 0.224, 0.225])

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img_rgb, target_size)

    # Normalize
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
    img_normalized = (img_float - mean) / std

    # Convert to [1, 3, H, W]
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)

    return img_tensor


def run_inference(session, image_tensor):
    """Run ONNX inference"""
    input_name = session.get_inputs()[0].name

    start = time.time()
    outputs = session.run(None, {input_name: image_tensor})
    elapsed = time.time() - start

    return outputs, elapsed


def decode_outputs(outputs, config):
    """
    Decode model outputs to bounding boxes and labels
    Table Transformer outputs: logits and boxes

    The model outputs raw logits (not probabilities). We need to:
    1. Apply softmax to convert logits to probabilities
    2. Filter out "no-object" class (usually last class)
    3. Apply confidence threshold on probabilities
    """
    # outputs[0]: logits [batch, num_queries, num_classes]
    # outputs[1]: boxes [batch, num_queries, 4] in [cx, cy, w, h] format normalized [0,1]

    logits = outputs[0]
    boxes = outputs[1]

    # Apply softmax to get probabilities
    # Subtract max for numerical stability
    logits_stable = logits[0] - np.max(logits[0], axis=-1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Get predicted classes and their probabilities
    pred_classes = np.argmax(probs, axis=-1)
    pred_scores = np.max(probs, axis=-1)

    # Filter by confidence threshold (now on probabilities)
    # Also filter out "no-object" class (typically class index num_classes-1)
    num_classes = probs.shape[-1]
    no_object_class = num_classes - 1

    threshold = 0.7  # Confidence threshold on probabilities
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


def draw_boxes(image_path, boxes, classes, scores, output_path):
    """
    Draw bounding boxes on image
    boxes: [N, 4] in normalized [cx, cy, w, h] format
    """
    # Load original image
    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w = img.shape[:2]

    # Calculate font scale and thickness based on image size
    # Scale factor: normalize to 1000px width
    scale_factor = w / 1000.0
    font_scale = 0.4 * scale_factor
    font_thickness = max(1, int(1 * scale_factor))
    box_thickness = max(2, int(2 * scale_factor))

    # Colors for different classes (BGR format)
    colors = [
        (0, 255, 0),     # Green - class 0 (table)
        (255, 0, 0),     # Blue - class 1 (table column)
        (0, 0, 255),     # Red - class 2 (table row)
        (255, 255, 0),   # Cyan - class 3 (table column header)
        (255, 0, 255),   # Magenta - class 4 (table projected row header)
        (0, 255, 255),   # Yellow - class 5 (table spanning cell)
        (128, 0, 128),   # Purple - class 6
    ]

    # Draw each box
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        # Convert from [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixel coordinates
        cx, cy, bw, bh = box
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        # Get color for this class
        color = colors[int(cls) % len(colors)]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        # Draw label with background for better visibility
        label = f"C{int(cls)}:{score:.2f}"

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Draw background rectangle for text
        label_y = y1 - 5 if y1 - 5 - text_height > 0 else y1 + text_height + 5
        cv2.rectangle(img, (x1, label_y - text_height - 2), (x1 + text_width, label_y + 2), color, -1)

        # Draw text
        cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Save
    cv2.imwrite(str(output_path), img)
    print(f"  Visualization saved: {output_path.name}")


def main():
    print("="*70)
    print("TABLE TRANSFORMER ONNX INFERENCE - FINTABNET BENCHMARK")
    print("="*70)

    # Load configs
    print("\nLoading configurations...")
    config, preprocessor_config = load_configs()
    print(f"✅ Config loaded")
    print(f"   Model: {config.get('_name_or_path', 'table-transformer')}")
    print(f"   Num labels: {config.get('num_labels', 'N/A')}")

    # Load ONNX model
    print(f"\nLoading ONNX model from {ONNX_MODEL_PATH}...")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    print(f"✅ ONNX model loaded")

    # Print model info
    input_info = session.get_inputs()[0]
    print(f"   Input name: {input_info.name}")
    print(f"   Input shape: {input_info.shape}")
    print(f"   Input type: {input_info.type}")

    # Get benchmark images
    image_files = sorted(DATASET_DIR.glob("*.png"))
    print(f"\nFound {len(image_files)} images in benchmark dataset")

    if len(image_files) == 0:
        print(f"❌ No images found in {DATASET_DIR}")
        return

    # Run inference on all images
    results = []
    total_time = 0.0

    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'-'*70}")
        print(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        print(f"{'-'*70}")

        # Preprocess
        img_tensor = preprocess_image(image_path, preprocessor_config)
        print(f"Image preprocessed: {img_tensor.shape}")

        # Run inference
        outputs, elapsed = run_inference(session, img_tensor)
        total_time += elapsed
        print(f"Inference time: {elapsed:.3f}s")

        # Decode outputs
        detections = decode_outputs(outputs, config)
        print(f"Detections: {detections['num_detections']}")
        print(f"  Boxes shape: {detections['boxes'].shape}")
        print(f"  Classes: {detections['classes'][:10]}")  # First 10
        print(f"  Scores (first 5): {detections['scores'][:5]}")

        # Draw visualizations
        vis_path = VISUALIZATIONS_DIR / f"{image_path.stem}_detections.png"
        draw_boxes(image_path, detections['boxes'], detections['classes'], detections['scores'], vis_path)

        # Store result
        result = {
            'image': image_path.name,
            'num_detections': int(detections['num_detections']),
            'inference_time': float(elapsed),
            'boxes': detections['boxes'].tolist(),
            'classes': detections['classes'].tolist(),
            'scores': detections['scores'].tolist()
        }
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTotal images: {len(results)}")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average inference time: {total_time/len(results):.3f}s")
    print(f"Average detections per image: {np.mean([r['num_detections'] for r in results]):.1f}")

    # Save results
    output_file = OUTPUT_DIR / "table_transformer_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model': str(ONNX_MODEL_PATH),
            'dataset': str(DATASET_DIR),
            'total_images': len(results),
            'total_time': total_time,
            'avg_time': total_time / len(results),
            'results': results
        }, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
