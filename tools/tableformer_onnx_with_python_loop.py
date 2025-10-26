#!/usr/bin/env python3
"""
TableFormer ONNX Inference with Python Autoregressive Loop

This script uses the 4 ONNX component models with the autoregressive loop
implemented in Python (NOT inside ONNX).

Architecture:
1. Encoder ONNX: image → encoder features
2. Tag Transformer Encoder ONNX: encoder features → memory
3. Tag Transformer Decoder Step ONNX: memory + decoded_tags → next_tag_logits + tag_hidden
4. BBox Decoder ONNX: encoder features + tag_hidden_states → bboxes

The loop is implemented in Python for maximum flexibility and correctness.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class TableFormerONNXWithPythonLoop:
    """TableFormer inference using ONNX components with Python autoregressive loop."""

    def __init__(self, model_dir):
        """
        Initialize ONNX models.

        Args:
            model_dir: Directory containing the 4 ONNX component files
        """
        self.model_dir = Path(model_dir)

        print("Loading ONNX component models...")

        # Load all 4 components
        self.encoder_session = ort.InferenceSession(
            str(self.model_dir / "tableformer_fast_encoder.onnx"),
            providers=["CPUExecutionProvider"]
        )
        print("  ✅ Encoder loaded")

        self.tag_encoder_session = ort.InferenceSession(
            str(self.model_dir / "tableformer_fast_tag_transformer_encoder.onnx"),
            providers=["CPUExecutionProvider"]
        )
        print("  ✅ Tag Transformer Encoder loaded")

        self.tag_decoder_session = ort.InferenceSession(
            str(self.model_dir / "tableformer_fast_tag_transformer_decoder_step.onnx"),
            providers=["CPUExecutionProvider"]
        )
        print("  ✅ Tag Transformer Decoder Step loaded")

        self.bbox_decoder_session = ort.InferenceSession(
            str(self.model_dir / "tableformer_fast_bbox_decoder.onnx"),
            providers=["CPUExecutionProvider"]
        )
        print("  ✅ BBox Decoder loaded")

        # Load config and word map
        with open(self.model_dir / "tableformer_fast_config.json") as f:
            self.config = json.load(f)

        with open(self.model_dir / "tableformer_fast_wordmap.json") as f:
            word_map_data = json.load(f)
            self.word_map = word_map_data["word_map_tag"]
            self.inv_word_map = {v: k for k, v in self.word_map.items()}

        # Special tokens
        self.start_token = self.word_map["<start>"]
        self.end_token = self.word_map["<end>"]
        self.pad_token = self.word_map["<pad>"]

        # Tokens that need bbox
        self.bbox_tokens = {
            self.word_map.get("fcel", -1),
            self.word_map.get("ecel", -1),
            self.word_map.get("ched", -1),
            self.word_map.get("rhed", -1),
            self.word_map.get("srow", -1),
            self.word_map.get("nl", -1),
            self.word_map.get("ucel", -1),
        }

        print(f"✅ All models loaded")
        print(f"   Vocabulary size: {len(self.word_map)}")
        print(f"   Special tokens: <start>={self.start_token}, <end>={self.end_token}")

    def preprocess_image(self, image_path):
        """Preprocess image for TableFormer."""
        img_size = self.config['dataset']['resized_image']
        mean = np.array(
            self.config['dataset']['image_normalization']['mean'],
            dtype=np.float32
        ).reshape(1, 1, 3)
        std = np.array(
            self.config['dataset']['image_normalization']['std'],
            dtype=np.float32
        ).reshape(1, 1, 3)

        # Read and resize
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))

        # Normalize
        img_float = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_float - mean) / std

        # Convert to NCHW format
        img_tensor = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)

        return img_tensor, img, (orig_w, orig_h)

    def predict(self, image_path, max_steps=500):
        """
        Run full TableFormer prediction with autoregressive loop in Python.

        Args:
            image_path: Path to input image
            max_steps: Maximum number of decoding steps

        Returns:
            dict with prediction results
        """
        start_time = time.time()

        # Preprocess
        img_tensor, orig_img, orig_size = self.preprocess_image(image_path)

        print(f"\n{'='*70}")
        print(f"INFERENCE: {Path(image_path).name}")
        print(f"{'='*70}")

        # Step 1: Run Encoder
        t0 = time.time()
        encoder_out = self.encoder_session.run(None, {'images': img_tensor})[0]
        t1 = time.time()
        print(f"\n[1/4] Encoder: {t1-t0:.3f}s → shape {encoder_out.shape}")

        # Step 2: Run Tag Transformer Encoder
        t0 = time.time()
        memory = self.tag_encoder_session.run(None, {'encoder_out': encoder_out})[0]
        t1 = time.time()
        print(f"[2/4] Tag Transformer Encoder: {t1-t0:.3f}s → memory shape {memory.shape}")

        # Step 3: Autoregressive Tag Decoding (PYTHON LOOP!)
        print(f"[3/4] Autoregressive Decoding (Python loop)...")
        t0 = time.time()

        decoded_tags_ids = [self.start_token]
        decoded_tags_str = []
        tag_hidden_states = []

        # Tracking for bbox prediction (following TableFormer logic)
        skip_next_tag = True
        prev_tag_ucel = False
        first_lcel = True
        line_num = 0
        cur_bbox_ind = -1
        bbox_ind = 0
        bboxes_to_merge = {}

        step = 0
        while step < max_steps:
            # Prepare input: all decoded tags so far (seq_len, 1)
            decoded_input = np.array(decoded_tags_ids, dtype=np.int64).reshape(-1, 1)

            # Run decoder step
            outputs = self.tag_decoder_session.run(
                None,
                {
                    'memory': memory,
                    'decoded_tags': decoded_input
                }
            )
            logits = outputs[0]  # (1, vocab_size)
            tag_hidden = outputs[1]  # (1, 512)

            # Get predicted tag
            new_tag_id = int(np.argmax(logits[0]))
            new_tag_str = self.inv_word_map.get(new_tag_id, f"<unk:{new_tag_id}>")

            # Structure error correction (from TableFormer source)
            if line_num == 0 and new_tag_id == self.word_map.get("xcel", -1):
                new_tag_id = self.word_map.get("lcel", -1)
                new_tag_str = "lcel"

            if prev_tag_ucel and new_tag_id == self.word_map.get("lcel", -1):
                new_tag_id = self.word_map.get("fcel", -1)
                new_tag_str = "fcel"

            # Check for end token
            if new_tag_id == self.end_token:
                break

            decoded_tags_ids.append(new_tag_id)
            decoded_tags_str.append(new_tag_str)

            # Track which tags need bbox
            if not skip_next_tag:
                if new_tag_id in self.bbox_tokens:
                    tag_hidden_states.append(tag_hidden[0])  # (512,)
                    if not first_lcel:
                        bboxes_to_merge[cur_bbox_ind] = bbox_ind
                    bbox_ind += 1

            # Handle lcel (horizontal span)
            lcel_token = self.word_map.get("lcel", -1)
            if new_tag_id != lcel_token:
                first_lcel = True
            else:
                if first_lcel:
                    tag_hidden_states.append(tag_hidden[0])
                    first_lcel = False
                    cur_bbox_ind = bbox_ind
                    bboxes_to_merge[cur_bbox_ind] = -1
                    bbox_ind += 1

            # Update skip flag
            nl_token = self.word_map.get("nl", -1)
            ucel_token = self.word_map.get("ucel", -1)
            xcel_token = self.word_map.get("xcel", -1)
            if new_tag_id in [nl_token, ucel_token, xcel_token]:
                skip_next_tag = True
                if new_tag_id == nl_token:
                    line_num += 1
            else:
                skip_next_tag = False

            # Track ucel
            prev_tag_ucel = (new_tag_id == ucel_token)

            step += 1

        t1 = time.time()
        print(f"      Steps: {step}, Tags: {len(decoded_tags_str)}, Time: {t1-t0:.3f}s")
        print(f"      First 10 tags: {decoded_tags_str[:10]}")

        # Step 4: BBox Prediction
        t0 = time.time()
        bbox_classes_list = []
        bbox_coords_list = []

        if len(tag_hidden_states) > 0:
            # Stack all hidden states (N, 512)
            tag_hidden_batch = np.stack(tag_hidden_states, axis=0).astype(np.float32)

            # Run bbox decoder
            bbox_classes, bbox_coords = self.bbox_decoder_session.run(
                None,
                {
                    'encoder_out': encoder_out,
                    'tag_hidden_states': tag_hidden_batch
                }
            )

            # Merge bboxes for spans
            bbox_classes_merged, bbox_coords_merged = self._merge_bboxes(
                bbox_classes, bbox_coords, bboxes_to_merge
            )

            bbox_classes_list = bbox_classes_merged
            bbox_coords_list = bbox_coords_merged

        t1 = time.time()
        print(f"[4/4] BBox Decoding: {t1-t0:.3f}s → {len(bbox_coords_list)} cells")

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ TOTAL TIME: {total_time:.3f}s")
        print(f"{'='*70}")

        return {
            'tags': decoded_tags_str,
            'bbox_classes': np.array(bbox_classes_list) if bbox_classes_list else np.array([]),
            'bbox_coords': np.array(bbox_coords_list) if bbox_coords_list else np.array([]),
            'inference_time': total_time,
            'original_image': orig_img,
            'original_size': orig_size
        }

    def _merge_bboxes(self, bbox_classes, bbox_coords, bboxes_to_merge):
        """Merge bounding boxes for spanning cells."""
        classes_merged = []
        coords_merged = []

        for box_ind in range(len(bbox_coords)):
            if box_ind in bboxes_to_merge:
                merge_ind = bboxes_to_merge[box_ind]
                if merge_ind != -1:
                    # Merge with previous bbox
                    prev_coords = coords_merged[merge_ind]
                    curr_coords = bbox_coords[box_ind]

                    # Take union
                    x1 = min(prev_coords[0], curr_coords[0])
                    y1 = min(prev_coords[1], curr_coords[1])
                    x2 = max(prev_coords[2], curr_coords[2])
                    y2 = max(prev_coords[3], curr_coords[3])

                    coords_merged[merge_ind] = np.array([x1, y1, x2, y2], dtype=np.float32)
                    continue

            classes_merged.append(bbox_classes[box_ind])
            coords_merged.append(bbox_coords[box_ind])

        return classes_merged, coords_merged

    def visualize(self, result, output_path):
        """Visualize prediction with bounding boxes."""
        img = result['original_image'].copy()
        orig_w, orig_h = result['original_size']
        img_h, img_w = img.shape[:2]

        # Scale visualization
        scale = 2
        img_vis = cv2.resize(img, (img_w * scale, img_h * scale))

        # Draw bboxes
        for i, bbox in enumerate(result['bbox_coords']):
            # BBox coords are normalized [0, 1] relative to resized image (448x448)
            x1 = int(bbox[0] * img_w * scale)
            y1 = int(bbox[1] * img_h * scale)
            x2 = int(bbox[2] * img_w * scale)
            y2 = int(bbox[3] * img_h * scale)

            # Draw rectangle
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw cell number
            cv2.putText(
                img_vis, str(i), (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )

        # Save
        cv2.imwrite(str(output_path), img_vis)
        print(f"\n✅ Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TableFormer ONNX Inference with Python Autoregressive Loop"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/onnx-components",
        help="Directory containing ONNX component models"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization with bboxes"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for visualization"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum decoding steps"
    )

    args = parser.parse_args()

    # Initialize inference
    inference = TableFormerONNXWithPythonLoop(args.model_dir)

    # Run prediction
    result = inference.predict(args.image, max_steps=args.max_steps)

    # Print summary
    print(f"\n📊 RESULTS:")
    print(f"   Tags: {len(result['tags'])}")
    print(f"   Cells: {len(result['bbox_coords'])}")
    print(f"   Time: {result['inference_time']:.3f}s")

    # Visualize
    if args.visualize:
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.image)
            output_path = input_path.parent / f"{input_path.stem}_tableformer_onnx.png"

        inference.visualize(result, output_path)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
