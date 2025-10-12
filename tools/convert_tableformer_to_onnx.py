#!/usr/bin/env python3
"""
TableFormer PyTorch to ONNX Conversion Script

Converts the official Docling TableFormer models (safetensors) to ONNX format
for use in the .NET backend.

Usage:
    python convert_tableformer_to_onnx.py --model fast --output ./models/onnx/
    python convert_tableformer_to_onnx.py --model accurate --output ./models/onnx/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from safetensors.torch import load_file

# Add docling-ibm-models to path
sys.path.insert(0, "/tmp/docling-ibm-models")

from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs
from docling_ibm_models.tableformer.models.table04_rs.encoder04_rs import Encoder04
from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import Tag_Transformer
from docling_ibm_models.tableformer.models.table04_rs.bbox_decoder_rs import BBoxDecoder


class TableFormerONNXWrapper(nn.Module):
    """
    Wrapper for TableFormer model to make it ONNX-exportable.

    The original model has an autoregressive loop which is not directly
    exportable to ONNX. This wrapper implements the full prediction loop
    inside the forward method.
    """

    def __init__(self, model, config, word_map, max_steps=1024):
        super().__init__()
        self.model = model
        self.config = config
        self.word_map = word_map["word_map_tag"]
        self.max_steps = max_steps
        self.device = next(model.parameters()).device

        # Pre-compute special tokens
        self.start_token = self.word_map["<start>"]
        self.end_token = self.word_map["<end>"]
        self.pad_token = self.word_map["<pad>"]

        # Tokens that require bbox prediction
        self.bbox_tokens = set([
            self.word_map.get("fcel", -1),
            self.word_map.get("ecel", -1),
            self.word_map.get("ched", -1),
            self.word_map.get("rhed", -1),
            self.word_map.get("srow", -1),
            self.word_map.get("nl", -1),
            self.word_map.get("ucel", -1),
        ])

    def forward(self, images):
        """
        Forward pass with full autoregressive loop.

        Args:
            images: (batch_size, 3, 448, 448) input images

        Returns:
            tags: (batch_size, max_steps) predicted tag sequence (padded)
            bbox_classes: (batch_size, max_steps, 2) bbox class predictions (padded)
            bbox_coords: (batch_size, max_steps, 4) bbox coordinates (padded)
            valid_length: (batch_size,) actual length of predictions (before padding)
        """
        batch_size = images.size(0)
        assert batch_size == 1, "Currently only batch_size=1 is supported"

        # Encoder
        enc_out = self.model._encoder(images)  # (1, 28, 28, 256)

        # Prepare for tag transformer
        encoder_out = self.model._tag_transformer._input_filter(
            enc_out.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)  # (1, 28, 28, 512)

        encoder_dim = encoder_out.size(-1)
        enc_inputs = encoder_out.view(batch_size, -1, encoder_dim)
        enc_inputs = enc_inputs.permute(1, 0, 2)  # (784, 1, 512)
        positions = enc_inputs.shape[0]

        n_heads = self.model._tag_transformer._n_heads
        encoder_mask = torch.zeros(
            (batch_size * n_heads, positions, positions),
            device=self.device,
            dtype=torch.bool
        )

        # Encode once
        memory = self.model._tag_transformer._encoder(enc_inputs, mask=encoder_mask)

        # Initialize outputs (padded)
        # Note: We'll set bbox_classes dimension dynamically after first bbox prediction
        output_tags = torch.full(
            (batch_size, self.max_steps),
            self.pad_token,
            dtype=torch.long,
            device=self.device
        )
        output_bbox_classes = None  # Will be initialized after first bbox prediction
        output_bbox_coords = torch.zeros(
            (batch_size, self.max_steps, 4),
            dtype=torch.float32,
            device=self.device
        )
        valid_length = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Autoregressive decoding
        decoded_tags = torch.LongTensor([[self.start_token]]).to(self.device)
        cache = None
        tag_H_buf = []

        # Tracking for bbox prediction
        skip_next_tag = True
        prev_tag_ucel = False
        first_lcel = True
        line_num = 0
        cur_bbox_ind = -1
        bbox_ind = 0
        bboxes_to_merge = {}

        step = 0
        while step < self.max_steps:
            # Decode one step
            decoded_embedding = self.model._tag_transformer._embedding(decoded_tags)
            decoded_embedding = self.model._tag_transformer._positional_encoding(
                decoded_embedding
            )
            decoded, cache = self.model._tag_transformer._decoder(
                decoded_embedding,
                memory,
                cache,
                memory_key_padding_mask=encoder_mask,
            )
            logits = self.model._tag_transformer._fc(decoded[-1, :, :])
            new_tag = logits.argmax(1).item()

            # Structure error correction
            if line_num == 0 and new_tag == self.word_map.get("xcel", -1):
                new_tag = self.word_map.get("lcel", -1)

            if prev_tag_ucel and new_tag == self.word_map.get("lcel", -1):
                new_tag = self.word_map.get("fcel", -1)

            # End of generation
            if new_tag == self.end_token:
                output_tags[0, step] = new_tag
                valid_length[0] = step + 1
                break

            output_tags[0, step] = new_tag

            # Track which tags need bbox
            if not skip_next_tag:
                if new_tag in self.bbox_tokens:
                    tag_H_buf.append(decoded[-1, :, :])
                    if not first_lcel:
                        bboxes_to_merge[cur_bbox_ind] = bbox_ind
                    bbox_ind += 1

            # Handle lcel (horizontal span)
            lcel_token = self.word_map.get("lcel", -1)
            if new_tag != lcel_token:
                first_lcel = True
            else:
                if first_lcel:
                    tag_H_buf.append(decoded[-1, :, :])
                    first_lcel = False
                    cur_bbox_ind = bbox_ind
                    bboxes_to_merge[cur_bbox_ind] = -1
                    bbox_ind += 1

            # Update skip flag
            nl_token = self.word_map.get("nl", -1)
            ucel_token = self.word_map.get("ucel", -1)
            xcel_token = self.word_map.get("xcel", -1)
            if new_tag in [nl_token, ucel_token, xcel_token]:
                skip_next_tag = True
            else:
                skip_next_tag = False

            # Track ucel
            prev_tag_ucel = (new_tag == ucel_token)

            # Append to decoded sequence
            decoded_tags = torch.cat([
                decoded_tags,
                torch.LongTensor([[new_tag]]).to(self.device)
            ], dim=0)

            step += 1

        # If we didn't hit end token, record length
        if valid_length[0] == 0:
            valid_length[0] = step

        # BBox prediction for all collected cells
        if len(tag_H_buf) > 0 and self.model._bbox:
            bbox_classes, bbox_coords = self.model._bbox_decoder.inference(
                enc_out, tag_H_buf
            )

            # Merge bboxes for spans
            bbox_classes_merged, bbox_coords_merged = self._merge_bboxes(
                bbox_classes, bbox_coords, bboxes_to_merge
            )

            # Copy to output (padded)
            num_bboxes = min(len(bbox_classes_merged), self.max_steps)
            if num_bboxes > 0:
                # Initialize output_bbox_classes with actual dimension from model output
                if output_bbox_classes is None:
                    actual_num_classes = bbox_classes_merged[0].shape[0]
                    output_bbox_classes = torch.zeros(
                        (batch_size, self.max_steps, actual_num_classes),
                        dtype=torch.float32,
                        device=self.device
                    )

                output_bbox_classes[0, :num_bboxes] = torch.stack(bbox_classes_merged[:num_bboxes])
                output_bbox_coords[0, :num_bboxes] = torch.stack(bbox_coords_merged[:num_bboxes])

        # If no bboxes were predicted, initialize with default dimension (3)
        if output_bbox_classes is None:
            output_bbox_classes = torch.zeros(
                (batch_size, self.max_steps, 3),
                dtype=torch.float32,
                device=self.device
            )

        return output_tags, output_bbox_classes, output_bbox_coords, valid_length

    def _merge_bboxes(self, bbox_classes, bbox_coords, bboxes_to_merge):
        """Merge bounding boxes for spanning cells."""
        classes_merged = []
        coords_merged = []
        boxes_to_skip = []

        for box_ind in range(len(bbox_coords)):
            box1 = bbox_coords[box_ind]
            cls1 = bbox_classes[box_ind]

            if box_ind in bboxes_to_merge:
                box2_ind = bboxes_to_merge[box_ind]
                if box2_ind >= 0 and box2_ind < len(bbox_coords):
                    box2 = bbox_coords[box2_ind]
                    boxes_to_skip.append(box2_ind)
                    boxm = self._merge_two_boxes(box1, box2)
                    coords_merged.append(boxm)
                    classes_merged.append(cls1)
                else:
                    coords_merged.append(box1)
                    classes_merged.append(cls1)
            else:
                if box_ind not in boxes_to_skip:
                    coords_merged.append(box1)
                    classes_merged.append(cls1)

        return classes_merged, coords_merged

    def _merge_two_boxes(self, bbox1, bbox2):
        """Merge two bboxes in [cx, cy, w, h] format."""
        # Convert to corners
        left1 = bbox1[0] - bbox1[2] / 2
        top1 = bbox1[1] - bbox1[3] / 2
        right1 = bbox1[0] + bbox1[2] / 2
        bottom1 = bbox1[1] + bbox1[3] / 2

        left2 = bbox2[0] - bbox2[2] / 2
        top2 = bbox2[1] - bbox2[3] / 2
        right2 = bbox2[0] + bbox2[2] / 2
        bottom2 = bbox2[1] + bbox2[3] / 2

        # Merge
        new_left = min(left1, left2)
        new_top = min(top1, top2)
        new_right = max(right1, right2)
        new_bottom = max(bottom1, bottom2)

        # Convert back to centroid format
        new_w = new_right - new_left
        new_h = new_bottom - new_top
        new_cx = new_left + new_w / 2
        new_cy = new_top + new_h / 2

        return torch.tensor([new_cx, new_cy, new_w, new_h], device=bbox1.device)


def load_tableformer_model(model_path, config_path, device="cpu"):
    """Load TableFormer model from safetensors."""
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Add required fields for model initialization
    if "save_dir" not in config.get("model", {}):
        config.setdefault("model", {})["save_dir"] = "/tmp/tableformer"

    print(f"Loading model weights from {model_path}")
    state_dict = load_file(model_path)

    # Prepare init_data with word_map
    init_data = {
        "word_map": config["dataset_wordmap"]
    }

    # Create model
    print("Creating TableModel04_rs...")
    model = TableModel04_rs(config, init_data, device)

    # Load weights
    print("Loading state dict...")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("Model loaded successfully!")
    return model, config


def export_to_onnx(model, config, output_path, opset_version=17):
    """Export TableFormer model to ONNX."""
    device = next(model.parameters()).device
    word_map = config["dataset_wordmap"]
    max_steps = config["predict"]["max_steps"]

    print(f"Creating ONNX wrapper (max_steps={max_steps})...")
    wrapper = TableFormerONNXWrapper(model, config, word_map, max_steps)
    wrapper.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 448, 448, device=device)

    print("Running forward pass to test...")
    with torch.no_grad():
        test_output = wrapper(dummy_input)
        tags, bbox_classes, bbox_coords, valid_length = test_output
        print(f"  Tags shape: {tags.shape}")
        print(f"  BBox classes shape: {bbox_classes.shape}")
        print(f"  BBox coords shape: {bbox_coords.shape}")
        print(f"  Valid length: {valid_length.item()}")

    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["tags", "bbox_classes", "bbox_coords", "valid_length"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "tags": {0: "batch_size"},
            "bbox_classes": {0: "batch_size"},
            "bbox_coords": {0: "batch_size"},
            "valid_length": {0: "batch_size"},
        },
    )

    print(f"ONNX model saved to {output_path}")
    return test_output


def validate_onnx_model(onnx_path, pytorch_output, dummy_input):
    """Validate ONNX model against PyTorch output."""
    print("\nValidating ONNX model...")

    # Load ONNX model
    print("Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid!")

    # Run with ONNX Runtime
    print("Running inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    ort_inputs = {"images": dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Compare outputs
    tags_pt, bbox_classes_pt, bbox_coords_pt, valid_length_pt = pytorch_output
    tags_ort, bbox_classes_ort, bbox_coords_ort, valid_length_ort = ort_outputs

    print("\nNumerical validation:")

    # Tags (exact match expected)
    tags_match = (tags_pt.cpu().numpy() == tags_ort).all()
    print(f"  Tags match: {tags_match}")

    # Valid length (exact match)
    vl_match = (valid_length_pt.cpu().numpy() == valid_length_ort).all()
    print(f"  Valid length match: {vl_match}")

    # BBox classes (close match)
    bbox_classes_diff = abs(bbox_classes_pt.cpu().numpy() - bbox_classes_ort).max()
    print(f"  BBox classes max diff: {bbox_classes_diff:.6f}")

    # BBox coords (close match)
    bbox_coords_diff = abs(bbox_coords_pt.cpu().numpy() - bbox_coords_ort).max()
    print(f"  BBox coords max diff: {bbox_coords_diff:.6f}")

    # Overall validation
    is_valid = (
        tags_match and
        vl_match and
        bbox_classes_diff < 1e-4 and
        bbox_coords_diff < 1e-4
    )

    if is_valid:
        print("\n✅ Validation PASSED! ONNX model matches PyTorch output.")
    else:
        print("\n❌ Validation FAILED! Differences detected.")

    return is_valid


def main():
    parser = argparse.ArgumentParser(description="Convert TableFormer to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["fast", "accurate"],
        help="Model variant to convert"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for conversion"
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "models" / "tableformer" / args.model / "model_artifacts" / "tableformer" / args.model
    model_path = model_dir / f"tableformer_{args.model}.safetensors"
    config_path = model_dir / "tm_config.json"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tableformer_{args.model}.onnx"

    print(f"========================================")
    print(f"TableFormer to ONNX Conversion")
    print(f"========================================")
    print(f"Model variant: {args.model}")
    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")
    print(f"Output path: {output_path}")
    print(f"Device: {args.device}")
    print(f"========================================\n")

    # Load model
    device = torch.device(args.device)
    model, config = load_tableformer_model(str(model_path), str(config_path), device)

    # Create dummy input for export and validation
    dummy_input = torch.randn(1, 3, 448, 448, device=device)

    # Export to ONNX
    pytorch_output = export_to_onnx(model, config, str(output_path), args.opset)

    # Validate
    is_valid = validate_onnx_model(str(output_path), pytorch_output, dummy_input)

    # Save config alongside ONNX
    output_config_path = output_dir / f"tableformer_{args.model}_config.json"
    print(f"\nSaving config to {output_config_path}")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n========================================")
    if is_valid:
        print("✅ Conversion completed successfully!")
    else:
        print("⚠️  Conversion completed with validation warnings")
    print(f"ONNX model: {output_path}")
    print(f"Config: {output_config_path}")
    print("========================================")

    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
