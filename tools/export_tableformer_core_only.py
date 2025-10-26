#!/usr/bin/env python3
"""
Export TableFormer CORE only (no autoregressive loop)

This exports ONLY the forward pass of the model without any loop:
- Input: image + decoded_sequence_so_far
- Output: logits for next token + bbox predictions

The autoregressive loop will be implemented separately in Python.
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file

sys.path.insert(0, "/tmp/docling-ibm-models")
from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs


class TableFormerCoreONNX(nn.Module):
    """
    TableFormer CORE module - single forward pass without loop.

    This wraps the model to do ONE decoding step:
    - Encode image
    - Decode given sequence
    - Return next token logits + bbox for last decoded position
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, images, decoded_tags):
        """
        Single forward pass.

        Args:
            images: (B, 3, 448, 448) input images
            decoded_tags: (seq_len, B) previously decoded tag indices

        Returns:
            next_tag_logits: (B, vocab_size) logits for next tag
            last_bbox_classes: (B, 3) bbox class probs for last position
            last_bbox_coords: (B, 4) bbox coordinates for last position
        """
        batch_size = images.size(0)
        device = images.device

        # 1. Encoder
        enc_out = self.model._encoder(images)  # (B, 28, 28, 256)

        # 2. Tag Transformer Encoder
        encoder_out = self.model._tag_transformer._input_filter(
            enc_out.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)  # (B, 28, 28, 512)

        encoder_dim = encoder_out.size(-1)
        enc_inputs = encoder_out.view(batch_size, -1, encoder_dim)
        enc_inputs = enc_inputs.permute(1, 0, 2)  # (784, B, 512)
        positions = enc_inputs.shape[0]

        n_heads = self.model._tag_transformer._n_heads
        encoder_mask = torch.zeros(
            (batch_size * n_heads, positions, positions),
            device=device,
            dtype=torch.bool
        )

        memory = self.model._tag_transformer._encoder(enc_inputs, mask=encoder_mask)

        # 3. Tag Transformer Decoder (for given sequence)
        decoded_embedding = self.model._tag_transformer._embedding(decoded_tags)
        decoded_embedding = self.model._tag_transformer._positional_encoding(decoded_embedding)

        # NOTE: No cache for simplicity in ONNX
        decoded, _ = self.model._tag_transformer._decoder(
            decoded_embedding,
            memory,
            cache=None,
            memory_key_padding_mask=encoder_mask,
        )

        # Get logits for NEXT token (after last position in sequence)
        next_tag_logits = self.model._tag_transformer._fc(decoded[-1, :, :])  # (B, vocab_size)

        # 4. BBox for last decoded position
        last_tag_hidden = decoded[-1, :, :]  # (B, 512)

        if self.model._bbox:
            bbox_classes, bbox_coords = self.model._bbox_decoder.inference(
                enc_out, [last_tag_hidden]
            )
            # bbox_classes: list of (3,), bbox_coords: list of (4,)

            if len(bbox_classes) > 0:
                last_bbox_classes = bbox_classes[0].unsqueeze(0)  # (1, 3)
                last_bbox_coords = bbox_coords[0].unsqueeze(0)    # (1, 4)
            else:
                last_bbox_classes = torch.zeros((batch_size, 3), device=device)
                last_bbox_coords = torch.zeros((batch_size, 4), device=device)
        else:
            last_bbox_classes = torch.zeros((batch_size, 3), device=device)
            last_bbox_coords = torch.zeros((batch_size, 4), device=device)

        return next_tag_logits, last_bbox_classes, last_bbox_coords


def load_model(model_path, config_path):
    """Load TableFormer model from safetensors."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Add required fields for model initialization
    if "save_dir" not in config.get("model", {}):
        config.setdefault("model", {})["save_dir"] = "/tmp/tableformer"

    # Create model
    model = TableModel04_rs(config, init_data={}, device="cpu")

    # Load weights
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def export_core(model, config, output_path, opset_version=17):
    """Export core model to ONNX."""
    core_model = TableFormerCoreONNX(model, config)
    core_model.eval()

    # Dummy inputs
    batch_size = 1
    seq_len = 5  # Example: <start> + 4 decoded tokens

    dummy_images = torch.randn(batch_size, 3, 448, 448)
    word_map = config['dataset_wordmap']['word_map_tag']
    dummy_decoded_tags = torch.randint(0, len(word_map), (seq_len, batch_size), dtype=torch.long)

    print(f"\nExporting TableFormer Core to ONNX...")
    print(f"  Dummy image shape: {dummy_images.shape}")
    print(f"  Dummy tags shape: {dummy_decoded_tags.shape}")

    # Test forward pass
    with torch.no_grad():
        logits, bbox_cls, bbox_coords = core_model(dummy_images, dummy_decoded_tags)
        print(f"\n  Test forward pass:")
        print(f"    Next tag logits: {logits.shape}")
        print(f"    BBox classes: {bbox_cls.shape}")
        print(f"    BBox coords: {bbox_coords.shape}")

    # Export
    print(f"\n  Exporting to {output_path}...")
    torch.onnx.export(
        core_model,
        (dummy_images, dummy_decoded_tags),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images', 'decoded_tags'],
        output_names=['next_tag_logits', 'last_bbox_classes', 'last_bbox_coords'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'decoded_tags': {0: 'seq_len', 1: 'batch_size'},
            'next_tag_logits': {0: 'batch_size'},
            'last_bbox_classes': {0: 'batch_size'},
            'last_bbox_coords': {0: 'batch_size'}
        }
    )

    print(f"✅ Export complete!")

    # Validate ONNX
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model is valid")

    return logits, bbox_cls, bbox_coords


def validate_onnx(onnx_path, pytorch_outputs, dummy_images, dummy_decoded_tags):
    """Validate ONNX model against PyTorch."""
    import onnxruntime as ort
    import numpy as np

    print(f"\n{'='*70}")
    print("VALIDATING ONNX MODEL")
    print(f"{'='*70}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    onnx_outputs = session.run(
        None,
        {
            'images': dummy_images.numpy(),
            'decoded_tags': dummy_decoded_tags.numpy()
        }
    )

    logits_onnx, bbox_cls_onnx, bbox_coords_onnx = onnx_outputs
    logits_pt, bbox_cls_pt, bbox_coords_pt = pytorch_outputs

    # Compare
    logits_diff = np.abs(logits_pt.numpy() - logits_onnx).max()
    bbox_cls_diff = np.abs(bbox_cls_pt.numpy() - bbox_cls_onnx).max()
    bbox_coords_diff = np.abs(bbox_coords_pt.numpy() - bbox_coords_onnx).max()

    print(f"\nNumerical differences:")
    print(f"  Next tag logits: max_diff = {logits_diff:.6f}")
    print(f"  BBox classes:    max_diff = {bbox_cls_diff:.6f}")
    print(f"  BBox coords:     max_diff = {bbox_coords_diff:.6f}")

    # Check predictions match
    pt_pred = logits_pt.argmax(1).item()
    onnx_pred = logits_onnx.argmax(1)[0]

    print(f"\nPredicted next tag:")
    print(f"  PyTorch: {pt_pred}")
    print(f"  ONNX:    {onnx_pred}")
    print(f"  Match:   {'✅' if pt_pred == onnx_pred else '❌'}")

    # Overall validation
    threshold = 1e-4
    is_valid = (
        logits_diff < threshold and
        bbox_cls_diff < threshold and
        bbox_coords_diff < threshold and
        pt_pred == onnx_pred
    )

    if is_valid:
        print(f"\n{'='*70}")
        print("✅ VALIDATION PASSED!")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("❌ VALIDATION FAILED - Differences exceed threshold")
        print(f"{'='*70}")

    return is_valid


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export TableFormer Core (no loop) to ONNX")
    parser.add_argument("--model", default="models/model_artifacts/tableformer/fast/tableformer_fast.safetensors")
    parser.add_argument("--config", default="models/model_artifacts/tableformer/fast/tm_config.json")
    parser.add_argument("--output", default="models/onnx-core/tableformer_fast_core.onnx")
    parser.add_argument("--opset", type=int, default=17)

    args = parser.parse_args()

    print("="*70)
    print("TABLEFORMER CORE-ONLY ONNX EXPORT")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Opset: {args.opset}")

    # Create output dir
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model...")
    model, config = load_model(args.model, args.config)
    print(f"✅ Model loaded")

    # Export
    batch_size = 1
    seq_len = 5
    dummy_images = torch.randn(batch_size, 3, 448, 448)
    word_map = config['dataset_wordmap']['word_map_tag']
    dummy_decoded_tags = torch.randint(0, len(word_map), (seq_len, batch_size), dtype=torch.long)

    pytorch_outputs = export_core(model, config, str(output_path), opset_version=args.opset)

    # Validate
    is_valid = validate_onnx(str(output_path), pytorch_outputs, dummy_images, dummy_decoded_tags)

    # Save metadata
    metadata = {
        "architecture": "core_only",
        "notes": "This ONNX model contains ONLY the core forward pass. Autoregressive loop must be implemented separately.",
        "word_map": config['dataset_wordmap']['word_map_tag'],
        "image_size": config['dataset']['resized_image'],
        "image_normalization": config['dataset']['image_normalization']
    }

    metadata_path = output_path.parent / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Metadata saved to {metadata_path}")

    if is_valid:
        import os
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n{'='*70}")
        print("✅ EXPORT SUCCESSFUL!")
        print(f"{'='*70}")
        print(f"Model size: {size_mb:.1f} MB")
        print(f"Output: {output_path}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
