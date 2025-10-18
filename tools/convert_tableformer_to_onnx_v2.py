#!/usr/bin/env python3
"""
TableFormer PyTorch to ONNX Conversion Script - Version 2

This version exports TableFormer as SEPARATE ONNX models:
1. Encoder: images → encoder features
2. Decoder (single step): encoder features + decoded_tags → next_tag + bbox

The autoregressive loop will be implemented in .NET C# code.

This approach avoids ONNX tracing issues with dynamic loops and .item() calls.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from safetensors.torch import load_file

# Add docling-ibm-models to path
sys.path.insert(0, "/tmp/docling-ibm-models")

from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs


class TableFormerEncoder(nn.Module):
    """
    Encoder-only module for ONNX export.

    Input: images (batch, 3, 448, 448)
    Output: encoder_features (batch, 28, 28, 256)
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model._encoder

    def forward(self, images):
        """
        Args:
            images: (B, 3, 448, 448)
        Returns:
            encoder_features: (B, 28, 28, 256)
        """
        return self.encoder(images)


class TableFormerDecoderStep(nn.Module):
    """
    Single-step decoder for ONNX export.

    This performs ONE autoregressive decoding step:
    - Takes encoder features and previous decoded tags
    - Returns logits for next tag and bbox predictions

    The autoregressive loop will be implemented in C# .NET.
    """

    def __init__(self, model, config, word_map):
        super().__init__()
        self.model = model
        self.config = config
        self.word_map = word_map["word_map_tag"]

        # Cache transformer components
        self.tag_transformer = model._tag_transformer
        self.bbox_decoder = model._bbox_decoder if model._bbox else None

    def forward(self, encoder_out, decoded_tags):
        """
        Single decoder step.

        Args:
            encoder_out: (B, 28, 28, 256) encoder features
            decoded_tags: (seq_len, B) previously decoded tags

        Returns:
            tag_logits: (B, vocab_size) logits for next tag
            bbox_classes: (B, 3) bbox class predictions for last tag
            bbox_coords: (B, 4) bbox coordinate predictions for last tag
        """
        batch_size = encoder_out.size(0)

        # Prepare encoder output for transformer
        encoder_out_transformed = self.tag_transformer._input_filter(
            encoder_out.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)  # (B, 28, 28, 512)

        encoder_dim = encoder_out_transformed.size(-1)
        enc_inputs = encoder_out_transformed.view(batch_size, -1, encoder_dim)
        enc_inputs = enc_inputs.permute(1, 0, 2)  # (784, B, 512)
        positions = enc_inputs.shape[0]

        # Encoder mask (all zeros for no masking)
        n_heads = self.tag_transformer._n_heads
        encoder_mask = torch.zeros(
            (batch_size * n_heads, positions, positions),
            device=encoder_out.device,
            dtype=torch.bool
        )

        # Encode once
        memory = self.tag_transformer._encoder(enc_inputs, mask=encoder_mask)

        # Decode
        decoded_embedding = self.tag_transformer._embedding(decoded_tags)
        decoded_embedding = self.tag_transformer._positional_encoding(decoded_embedding)

        # NOTE: We don't use cache here for simplicity in ONNX export
        # The C# implementation can add caching for efficiency
        decoded, _ = self.tag_transformer._decoder(
            decoded_embedding,
            memory,
            cache=None,
            memory_key_padding_mask=encoder_mask,
        )

        # Get logits for the last position
        tag_logits = self.tag_transformer._fc(decoded[-1, :, :])  # (B, vocab_size)

        # Predict bbox for the last decoded position
        if self.bbox_decoder is not None:
            tag_H = decoded[-1, :, :].unsqueeze(0)  # (1, B, hidden_dim)
            bbox_classes, bbox_coords = self.bbox_decoder.inference(
                encoder_out, [tag_H.squeeze(0)]
            )
            # bbox_classes: list of (3,) tensors
            # bbox_coords: list of (4,) tensors

            if len(bbox_classes) > 0:
                bbox_classes_out = bbox_classes[0].unsqueeze(0)  # (1, 3)
                bbox_coords_out = bbox_coords[0].unsqueeze(0)    # (1, 4)
            else:
                bbox_classes_out = torch.zeros((batch_size, 3), device=encoder_out.device)
                bbox_coords_out = torch.zeros((batch_size, 4), device=encoder_out.device)
        else:
            bbox_classes_out = torch.zeros((batch_size, 3), device=encoder_out.device)
            bbox_coords_out = torch.zeros((batch_size, 4), device=encoder_out.device)

        return tag_logits, bbox_classes_out, bbox_coords_out


def load_tableformer_model(model_path, config_path, device="cpu"):
    """Load TableFormer model from safetensors."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model = TableModel04_rs(config, device=device)

    # Load weights
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def export_encoder(model, output_path, opset_version=17):
    """Export encoder to ONNX."""
    encoder_module = TableFormerEncoder(model)
    encoder_module.eval()

    dummy_input = torch.randn(1, 3, 448, 448)

    print(f"\nExporting encoder to ONNX (opset {opset_version})...")
    torch.onnx.export(
        encoder_module,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['encoder_features'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'encoder_features': {0: 'batch_size'}
        }
    )

    print(f"✅ Encoder ONNX model saved to {output_path}")

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ Encoder ONNX model is valid")

    return encoder_module(dummy_input)


def export_decoder_step(model, config, output_path, encoder_features, opset_version=17):
    """Export single-step decoder to ONNX."""
    word_map = config['dataset_wordmap']
    decoder_module = TableFormerDecoderStep(model, config, word_map)
    decoder_module.eval()

    # Dummy inputs
    batch_size = 1
    seq_len = 5  # Example sequence length
    dummy_decoded_tags = torch.randint(
        0, len(word_map["word_map_tag"]), (seq_len, batch_size), dtype=torch.long
    )

    print(f"\nExporting decoder step to ONNX (opset {opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            decoder_module,
            (encoder_features, dummy_decoded_tags),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['encoder_features', 'decoded_tags'],
            output_names=['tag_logits', 'bbox_classes', 'bbox_coords'],
            dynamic_axes={
                'encoder_features': {0: 'batch_size'},
                'decoded_tags': {0: 'seq_len', 1: 'batch_size'},
                'tag_logits': {0: 'batch_size'},
                'bbox_classes': {0: 'batch_size'},
                'bbox_coords': {0: 'batch_size'}
            }
        )

    print(f"✅ Decoder ONNX model saved to {output_path}")

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ Decoder ONNX model is valid")


def validate_models(encoder_path, decoder_path, model, config):
    """Validate ONNX models against PyTorch."""
    print("\n" + "="*70)
    print("VALIDATING ONNX MODELS")
    print("="*70)

    # Load ONNX models
    encoder_session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])

    # Test input
    test_image = torch.randn(1, 3, 448, 448)

    # PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        encoder_out_pt = model._encoder(test_image)

        # Prepare for tag transformer
        encoder_out_transformed = model._tag_transformer._input_filter(
            encoder_out_pt.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        # Start token
        word_map = config['dataset_wordmap']['word_map_tag']
        start_token = word_map["<start>"]
        decoded_tags = torch.LongTensor([[start_token]])

        # Get first prediction
        encoder_dim = encoder_out_transformed.size(-1)
        enc_inputs = encoder_out_transformed.view(1, -1, encoder_dim).permute(1, 0, 2)
        memory = model._tag_transformer._encoder(enc_inputs)

        decoded_embedding = model._tag_transformer._embedding(decoded_tags)
        decoded_embedding = model._tag_transformer._positional_encoding(decoded_embedding)
        decoded, _ = model._tag_transformer._decoder(decoded_embedding, memory, cache=None)
        tag_logits_pt = model._tag_transformer._fc(decoded[-1, :, :])

    # ONNX inference
    print("Running ONNX inference...")
    encoder_out_onnx = encoder_session.run(None, {'images': test_image.numpy()})[0]

    tag_logits_onnx, bbox_classes_onnx, bbox_coords_onnx = decoder_session.run(
        None,
        {
            'encoder_features': encoder_out_onnx,
            'decoded_tags': decoded_tags.numpy()
        }
    )

    # Compare
    print("\n" + "-"*70)
    print("COMPARISON")
    print("-"*70)

    encoder_diff = np.abs(encoder_out_pt.numpy() - encoder_out_onnx).max()
    print(f"Encoder max diff: {encoder_diff:.6f}")

    tag_logits_diff = np.abs(tag_logits_pt.numpy() - tag_logits_onnx).max()
    print(f"Tag logits max diff: {tag_logits_diff:.6f}")

    # Check predicted tags match
    pt_pred = tag_logits_pt.argmax(1).item()
    onnx_pred = tag_logits_onnx.argmax(1)[0]
    print(f"PyTorch predicted tag: {pt_pred}")
    print(f"ONNX predicted tag: {onnx_pred}")
    print(f"Match: {'✅' if pt_pred == onnx_pred else '❌'}")

    # Overall validation
    if encoder_diff < 1e-4 and tag_logits_diff < 1e-4 and pt_pred == onnx_pred:
        print("\n✅ VALIDATION SUCCESSFUL!")
        return True
    else:
        print("\n❌ VALIDATION FAILED - Differences detected")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert TableFormer to ONNX (v2 - split architecture)")
    parser.add_argument("--model", type=str, default="fast", choices=["fast", "accurate"],
                       help="Model variant to convert")
    parser.add_argument("--model-dir", type=str,
                       default="/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/models-safetensors",
                       help="Directory containing safetensors models")
    parser.add_argument("--output-dir", type=str,
                       default="/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/models-onnx-v2",
                       help="Output directory for ONNX models")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")

    args = parser.parse_args()

    # Paths
    model_dir = Path(args.model_dir) / args.model
    model_path = model_dir / f"tableformer_{args.model}.safetensors"
    config_path = model_dir / "tm_config.json"

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_output = output_dir / f"tableformer_{args.model}_encoder.onnx"
    decoder_output = output_dir / f"tableformer_{args.model}_decoder_step.onnx"

    print("="*70)
    print("TABLEFORMER → ONNX CONVERSION (V2 - Split Architecture)")
    print("="*70)
    print(f"\nModel: {model_path}")
    print(f"Config: {config_path}")
    print(f"Output encoder: {encoder_output}")
    print(f"Output decoder: {decoder_output}")
    print()

    # Load model
    print("Loading TableFormer model...")
    model, config = load_tableformer_model(str(model_path), str(config_path), device="cpu")
    print("✅ Model loaded")

    # Export encoder
    encoder_features = export_encoder(model, str(encoder_output), opset_version=args.opset)

    # Export decoder step
    export_decoder_step(model, config, str(decoder_output), encoder_features, opset_version=args.opset)

    # Validate
    is_valid = validate_models(str(encoder_output), str(decoder_output), model, config)

    # Save metadata
    metadata = {
        "model_variant": args.model,
        "opset_version": args.opset,
        "encoder_model": str(encoder_output.name),
        "decoder_model": str(decoder_output.name),
        "architecture": "split",
        "notes": "Encoder and decoder are separate models. Autoregressive loop must be implemented in application code.",
        "word_map": config['dataset_wordmap']['word_map_tag'],
        "image_size": config['dataset']['resized_image'],
        "image_normalization": config['dataset']['image_normalization']
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Metadata saved to {metadata_path}")

    if is_valid:
        print("\n" + "="*70)
        print("✅ CONVERSION SUCCESSFUL!")
        print("="*70)
        import os
        enc_size = os.path.getsize(encoder_output) / (1024 * 1024)
        dec_size = os.path.getsize(decoder_output) / (1024 * 1024)
        print(f"Encoder size: {enc_size:.1f} MB")
        print(f"Decoder size: {dec_size:.1f} MB")
        print(f"Total size: {enc_size + dec_size:.1f} MB")
        return 0
    else:
        print("\n" + "="*70)
        print("❌ CONVERSION FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
