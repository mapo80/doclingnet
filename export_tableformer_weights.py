#!/usr/bin/env python3
"""
Export TableFormer weights from safetensors to individual .pt files for C# loading.
"""

import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file


def export_weights(safetensors_path, output_dir):
    """Export all weights to individual PyTorch files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading safetensors from: {safetensors_path}")
    state_dict = load_file(safetensors_path)
    print(f"Total tensors: {len(state_dict)}\n")

    # Group tensors by module
    encoder_tensors = {}
    tag_transformer_tensors = {}
    bbox_decoder_tensors = {}
    other_tensors = {}

    for name, tensor in state_dict.items():
        if name.startswith("_encoder."):
            encoder_tensors[name] = tensor
        elif name.startswith("_tag_transformer."):
            tag_transformer_tensors[name] = tensor
        elif name.startswith("_bbox_decoder."):
            bbox_decoder_tensors[name] = tensor
        else:
            other_tensors[name] = tensor

    print(f"Encoder tensors: {len(encoder_tensors)}")
    print(f"Tag Transformer tensors: {len(tag_transformer_tensors)}")
    print(f"BBox Decoder tensors: {len(bbox_decoder_tensors)}")
    print(f"Other tensors: {len(other_tensors)}\n")

    # Save grouped state dicts
    encoder_dict = {k: v for k, v in encoder_tensors.items()}
    tag_transformer_dict = {k: v for k, v in tag_transformer_tensors.items()}
    bbox_decoder_dict = {k: v for k, v in bbox_decoder_tensors.items()}

    encoder_path = output_dir / "encoder_weights.pt"
    tag_transformer_path = output_dir / "tag_transformer_weights.pt"
    bbox_decoder_path = output_dir / "bbox_decoder_weights.pt"
    full_path = output_dir / "full_model_weights.pt"

    print("Saving weight files...")
    torch.save(encoder_dict, encoder_path)
    print(f"✅ Encoder weights saved to: {encoder_path}")

    torch.save(tag_transformer_dict, tag_transformer_path)
    print(f"✅ Tag Transformer weights saved to: {tag_transformer_path}")

    torch.save(bbox_decoder_dict, bbox_decoder_path)
    print(f"✅ BBox Decoder weights saved to: {bbox_decoder_path}")

    torch.save(state_dict, full_path)
    print(f"✅ Full model weights saved to: {full_path}")

    # Save tensor name mapping
    mapping_path = output_dir / "tensor_mapping.txt"
    with open(mapping_path, "w") as f:
        f.write("=== ENCODER TENSORS ===\n")
        for name in sorted(encoder_tensors.keys()):
            shape = tuple(encoder_tensors[name].shape)
            f.write(f"{name}: {shape}\n")

        f.write("\n=== TAG TRANSFORMER TENSORS ===\n")
        for name in sorted(tag_transformer_tensors.keys()):
            shape = tuple(tag_transformer_tensors[name].shape)
            f.write(f"{name}: {shape}\n")

        f.write("\n=== BBOX DECODER TENSORS ===\n")
        for name in sorted(bbox_decoder_tensors.keys()):
            shape = tuple(bbox_decoder_tensors[name].shape)
            f.write(f"{name}: {shape}\n")

    print(f"✅ Tensor mapping saved to: {mapping_path}\n")

    # Print key tensor shapes for verification
    print("Key tensor shapes:")
    print("-" * 70)

    # Encoder projection
    proj_keys = [k for k in encoder_tensors.keys() if "projection" in k or "_resnet.11" in k]
    if proj_keys:
        print("\nEncoder Projection Layers:")
        for k in sorted(proj_keys):
            print(f"  {k}: {tuple(encoder_tensors[k].shape)}")

    # Tag transformer input filter
    input_filter_keys = [k for k in tag_transformer_tensors.keys() if "_input_filter" in k]
    if input_filter_keys:
        print(f"\nTag Transformer Input Filter Layers: ({len(input_filter_keys)} tensors)")
        for k in sorted(input_filter_keys)[:5]:  # First 5
            print(f"  {k}: {tuple(tag_transformer_tensors[k].shape)}")

    # Decoder layers
    decoder_keys = [k for k in tag_transformer_tensors.keys() if "_decoder.layers.0" in k]
    if decoder_keys:
        print(f"\nTag Transformer Decoder Layer 0: ({len(decoder_keys)} tensors)")
        for k in sorted(decoder_keys)[:5]:  # First 5
            print(f"  {k}: {tuple(tag_transformer_tensors[k].shape)}")

    print("\n" + "=" * 70)
    print("Export complete!")


def main():
    parser = argparse.ArgumentParser(description="Export TableFormer weights for C# loading")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to safetensors model file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for exported weights",
    )

    args = parser.parse_args()
    export_weights(args.model, args.output)


if __name__ == "__main__":
    main()
