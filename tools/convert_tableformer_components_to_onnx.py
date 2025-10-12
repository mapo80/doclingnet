#!/usr/bin/env python3
"""
TableFormer Component-wise PyTorch to ONNX Conversion Script

Exports TableFormer components separately to avoid autoregressive loop issues:
1. Encoder: Image -> Encoder features
2. Tag Transformer Encoder: Encoder features -> Memory
3. Tag Transformer Decoder (single step): Memory + previous tags -> next tag logits
4. BBox Decoder: Encoder features + tag hidden states -> bbox predictions

The autoregressive loop will be implemented in C# .NET.

Usage:
    python convert_tableformer_components_to_onnx.py --model fast --output ./models/onnx/
    python convert_tableformer_components_to_onnx.py --model accurate --output ./models/onnx/
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


class EncoderWrapper(nn.Module):
    """Wrapper for Encoder component."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model._encoder

    def forward(self, images):
        """
        Args:
            images: (batch_size, 3, 448, 448)
        Returns:
            encoder_out: (batch_size, 28, 28, 256)
        """
        return self.encoder(images)


class TagTransformerEncoderWrapper(nn.Module):
    """Wrapper for Tag Transformer Encoder."""

    def __init__(self, model):
        super().__init__()
        self.tag_transformer = model._tag_transformer
        self.n_heads = model._tag_transformer._n_heads

    def forward(self, encoder_out):
        """
        Args:
            encoder_out: (batch_size, 28, 28, 256)
        Returns:
            memory: (784, batch_size, 512)
        """
        batch_size = encoder_out.size(0)

        # Apply input filter
        filtered = self.tag_transformer._input_filter(
            encoder_out.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)  # (batch_size, 28, 28, 512)

        # Reshape for transformer
        encoder_dim = filtered.size(-1)
        enc_inputs = filtered.view(batch_size, -1, encoder_dim)
        enc_inputs = enc_inputs.permute(1, 0, 2)  # (784, batch_size, 512)

        # Create mask (all zeros = no masking)
        positions = enc_inputs.shape[0]
        encoder_mask = torch.zeros(
            (batch_size * self.n_heads, positions, positions),
            device=encoder_out.device,
            dtype=torch.bool
        )

        # Encode
        memory = self.tag_transformer._encoder(enc_inputs, mask=encoder_mask)
        return memory


class TagTransformerDecoderStepWrapper(nn.Module):
    """Wrapper for single step of Tag Transformer Decoder."""

    def __init__(self, model):
        super().__init__()
        self.tag_transformer = model._tag_transformer
        self.embedding = model._tag_transformer._embedding
        self.positional_encoding = model._tag_transformer._positional_encoding
        self.decoder = model._tag_transformer._decoder
        self.fc = model._tag_transformer._fc

    def forward(self, decoded_tags, memory, encoder_mask):
        """
        Args:
            decoded_tags: (seq_len, batch_size) - sequence of tag indices so far
            memory: (784, batch_size, 512) - encoder memory
            encoder_mask: (batch_size * n_heads, 784, 784) - encoder attention mask
        Returns:
            logits: (batch_size, vocab_size) - logits for next tag
            tag_hidden: (batch_size, 512) - hidden state for bbox prediction
        """
        # Embed and encode positions
        decoded_embedding = self.embedding(decoded_tags)
        decoded_embedding = self.positional_encoding(decoded_embedding)

        # Decode (with cache=None for simplicity, could be optimized)
        decoded, _ = self.decoder(
            decoded_embedding,
            memory,
            cache=None,
            memory_key_padding_mask=encoder_mask,
        )

        # Get logits for last position
        logits = self.fc(decoded[-1, :, :])
        tag_hidden = decoded[-1, :, :]

        return logits, tag_hidden


class BBoxDecoderWrapper(nn.Module):
    """Wrapper for BBox Decoder."""

    def __init__(self, model):
        super().__init__()
        self.bbox_decoder = model._bbox_decoder

    def forward(self, encoder_out, tag_hiddens):
        """
        Args:
            encoder_out: (batch_size, 28, 28, 256)
            tag_hiddens: (num_cells, batch_size, 512) - hidden states for cells
        Returns:
            bbox_classes: (num_cells, num_classes+1)
            bbox_coords: (num_cells, 4)
        """
        # Convert tag_hiddens from (num_cells, batch_size, 512) to list of (batch_size, 512)
        tag_H_list = [tag_hiddens[i] for i in range(tag_hiddens.size(0))]

        # Run bbox decoder inference
        # Returns: bbox_classes (num_cells, num_classes+1), bbox_coords (num_cells, 4)
        bbox_classes, bbox_coords = self.bbox_decoder.inference(encoder_out, tag_H_list)

        return bbox_classes, bbox_coords


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


def export_encoder(model, output_path, device="cpu"):
    """Export Encoder to ONNX."""
    print("\n--- Exporting Encoder ---")
    wrapper = EncoderWrapper(model)
    wrapper.eval()

    # Dummy input: batch_size=1, 3 channels, 448x448
    dummy_input = torch.randn(1, 3, 448, 448, device=device)

    print("Testing forward pass...")
    with torch.no_grad():
        output = wrapper(dummy_input)
        print(f"  Output shape: {output.shape}")  # Should be (1, 28, 28, 256)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["encoder_out"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "encoder_out": {0: "batch_size"},
        },
    )
    print("✅ Encoder exported successfully!")
    return output


def export_tag_transformer_encoder(model, output_path, device="cpu"):
    """Export Tag Transformer Encoder to ONNX."""
    print("\n--- Exporting Tag Transformer Encoder ---")
    wrapper = TagTransformerEncoderWrapper(model)
    wrapper.eval()

    # Dummy input: encoder output (1, 28, 28, 256)
    dummy_input = torch.randn(1, 28, 28, 256, device=device)

    print("Testing forward pass...")
    with torch.no_grad():
        output = wrapper(dummy_input)
        print(f"  Output shape: {output.shape}")  # Should be (784, 1, 512)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["encoder_out"],
        output_names=["memory"],
        dynamic_axes={
            "encoder_out": {0: "batch_size"},
            "memory": {1: "batch_size"},
        },
    )
    print("✅ Tag Transformer Encoder exported successfully!")
    return output


def export_tag_transformer_decoder_step(model, word_map, output_path, device="cpu"):
    """Export Tag Transformer Decoder (single step) to ONNX."""
    print("\n--- Exporting Tag Transformer Decoder Step ---")
    wrapper = TagTransformerDecoderStepWrapper(model)
    wrapper.eval()

    # Dummy inputs for single decoding step
    # Start with just the start token
    decoded_tags = torch.LongTensor([[word_map["word_map_tag"]["<start>"]]]).to(device)  # (1, 1)
    memory = torch.randn(784, 1, 512, device=device)  # (784, 1, 512)
    n_heads = model._tag_transformer._n_heads
    encoder_mask = torch.zeros((1 * n_heads, 784, 784), device=device, dtype=torch.bool)

    print("Testing forward pass...")
    with torch.no_grad():
        logits, tag_hidden = wrapper(decoded_tags, memory, encoder_mask)
        print(f"  Logits shape: {logits.shape}")  # Should be (1, vocab_size)
        print(f"  Tag hidden shape: {tag_hidden.shape}")  # Should be (1, 512)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        (decoded_tags, memory, encoder_mask),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["decoded_tags", "memory", "encoder_mask"],
        output_names=["logits", "tag_hidden"],
        dynamic_axes={
            "decoded_tags": {0: "seq_len", 1: "batch_size"},
            "memory": {1: "batch_size"},
            "encoder_mask": {0: "batch_n_heads"},
            "logits": {0: "batch_size"},
            "tag_hidden": {0: "batch_size"},
        },
    )
    print("✅ Tag Transformer Decoder Step exported successfully!")
    return logits, tag_hidden


def export_bbox_decoder(model, output_path, device="cpu"):
    """Export BBox Decoder to ONNX."""
    print("\n--- Exporting BBox Decoder ---")
    wrapper = BBoxDecoderWrapper(model)
    wrapper.eval()

    # Dummy inputs
    encoder_out = torch.randn(1, 28, 28, 256, device=device)
    tag_hiddens = torch.randn(10, 1, 512, device=device)  # 10 cells example

    print("Testing forward pass...")
    with torch.no_grad():
        bbox_classes, bbox_coords = wrapper(encoder_out, tag_hiddens)
        print(f"  BBox classes shape: {bbox_classes.shape}")  # Should be (10, num_classes+1)
        print(f"  BBox coords shape: {bbox_coords.shape}")    # Should be (10, 4)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        (encoder_out, tag_hiddens),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["encoder_out", "tag_hiddens"],
        output_names=["bbox_classes", "bbox_coords"],
        dynamic_axes={
            "encoder_out": {0: "batch_size"},
            "tag_hiddens": {0: "num_cells", 1: "batch_size"},
            "bbox_classes": {0: "num_cells"},
            "bbox_coords": {0: "num_cells"},
        },
    )
    print("✅ BBox Decoder exported successfully!")
    return bbox_classes, bbox_coords


def validate_onnx_models(output_dir, model_name):
    """Quick validation that ONNX models can be loaded."""
    print("\n--- Validating ONNX Models ---")

    components = [
        "encoder",
        "tag_transformer_encoder",
        "tag_transformer_decoder_step",
        "bbox_decoder"
    ]

    all_valid = True
    for component in components:
        onnx_path = output_dir / f"{model_name}_{component}.onnx"
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"  ✅ {component}: Valid")
        except Exception as e:
            print(f"  ❌ {component}: {e}")
            all_valid = False

    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Convert TableFormer components to ONNX")
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
        help="Output directory for ONNX models"
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
    model_dir = base_dir / "models" / "model_artifacts" / "tableformer" / args.model
    model_path = model_dir / f"tableformer_{args.model}.safetensors"
    config_path = model_dir / "tm_config.json"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"========================================")
    print(f"TableFormer Component-wise ONNX Conversion")
    print(f"========================================")
    print(f"Model variant: {args.model}")
    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {args.device}")
    print(f"========================================")

    # Load model
    device = torch.device(args.device)
    model, config = load_tableformer_model(str(model_path), str(config_path), device)

    # Export each component
    model_name = f"tableformer_{args.model}"

    export_encoder(
        model,
        output_dir / f"{model_name}_encoder.onnx",
        device
    )

    export_tag_transformer_encoder(
        model,
        output_dir / f"{model_name}_tag_transformer_encoder.onnx",
        device
    )

    export_tag_transformer_decoder_step(
        model,
        config["dataset_wordmap"],
        output_dir / f"{model_name}_tag_transformer_decoder_step.onnx",
        device
    )

    export_bbox_decoder(
        model,
        output_dir / f"{model_name}_bbox_decoder.onnx",
        device
    )

    # Validate all models
    all_valid = validate_onnx_models(output_dir, model_name)

    # Save config and word map
    output_config_path = output_dir / f"{model_name}_config.json"
    print(f"\nSaving config to {output_config_path}")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)

    word_map_path = output_dir / f"{model_name}_wordmap.json"
    print(f"Saving word map to {word_map_path}")
    with open(word_map_path, "w") as f:
        json.dump(config["dataset_wordmap"], f, indent=2)

    print("\n========================================")
    if all_valid:
        print("✅ All components converted successfully!")
    else:
        print("⚠️  Some components failed validation")
    print(f"Output directory: {output_dir}")
    print("========================================")

    print("\nNext steps:")
    print("1. Implement autoregressive loop in C# .NET")
    print("2. Load all 4 ONNX models in .NET")
    print("3. Implement OTSL tag generation logic")
    print("4. Implement bbox merging for spanning cells")

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
