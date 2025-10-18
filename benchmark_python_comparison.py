#!/usr/bin/env python3
"""
Simple Python TableFormer inference for comparison with C# implementation.
Uses the docling_ibm_models library directly.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import (
    TableModel04_rs,
)


def load_config(config_path):
    """Load tm_config.json."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def preprocess_image(image_path, target_size=448):
    """
    Preprocess image for TableFormer.

    Args:
        image_path: Path to input image
        target_size: Target size for resizing (default 448x448)

    Returns:
        Preprocessed tensor of shape (1, 3, 448, 448)
    """
    # Load and resize image
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    # Convert to tensor and normalize
    img_tensor = torch.tensor(list(img.getdata())).float().reshape(target_size, target_size, 3)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # (3, H, W)

    # Normalize using config values
    mean = torch.tensor([0.94247851, 0.94254675, 0.94292611]).view(3, 1, 1)
    std = torch.tensor([0.17910956, 0.17940403, 0.17931663]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

    return img_tensor, orig_size


def decode_tags(tag_sequence, word_map):
    """Decode tag sequence to tag names."""
    reverse_map = {v: k for k, v in word_map.items()}
    return [reverse_map.get(int(tag), f"<unk_{tag}>") for tag in tag_sequence]


def main():
    parser = argparse.ArgumentParser(description="Python TableFormer inference for comparison")
    parser.add_argument("--model", required=True, help="Path to safetensors model")
    parser.add_argument("--config", required=True, help="Path to tm_config.json")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to output results file")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu/cuda)")

    args = parser.parse_args()

    print("=" * 70)
    print("PYTHON TABLEFORMER INFERENCE (for C# comparison)")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Config: {args.config}")
    print(f"Image: {args.image}")
    print(f"Device: {args.device}\n")

    # Load config
    config = load_config(args.config)
    word_map_tag = config["dataset_wordmap"]["word_map_tag"]

    # Add required fields if missing
    if "save_dir" not in config["model"]:
        config["model"]["save_dir"] = str(Path(args.model).parent)

    # Load model weights
    print("Loading model...")
    state_dict = load_file(args.model)

    # Create model with word_map in init_data
    init_data = {
        "word_map": {
            "word_map_tag": word_map_tag,
            "word_map_cell": config["dataset_wordmap"]["word_map_cell"],
        }
    }

    device = torch.device(args.device)
    model = TableModel04_rs(config, init_data=init_data, device=device)

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Model loaded successfully\n")

    # Preprocess image
    print("Preprocessing image...")
    img_tensor, orig_size = preprocess_image(args.image)
    print(f"  Original size: {orig_size[0]}x{orig_size[1]}")
    print(f"  Resized to: {img_tensor.shape[2]}x{img_tensor.shape[3]}")
    print(f"  Tensor shape: {img_tensor.shape}\n")

    img_tensor = img_tensor.to(device)

    # Run inference
    print("Running inference...")
    start_time = time.time()

    with torch.no_grad():
        # Call model's predict method with required arguments
        # max_steps=1024 (from config), k=1 for greedy decoding
        result = model.predict(img_tensor, max_steps=1024, k=1)

    inference_time = time.time() - start_time
    print(f"✅ Inference completed in {inference_time:.2f}s\n")

    # Extract results - predict returns a tuple of multiple values
    tag_sequence = result[0]  # First element is tag sequence
    bbox_coords = result[1] if len(result) > 1 else None  # Second element might be bboxes

    # Convert to lists if they are tensors
    if torch.is_tensor(tag_sequence):
        tag_sequence = tag_sequence.cpu().tolist()
    if bbox_coords is not None and torch.is_tensor(bbox_coords):
        bbox_coords = bbox_coords.cpu().numpy()

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Tag sequence length: {len(tag_sequence)}")

    if bbox_coords is not None:
        print(f"Number of bounding boxes: {bbox_coords.shape[0]}")

    # Decode tags
    tag_names = decode_tags(tag_sequence, word_map_tag)
    print(f"\nFirst 50 tags: {' '.join(tag_names[:50])}")
    print(f"Last 50 tags: {' '.join(tag_names[-50:])}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(f"Python TableFormer Results: {Path(args.image).name}\n")
            f.write(f"Inference Time: {inference_time:.2f}s\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Tag Sequence:\n")
            f.write(" ".join(tag_names) + "\n\n")

            f.write(f"Number of Tags: {len(tag_sequence)}\n\n")

            if bbox_coords is not None:
                f.write(f"Number of Bounding Boxes: {bbox_coords.shape[0]}\n\n")
                f.write("Bounding Boxes:\n")
                for i, bbox in enumerate(bbox_coords[:100]):  # First 100 boxes
                    # Handle variable bbox format
                    if len(bbox) == 4:
                        f.write(f"  Box {i}: cx={bbox[0]:.4f}, cy={bbox[1]:.4f}, "
                               f"w={bbox[2]:.4f}, h={bbox[3]:.4f}\n")
                    else:
                        f.write(f"  Box {i}: {bbox}\n")

        print(f"\n✅ Results saved to: {output_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
