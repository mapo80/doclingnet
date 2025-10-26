#!/usr/bin/env python3
"""
Create a simple test to verify C# implementation produces same output as Python
by running both on the same input and comparing intermediate outputs.
"""

import argparse
import json
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from pathlib import Path

from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs


def preprocess_image(image_path, target_size=448):
    """Preprocess image for TableFormer."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    img_tensor = torch.tensor(list(img.getdata())).float().reshape(target_size, target_size, 3)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0

    mean = torch.tensor([0.94247851, 0.94254675, 0.94292611]).view(3, 1, 1)
    std = torch.tensor([0.17910956, 0.17940403, 0.17931663]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0)


def save_intermediate_outputs(model, img_tensor, output_dir):
    """
    Run inference and save intermediate outputs for comparison.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running Python inference...")

    with torch.no_grad():
        # 1. Encoder output
        encoder_out = model._encoder(img_tensor)
        print(f"Encoder output shape: {encoder_out.shape}")
        np.save(output_dir / "encoder_output.npy", encoder_out.cpu().numpy())

        # 2. Full inference
        result = model.predict(img_tensor, max_steps=1024, k=1)
        tag_sequence = result[0]

        print(f"\nFull tag sequence length: {len(tag_sequence)}")
        print(f"First 20 tags: {tag_sequence[:20]}")
        print(f"Last 20 tags: {tag_sequence[-20:]}")

        # Save tag sequence
        np.save(output_dir / "full_tag_sequence.npy", np.array(tag_sequence))

    print(f"\n✅ All outputs saved to: {output_dir}")

    return tag_sequence


def main():
    parser = argparse.ArgumentParser(description="Create test data with Python ground truth")
    parser.add_argument("--model", required=True, help="Path to safetensors model")
    parser.add_argument("--config", required=True, help="Path to tm_config.json")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--output", required=True, help="Output directory for test data")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    word_map_tag = config["dataset_wordmap"]["word_map_tag"]

    if "save_dir" not in config["model"]:
        config["model"]["save_dir"] = str(Path(args.model).parent)

    # Load model
    print("Loading Python model...")
    state_dict = load_file(args.model)

    init_data = {
        "word_map": {
            "word_map_tag": word_map_tag,
            "word_map_cell": config["dataset_wordmap"]["word_map_cell"],
        }
    }

    device = torch.device("cpu")
    model = TableModel04_rs(config, init_data=init_data, device=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("✅ Model loaded\n")

    # Preprocess image
    print(f"Preprocessing image: {args.image}")
    img_tensor = preprocess_image(args.image)
    print(f"Image tensor shape: {img_tensor.shape}\n")

    # Save input
    output_dir = Path(args.output)
    np.save(output_dir / "input_image.npy", img_tensor.cpu().numpy())

    # Run and save intermediate outputs
    tag_sequence = save_intermediate_outputs(model, img_tensor, output_dir)

    # Save metadata
    metadata = {
        "image_path": str(args.image),
        "image_shape": list(img_tensor.shape),
        "tag_sequence_length": len(tag_sequence),
        "first_20_tags": [int(t) for t in tag_sequence[:20]],
        "last_20_tags": [int(t) for t in tag_sequence[-20:]],
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Test data created successfully!")
    print(f"   Use these files to verify C# implementation produces identical outputs")


if __name__ == "__main__":
    main()
