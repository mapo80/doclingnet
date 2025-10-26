#!/usr/bin/env python3
"""
Simplified Python reference inference - direct model loading.
"""
import sys
import json
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/politom/.pyenv/versions/3.11.8/lib/python3.11/site-packages")

from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs

def main():
    print("=== Python Reference Inference (Simplified) ===\n")

    # Paths
    config_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json"
    weights_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors"
    image_path = "/Users/politom/Documents/Workspace/personal/doclingnet/dataset/2305.03393v1-pg9-img.png"

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Loading model from scratch...")
    device = torch.device('cpu')

    # Prepare init_data (vocab, etc.)
    init_data = {
        'tag_vocab': config['inference']['otsl_compact_vocab'],
        'tag_decode': {v: k for k, v in config['inference']['otsl_compact_vocab'].items()},
    }

    # Initialize model
    model = TableModel04_rs(config, init_data, device)

    # Load weights from safetensors
    from safetensors.torch import load_file
    state_dict = load_file(weights_path)

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("✓ Model loaded\n")

    # Verify PositionalEncoding shape
    pe_module = model._tag_transformer._positional_encoding
    pe_buffer = pe_module.pe
    print(f"PositionalEncoding buffer shape: {list(pe_buffer.shape)}")
    print(f"Expected: [1024, 1, 512]")
    print(f"Match: {list(pe_buffer.shape) == [1024, 1, 512]}\n")

    # Load and preprocess image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((896, 896), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    print(f"Image tensor shape: {list(img_tensor.shape)}\n")

    # Run inference
    print("Running inference (max 100 steps)...")
    with torch.no_grad():
        result = model.predict(img_tensor, max_steps=100, k=1, return_attention=False)

    # Extract tokens
    tags = result['tags'][0]
    tokens = [model._tag_decode[tag] for tag in tags]

    print(f"\n✓ Generated {len(tokens)} tokens")
    print(f"  First 50: {' '.join(tokens[:50])}")
    print(f"  Contains <end>: {'<end>' in tokens}")
    print(f"  Unique tokens: {len(set(tokens))}")

    # Check repetition
    has_rep = False
    if len(tokens) >= 5:
        for i in range(min(len(tokens) - 5, 100)):
            if tokens[i] == tokens[i+1] == tokens[i+2] == tokens[i+3] == tokens[i+4]:
                has_rep = True
                print(f"  ⚠️ Repetition at {i}: {tokens[i]}")
                break

    if not has_rep:
        print("  ✓ No repetition")

    # Save output
    with open("python_reference_output.json", 'w') as f:
        json.dump({
            "tokens": tokens,
            "token_count": len(tokens),
            "has_end_token": "<end>" in tokens,
            "unique_tokens": len(set(tokens)),
            "has_repetition": has_rep
        }, f, indent=2)

    print("\n✓ Output saved to python_reference_output.json")

    if not has_rep and "<end>" in tokens:
        print("\n✅ SUCCESS")
        return 0
    elif has_rep:
        print("\n❌ FAILURE: Repetition")
        return 1
    else:
        print("\n⚠️  PARTIAL: No <end>")
        return 2

if __name__ == "__main__":
    sys.exit(main())
