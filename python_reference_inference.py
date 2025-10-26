#!/usr/bin/env python3
"""
Python reference inference script for TableFormer.
Generates reference output to compare with C# implementation.
"""
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add docling to path
sys.path.insert(0, "/Users/politom/.pyenv/versions/3.11.8/lib/python3.11/site-packages")

from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs


def load_and_preprocess_image(image_path, target_size=448):
    """Load and preprocess image to match C# preprocessing."""
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    # Convert to numpy and normalize
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    # Normalize with ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_array = (img_array - mean) / std

    # Convert to CHW format and add batch dimension
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)

    return img_tensor

def main():
    print("=== Python Reference Inference ===\n")

    # Paths
    config_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json"
    weights_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors"
    image_path = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/benchmark/HAL.2004.page_82.pdf_125315.png"

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Add save_dir to config for base_model.py
    config["model"]["save_dir"] = str(Path(weights_path).parent)

    print(f"Config: {json.dumps(config, indent=2)}\n")

    # Load model
    print("Loading model...")
    device = torch.device('cpu')

    # Initialize model directly
    # Prepare init_data for model constructor
    init_data_for_model = {
        "word_map": config["dataset_wordmap"]
    }

    model = TableModel04_rs(config, init_data=init_data_for_model, device=device)

    # Load weights from safetensors file
    from safetensors.torch import load_file
    state_dict = load_file(weights_path, device="cpu")
    model.load_state_dict(state_dict)

    model.to(device)
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
    img_tensor = load_and_preprocess_image(image_path)
    print(f"Image tensor shape: {list(img_tensor.shape)}\n")

    # Save ground truth tensors for comparison
    debug_dir = Path("/Users/politom/Documents/Workspace/personal/doclingnet/debug")
    debug_dir.mkdir(exist_ok=True)
    np.save(debug_dir / "python_preprocessed_image.npy", img_tensor.numpy())

    # Run inference
    print("Running inference (max 100 steps for debug)...")
    with torch.no_grad():
        # Get encoder output first
        enc_out = model._encoder(img_tensor)
        np.save(debug_dir / "python_encoder_output.npy", enc_out.numpy())

        # Manually run first step to get logits
        enc_proj = model._tag_transformer._input_filter(enc_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        enc_flat = enc_proj.reshape(enc_proj.shape[0], -1, enc_proj.shape[-1])
        memory = model._tag_transformer._encoder(enc_flat.permute(1, 0, 2))

        start_token = torch.tensor([[init_data_for_model["word_map"]["word_map_tag"]["<start>"]]], device=device)
        tgt_embed = model._tag_transformer._embedding(start_token)
        tgt_pos = model._tag_transformer._positional_encoding(tgt_embed.permute(1,0,2))
        
        pred, _ = model._tag_transformer._decoder(tgt_pos, memory, None)
        scores = model._tag_transformer._fc(pred)
        np.save(debug_dir / "python_first_step_logits.npy", scores.numpy())

        # Run full prediction for sequence comparison
        result = model.predict(img_tensor, max_steps=100, k=1, return_attention=False)

    # Extract results
    # Create reverse map for decoding tokens
    reverse_word_map_tag = {v: k for k, v in init_data_for_model["word_map"]["word_map_tag"].items()}

    tags = result[0]  # First batch, list of tags
    tokens = [reverse_word_map_tag[tag] for tag in tags]

    print(f"\n✓ Generated {len(tokens)} tokens")
    print(f"  Tokens: {' '.join(tokens[:50])}")  # First 50 tokens
    if len(tokens) > 50:
        print(f"  ... (showing first 50 of {len(tokens)} total)")
    print(f"\n  Contains <end>: {'<end>' in tokens}")
    print(f"  Unique tokens: {len(set(tokens))}")

    # Check for repetitive pattern
    has_repetition = False
    repetition_start = -1
    if len(tokens) >= 5:
        for i in range(min(len(tokens) - 5, 100)):
            if (tokens[i] == tokens[i+1] == tokens[i+2] ==
                tokens[i+3] == tokens[i+4]):
                has_repetition = True
                repetition_start = i
                print(f"  ⚠️ Repetitive pattern detected at position {i}: {tokens[i]}")
                break

    if not has_repetition:
        print(f"  ✓ No repetitive patterns detected")

    # Save reference output
    output = {
        "tokens": tokens,
        "token_count": len(tokens),
        "has_end_token": "<end>" in tokens,
        "unique_token_count": len(set(tokens)),
        "has_repetition": has_repetition,
        "repetition_start": repetition_start if has_repetition else None,
        "first_50_tokens": tokens[:50]
    }

    output_path = "/Users/politom/Documents/Workspace/personal/doclingnet/python_reference_output.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Reference output saved to: {output_path}")

    # Verdict
    print("\n=== Verdict ===")
    if not has_repetition and "<end>" in tokens:
        print("✅ SUCCESS: Model generates diverse tokens and terminates with <end>")
        return 0
    elif has_repetition:
        print("❌ FAILURE: Repetitive pattern detected")
        return 1
    else:
        print("⚠️  PARTIAL: No repetitive pattern but <end> not generated (check max_steps)")
        return 2

if __name__ == "__main__":
    sys.exit(main())
