#!/usr/bin/env python3
"""
Debug inference step-by-step: extract logits for first 10 steps to compare with C#.
"""
import sys
sys.path.insert(0, '/Users/politom/.cache/huggingface/modules/transformers_modules/JPQD/tableformer-jpqd-ft-v1')

import torch
import torch.nn.functional as F
from modeling_tableformer import TableFormer
from PIL import Image
import torchvision.transforms as transforms

# Load model
config_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json"
safetensors_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors"

print("Loading model...")
model = TableFormer.from_pretrained_tableformer(
    config_path=config_path,
    safetensors_path=safetensors_path
)
model.eval()

# Load and preprocess image
image_path = "/Users/politom/Documents/Workspace/personal/doclingnet/dataset/test-fase3-validation/HAL.2004.page_82.pdf_125317.png"
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0)  # (1, 3, 448, 448)

print(f"Image shape: {image_tensor.shape}")
print(f"Image range: [{image_tensor.min():.6f}, {image_tensor.max():.6f}]")

# Word map
word_map_tag = model.word_map_tag
id_to_word = {v: k for k, v in word_map_tag.items()}

print(f"\nWord map: {word_map_tag}")

# Encode image
with torch.no_grad():
    enc_out = model._encoder(image_tensor)
    print(f"\nEncoder output shape: {enc_out.shape}")
    print(f"Encoder output range: [{enc_out.min():.6f}, {enc_out.max():.6f}]")

    # Start with <start> token
    start_token = word_map_tag["<start>"]
    decoded_tags = torch.tensor([[start_token]], dtype=torch.long)

    print(f"\nStarting inference with <start> token (id={start_token})")
    print("=" * 80)

    # Run first 10 steps
    for step in range(10):
        # Forward through tag transformer
        predictions, decoder_output, cache = model._tag_transformer(enc_out, decoded_tags)

        # Get logits for last token
        last_logits = predictions[0, -1, :]  # (vocab_size,)
        predicted_token = last_logits.argmax().item()

        print(f"\n[DEBUG Step {step}] {'=' * 60}")
        print(f"  Current sequence: {[id_to_word.get(t.item(), f'id{t}') for t in decoded_tags[0]]}")
        print(f"  Sequence length: {len(decoded_tags[0])}")

        # Show all logits
        print(f"  All logits:")
        for token_id in sorted(word_map_tag.values()):
            token_name = id_to_word[token_id]
            logit = last_logits[token_id].item()
            marker = " ← PREDICTED" if token_id == predicted_token else ""
            print(f"    [{token_id:02d}] {token_name:<10}: {logit:8.4f}{marker}")

        # Top-5
        top5_values, top5_indices = last_logits.topk(5)
        print(f"\n  Top-5 logits:")
        for val, idx in zip(top5_values, top5_indices):
            print(f"    {id_to_word[idx.item()]}: {val.item():.4f}")

        # <end> analysis
        end_token_id = word_map_tag["<end>"]
        end_logit = last_logits[end_token_id].item()
        sorted_logits = sorted(last_logits.tolist(), reverse=True)
        end_rank = sorted_logits.index(end_logit) + 1

        print(f"\n  <end> token analysis:")
        print(f"    Logit: {end_logit:.4f}")
        print(f"    Rank: {end_rank}/{len(word_map_tag)}")
        print(f"    Distance from max: {last_logits.max().item() - end_logit:.4f}")

        # Add predicted token to sequence
        predicted_tensor = torch.tensor([[predicted_token]], dtype=torch.long)
        decoded_tags = torch.cat([decoded_tags, predicted_tensor], dim=1)

        # Stop if <end> predicted
        if predicted_token == end_token_id:
            print(f"\n<end> token predicted at step {step}. Stopping.")
            break

print("\n" + "=" * 80)
print("Done!")
