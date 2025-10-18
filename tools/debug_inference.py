"""
Debug script to compare Python vs C# inference step by step
"""
import torch
import numpy as np
from safetensors.torch import load_file
from PIL import Image
import json

# Load config
config_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Load weights
weights_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors"
state_dict = load_file(weights_path)

# Load test image
img_path = "/Users/politom/Documents/Workspace/personal/doclingnet/dataset/fintabnet-sample/HAL.2004.page_82.pdf_125317.png"
img = Image.open(img_path).convert('RGB')
print(f"Image size: {img.size}")

# Resize to 448x448
img = img.resize((448, 448))
img_array = np.array(img) / 255.0
print(f"Image array shape: {img_array.shape}")
print(f"Image array min/max: {img_array.min():.4f} / {img_array.max():.4f}")

# Check first few embedding weights
embedding_key = "_tag_transformer._embedding.weight"
if embedding_key in state_dict:
    emb_weights = state_dict[embedding_key]
    print(f"\nEmbedding weights shape: {emb_weights.shape}")
    print(f"Embedding[0] (start token) first 10 values:")
    print(emb_weights[0, :10])
    print(f"\nEmbedding[1] (nl token) first 10 values:")
    print(emb_weights[1, :10])

# Check encoder output dimension
enc_weight_key = "_encoder._resnet.6.0.conv1.weight"
if enc_weight_key in state_dict:
    enc_w = state_dict[enc_weight_key]
    print(f"\nEncoder layer 6.0 conv1 weight shape: {enc_w.shape}")
    print(f"  Input channels: {enc_w.shape[1]}")
    print(f"  Output channels: {enc_w.shape[0]}")

# Check input_filter dimensions
if_key = "_tag_transformer._input_filter.0.conv1.weight"
if if_key in state_dict:
    if_w = state_dict[if_key]
    print(f"\nTagTransformer input_filter.0.conv1 weight shape: {if_w.shape}")
    print(f"  Input channels: {if_w.shape[1]}")
    print(f"  Output channels: {if_w.shape[0]}")

# Check final fc layer
fc_key = "_tag_transformer._fc.weight"
if fc_key in state_dict:
    fc_w = state_dict[fc_key]
    print(f"\nTagTransformer fc weight shape: {fc_w.shape}")
    print(f"  Input features: {fc_w.shape[1]}")
    print(f"  Output classes (vocab_size): {fc_w.shape[0]}")
    
    # Show first row (logits for token 0 = <start>)
    print(f"\nFC weight[0] (for token 0) first 10 values:")
    print(fc_w[0, :10])

print("\n" + "="*60)
print("Weight statistics summary:")
print("="*60)
for key in sorted(state_dict.keys())[:20]:
    t = state_dict[key]
    print(f"{key:60s} shape={str(list(t.shape)):20s} mean={t.mean():.6f} std={t.std():.6f}")
