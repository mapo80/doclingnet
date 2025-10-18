#!/usr/bin/env python3
"""
Extract transformer decoder output at step 0 (after predicting first token after <start>).
This verifies that the C# transformer logic matches Python.
"""
import json
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import math

# PositionalEncoding from Python reference
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Load image and preprocess
image_path = "/Users/politom/Documents/Workspace/personal/doclingnet/dataset/test-fase3-validation/HAL.2004.page_82.pdf_125317.png"
image = Image.open(image_path).convert('RGB')

# Resize to 448x448
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 448, 448)

print(f"Image tensor shape: {image_tensor.shape}")
print(f"Image tensor range: [{image_tensor.min():.6f}, {image_tensor.max():.6f}]")

# Simulate encoder output (we'll use random values since we just want to test decoder)
# In reality we'd load the actual encoder output from the previous test
encoder_out = torch.randn(1, 196, 512)  # (batch, 14*14, 512)
print(f"\nEncoder output shape: {encoder_out.shape}")
print(f"Encoder output range: [{encoder_out.min():.6f}, {encoder_out.max():.6f}]")

# Word map (from config)
word_map_tag = {
    "<start>": 0,
    "ecel": 1,
    "xcel": 2,
    "nl": 3,
    "lcel": 4,
    "fcel": 5,
    "rcel": 6,
    "ched": 7,
    "srow": 8,
    "scel": 9,
    "<end>": 10,
    "<pad>": 11,
    "empty": 12
}

# Create embedding layer (512 dims)
embed_dim = 512
vocab_size = len(word_map_tag)
embedding = nn.Embedding(vocab_size, embed_dim)

# Initialize with random weights (in real test we'd load actual weights)
nn.init.xavier_uniform_(embedding.weight)

# Create positional encoding
pos_encoder = PositionalEncoding(d_model=512, dropout=0.1, max_len=1024)
pos_encoder.eval()

# Step 0: Start with <start> token
start_token = torch.tensor([[word_map_tag["<start>"]]])  # Shape: (1, 1)
print(f"\nStart token: {start_token}")

# Embed the start token
with torch.no_grad():
    embedded = embedding(start_token)  # Shape: (1, 1, 512)
    print(f"Embedded shape: {embedded.shape}")
    print(f"Embedded[0,0,:20]: {embedded[0, 0, :20].tolist()}")

    # Apply positional encoding
    pos_encoded = pos_encoder(embedded)  # Shape: (1, 1, 512)
    print(f"\nAfter PE shape: {pos_encoded.shape}")
    print(f"After PE[0,0,:20]: {pos_encoded[0, 0, :20].tolist()}")

    # Create a simple transformer decoder layer for testing
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
    decoder_layer.eval()

    # Create causal mask (1x1 for single token - no masking needed)
    tgt_mask = None  # No mask needed for first token

    # Pass through decoder
    decoder_out = decoder_layer(pos_encoded, encoder_out, tgt_mask=tgt_mask)
    print(f"\nDecoder output shape: {decoder_out.shape}")
    print(f"Decoder output[0,0,:20]: {decoder_out[0, 0, :20].tolist()}")
    print(f"Decoder output range: [{decoder_out.min():.6f}, {decoder_out.max():.6f}]")

    # Apply FC layer to get logits
    fc = nn.Linear(512, vocab_size)
    logits = fc(decoder_out)  # Shape: (1, 1, vocab_size)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits[0,0]: {logits[0, 0].tolist()}")

    # Get prediction
    predicted_token = logits.argmax(dim=-1)
    print(f"Predicted token ID: {predicted_token.item()}")

    # Find token name
    id_to_word = {v: k for k, v in word_map_tag.items()}
    print(f"Predicted token: {id_to_word[predicted_token.item()]}")

# Save outputs for comparison
output = {
    "start_token": start_token.tolist(),
    "embedded_first_20": embedded[0, 0, :20].tolist(),
    "embedded_full": embedded[0, 0].tolist(),
    "pos_encoded_first_20": pos_encoded[0, 0, :20].tolist(),
    "pos_encoded_full": pos_encoded[0, 0].tolist(),
    "decoder_out_first_20": decoder_out[0, 0, :20].tolist(),
    "decoder_out_full": decoder_out[0, 0].tolist(),
    "logits": logits[0, 0].tolist(),
    "predicted_token": predicted_token.item(),
}

output_path = "/Users/politom/Documents/Workspace/personal/doclingnet/debug/python_transformer_step0.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved to {output_path}")
