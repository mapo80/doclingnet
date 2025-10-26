#!/usr/bin/env python3
"""
Extract positional encoding values at positions 0 and 50 from the TableFormer model.
This verifies that C# PositionalEncoding implementation matches Python.
"""
import json
import torch
import torch.nn as nn
import math

# PositionalEncoding from Python reference implementation
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

# Create PE with d_model=512, max_len=1024
print("Creating PositionalEncoding with d_model=512, max_len=1024...")
pos_encoder = PositionalEncoding(d_model=512, dropout=0.1, max_len=1024)
pos_encoder.eval()

# Get the PE buffer
pe = pos_encoder.pe  # Shape: (1, 1024, 512)

print(f"PE shape: {pe.shape}")
print(f"PE dtype: {pe.dtype}")

# Extract position 0 and position 50
pe_pos0 = pe[0, 0, :].cpu().numpy()  # First position, all dimensions
pe_pos50 = pe[0, 50, :].cpu().numpy()  # Position 50, all dimensions

print(f"\nPE[0] shape: {pe_pos0.shape}")
print(f"PE[0] first 20 values: {[f'{x:.6f}' for x in pe_pos0[:20]]}")
print(f"PE[0] range: [{pe_pos0.min():.6f}, {pe_pos0.max():.6f}]")

print(f"\nPE[50] shape: {pe_pos50.shape}")
print(f"PE[50] first 20 values: {[f'{x:.6f}' for x in pe_pos50[:20]]}")
print(f"PE[50] range: [{pe_pos50.min():.6f}, {pe_pos50.max():.6f}]")

# Save to JSON
output = {
    "pe_shape": list(pe.shape),
    "pe_pos0_shape": list(pe_pos0.shape),
    "pe_pos50_shape": list(pe_pos50.shape),
    "pe_pos0_first_20": pe_pos0[:20].tolist(),
    "pe_pos0_full": pe_pos0.tolist(),
    "pe_pos50_first_20": pe_pos50[:20].tolist(),
    "pe_pos50_full": pe_pos50.tolist(),
    "pe_pos0_range": [float(pe_pos0.min()), float(pe_pos0.max())],
    "pe_pos50_range": [float(pe_pos50.min()), float(pe_pos50.max())],
    "scale": float(pos_encoder.scale.item()),
}

output_path = "/Users/politom/Documents/Workspace/personal/doclingnet/debug/python_pe_check.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved to {output_path}")
