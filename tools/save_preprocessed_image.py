"""
Save preprocessed image tensor to compare with C#
"""
import torch
import numpy as np
from PIL import Image
import json

# Load test image
img_path = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/benchmark/HAL.2004.page_82.pdf_125317.png"
img = Image.open(img_path).convert('RGB')
print(f"Original image size: {img.size}")

# Resize to 448x448
img = img.resize((448, 448))
print(f"Resized to: {img.size}")

# Convert to numpy array
img_array = np.array(img)
print(f"Image array shape: {img_array.shape}")
print(f"Image array dtype: {img_array.dtype}")
print(f"Image array range: [{img_array.min()}, {img_array.max()}]")

# Normalize with ImageNet stats from config
mean = np.array([0.94247851, 0.94254675, 0.94292611])
std = np.array([0.17910956, 0.17940403, 0.17931663])

# Convert to float [0, 1] and normalize
img_normalized = (img_array / 255.0 - mean) / std

print(f"\nNormalized image shape: {img_normalized.shape}")
print(f"Normalized image dtype: {img_normalized.dtype}")
print(f"Normalized image range: [{img_normalized.min():.4f}, {img_normalized.max():.4f}]")

# Print sample pixels
print(f"\nSample pixels at (0,0):")
print(f"  R: {img_array[0, 0, 0]} -> normalized: {img_normalized[0, 0, 0]:.6f}")
print(f"  G: {img_array[0, 0, 1]} -> normalized: {img_normalized[0, 0, 1]:.6f}")
print(f"  B: {img_array[0, 0, 2]} -> normalized: {img_normalized[0, 0, 2]:.6f}")

print(f"\nSample pixels at (224,224):")
print(f"  R: {img_array[224, 224, 0]} -> normalized: {img_normalized[224, 224, 0]:.6f}")
print(f"  G: {img_array[224, 224, 1]} -> normalized: {img_normalized[224, 224, 1]:.6f}")
print(f"  B: {img_array[224, 224, 2]} -> normalized: {img_normalized[224, 224, 2]:.6f}")

# Convert to tensor format (C, H, W)
img_tensor = torch.from_numpy(img_normalized).float()
img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
print(f"\nTensor shape after permute: {img_tensor.shape}")

# Save first 100 values from each channel for comparison
output = {
    "shape": list(img_tensor.shape),
    "channel_0_first_100": img_tensor[0].flatten()[:100].tolist(),
    "channel_1_first_100": img_tensor[1].flatten()[:100].tolist(),
    "channel_2_first_100": img_tensor[2].flatten()[:100].tolist(),
    "mean": mean.tolist(),
    "std": std.tolist(),
}

output_path = "/Users/politom/Documents/Workspace/personal/doclingnet/debug/python_preprocessed_image.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to: {output_path}")
