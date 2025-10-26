"""
Save encoder output to compare with C#
"""
import torch
import torch.nn as nn
from safetensors.torch import load_file
from PIL import Image
import numpy as np
import json

# Load model weights
weights_path = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors"
state_dict = load_file(weights_path)

# Extract encoder weights
encoder_state = {k.replace("_encoder._resnet.", ""): v for k, v in state_dict.items() if k.startswith("_encoder._resnet.")}

print(f"Loaded {len(encoder_state)} encoder weights")
print("Sample keys:", list(encoder_state.keys())[:5])

# Load and preprocess image
img_path = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/benchmark/HAL.2004.page_82.pdf_125317.png"
img = Image.open(img_path).convert('RGB')
img = img.resize((448, 448))

# Normalize
mean = np.array([0.94247851, 0.94254675, 0.94292611])
std = np.array([0.17910956, 0.17940403, 0.17931663])
img_array = np.array(img) / 255.0
img_normalized = (img_array - mean) / std

# Convert to tensor (C, H, W)
img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1).unsqueeze(0)
print(f"Input tensor shape: {img_tensor.shape}")

# Build a simple ResNet-18 encoder (just layers needed for output)
# We'll use the actual PyTorch ResNet18 and load weights into it
from torchvision.models import resnet18

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pretrained ResNet18 structure
        resnet = resnet18(pretrained=False)
        
        # We need layers up to layer4 (which outputs 512 channels)
        # But TableFormer uses layer3 output (256 channels) + adaptive pool
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        
        # TableFormer uses AdaptiveAvgPool2d(14)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(14)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Adaptive pool to 14x14
        x = self.adaptive_pool(x)
        
        return x

# Create encoder
encoder = Encoder()
encoder.eval()

# Load weights - match TableFormer naming
# TableFormer uses: _encoder._resnet.0. for conv1, _encoder._resnet.1. for bn1, etc.
# Map: 0 -> conv1, 1 -> bn1, 4 -> layer1, 5 -> layer2, 6 -> layer3

def load_encoder_weights(encoder, state_dict):
    # conv1: 0.weight
    encoder.conv1.weight.data = state_dict["0.weight"]
    
    # bn1: 1.weight, 1.bias
    encoder.bn1.weight.data = state_dict["1.weight"]
    encoder.bn1.bias.data = state_dict["1.bias"]
    encoder.bn1.running_mean.data = state_dict["1.running_mean"]
    encoder.bn1.running_var.data = state_dict["1.running_var"]
    
    # layer1 (4.x.x)
    for i in range(2):  # 2 BasicBlocks
        for param in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                      'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']:
            key = f"4.{i}.{param}"
            if key in state_dict:
                parts = param.split('.')
                layer = getattr(getattr(encoder.layer1[i], parts[0]), parts[1]) if len(parts) > 1 else getattr(encoder.layer1[i], parts[0])
                if hasattr(layer, 'data'):
                    layer.data = state_dict[key]
                else:
                    setattr(layer, parts[1], state_dict[key])
    
    print("Loaded encoder weights")

load_encoder_weights(encoder, encoder_state)

# Run encoder
with torch.no_grad():
    output = encoder(img_tensor)

print(f"Encoder output shape: {output.shape}")
print(f"Encoder output range: [{output.min():.6f}, {output.max():.6f}]")
print(f"Encoder output mean: {output.mean():.6f}")
print(f"Encoder output std: {output.std():.6f}")

# Save first 100 values from first channel
output_flat = output[0, 0].flatten()
save_data = {
    "shape": list(output.shape),
    "first_100_values": output_flat[:100].tolist(),
    "stats": {
        "min": float(output.min()),
        "max": float(output.max()),
        "mean": float(output.mean()),
        "std": float(output.std())
    }
}

output_path = "/Users/politom/Documents/Workspace/personal/doclingnet/debug/python_encoder_output.json"
with open(output_path, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to: {output_path}")
