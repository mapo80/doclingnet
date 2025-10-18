# TorchSharp Quick Start Guide

**Target**: Get a minimal TorchSharp implementation running in 1 day

---

## Step 1: Setup (15 minutes)

### Install TorchSharp

```bash
cd dotnet/TableFormerSdk
dotnet add package TorchSharp --version 0.105.1
dotnet add package TorchSharp-cpu --version 0.105.1
dotnet restore
```

### Download Model

```bash
cd ../..
huggingface-cli download ds4sd/docling-models \
  model_artifacts/tableformer/fast/tableformer_fast.safetensors \
  model_artifacts/tableformer/fast/tm_config.json \
  --local-dir models
```

---

## Step 2: Minimal C# Example (30 minutes)

Create `dotnet/TableFormerSdk/Examples/TorchSharpMinimal.cs`:

```csharp
using System;
using TorchSharp;
using static TorchSharp.torch;
using Newtonsoft.Json.Linq;

namespace TableFormerSdk.Examples
{
    /// <summary>
    /// Minimal TorchSharp example - loads safetensors and runs forward pass
    /// </summary>
    public class TorchSharpMinimal
    {
        public static void Run()
        {
            Console.WriteLine("=== TorchSharp Minimal Example ===\n");

            // 1. Load configuration
            var configPath = "models/model_artifacts/tableformer/fast/tm_config.json";
            var config = JObject.Parse(File.ReadAllText(configPath));
            Console.WriteLine($"✅ Config loaded: {configPath}");

            // 2. Create dummy model (simplified - no real architecture yet)
            var model = CreateDummyModel(config);
            Console.WriteLine("✅ Dummy model created");

            // 3. Load safetensors weights
            var modelPath = "models/model_artifacts/tableformer/fast/tableformer_fast.safetensors";
            LoadSafetensorsWeights(model, modelPath);
            Console.WriteLine($"✅ Weights loaded: {modelPath}");

            // 4. Create dummy input
            var dummyImage = torch.randn(1, 3, 448, 448);
            Console.WriteLine($"✅ Dummy input created: {dummyImage.shape}");

            // 5. Forward pass
            using var noGrad = torch.no_grad();
            var output = model.forward(dummyImage);
            Console.WriteLine($"✅ Forward pass complete: {output.shape}");

            Console.WriteLine("\n=== Success! TorchSharp is working ===");
        }

        private static Module<Tensor, Tensor> CreateDummyModel(JObject config)
        {
            // Simplified model - just one Conv2d layer for testing
            return nn.Sequential(
                ("conv1", nn.Conv2d(3, 64, kernelSize: 3, padding: 1)),
                ("relu", nn.ReLU())
            );
        }

        private static void LoadSafetensorsWeights(Module model, string path)
        {
            try
            {
                // TorchSharp can load safetensors via libtorch
                var stateDict = torch.load(path);
                model.load_state_dict(stateDict);
                Console.WriteLine($"  Loaded {stateDict.Count} parameters");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ⚠️  Weight loading not yet implemented: {ex.Message}");
                Console.WriteLine("  → This is expected for minimal example");
            }
        }
    }
}
```

### Test It

```bash
dotnet run --project dotnet/TableFormerSdk/TableFormerSdk.csproj
```

---

## Step 3: Python Reference Implementation (15 minutes)

Create a simple Python script to understand the flow:

`scripts/torchsharp_reference.py`:

```python
#!/usr/bin/env python3
"""
Simple reference implementation showing what TorchSharp needs to replicate
"""
import sys
import torch
from safetensors.torch import load_file

sys.path.insert(0, "/tmp/docling-ibm-models")
from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs

def main():
    print("=== Python Reference Implementation ===\n")

    # 1. Load config
    import json
    with open("models/model_artifacts/tableformer/fast/tm_config.json") as f:
        config = json.load(f)

    # Add required fields
    config.setdefault("model", {})["save_dir"] = "/tmp"

    print("✅ Config loaded")

    # 2. Create model
    model = TableModel04_rs(config, init_data={}, device="cpu")
    print("✅ Model created")
    print(f"   Architecture: {type(model).__name__}")
    print(f"   Components:")
    print(f"     - Encoder: {type(model._encoder).__name__}")
    print(f"     - Tag Transformer: {type(model._tag_transformer).__name__}")
    print(f"     - BBox Decoder: {type(model._bbox_decoder).__name__}")

    # 3. Load weights
    state_dict = load_file("models/model_artifacts/tableformer/fast/tableformer_fast.safetensors")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Weights loaded ({len(state_dict)} parameters)")

    # 4. Forward pass
    dummy_input = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✅ Forward pass complete")
    print(f"   Sequence length: {len(output['sequence'])}")
    print(f"   BBox count: {output['bbox_coords'].shape[0]}")
    print(f"   First 5 tags: {output['decoded_tags'][:5]}")

    print("\n=== What TorchSharp needs to implement ===")
    print("1. TableModel04_rs → TableModel04.cs")
    print("2. Encoder04 → Encoder04.cs")
    print("3. Tag_Transformer → TagTransformer.cs")
    print("4. BBoxDecoder → BBoxDecoder.cs")
    print("5. Autoregressive loop in forward()")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python3 scripts/torchsharp_reference.py
```

---

## Step 4: API Mapping Cheat Sheet

### PyTorch → TorchSharp Common Patterns

| PyTorch (Python) | TorchSharp (C#) |
|------------------|-----------------|
| `import torch` | `using TorchSharp;` |
| `import torch.nn as nn` | `using static TorchSharp.torch.nn;` |
| `class MyModel(nn.Module):` | `class MyModel : Module<Tensor, Tensor>` |
| `def __init__(self):` | `public MyModel() : base("MyModel")` |
| `self.conv = nn.Conv2d(...)` | `_conv = Conv2d(...);` |
| `def forward(self, x):` | `public override Tensor forward(Tensor x)` |
| `x = self.conv(x)` | `x = _conv.forward(x);` |
| `x = F.relu(x)` | `x = functional.relu(x);` |
| `x.argmax(dim=1)` | `x.argmax(dim: 1)` |
| `torch.no_grad()` | `using var _ = torch.no_grad();` |
| `model.eval()` | `model.eval();` |
| `torch.tensor([1, 2, 3])` | `torch.tensor(new long[] {1, 2, 3})` |
| `x.reshape(-1, 10)` | `x.reshape(-1, 10)` |
| `torch.stack(tensors)` | `torch.stack(tensors.ToArray())` |

### Example: Conv2d Layer

**Python**:
```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
x = self.conv1(x)
```

**C# TorchSharp**:
```csharp
_conv1 = Conv2d(3, 64, kernelSize: 7, stride: 2, padding: 3);
x = _conv1.forward(x);
```

### Example: Linear Layer

**Python**:
```python
self.fc = nn.Linear(512, 128)
x = self.fc(x)
```

**C# TorchSharp**:
```csharp
_fc = Linear(512, 128);
x = _fc.forward(x);
```

### Example: Embedding

**Python**:
```python
self.embedding = nn.Embedding(vocab_size, d_model)
x = self.embedding(tokens)
```

**C# TorchSharp**:
```csharp
_embedding = Embedding(vocabSize, dModel);
x = _embedding.forward(tokens);
```

---

## Step 5: Port One Component (2 hours)

Let's port the simplest component: **PositionalEncoding**

### Python (from transformer_rs.py):

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
```

### C# TorchSharp:

```csharp
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TableFormerSdk.Models
{
    public class PositionalEncoding : Module<Tensor, Tensor>
    {
        private readonly Dropout _dropout;
        private readonly Tensor _pe;

        public PositionalEncoding(long dModel, double dropout = 0.1, long maxLen = 5000)
            : base("PositionalEncoding")
        {
            _dropout = Dropout(dropout);

            // Create positional encoding
            var position = torch.arange(maxLen).unsqueeze(1);
            var divTerm = torch.exp(
                torch.arange(0, dModel, 2) * (-Math.Log(10000.0) / dModel)
            );

            var pe = torch.zeros(maxLen, 1, dModel);
            pe[.., 0, torch.arange(0, dModel, 2)] = torch.sin(position * divTerm);
            pe[.., 0, torch.arange(1, dModel, 2)] = torch.cos(position * divTerm);

            register_buffer("pe", pe);
            _pe = pe;
        }

        public override Tensor forward(Tensor x)
        {
            x = x + _pe[torch.arange(x.size(0))];
            return _dropout.forward(x);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _dropout?.Dispose();
                _pe?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
```

### Test It:

```csharp
var pe = new PositionalEncoding(dModel: 512, dropout: 0.1);
var input = torch.randn(10, 1, 512);  // (seq_len, batch, d_model)
var output = pe.forward(input);
Console.WriteLine($"Input: {input.shape}, Output: {output.shape}");
```

---

## Step 6: Next Steps

After completing this quick start, you're ready for:

1. **Port Encoder04** (see full plan for details)
2. **Port TagTransformer**
3. **Port BBoxDecoder**
4. **Integrate everything into TableModel04**
5. **Add safetensors loading**
6. **Test against Python baseline**

---

## Troubleshooting

### Issue: "TorchSharp not found"

```bash
dotnet add package TorchSharp --version 0.105.1
dotnet add package TorchSharp-cpu --version 0.105.1
dotnet restore
```

### Issue: "libtorch not loading"

- Make sure TorchSharp-cpu (or TorchSharp-cuda) is installed
- Check platform-specific package is available
- Try cleaning: `dotnet clean && dotnet restore`

### Issue: "Safetensors loading fails"

TorchSharp uses libtorch which should support safetensors. If not:

```python
# Convert safetensors to .pth format
import torch
from safetensors.torch import load_file

state_dict = load_file("tableformer_fast.safetensors")
torch.save(state_dict, "tableformer_fast.pth")
```

Then load in C#:
```csharp
var stateDict = torch.load("tableformer_fast.pth");
model.load_state_dict(stateDict);
```

---

## Resources

- **TorchSharp Examples**: https://github.com/dotnet/TorchSharp/tree/main/test
- **API Documentation**: https://github.com/dotnet/TorchSharp/tree/main/docfx
- **PyTorch Docs (reference)**: https://pytorch.org/docs/stable/
- **Full Adoption Plan**: See `TORCHSHARP_ADOPTION_PLAN.md`

---

**Ready to start?** Begin with Step 1 and work through each step sequentially!
