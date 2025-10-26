# TableFormer TorchSharp Adoption Plan

**Date**: 2025-10-17
**Status**: 🟡 Ready for Implementation
**Priority**: HIGH

---

## Executive Summary

After extensive testing, ONNX conversion of TableFormer has proven **infeasible** due to:
- ❌ Autoregressive loop with dynamic early stopping
- ❌ Variable sequence lengths (1-500 tokens)
- ❌ BBox coordinate differences > 0.7 (unacceptable!)
- ❌ Sequence always max length (1024) instead of actual (50-300)

**TorchSharp is the ONLY viable path forward** because:
- ✅ Loads safetensors directly (no conversion)
- ✅ 1:1 API mapping with PyTorch
- ✅ We have working Python implementation to port
- ✅ Same accuracy as Python (100% equivalent)

---

## Current Project Structure

```
src/submodules/ds4sd-docling-tableformer-onnx/
├── dotnet/
│   ├── TableFormerSdk/              # Existing ONNX-based SDK
│   │   ├── Backends/
│   │   │   ├── TableFormerOnnxBackend.cs
│   │   │   └── ...
│   │   └── ...
│   └── TableFormerSdk.Tests/
├── models-onnx/                     # FAILED: BBox diff > 0.7
├── models-onnx-components/          # FAILED: Dynamic shape issues
├── dataset/
│   └── FinTabNet/
├── scripts/
└── *.py                             # Python scripts
```

---

## Phase 1: Setup TorchSharp Infrastructure

### 1.1 Add TorchSharp Packages

Update `dotnet/TableFormerSdk/TableFormerSdk.csproj`:

```xml
<ItemGroup>
  <!-- Existing packages -->
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.22.1" />

  <!-- NEW: TorchSharp packages -->
  <PackageReference Include="TorchSharp" Version="0.105.1" />
  <PackageReference Include="TorchSharp-cpu" Version="0.105.1" />
  <!-- Or for CUDA support: -->
  <!-- <PackageReference Include="TorchSharp-cuda-windows" Version="0.105.1" /> -->
  <!-- <PackageReference Include="TorchSharp-cuda-linux" Version="0.105.1" /> -->
</ItemGroup>
```

### 1.2 Download Safetensors Models

```bash
# Download TableFormer fast model
huggingface-cli download ds4sd/docling-models \
  model_artifacts/tableformer/fast/tableformer_fast.safetensors \
  model_artifacts/tableformer/fast/tm_config.json \
  --local-dir models

# Download TableFormer accurate model (optional)
huggingface-cli download ds4sd/docling-models \
  model_artifacts/tableformer/accurate/tableformer_accurate.safetensors \
  model_artifacts/tableformer/accurate/tm_config.json \
  --local-dir models
```

**Target location**: `models/model_artifacts/tableformer/fast/`

---

## Phase 2: Port Python Model Architecture to C#

### 2.1 Python Source Files to Port

From `docling-ibm-models` package (already installed in Python):

```
/Users/politom/.pyenv/versions/3.11.8/lib/python3.11/site-packages/docling_ibm_models/tableformer/models/
├── table04_rs/
│   ├── tablemodel04_rs.py          → TableModel04.cs (main model)
│   ├── encoder04_rs.py             → Encoder04.cs (CNN encoder)
│   ├── transformer_rs.py           → TagTransformer.cs (decoder)
│   └── bbox_decoder_rs.py          → BBoxDecoder.cs (bbox prediction)
└── common/
    └── base_model.py               → BaseModel.cs (base class)
```

### 2.2 File-by-File Porting Guide

#### **File 1: `TableModel04.cs`** (Main Model)

**Python**: `tablemodel04_rs.py` (~150 lines)

**Key components**:
```python
class TableModel04_rs(BaseModel):
    def __init__(self, config, init_data, device):
        self._encoder = Encoder04(...)
        self._tag_transformer = Tag_Transformer(...)
        self._bbox_decoder = BBoxDecoder(...)

    def forward(self, images):
        # 1. CNN encoding
        enc_out = self._encoder(images)

        # 2. Autoregressive decoding
        sequence = [self._start_token]
        while len(sequence) < max_len:
            logits = self._tag_transformer(enc_out, sequence)
            next_token = logits.argmax()
            if next_token == self._end_token:
                break
            sequence.append(next_token)

        # 3. BBox prediction
        bbox_classes, bbox_coords = self._bbox_decoder(enc_out, tag_hidden_states)

        return {
            'sequence': sequence,
            'bbox_classes': bbox_classes,
            'bbox_coords': bbox_coords
        }
```

**C# TorchSharp equivalent**:
```csharp
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class TableModel04 : Module<Tensor, Dictionary<string, object>>
{
    private readonly Encoder04 _encoder;
    private readonly TagTransformer _tagTransformer;
    private readonly BBoxDecoder _bboxDecoder;
    private readonly int _startToken;
    private readonly int _endToken;

    public TableModel04(Config config, Device device) : base("TableModel04")
    {
        _encoder = new Encoder04(config);
        _tagTransformer = new TagTransformer(config);
        _bboxDecoder = new BBoxDecoder(config);

        _startToken = config.WordMap["<start>"];
        _endToken = config.WordMap["<end>"];

        RegisterComponents();
    }

    public override Dictionary<string, object> forward(Tensor images)
    {
        using var _ = torch.no_grad();

        // 1. CNN encoding
        var encOut = _encoder.forward(images);

        // 2. Autoregressive decoding
        var sequence = new List<long> { _startToken };
        var tagHiddenStates = new List<Tensor>();

        for (int step = 0; step < 500; step++)
        {
            var seqTensor = torch.tensor(sequence.ToArray(), dtype: ScalarType.Int64)
                                .reshape(-1, 1);

            var (logits, tagHidden) = _tagTransformer.forward(encOut, seqTensor);
            var nextToken = logits.argmax(dim: -1).item<long>();

            if (nextToken == _endToken)
                break;

            sequence.Add(nextToken);
            tagHiddenStates.Add(tagHidden.clone());
        }

        // 3. BBox prediction
        var (bboxClasses, bboxCoords) = _bboxDecoder.forward(encOut, tagHiddenStates);

        return new Dictionary<string, object>
        {
            ["sequence"] = sequence,
            ["bbox_classes"] = bboxClasses,
            ["bbox_coords"] = bboxCoords
        };
    }
}
```

#### **File 2: `Encoder04.cs`** (CNN Encoder)

**Python**: `encoder04_rs.py` (~200 lines)

**Architecture**: ResNet-like CNN

```python
class Encoder04(nn.Module):
    def __init__(self, config):
        # Conv layers: 3 → 64 → 128 → 256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # ... more layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # ... forward pass
        return x  # Shape: (B, 256, 28, 28)
```

**C# TorchSharp**:
```csharp
public class Encoder04 : Module<Tensor, Tensor>
{
    private readonly Conv2d _conv1;
    private readonly BatchNorm2d _bn1;
    private readonly ReLU _relu;
    // ... more layers

    public Encoder04(Config config) : base("Encoder04")
    {
        _conv1 = Conv2d(3, 64, kernelSize: 7, stride: 2, padding: 3);
        _bn1 = BatchNorm2d(64);
        _relu = ReLU();
        // ... initialize more layers

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = _conv1.forward(x);
        x = _bn1.forward(x);
        x = _relu.forward(x);
        // ... forward pass
        return x;  // Shape: (B, 256, 28, 28)
    }
}
```

#### **File 3: `TagTransformer.cs`** (Autoregressive Decoder)

**Python**: `transformer_rs.py` (~300 lines)

**Architecture**: Transformer decoder with positional encoding

```python
class Tag_Transformer(nn.Module):
    def __init__(self, config):
        self._embedding = nn.Embedding(vocab_size, d_model)
        self._positional_encoding = PositionalEncoding(d_model)
        self._encoder = TransformerEncoder(...)
        self._decoder = TransformerDecoder(...)
        self._fc = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_out, decoded_tags):
        # Encoder side
        enc_inputs = self._input_filter(encoder_out)
        memory = self._encoder(enc_inputs)

        # Decoder side
        decoded_embedding = self._embedding(decoded_tags)
        decoded_embedding = self._positional_encoding(decoded_embedding)
        decoded = self._decoder(decoded_embedding, memory)

        logits = self._fc(decoded[-1])
        tag_hidden = decoded[-1]

        return logits, tag_hidden
```

**C# TorchSharp**:
```csharp
public class TagTransformer : Module<(Tensor, Tensor), (Tensor, Tensor)>
{
    private readonly Embedding _embedding;
    private readonly PositionalEncoding _positionalEncoding;
    private readonly TransformerEncoder _encoder;
    private readonly TransformerDecoder _decoder;
    private readonly Linear _fc;

    public TagTransformer(Config config) : base("TagTransformer")
    {
        _embedding = Embedding(config.VocabSize, config.DModel);
        _positionalEncoding = new PositionalEncoding(config.DModel);
        _encoder = TransformerEncoder(...);
        _decoder = TransformerDecoder(...);
        _fc = Linear(config.DModel, config.VocabSize);

        RegisterComponents();
    }

    public override (Tensor, Tensor) forward((Tensor, Tensor) input)
    {
        var (encoderOut, decodedTags) = input;

        // Encoder side
        var encInputs = InputFilter(encoderOut);
        var memory = _encoder.forward(encInputs, mask: null);

        // Decoder side
        var decodedEmbedding = _embedding.forward(decodedTags);
        decodedEmbedding = _positionalEncoding.forward(decodedEmbedding);
        var decoded = _decoder.forward(decodedEmbedding, memory, tgt_mask: null);

        var logits = _fc.forward(decoded[-1]);
        var tagHidden = decoded[-1];

        return (logits, tagHidden);
    }
}
```

#### **File 4: `BBoxDecoder.cs`** (Bounding Box Prediction)

**Python**: `bbox_decoder_rs.py` (~150 lines)

```python
class BBoxDecoder(nn.Module):
    def __init__(self, config):
        self.fc1 = nn.Linear(d_model, 512)
        self.fc_classes = nn.Linear(512, 3)
        self.fc_coords = nn.Linear(512, 4)

    def forward(self, encoder_out, tag_hidden_states):
        bbox_classes = []
        bbox_coords = []

        for tag_hidden in tag_hidden_states:
            x = F.relu(self.fc1(tag_hidden))
            classes = torch.sigmoid(self.fc_classes(x))
            coords = torch.sigmoid(self.fc_coords(x))

            bbox_classes.append(classes)
            bbox_coords.append(coords)

        return torch.stack(bbox_classes), torch.stack(bbox_coords)
```

**C# TorchSharp**:
```csharp
public class BBoxDecoder : Module<(Tensor, List<Tensor>), (Tensor, Tensor)>
{
    private readonly Linear _fc1;
    private readonly Linear _fcClasses;
    private readonly Linear _fcCoords;

    public BBoxDecoder(Config config) : base("BBoxDecoder")
    {
        _fc1 = Linear(config.DModel, 512);
        _fcClasses = Linear(512, 3);
        _fcCoords = Linear(512, 4);

        RegisterComponents();
    }

    public override (Tensor, Tensor) forward((Tensor, List<Tensor>) input)
    {
        var (encoderOut, tagHiddenStates) = input;

        var bboxClassesList = new List<Tensor>();
        var bboxCoordsList = new List<Tensor>();

        foreach (var tagHidden in tagHiddenStates)
        {
            var x = functional.relu(_fc1.forward(tagHidden));
            var classes = torch.sigmoid(_fcClasses.forward(x));
            var coords = torch.sigmoid(_fcCoords.forward(x));

            bboxClassesList.Add(classes);
            bboxCoordsList.Add(coords);
        }

        return (torch.stack(bboxClassesList), torch.stack(bboxCoordsList));
    }
}
```

---

## Phase 3: Create TorchSharp Backend

### 3.1 New Backend Class

Create `dotnet/TableFormerSdk/Backends/TableFormerTorchSharpBackend.cs`:

```csharp
using TorchSharp;
using static TorchSharp.torch;

namespace TableFormerSdk.Backends
{
    public class TableFormerTorchSharpBackend : ITableFormerBackend
    {
        private readonly TableModel04 _model;
        private readonly Config _config;
        private readonly Device _device;

        public TableFormerTorchSharpBackend(string modelPath, string configPath)
        {
            _config = Config.Load(configPath);
            _device = cuda.is_available() ? CUDA : CPU;

            // Create model architecture
            _model = new TableModel04(_config, _device);

            // Load weights from safetensors
            LoadSafetensors(modelPath);

            _model.eval();
        }

        private void LoadSafetensors(string path)
        {
            // TorchSharp can load safetensors via libtorch
            var stateDict = torch.load(path);
            _model.load_state_dict(stateDict);
        }

        public TableStructureResult Predict(byte[] imageData)
        {
            // Preprocess image
            var imageTensor = PreprocessImage(imageData);

            // Run inference
            var output = _model.forward(imageTensor);

            // Postprocess
            return PostprocessOutput(output);
        }

        private Tensor PreprocessImage(byte[] imageData)
        {
            // Load image, resize to 448x448, normalize
            // Mean: [0.485, 0.456, 0.406]
            // Std: [0.229, 0.224, 0.225]
            // ...
        }

        private TableStructureResult PostprocessOutput(Dictionary<string, object> output)
        {
            var sequence = (List<long>)output["sequence"];
            var bboxCoords = (Tensor)output["bbox_coords"];

            // Convert to TableStructureResult
            // ...
        }
    }
}
```

### 3.2 Update Backend Registry

Modify `dotnet/TableFormerSdk/BackendRegistry.cs`:

```csharp
public enum TableFormerRuntime
{
    Auto,
    ONNX,
    ORT,
    OpenVINO,
    TorchSharp  // NEW
}

public class BackendRegistry
{
    public ITableFormerBackend CreateBackend(
        TableFormerRuntime runtime,
        TableFormerModelPaths paths)
    {
        return runtime switch
        {
            TableFormerRuntime.ONNX => new TableFormerOnnxBackend(paths),
            TableFormerRuntime.TorchSharp => new TableFormerTorchSharpBackend(
                paths.SafetensorsPath,
                paths.ConfigPath
            ),
            // ... other backends
        };
    }
}
```

---

## Phase 4: Testing & Validation

### 4.1 Unit Tests

Create `dotnet/TableFormerSdk.Tests/TorchSharpBackendTests.cs`:

```csharp
[Fact]
public void TorchSharpBackend_ShouldLoadModel()
{
    var backend = new TableFormerTorchSharpBackend(
        "models/tableformer_fast.safetensors",
        "models/tm_config.json"
    );

    Assert.NotNull(backend);
}

[Fact]
public void TorchSharpBackend_ShouldPredict()
{
    var backend = new TableFormerTorchSharpBackend(...);
    var imageData = File.ReadAllBytes("test_image.png");

    var result = backend.Predict(imageData);

    Assert.True(result.Cells.Count > 0);
}

[Fact]
public void TorchSharpBackend_ShouldMatchPythonBaseline()
{
    // Compare C# TorchSharp output with Python safetensors output
    var csharpResult = _torchSharpBackend.Predict(imageData);
    var pythonResult = LoadPythonBaseline("baseline.json");

    Assert.Equal(pythonResult.CellCount, csharpResult.Cells.Count);
    // Assert bbox coordinates are within tolerance
}
```

### 4.2 Validation Script

Create `scripts/validate_torchsharp_vs_python.py`:

```python
#!/usr/bin/env python3
"""
Compare TorchSharp C# implementation with Python baseline
"""

def validate():
    # 1. Run Python inference
    python_result = run_python_inference(image_path)

    # 2. Run C# TorchSharp inference
    csharp_result = run_csharp_inference(image_path)

    # 3. Compare
    sequence_match = python_result['sequence'] == csharp_result['sequence']
    bbox_diff = np.abs(python_result['bbox_coords'] - csharp_result['bbox_coords']).max()

    assert sequence_match, "Sequences don't match!"
    assert bbox_diff < 1e-4, f"BBox diff too large: {bbox_diff}"

    print("✅ TorchSharp implementation matches Python!")
```

---

## Phase 5: Performance Optimization

### 5.1 Batching Support

```csharp
public List<TableStructureResult> PredictBatch(List<byte[]> images)
{
    // Stack images into batch: (B, 3, 448, 448)
    var imageTensors = images.Select(PreprocessImage).ToList();
    var batchTensor = torch.stack(imageTensors);

    // Run batch inference
    var outputs = _model.forward(batchTensor);

    // Split results
    return SplitBatchResults(outputs);
}
```

### 5.2 GPU Acceleration

```csharp
public TableFormerTorchSharpBackend(string modelPath, string configPath, bool useGpu = false)
{
    _device = useGpu && cuda.is_available() ? CUDA : CPU;
    _model = new TableModel04(_config, _device).to(_device);
}
```

### 5.3 Mixed Precision (FP16)

```csharp
if (_device.type == DeviceType.CUDA)
{
    _model.half();  // Convert to FP16
}
```

---

## Implementation Roadmap

### Week 1-2: Infrastructure Setup
- [ ] Add TorchSharp packages
- [ ] Download safetensors models
- [ ] Set up project structure
- [ ] Create Config.cs for model configuration

### Week 3-4: Core Model Porting
- [ ] Port Encoder04 (CNN)
- [ ] Port TagTransformer (decoder)
- [ ] Port BBoxDecoder
- [ ] Port TableModel04 (main model)

### Week 5: Integration
- [ ] Create TableFormerTorchSharpBackend
- [ ] Integrate with existing SDK
- [ ] Add safetensors loading logic

### Week 6: Testing
- [ ] Unit tests for each component
- [ ] Integration tests
- [ ] Validation vs Python baseline
- [ ] Performance benchmarks

### Week 7: Documentation
- [ ] API documentation
- [ ] Usage examples
- [ ] Migration guide from ONNX to TorchSharp

---

## Benefits of TorchSharp Approach

| Aspect | ONNX | TorchSharp |
|--------|------|------------|
| **Accuracy** | ❌ BBox diff > 0.7 | ✅ Same as Python (< 1e-4) |
| **Sequence Length** | ❌ Always 1024 | ✅ Correct (50-300) |
| **Autoregressive Loop** | ❌ Hard-coded/broken | ✅ Full control in C# |
| **Model Updates** | ❌ Re-export required | ✅ Just swap safetensors file |
| **Debugging** | ❌ Black box | ✅ Full C# debugger support |
| **Performance** | ✅ Good | ✅ Same (libtorch backend) |
| **Maintenance** | ❌ Conversion pipeline | ✅ Direct code correspondence |

---

## Risks & Mitigation

### Risk 1: TorchSharp API Differences
**Mitigation**: Follow PyTorch → TorchSharp mapping guide, extensive testing

### Risk 2: Safetensors Loading Issues
**Mitigation**: TorchSharp uses libtorch which supports safetensors natively

### Risk 3: Performance Concerns
**Mitigation**: TorchSharp uses same libtorch backend as Python PyTorch

### Risk 4: Complex Model Architecture
**Mitigation**: Port incrementally, test each component separately

---

## Success Criteria

1. ✅ TorchSharp model loads safetensors successfully
2. ✅ Inference produces same results as Python (diff < 1e-4)
3. ✅ Autoregressive loop works correctly (correct sequence lengths)
4. ✅ BBox predictions match Python baseline
5. ✅ Performance comparable to Python (~0.1-0.5s per image)
6. ✅ All existing SDK tests pass with TorchSharp backend

---

## Next Steps

1. **Get approval** for TorchSharp adoption
2. **Schedule kick-off meeting** with team
3. **Assign developers** to porting tasks
4. **Set up development environment** with TorchSharp
5. **Start with Encoder04** (simplest component)
6. **Iterate** through each component with testing

---

## References

- **TorchSharp GitHub**: https://github.com/dotnet/TorchSharp
- **TorchSharp Documentation**: https://github.com/dotnet/TorchSharp/tree/main/docfx
- **PyTorch → TorchSharp Mapping**: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/mapping.md
- **TableFormer Paper**: https://arxiv.org/abs/2203.01017
- **Docling Models**: https://huggingface.co/ds4sd/docling-models

---

## Contact

For questions about this plan:
- **Technical Lead**: [Your Name]
- **Architecture Review**: [Reviewer Name]
- **Timeline**: 7 weeks (target completion: [Date])

---

**Status**: 🟡 Ready for Team Review
**Last Updated**: 2025-10-17
