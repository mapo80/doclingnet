# Phase 2 Completion Summary - ONNX Conversion

**Date**: 2025-10-12
**Status**: ✅ **COMPLETED**
**Duration**: ~2 hours

## Objective

Convert official Docling TableFormer models (SafeTensors) to ONNX format for use in .NET backend.

## Challenges Encountered

### Challenge 1: Bbox Classes Dimension Mismatch

**Problem**: Config specified `bbox_classes: 2`, but model actually outputs `num_classes + 1 = 3` classes.

**Error**:
```
RuntimeError: The expanded size of the tensor (2) must match the existing size (3) at non-singleton dimension 1.
```

**Root Cause**: In `bbox_decoder_rs.py` line 117:
```python
self._class_embed = nn.Linear(512, self._num_classes + 1)
```

**Solution**: Modified conversion script to dynamically detect actual output dimensions instead of trusting config values.

### Challenge 2: Autoregressive Loop Export Timeout

**Problem**: Initial approach tried to export the entire autoregressive loop as a single ONNX model. This caused:
- 5+ minute timeout with no completion
- Multiple `TracerWarning` messages about dynamic control flow
- ONNX tracer unable to handle Python while loops with tensor-dependent conditionals

**Error Symptoms**:
```
TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect
TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect
```

**Root Cause**: ONNX's static graph representation cannot efficiently handle:
- Unbounded while loops (`while step < max_steps`)
- Conditionals based on tensor values (`if new_tag == end_token`)
- Dynamic control flow (`if not skip_next_tag`)
- Variable-length sequences

**Solution**: **Component-wise export strategy** - Split model into 4 independent ONNX models:
1. **Encoder** - Image preprocessing and feature extraction
2. **Tag Transformer Encoder** - Encode image features to memory
3. **Tag Transformer Decoder Step** - Single autoregressive decoding step
4. **BBox Decoder** - Predict bounding boxes for detected cells

The autoregressive loop logic will be implemented in C# .NET code, giving full control and performance.

## Files Created

### Conversion Scripts

1. **tools/convert_tableformer_to_onnx.py** (deprecated)
   - Initial monolithic approach
   - Not functional due to autoregressive loop issues
   - Kept for reference

2. **tools/convert_tableformer_components_to_onnx.py** (✅ working)
   - Component-wise export strategy
   - Successfully exports all 4 components
   - Includes validation and test code

### ONNX Models (Fast Variant)

All models exported to: `models/tableformer-onnx/`

| File | Size | Description |
|------|------|-------------|
| `tableformer_fast_encoder.onnx` | 11 MB | ResNet-18 encoder: (1,3,448,448) → (1,28,28,256) |
| `tableformer_fast_tag_transformer_encoder.onnx` | 64 MB | Transformer encoder: (1,28,28,256) → (784,1,512) |
| `tableformer_fast_tag_transformer_decoder_step.onnx` | 26 MB | Single decoder step: tags + memory → logits + hidden |
| `tableformer_fast_bbox_decoder.onnx` | 38 MB | BBox predictor: encoder_out + hiddens → classes + coords |

**Total size**: ~139 MB

### ONNX Models (Accurate Variant)

Salvati nella stessa cartella per garantire parità di esecuzione con la modalità Fast:

| File | Size | Description |
|------|------|-------------|
| `tableformer_accurate_encoder.onnx` | 15 MB | ResNet-18 encoder con pesi Accurate |
| `tableformer_accurate_tag_transformer_encoder.onnx` | 88 MB | Transformer encoder Accurate |
| `tableformer_accurate_tag_transformer_decoder_step.onnx` | 34 MB | Decoder autoregressivo Accurate |
| `tableformer_accurate_bbox_decoder.onnx` | 52 MB | Predizione bounding box Accurate |

**Total size**: ~189 MB

### Configuration Files

- `tableformer_fast_config.json` - Full model configuration
- `tableformer_fast_wordmap.json` - OTSL token vocabulary mapping

## Model Architecture (Component View)

```
Input Image (1, 3, 448, 448)
    ↓
[1] Encoder (encoder.onnx)
    ↓
Encoder Features (1, 28, 28, 256)
    ↓
[2] Tag Transformer Encoder (tag_transformer_encoder.onnx)
    ↓
Memory (784, 1, 512)
    ↓
    ├─→ [3] Tag Transformer Decoder Step (tag_transformer_decoder_step.onnx)
    │   ↓  (autoregressive loop in C# - call N times)
    │   Previous Tags → Next Tag Logits + Hidden State
    │   ↓
    │   Collect all tag hidden states
    │
    └─→ [4] BBox Decoder (bbox_decoder.onnx)
        Encoder Features + Tag Hidden States → BBox Classes + Coords
```

## Technical Details

### Input/Output Specifications

#### 1. Encoder
- **Input**: `images` (batch, 3, 448, 448) - float32
- **Output**: `encoder_out` (batch, 28, 28, 256) - float32

#### 2. Tag Transformer Encoder
- **Input**: `encoder_out` (batch, 28, 28, 256) - float32
- **Output**: `memory` (784, batch, 512) - float32

#### 3. Tag Transformer Decoder Step
- **Inputs**:
  - `decoded_tags` (seq_len, batch) - int64 (sequence of tag indices so far)
  - `memory` (784, batch, 512) - float32
  - `encoder_mask` (batch * n_heads, 784, 784) - bool
- **Outputs**:
  - `logits` (batch, 13) - float32 (vocabulary size = 13 OTSL tokens)
  - `tag_hidden` (batch, 512) - float32 (hidden state for bbox prediction)

#### 4. BBox Decoder
- **Inputs**:
  - `encoder_out` (batch, 28, 28, 256) - float32
  - `tag_hiddens` (num_cells, batch, 512) - float32
- **Outputs**:
  - `bbox_classes` (num_cells, 3) - float32 (logits for 3 classes)
  - `bbox_coords` (num_cells, 4) - float32 (cx, cy, w, h normalized 0-1)

### OTSL Vocabulary

Word map contains 13 tokens:
- `<start>` - Start of sequence
- `<end>` - End of sequence
- `<pad>` - Padding token
- `fcel` - First cell in row
- `ecel` - Empty cell
- `lcel` - Linked cell (horizontal span)
- `xcel` - Cross cell (vertical span continuation)
- `ucel` - Up cell (vertical span start)
- `nl` - New line (row separator)
- `ched` - Column header
- `rhed` - Row header
- `srow` - Spanning row

## Validation Results

All 4 ONNX models validated successfully:
- ✅ `encoder`: Valid
- ✅ `tag_transformer_encoder`: Valid
- ✅ `tag_transformer_decoder_step`: Valid
- ✅ `bbox_decoder`: Valid

ONNX checker confirms:
- Correct graph structure
- Valid operators (opset 17)
- Proper input/output shapes
- No missing dependencies

## Next Steps (Phase 3-4)

### Phase 3: Cleanup (estimated 2-3 hours)
1. Remove old ONNX models:
   - `src/submodules/ds4sd-docling-tableformer-onnx/models/encoder.onnx`
   - `src/submodules/ds4sd-docling-tableformer-onnx/models/bbox_decoder.onnx`
   - `src/submodules/ds4sd-docling-tableformer-onnx/models/decoder.onnx`
   - `src/submodules/ds4sd-docling-tableformer-onnx/models/tableformer-fast-encoder.onnx`

2. Remove obsolete backends:
   - `TableFormerPipelineBackend.cs`
   - OpenVINO-related code (if any)

3. Clean up old SDK code in `src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerSdk/`

### Phase 4: New .NET Implementation (estimated 8-12 hours)

#### 4.1 Create New Backend Structure
```csharp
// New files to create:
- TableFormerOnnxComponents.cs         // ONNX session wrappers
- TableFormerAutoregressive.cs         // Autoregressive loop logic
- OtslParser.cs                        // OTSL → Table structure
- TableFormerOnnxBackend.cs (rewrite) // Main integration
```

#### 4.2 Implementation Tasks

1. **Load ONNX Models** (2 hours)
   - Create InferenceSession for each component
   - Handle model paths and initialization
   - Implement proper disposal pattern

2. **Autoregressive Loop** (3-4 hours)
   - Implement tag generation loop (max 1024 steps)
   - Structure error correction logic:
     - First line: xcel → lcel
     - After ucel: lcel → fcel
   - Collect tag hidden states for cells requiring bboxes
   - Early stopping on `<end>` token

3. **BBox Prediction** (2 hours)
   - Call BBox decoder with collected hidden states
   - Implement bbox merging for spanning cells (lcel, ucel)
   - Convert from [cx, cy, w, h] to [x1, y1, x2, y2]

4. **OTSL Parser** (2-3 hours)
   - Parse OTSL tag sequence to table structure
   - Build cell grid with row/column spans
   - Handle special cases: ched, rhed, nl, srow
   - Group cells into rows and columns

5. **Coordinate Transformation** (1 hour)
   - Scale bboxes from normalized 0-1 to actual table coordinates
   - Apply table bounding box offset
   - Clamp to table boundaries

6. **Integration** (1-2 hours)
   - Update `TableFormerTableStructureService.cs`
   - Wire up new backend
   - Update options and configuration

## Key Insights

1. **Component-wise export is essential** for models with complex control flow
2. **Config values cannot always be trusted** - always validate against actual model outputs
3. **Dynamic dimensions require special handling** in ONNX export
4. **Autoregressive loops are best implemented in application code**, not in ONNX graphs
5. **PyTorch tracing has limitations** with control flow - component export avoids these

## Files Modified

- `tools/convert_tableformer_to_onnx.py` (created, deprecated)
- `tools/convert_tableformer_components_to_onnx.py` (created, working)

## References

- Original Python implementation: `/tmp/docling-ibm-models/docling_ibm_models/tableformer/models/table04_rs/`
- Architecture analysis: `docs/TABLEFORMER_ARCHITECTURE_ANALYSIS.md`
- Phase 1 summary: `FASE1_COMPLETED_SUMMARY.md`
- Intervention plan: `PIANO_INTERVENTO_TABLEFORMER_NEW.md`
