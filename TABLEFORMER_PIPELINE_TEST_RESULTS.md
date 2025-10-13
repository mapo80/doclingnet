# TableFormer Pipeline Test Results

**Test Date**: 2025-10-13
**Test Image**: `dataset/2305.03393v1-pg9-img.png` (301KB, 1275x1650px)
**Content**: Academic paper page with one table (Table 1 showing HPO results)

## Test Summary

### .NET Pipeline Execution

**Command**:
```bash
export TABLEFORMER_FAST_MODELS_PATH="/path/to/models"
dotnet run --project src/Docling.Tooling/Docling.Tooling.csproj -- convert \
  --input dataset/2305.03393v1-pg9-img.png \
  --output test-output \
  --table-mode fast \
  --table-debug
```

**Results**:

| Stage | Duration | Status |
|-------|----------|--------|
| Page Preprocessing | 25ms | ✅ Success |
| Layout Analysis | 1,250ms | ✅ Success |
| Table Structure | 19ms | ⚠️ **FAILED** |
| OCR | 10,859ms | ✅ Success |
| Page Assembly | 21ms | ✅ Success |
| Image Export | 71ms | ✅ Success |
| Markdown Serialization | 0ms | ✅ Success |
| **TOTAL** | **12,598ms (~12.6s)** | ⚠️ Partial |

### Layout Detection Results

✅ **Layout detection worked correctly**:
- Detected **12 layout regions**
- Found **1 table region** at correct position
- Detected **11 text regions** (paragraphs, headers)

### TableFormer Results

❌ **TableFormer inference FAILED**:

**Error**:
```
TableFormer inference error: [ErrorCode:InvalidArgument]
Input name: 'encoder_mask' is not in the metadata
```

**Issue**: The ONNX model exported for the tag_transformer_decoder_step is expecting an input named `encoder_mask`, but this input is missing from the model metadata or is not being provided correctly by the .NET code.

**Result**:
- Table structure inferred with **0 cells** (should have detected ~40 cells from Table 1)
- TableFormer backend initialized successfully but inference failed
- Fell back to stub behavior (no structure extraction)

## Technical Details

### Model Loading

✅ **Model paths configured correctly**:
```
TABLEFORMER_FAST_MODELS_PATH=/path/to/models
```

✅ **Models loaded**:
- `tableformer_fast_encoder.onnx`
- `tableformer_fast_tag_transformer_encoder.onnx`
- `tableformer_fast_tag_transformer_decoder_step.onnx`
- `tableformer_fast_bbox_decoder.onnx`
- `tableformer_fast_config.json`
- `tableformer_fast_wordmap.json`

### Performance Breakdown

**Without OCR** (for table-only comparison):
- Layout Analysis: 1.25s
- Table Structure: 0.019s (failed)
- **Total**: ~1.27s

**With full pipeline**:
- Total time: 12.6s
- OCR dominated: 10.9s (86% of total time)

## Known Issues

### Issue #1: encoder_mask Input Missing

**Symptom**: ONNX Runtime error during tag_transformer_decoder_step inference

**Root Cause**: The decoder step model expects an `encoder_mask` input that is either:
1. Not exported correctly during ONNX conversion
2. Not being passed by the .NET inference code
3. Has incorrect shape or name in the model metadata

**Impact**: TableFormer cannot extract table structure (0 cells detected)

**Next Steps**:
1. ✅ Verify ONNX model export includes `encoder_mask` as input
2. Check Python reference implementation for how encoder_mask is generated
3. Update .NET inference code to provide encoder_mask with correct shape
4. Re-export ONNX models if necessary

### Issue #2: Python Comparison Not Available

Could not run Python reference implementation due to:
- `docling_ibm_models.tableformer.tableformer_predictor` module not found
- Python TableFormer API has changed or is not available as a simple predictor

**Workaround**: Need to check latest `docling_ibm_models` package or use lower-level model API.

## Expected vs Actual Results

### Expected (from image visual inspection):
- **1 table** detected by layout
- Table 1 contains:
  - **6 rows** (including header)
  - **7 columns** (enc-layers, dec-layers, Language, TEDs columns, mAP, Inference time)
  - Approximately **40-42 cells** total
  - Several cells with row/column spans

### Actual:
- ✅ **1 table** detected by layout (correct)
- ❌ **0 rows** detected by TableFormer
- ❌ **0 columns** detected by TableFormer
- ❌ **0 cells** extracted

## Comparison Status

| Metric | Python | .NET | Status |
|--------|--------|------|--------|
| Layout Detection | N/A | ✅ Working | - |
| TableFormer Loading | N/A | ✅ Working | - |
| TableFormer Inference | N/A | ❌ **FAILED** | Blocked |
| Cell Detection | N/A | 0 cells | Blocked |
| Performance | N/A | ~1.3s (table only) | Blocked |

**Conclusion**: Full Python vs .NET comparison is **BLOCKED** until the `encoder_mask` issue is resolved in the .NET TableFormer inference pipeline.

## Recommendations

### Priority 1: Fix encoder_mask Issue

1. **Investigate ONNX Export**:
   ```bash
   python tools/convert_tableformer_components_to_onnx.py
   ```
   - Verify decoder_step export includes encoder_mask input
   - Check input shapes and names match Python implementation

2. **Review .NET Inference Code**:
   - File: `src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerSdk/Backends/TableFormerOnnxBackend.cs`
   - Check autoregressive loop provides encoder_mask to decoder_step
   - Verify encoder_mask shape: should be `[seq_len, batch_size]` or `[batch_size, seq_len]`

3. **Compare with Python Reference**:
   - Check `docling_ibm_models/tableformer/models/table04_rs/tablemodel04_rs.py`
   - See how encoder_mask is generated and passed to decoder

### Priority 2: Enable Python Comparison

Once .NET is fixed:
1. Set up Python environment with correct `docling_ibm_models` version
2. Run same image through Python pipeline
3. Compare:
   - Cell detection accuracy
   - Row/column counts
   - Inference time
   - Memory usage

## Files Generated

- `netpipeline_output.txt` - Full .NET pipeline log
- `test-output/docling.md` - Generated markdown (incomplete due to TableFormer failure)
- `test-output/assets/page-0.png` - Exported page image

## Environment

- **.NET Version**: 9.0
- **Platform**: macOS (Darwin 23.0.0)
- **CPU**: Apple Silicon (ARM64)
- **ONNX Runtime**: 1.20.1
- **LayoutSdk**: Working (Onnx backend)
- **TableFormerSdk**: Partially working (loading OK, inference FAILED)
