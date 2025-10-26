# Table Detection Models Evaluation Summary

**Date**: October 16, 2025
**Task**: Evaluate table cell detection models on FinTabNet dataset

---

## Executive Summary

**Winner**: **SLANetPlus**
- **F1 Score**: 66.5%
- **Perfect cell count matching** on all 3 test images
- **Inference time**: 1.32s per image

---

## Models Evaluated

### 1. Table Transformer (Microsoft)
- **Source**: HuggingFace Xenova/table-transformer-structure-recognition-v1.1-all
- **Format**: ONNX (quantized, 24MB)
- **Task**: Table structure detection (table/row/column/header)
- **Result**: ❌ **0% match with ground truth** - Wrong task, detects structures not individual cells
- **Inference time**: 0.25s per image
- **Detections**: ~11.5 structures per image
- **Notes**:
  - Fast and efficient for table detection/cropping
  - Not suitable for cell-level ground truth comparison
  - Fixed softmax + thresholding bug during evaluation

### 2. SLANetPlus
- **Source**: PyPI package `slanet-plus-table`
- **Format**: PaddlePaddle (.pdmodel, 0.3MB)
- **Task**: Individual table cell detection
- **Result**: ✅ **66.5% F1 Score**
- **Inference time**: 1.32s per image
- **Key Features**:
  - Returns tuple: (HTML string, cell bboxes [N, 8], inference time)
  - Cell bboxes in 8-point format [x1,y1,x2,y2,x3,y3,x4,y4]
  - No OCR dependency required
  - Based on PaddlePaddle framework

---

## Detailed Results

### FinTabNet Dataset Evaluation

**Dataset**: 3 images with ground truth cell annotations

| Image | Cells (GT) | Cells (Detected) | Precision | Recall | F1 | Inference Time |
|-------|------------|------------------|-----------|--------|----|----|
| HAL.2009.page_77.pdf_125051.png | 30 | 30 | 0.867 | 0.867 | 0.867 | 1.01s |
| HAL.2015.page_43.pdf_125177.png | 128 | 128 | 0.648 | 0.648 | 0.648 | 2.30s |
| HAL.2017.page_79.pdf_125247.png | 9 | 9 | 0.222 | 0.222 | 0.222 | 0.66s |

**Overall Metrics**:
- Total TP: 111
- Total FP: 56
- Total FN: 56
- **Overall Precision**: 0.665
- **Overall Recall**: 0.665
- **Overall F1 Score**: 0.665

### Key Observations

1. **Perfect Cell Count Matching**: SLANetPlus detected exactly the same number of cells as ground truth on all 3 images (128/128, 30/30, 9/9)

2. **IoU Distribution**: Matched cells have IoU ranging from 0.5 to 1.0, with most matches above 0.6

3. **Performance Variation**:
   - Best performance: HAL.2009.page_77 (F1: 0.867)
   - Worst performance: HAL.2017.page_79 (F1: 0.222) - possibly more complex table structure

4. **Inference Speed**: Scales with table complexity (0.66s for 9 cells, 2.30s for 128 cells)

---

## Model Comparison

### SLANetPlus vs Table Transformer

| Aspect | SLANetPlus | Table Transformer |
|--------|------------|-------------------|
| **Task** | Cell detection | Structure detection |
| **F1 Score** | 66.5% | 0% (wrong task) |
| **Cell Count Match** | Perfect (167/167) | N/A |
| **Inference Time** | 1.32s avg | 0.25s avg |
| **Model Format** | PaddlePaddle | ONNX |
| **Model Size** | 0.3MB | 24MB |
| **Framework** | PaddlePaddle | PyTorch → ONNX |

---

## Technical Details

### SLANetPlus Architecture
```python
from slanet_plus_table import SLANetPlus

model = SLANetPlus()
html, cells, elapsed = model(image)

# Returns:
# - html: str - HTML table representation
# - cells: ndarray [N, 8] - Cell bboxes in 8-point format
# - elapsed: float - Inference time in seconds
```

**Model Location**: `/Users/politom/.pyenv/versions/3.11.8/lib/python3.11/site-packages/slanet_plus_table/models/inference.pdmodel`

**Dependencies**:
- PaddlePaddle framework
- No RapidOCR required (works standalone)

### Evaluation Methodology

**IoU Threshold**: 0.5

**Box Format Conversion**:
- SLANetPlus output: 8-point [x1,y1,x2,y2,x3,y3,x4,y4]
- Ground truth: 4-point [x, y, width, height]
- Conversion: Extract min/max x,y from 8 points → convert to xywh

**Metrics**:
- **True Positive (TP)**: Prediction matches GT with IoU ≥ 0.5
- **False Positive (FP)**: Prediction with no GT match
- **False Negative (FN)**: GT with no prediction match
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: 2 * (Precision * Recall) / (Precision + Recall)

---

## Visualizations

Generated comparison images with:
- **Green boxes**: Ground truth cells
- **Blue boxes**: Matched predictions (IoU ≥ 0.5)
- **Red boxes**: Unmatched predictions (false positives)

Saved to: `src/submodules/rapidtable-onnx/temp/*_slanetplus_comparison.png`

---

## Failed Approaches

### 1. RapidTable
- **Issue**: Requires RapidOCR despite documentation claiming optional
- **Error**: `'NoneType' object is not callable`
- **Attempts**: Multiple OCR format workarounds, all failed
- **Conclusion**: Library has dependency bug

### 2. TableFormer ONNX Conversion
- **Issue**: Autoregressive loop causes numerical drift in ONNX
- **Errors**:
  - BBox classes max diff: 1.217
  - BBox coords max diff: 0.717
- **Conclusion**: Removed all TableFormer files due to conversion difficulties

---

## Next Steps

### Option 1: PaddlePaddle to ONNX Conversion (Recommended)

**Advantages**:
- Enables .NET integration via ONNX Runtime
- No PaddlePaddle C# bindings required
- Proven conversion path exists

**Steps**:
1. Research Paddle2ONNX conversion tool
2. Convert inference.pdmodel to ONNX
3. Validate conversion accuracy
4. Create .NET SDK with ONNX Runtime
5. Port preprocessing/postprocessing to C#

**Challenges**:
- PaddlePaddle → ONNX conversion may introduce numerical differences
- Need to validate cell detection accuracy remains 66.5%

### Option 2: PaddlePaddle C# Bindings

**Advantages**:
- Use model directly without conversion
- Guaranteed same accuracy as Python

**Disadvantages**:
- PaddlePaddle C# bindings are less mature
- Additional dependency management complexity
- Limited community support

### Option 3: Continue Searching for ONNX-Native Models

**Considerations**:
- Table Transformer wrong task (structure not cells)
- RapidTable broken (OCR dependency bug)
- Other options to explore?

---

## Recommendation

**Proceed with Option 1: PaddlePaddle to ONNX Conversion**

**Rationale**:
1. SLANetPlus has proven 66.5% F1 score
2. Perfect cell count matching demonstrates reliability
3. ONNX Runtime provides best .NET integration
4. Model size is small (0.3MB) - conversion should be manageable
5. Paddle2ONNX is official and well-maintained

**Risk Mitigation**:
- Validate converted ONNX model against Python baseline on all 3 GT images
- Accept conversion only if F1 score remains ≥ 0.65
- If conversion fails, fall back to Option 2 (PaddlePaddle C# bindings)

---

## Files Generated

### Evaluation Scripts
- `src/submodules/rapidtable-onnx/tools/slanetplus_evaluation.py` - Main evaluation script
- `src/submodules/table-transformer-onnx/tools/table_transformer_onnx_inference.py` - Table Transformer test script
- `src/submodules/table-transformer-onnx/tools/evaluate_with_ground_truth.py` - Table Transformer GT comparison

### Results
- `src/submodules/rapidtable-onnx/temp/slanetplus_evaluation_results.json` - Detailed per-image metrics
- `/tmp/table_transformer_results/table_transformer_results.json` - Table Transformer results

### Visualizations
- `src/submodules/rapidtable-onnx/temp/HAL.2009.page_77.pdf_125051_slanetplus_comparison.png`
- `src/submodules/rapidtable-onnx/temp/HAL.2015.page_43.pdf_125177_slanetplus_comparison.png`
- `src/submodules/rapidtable-onnx/temp/HAL.2017.page_79.pdf_125247_slanetplus_comparison.png`

---

## Conclusion

SLANetPlus is the clear winner for table cell detection with 66.5% F1 score and perfect cell count matching. The next step is to convert the PaddlePaddle model to ONNX format for .NET integration.

**Status**: ✅ Evaluation complete - Ready for model conversion phase
