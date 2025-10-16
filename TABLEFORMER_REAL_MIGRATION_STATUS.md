# TableFormer Migration - Real Status Analysis

**Analysis Date**: 2025-10-16
**Branch**: `codex/update-tableformer-implementation-with-onnx-models`
**Analyzer**: Claude Code (Verification Task)

## Executive Summary

⚠️ **CRITICAL DISCREPANCY FOUND**: The migration status document claims Phase 1-5 are complete, but the actual implementation reveals a **significant gap** between documentation and reality.

### Key Findings

| Component | Documented Status | Actual Status | Gap Level |
|-----------|------------------|---------------|-----------|
| **ONNX Component Models** | ✅ Generated (4 files each) | ❌ **MISSING** - Only monolithic stubs exist | **CRITICAL** |
| **TableFormerOnnxBackend** | ✅ Full implementation | ⚠️ Simple stub with no real logic | **HIGH** |
| **Component Architecture** | ✅ 4-component pipeline | ❌ Single-model fallback | **CRITICAL** |
| **Config/WordMap Loading** | ✅ Implemented | ❓ Code exists but untested | **MEDIUM** |
| **Autoregressive Loop** | ✅ Complete with OTSL | ❓ Code exists but not wired | **HIGH** |
| **OTSL Parser** | ✅ Tested | ❓ Code exists but not integrated | **HIGH** |

## Detailed Analysis

### 1. ONNX Models Status

#### Expected (per documentation):
```
models/
├── tableformer_fast_encoder.onnx (11 MB)
├── tableformer_fast_tag_transformer_encoder.onnx (64 MB)
├── tableformer_fast_tag_transformer_decoder_step.onnx (26 MB)
├── tableformer_fast_bbox_decoder.onnx (38 MB)
├── tableformer_fast_config.json (7 KB)
├── tableformer_fast_wordmap.json (5 KB)
└── (same for accurate variant)
```

#### Reality:
```
src/submodules/ds4sd-docling-tableformer-onnx/models/
├── tableformer_accurate.onnx (1.0 MB) ← Single monolithic file
├── tableformer_accurate.yaml (2 KB)
├── tableformer_fast.onnx (132 bytes) ← LFS POINTER, not actual model
└── tableformer_fast.yaml (2 KB)
```

**Status**: ❌ **Component models DO NOT EXIST**

### 2. Backend Implementation

#### File: `TableFormerOnnxBackend.cs` (185 lines)

**Current Implementation**:
- Loads a single ONNX model (monolithic)
- Extracts 10 "features" (stub implementation)
- Simple post-processing: splits output in half for rows/columns
- **NO autoregressive loop**
- **NO OTSL parsing**
- **NO component-based architecture**

**Code Evidence** [TableFormerOnnxBackend.cs:52-64](src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerSdk/Backends/TableFormerOnnxBackend.cs#L52-L64):
```csharp
public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
{
    var features = TableFormerOnnxFeatureExtractor.ExtractFeatures(image, _featureLength);
    var inputTensor = new DenseTensor<long>(features, new[] { 1, features.Length });

    using var results = _session.Run(new[] { input });
    var output = results[0].AsTensor<float>().ToArray();

    return PostProcess(output, image.Width, image.Height);
}
```

This is a **placeholder stub**, not the documented 4-component architecture.

### 3. Component Classes Exist But Are NOT Used

#### File: `TableFormerComponents.cs` (740 lines)

This file contains COMPLETE implementations of:
- ✅ `TableFormerOnnxComponents` - 4-session ONNX wrapper
- ✅ `TableFormerAutoregressive` - Full autoregressive loop
- ✅ `OtslParser` - Complete OTSL parsing logic
- ✅ Helper classes for config, wordmap, normalization

**BUT**: These classes are **NOT WIRED** to the main backend!

The current `TableFormerOnnxBackend.cs` completely ignores this file and uses a simplified stub implementation instead.

### 4. Integration Status

#### Main Service: `TableFormerTableStructureService.cs`

The service correctly:
- ✅ Loads models via `TableFormerVariantModelPaths.FromDirectory()`
- ✅ Searches for models in correct submodule path
- ✅ Has metrics tracking and performance monitoring
- ✅ Handles overlay generation

**BUT**:
- ❌ Expects monolithic `.onnx` files (not component files)
- ❌ Uses stub backend that returns dummy cells
- ❌ No actual TableFormer inference happening

**Code Evidence** [TableFormerTableStructureService.cs:517-518](src/Docling.Models/Tables/TableFormerTableStructureService.cs#L517-L518):
```csharp
var submoduleModelsPath = Path.GetFullPath(
    Path.Combine(baseDirectory, "src", "submodules",
        "ds4sd-docling-tableformer-onnx", "models"));
```

This path exists, but contains wrong model format.

### 5. Model Path Configuration

#### Expected by `FromDirectory()`:
```csharp
var modelFile = Path.Combine(directory, $"{variantPrefix}.onnx");
var metadataFile = Path.Combine(directory, $"{variantPrefix}.yaml");
```

Looks for:
- `tableformer_fast.onnx` (monolithic)
- `tableformer_accurate.onnx` (monolithic)

**NOT** the component files documented in Phase 1-5!

### 6. Test Coverage

Build currently fails due to missing NuGet package source:
```
error NU1301: The local source '/Users/politom/Documents/Workspace/personal/doclingnet/packages/custom' doesn't exist.
```

Cannot verify test execution, but test files exist:
- `tests/Docling.Tests/Tables/TableFormerTableStructureServiceTests.cs`
- `tests/Docling.Tests/Tables/TableFormerQualityMetricsTests.cs`
- `src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerSdk.Tests/TableFormerSdkTests.cs`

## Root Cause Analysis

### Why This Happened

1. **Parallel Development Tracks**:
   - Component-based implementation was developed in `TableFormerComponents.cs`
   - Simple stub remained in `TableFormerOnnxBackend.cs`
   - Integration never completed

2. **Missing Model Generation**:
   - Python conversion script (`convert_tableformer_components_to_onnx.py`) exists
   - **BUT**: Component models were never actually generated and committed
   - Only monolithic models exist (possibly from old implementation)

3. **Documentation Ahead of Implementation**:
   - `TABLEFORMER_MIGRATION_STATUS.md` was updated optimistically
   - Actual integration work was not completed
   - Tests may not have been run end-to-end

## What Actually Works

✅ **Working Components**:
1. Model loading infrastructure (paths, config)
2. Service integration points (metrics, batch processing)
3. Image preprocessing (SKBitmap handling)
4. Coordinate transformation logic
5. Component classes (in TableFormerComponents.cs - isolated)

❌ **Not Working**:
1. Actual TableFormer inference (using stub)
2. Component-based ONNX execution
3. Autoregressive decoding
4. OTSL sequence generation and parsing
5. BBox extraction from model outputs

## Required Work to Complete Migration

### Phase A: Generate Component Models (Est: 2-3 hours)

1. Run Python conversion script to generate 4 ONNX files per variant
2. Verify file sizes match documentation
3. Generate `config.json` and `wordmap.json` files
4. Commit models with Git LFS
5. Update model paths to point to components

### Phase B: Wire Component Backend (Est: 4-6 hours)

1. Replace `TableFormerOnnxBackend.cs` with component-based version
2. Integrate `TableFormerOnnxComponents` class
3. Wire autoregressive loop
4. Connect OTSL parser
5. Add proper error handling

### Phase C: Integration Testing (Est: 3-4 hours)

1. Fix NuGet package source issue
2. Run existing test suite
3. Add integration tests with real images
4. Validate against Python golden outputs
5. Performance benchmarking

### Phase D: Validation & Documentation (Est: 2-3 hours)

1. Update migration status with REAL progress
2. Document actual architecture
3. Add troubleshooting guide
4. Performance tuning
5. Final QA

**Total Remaining Effort**: 11-16 hours

## Recommendations

### Immediate Actions

1. **Stop claiming Phase 1-5 are complete** - They are NOT
2. **Generate the component ONNX models** - This is blocking everything
3. **Replace stub backend** - Use the TableFormerComponents.cs code
4. **Run end-to-end test** - Verify actual TableFormer inference works

### Long-term

1. Add CI/CD checks to prevent documentation drift
2. Require integration tests to pass before marking phases complete
3. Add model file size validation in tests
4. Document stub vs. real implementation clearly

## Conclusion

The migration has **significant foundational work completed** (config, paths, component classes), but the **critical integration and model generation** are missing.

**Current Phase Status**:
- Phase 1-3 (Python side): ⚠️ **PARTIALLY COMPLETE** (code exists, models don't)
- Phase 4-5 (.NET implementation): ⚠️ **CODE READY, NOT INTEGRATED**
- Phase 6-8 (Testing, docs, validation): ❌ **BLOCKED**

**Actual Completion**: ~30-40% (not the documented 25% or implied 62%)

The good news: All the hard architectural work is done. The remaining work is mechanical (model generation + wiring).

---

**Next Steps**: See recommendations above. Priority is generating the component ONNX models.
