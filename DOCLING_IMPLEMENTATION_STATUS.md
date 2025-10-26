# DoclingNet - Implementation Status & Architecture

**Document Version:** 1.0
**Last Updated:** 2025-10-26
**Objective:** Convert images to Markdown by analyzing layout, extracting text with OCR, and parsing table structures

---

## Executive Summary

DoclingNet is a .NET 9.0 port of the Python Docling library. The project successfully integrates three core AI/ML components for document understanding:

1. **LayoutSdk** - Layout detection using ONNX models (Heron)
2. **EasyOcrNet** - OCR text extraction (CRAFT detection + CRNN recognition)
3. **TableFormerTorchSharpSdk** - Table structure recognition using TorchSharp

Current status: **Core components functional**, **Pipeline architecture in place**, **Markdown export needs integration**.

---

## Project Structure

```
src/
├── Docling.Cli/                    # ✅ FUNCTIONAL - Command-line interface
├── Docling.Core/                   # ✅ COMPLETE - Document model & primitives
├── Docling.Models/                 # ✅ COMPLETE - Data models
├── Docling.Export/                 # ⚠️  PARTIAL - Markdown serialization implemented
├── Docling.Pipelines/              # ⚠️  PARTIAL - Pipeline stages defined, not integrated
├── Docling.Tooling/                # ℹ️  UTILITY - Build/tooling support
└── submodules/
    ├── ds4sd-docling-layout-heron-onnx/    # ✅ Layout detection
    ├── ds4sd-docling-tableformer-onnx/     # ✅ Table structure (fixed SkiaSharp 3.x bug)
    └── easyocrnet/                          # ✅ OCR engine (updated to .NET 9.0)
```

---

## Implemented Components

### 1. ✅ Docling.Cli (Fully Functional)

**Purpose:** Command-line tool for testing and demonstrating all three AI components.

**Location:** [src/Docling.Cli/Program.cs](src/Docling.Cli/Program.cs) (681 lines)

**Features Implemented:**
- ✅ Layout detection with LayoutSdk (ONNX runtime)
- ✅ OCR extraction with EasyOcrNet (auto-download models from GitHub)
- ✅ Table extraction and structure recognition with TableFormer
- ✅ JSON output for OCR results and table structures
- ✅ Image cropping and region extraction
- ✅ Comprehensive logging with Serilog

**Key Methods:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `Main` | 33-159 | Entry point, orchestrates full pipeline |
| `ProcessLayoutItemsWithEasyOcrAsync` | 217-339 | OCR extraction on non-table elements |
| `CreateEasyOcrAsync` | 341-356 | Initialize OCR engine with auto-download |
| `ResolveEasyOcrModelDirectory` | 358-407 | Model path resolution with fallbacks |
| `ProcessTablesWithTableFormer` | 434-577 | TableFormer initialization and processing |
| `ProcessSingleTable` | 579-655 | Single table structure inference |
| `ExtractAndSaveRegion` | 161-174 | Crop and save image regions |
| `CropRegion` | 176-215 | Safe bitmap cropping with fallback |
| `FindModelPath` | 657-681 | Locate Heron ONNX model |

**Current Output:**
```
Input:  dataset/2305.03393v1-pg9-img.png
Output:
  - dataset/ocr_results/2305.03393v1-pg9-img_ocr_results.json (25KB JSON)
  - dataset/extracted_tables/2305.03393v1-pg9-img_table_1.png (extracted table image)
  - dataset/extracted_tables/table_structure_results.json (32KB, 46 cells detected)
  - dataset/2305.03393v1-pg9-img_ocr_text.txt (2.1KB formatted text)
```

**Performance:**
- Layout detection: ~463ms (12 elements detected)
- OCR per element: ~1-2s (depends on text density)
- TableFormer inference: ~1s (46 cells)

---

### 2. ✅ Docling.Core (Complete)

**Purpose:** Core document model representing structured output.

**Location:** [src/Docling.Core/](src/Docling.Core/)

**Key Classes:**

#### Document Model
- **`DoclingDocument`** - Main document container with items, pages, properties
- **`DocItem`** - Abstract base for all document items (paragraphs, tables, captions, etc.)
- **`DocItemKind`** - Enum: Text, Title, Section-header, Paragraph, Caption, Table, Picture, etc.
- **`DocItemProvenance`** - Tracks origin page/bbox for each item

#### Specialized Items
- **`ParagraphItem`** - Text paragraphs
- **`TableItem`** - Table with cells, rows, columns
- **`TableCellItem`** - Individual table cell with row/col span
- **`CaptionItem`** - Figure/table captions
- **`PictureItem`** - Images with references

#### Geometry
- **`BoundingBox`** - Rectangle (l, t, r, b)
- **`Point2D`** - 2D coordinates
- **`Polygon`** - Polygonal regions
- **`PageSize`** - Page dimensions

#### Primitives
- **`PageReference`** - Page metadata (number, dimensions, image path)
- **`ImageRef`** - Image references with URI/mimetype

**Design Principles:**
- Immutable where possible
- Builder pattern for document construction (`DoclingDocumentBuilder`)
- Deterministic ordering of items
- Query helpers for filtering by kind/page

---

### 3. ✅ Docling.Export (Partially Complete)

**Purpose:** Export DoclingDocument to various formats (primarily Markdown).

**Location:** [src/Docling.Export/](src/Docling.Export/)

**Implemented:**

#### Serialization
- **`MarkdownDocSerializer`** - Main markdown serializer
  - Converts DoclingDocument to markdown string
  - Handles tables (markdown table format)
  - Supports image exports
  - Configurable options (image alt text, table format)

#### Imaging
- **`ImageExportArtifact`** - Represents exported images with paths
- **`ImageExportOptions`** - Configuration for image export

**Key Features:**
- ✅ Markdown table generation from TableItem
- ✅ Image reference handling
- ✅ Hierarchical document structure preservation
- ✅ Configurable serialization options

**Example Output Structure:**
```markdown
# Section Header

Paragraph text...

| Col1 | Col2 |
|------|------|
| A    | B    |

![Caption](path/to/image.png)
```

---

### 4. ⚠️ Docling.Pipelines (Architecture Complete, Integration Pending)

**Purpose:** Orchestrate document processing as a series of pipeline stages.

**Location:** [src/Docling.Pipelines/](src/Docling.Pipelines/)

**Architecture:**

#### Core Pipeline
- **`ConvertPipeline`** - Sequential stage executor with logging
- **`PipelineContext`** - Shared state across stages (key-value store)
- **`IPipelineStage`** - Interface for all stages
- **`IPipelineObserver`** - Observer pattern for telemetry
- **`ConvertPipelineBuilder`** - Fluent builder for pipeline construction

#### Implemented Stages

| Stage | Status | Purpose |
|-------|--------|---------|
| **PagePreprocessingStage** | ⚠️ Defined | Load and preprocess images |
| **LayoutAnalysisStage** | ⚠️ Defined | Run LayoutSdk detection |
| **OcrStage** | ⚠️ Defined | Run EasyOcrNet extraction |
| **TableStructureInferenceStage** | ⚠️ Defined | Run TableFormer on tables |
| **PageAssemblyStage** | ⚠️ Defined | Assemble DoclingDocument from results |
| **MarkdownSerializationStage** | ✅ Implemented | Convert to markdown |
| **ImageExportStage** | ⚠️ Defined | Export extracted images |

**Pipeline Context Keys:**
```csharp
public static class PipelineContextKeys
{
    public const string Document = "doc:document";
    public const string LayoutResult = "layout:result";
    public const string OcrResult = "ocr:result";
    public const string TableResults = "table:results";
    public const string ImageExports = "export:images";
    public const string MarkdownSerializationResult = "md:result";
    public const string MarkdownSerializationCompleted = "md:completed";
}
```

**Options Classes:**
- `PipelineOptions` - Top-level config
- `LayoutOptions` - LayoutSdk configuration
- `TableStructureOptions` - TableFormer configuration
- `AcceleratorOptions` - Hardware acceleration (CPU/CUDA)
- `PreprocessingOptions` - Image preprocessing config
- `PictureDescriptionOptions` - Image captioning (future)

---

### 5. ✅ Submodules Integration

#### LayoutSdk
- **Source:** ds4sd-docling-layout-heron-onnx
- **Language:** C# with ONNX runtime
- **Model:** Heron (ResNet backbone + object detection head)
- **Status:** ✅ Fully functional
- **Features:**
  - Detects 12+ layout element types
  - Returns bounding boxes with confidence scores
  - ~460ms inference on typical page
- **Integration:** Direct usage in CLI, ready for pipeline

#### EasyOcrNet
- **Source:** easyocrnet (updated fork)
- **Language:** C# with ONNX runtime
- **Models:**
  - CRAFT (Character Region Awareness For Text detection)
  - CRNN (Convolutional Recurrent Neural Network recognition)
- **Status:** ✅ Fully functional
- **Updates Applied:**
  - ✅ Migrated to .NET 9.0
  - ✅ Updated dependencies (ML.OnnxRuntime 1.20.0, SkiaSharp 3.119.1)
  - ✅ Auto-download from GitHub releases
  - ✅ Fixed model directory resolution
- **Integration:** Working in CLI, ready for pipeline

#### TableFormerTorchSharpSdk
- **Source:** ds4sd-docling-tableformer-onnx
- **Language:** C# with TorchSharp
- **Model:** TableFormer (Transformer-based table structure recognition)
- **Status:** ✅ Fully functional (bug fixed)
- **Bug Fixed:** SkiaSharp 3.x premultiplied alpha issue
  - Changed from `bitmap.CopyTo()` to canvas rendering
  - File: [TableFormerDecodedPageImage.cs:79-96](src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerTorchSharpSdk/PagePreparation/TableFormerDecodedPageImage.cs#L79-L96)
- **Features:**
  - Detects table cells, rows, columns
  - Handles cell spans (row_span, col_span)
  - Identifies headers and row sections
  - Returns structured JSON with bounding boxes
- **Integration:** Working in CLI, ready for pipeline

---

## What's Working Right Now

### End-to-End Flow (CLI)
```
Image (PNG/JPEG)
    ↓
[LayoutSdk] → Detects 12 elements (tables, text, headers, captions)
    ↓
[EasyOcrNet] → Extracts text from non-table regions → JSON output
    ↓
[Image Cropping] → Extracts table images → PNG files
    ↓
[TableFormer] → Analyzes table structure → 46 cells detected
    ↓
[Output Files] → JSON results, extracted images, formatted text
```

### Verified Test Case
**Input:** `dataset/2305.03393v1-pg9-img.png` (academic paper page)

**Output:**
1. ✅ Layout detection: 12 elements (1 table, 6 text blocks, 2 section headers, 2 page headers, 1 caption)
2. ✅ OCR extraction: All text regions processed with confidence scores
3. ✅ Table extraction: 1 table cropped to separate image
4. ✅ Table structure: 46 cells identified with row/column indices and spans
5. ✅ Text file: Formatted output ordered by Y position (top-to-bottom)

**Performance:** Total ~5-6 seconds for full page processing

---

## What's Missing for Complete Pipeline

### 🔴 Critical (Required for Image→Markdown)

1. **Pipeline Integration**
   - **Status:** Architecture exists, stages defined, but NOT connected to actual implementations
   - **Work Required:**
     - Wire LayoutSdk into `LayoutAnalysisStage`
     - Wire EasyOcrNet into `OcrStage`
     - Wire TableFormer into `TableStructureInferenceStage`
     - Implement `PageAssemblyStage` to build `DoclingDocument` from raw results
     - Test end-to-end pipeline execution
   - **Estimated Effort:** 2-3 days
   - **Files to Modify:**
     - [src/Docling.Pipelines/Layout/LayoutAnalysisStage.cs](src/Docling.Pipelines/Layout/LayoutAnalysisStage.cs)
     - [src/Docling.Pipelines/Ocr/OcrStage.cs](src/Docling.Pipelines/Ocr/OcrStage.cs)
     - [src/Docling.Pipelines/Tables/TableStructureInferenceStage.cs](src/Docling.Pipelines/Tables/TableStructureInferenceStage.cs)
     - [src/Docling.Pipelines/Assembly/PageAssemblyStage.cs](src/Docling.Pipelines/Assembly/PageAssemblyStage.cs)

2. **DoclingDocument Assembly**
   - **Status:** `DoclingDocumentBuilder` exists but conversion logic missing
   - **Work Required:**
     - Map LayoutSdk boxes → DocItem instances
     - Map OCR results → ParagraphItem.Text
     - Map TableFormer results → TableItem with cells
     - Preserve spatial ordering (reading order)
     - Handle overlapping regions
   - **Estimated Effort:** 1-2 days
   - **Files to Modify:**
     - [src/Docling.Core/Documents/DoclingDocumentBuilder.cs](src/Docling.Core/Documents/DoclingDocumentBuilder.cs)
     - New file: `src/Docling.Pipelines/Assembly/ResultAssembler.cs`

3. **Markdown Export Integration**
   - **Status:** `MarkdownDocSerializer` implemented but not tested with real DoclingDocument
   - **Work Required:**
     - Test serializer with populated DoclingDocument
     - Verify table rendering
     - Handle edge cases (empty cells, long text, special chars)
     - Add table caption support
   - **Estimated Effort:** 1 day
   - **Files to Verify:**
     - [src/Docling.Export/Serialization/MarkdownDocSerializer.cs](src/Docling.Export/Serialization/MarkdownDocSerializer.cs)

### 🟡 Important (Enhances Quality)

4. **Reading Order Detection**
   - **Status:** Not implemented
   - **Work Required:**
     - Sort layout elements by position (top-to-bottom, left-to-right)
     - Handle multi-column layouts
     - Respect section hierarchy
   - **Estimated Effort:** 1-2 days

5. **Table-Text Integration**
   - **Status:** Tables and text processed separately
   - **Work Required:**
     - Perform OCR on table cells (use EasyOcr on cropped cells)
     - Integrate cell text into TableItem
     - Handle table captions (link Caption items to TableItem)
   - **Estimated Effort:** 1-2 days

6. **Error Handling & Recovery**
   - **Status:** Basic exceptions, no graceful degradation
   - **Work Required:**
     - Handle partial OCR failures
     - Continue on single-stage failure
     - Provide diagnostics for low-confidence regions
   - **Estimated Effort:** 1 day

### 🟢 Nice-to-Have (Future Enhancements)

7. **Configuration Management**
   - Unified config file (JSON/YAML)
   - Per-component tuning (confidence thresholds, model variants)
   - Environment variable overrides

8. **Batch Processing**
   - Multi-page document support
   - Parallel processing of pages
   - Progress reporting

9. **Visual Debugging**
   - Overlay bounding boxes on images
   - Confidence heatmaps
   - Debug output for each stage

10. **Additional Export Formats**
    - HTML
    - DOCX
    - JSON (structured)

---

## Technical Debt & Known Issues

### Fixed Issues ✅
1. ✅ **SkiaSharp 3.x Alpha Channel Bug** (TableFormer)
   - **Problem:** `bitmap.CopyTo()` didn't respect AlphaType.Unpremul
   - **Solution:** Use canvas rendering instead
   - **Status:** Fixed in [TableFormerDecodedPageImage.cs:79-96](src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerTorchSharpSdk/PagePreparation/TableFormerDecodedPageImage.cs#L79-L96)

2. ✅ **EasyOcrNet Model Auto-Download**
   - **Problem:** DirectoryNotFoundException when models missing
   - **Solution:** Create default directory, allow download to proceed
   - **Status:** Fixed in [Program.cs:358-407](src/Docling.Cli/Program.cs#L358-L407)

3. ✅ **GitHub Release Tag 404**
   - **Problem:** Hardcoded wrong release tag (`v1.0.0` → 404)
   - **Solution:** Use default from `GithubReleaseOptions` (`v2025.09.19`)
   - **Status:** Fixed in [Program.cs:353](src/Docling.Cli/Program.cs#L353)

### Remaining Issues ⚠️

1. **SkiaSharp API Deprecations**
   - **Issue:** 153 warnings for obsolete `SKFilterQuality`
   - **Impact:** Builds succeed but should migrate to `SKSamplingOptions`
   - **Priority:** Low (cosmetic)
   - **Location:** EasyOcrNet codebase

2. **TableFormer Accessibility**
   - **Issue:** Had to make internal classes public manually
   - **Impact:** Requires forking submodule
   - **Priority:** Medium (upstream contribution needed)
   - **Files:** TableFormerNeuralModel.cs, TableFormerSequenceDecoder.cs, TableFormerNeuralPrediction.cs

3. **OCR Quality**
   - **Issue:** Some OCR errors visible in output (e.g., "ZabNet" → should be "TabNet")
   - **Impact:** Accuracy concerns for production use
   - **Priority:** Medium (may need model fine-tuning or post-processing)

---

## Recommended Implementation Plan

### Phase 1: Core Pipeline (1 week)
**Goal:** Get Image → DoclingDocument working

1. **Day 1-2:** Wire stages to actual implementations
   - Connect LayoutSdk to LayoutAnalysisStage
   - Connect EasyOcrNet to OcrStage
   - Connect TableFormer to TableStructureInferenceStage

2. **Day 3-4:** Implement PageAssemblyStage
   - Map layout boxes to DocItems
   - Integrate OCR text
   - Handle table structures

3. **Day 5:** Testing & Debugging
   - Test with multiple images
   - Verify DoclingDocument structure
   - Fix edge cases

### Phase 2: Markdown Export (2-3 days)
**Goal:** Get DoclingDocument → Markdown working

1. **Day 1:** Test MarkdownDocSerializer
   - Create test DoclingDocuments
   - Verify table rendering
   - Test image references

2. **Day 2:** Integration & Polish
   - End-to-end pipeline test
   - Compare output to Python Docling
   - Refine formatting

3. **Day 3:** Reading Order & Quality
   - Implement basic reading order
   - Handle edge cases
   - Document limitations

### Phase 3: Polish & Extend (1 week)
**Goal:** Production-ready features

1. **Table-Text Integration:** OCR on table cells
2. **Error Handling:** Graceful degradation
3. **Configuration:** Unified config system
4. **Documentation:** Usage guide, API docs
5. **Tests:** Unit and integration tests

---

## Key Decisions & Design Notes

### 1. Why Pipeline Architecture?
- **Modularity:** Each stage is independent, testable
- **Observability:** Built-in telemetry via IPipelineObserver
- **Extensibility:** Easy to add new stages (e.g., image captioning)
- **Matching Python Docling:** Similar architecture to upstream

### 2. Why TorchSharp for TableFormer?
- **Model Compatibility:** TableFormer weights are PyTorch-native
- **Performance:** TorchSharp provides native bindings
- **Flexibility:** Easy to add custom layers if needed
- **Trade-off:** Larger dependency than pure ONNX

### 3. Why Separate CLI from Pipeline?
- **CLI:** Quick testing, debugging, demonstration
- **Pipeline:** Reusable library for integration
- **Allows:** CLI can evolve independently, pipeline stays stable

### 4. Document Model Design
- **Immutable Where Possible:** Thread-safe, predictable
- **Builder Pattern:** Fluent API for construction
- **Provenance Tracking:** Every item knows its source page/bbox
- **Query Helpers:** Easy filtering by kind, page, etc.

---

## Comparison to Python Docling

| Feature | Python Docling | DoclingNet | Status |
|---------|---------------|------------|--------|
| Layout Detection | ✅ Heron (PyTorch) | ✅ Heron (ONNX) | ✅ Equivalent |
| OCR | ✅ EasyOCR | ✅ EasyOcrNet | ✅ Equivalent |
| Table Structure | ✅ TableFormer | ✅ TableFormer (TorchSharp) | ✅ Equivalent |
| Markdown Export | ✅ | ✅ | ✅ Implemented |
| HTML Export | ✅ | ❌ | ⏳ Future |
| DOCX Export | ✅ | ❌ | ⏳ Future |
| Pipeline Architecture | ✅ | ✅ | ⚠️ Partial |
| Reading Order | ✅ | ❌ | ⏳ Planned |
| Multi-page Support | ✅ | ❌ | ⏳ Future |
| Image Captioning | ✅ (optional) | ❌ | ⏳ Future |

---

## Dependencies

### Core
- **.NET 9.0** (latest LTS)
- **SkiaSharp 3.119.1** (image processing)
- **Serilog** (logging)
- **System.Text.Json** (serialization)

### AI/ML
- **Microsoft.ML.OnnxRuntime 1.20.0** (LayoutSdk, EasyOcrNet)
- **TorchSharp** (TableFormer)

### Testing
- **xUnit** (unit tests)
- **FluentAssertions** (test assertions)

---

## Performance Metrics (Reference)

**Test Machine:** M1 Mac, 16GB RAM
**Test Image:** Academic paper page (1280x1810px)

| Component | Time | Notes |
|-----------|------|-------|
| Layout Detection | 463ms | 12 elements, ONNX CPU runtime |
| OCR (per element) | 1-2s | Depends on text density |
| Table Extraction | <100ms | Image cropping only |
| TableFormer | 1s | 46 cells, TorchSharp CPU |
| **Total** | **~5-6s** | Full page processing |

**Memory:** Peak ~800MB (includes model loading)

---

## Code Statistics

```
Total Projects: 20
Core Projects: 6 (Docling.*)
Submodule Projects: 14

Lines of Code:
  Docling.Cli:       681 lines (Program.cs)
  Docling.Core:      ~2,500 lines (16 files)
  Docling.Export:    ~800 lines (8 files)
  Docling.Pipelines: ~1,200 lines (27 files)
  Docling.Models:    ~400 lines (5 files)

Total Docling Code: ~5,500 lines (excluding submodules)
```

---

## Next Steps Summary

### To Complete Image→Markdown Pipeline:

1. ✅ **Submodules Working** (Done)
   - LayoutSdk
   - EasyOcrNet
   - TableFormer

2. 🔴 **Wire Pipeline Stages** (Critical)
   - Connect implementations to stages
   - Implement PageAssemblyStage
   - Test end-to-end flow

3. 🔴 **Test Markdown Export** (Critical)
   - Verify with real DoclingDocument
   - Fix table rendering issues
   - Handle edge cases

4. 🟡 **Add Reading Order** (Important)
   - Sort elements spatially
   - Handle multi-column layouts

5. 🟡 **Table Cell OCR** (Important)
   - Run EasyOcr on cells
   - Integrate into TableItem

6. 🟢 **Polish & Extend** (Nice-to-have)
   - Configuration management
   - Batch processing
   - Visual debugging

---

## Contact & References

**Project:** DoclingNet
**Original:** [Docling (Python)](https://github.com/DS4SD/docling)
**License:** MIT (assumed, verify with upstream)

**Key Files:**
- CLI Implementation: [src/Docling.Cli/Program.cs](src/Docling.Cli/Program.cs)
- Pipeline Core: [src/Docling.Pipelines/Internal/ConvertPipeline.cs](src/Docling.Pipelines/Internal/ConvertPipeline.cs)
- Document Model: [src/Docling.Core/Documents/DoclingDocument.cs](src/Docling.Core/Documents/DoclingDocument.cs)
- Markdown Export: [src/Docling.Export/Serialization/MarkdownDocSerializer.cs](src/Docling.Export/Serialization/MarkdownDocSerializer.cs)

**Test Data:**
- [dataset/2305.03393v1-pg9-img.png](dataset/2305.03393v1-pg9-img.png) - Academic paper page
- [dataset/ocr_results/](dataset/ocr_results/) - OCR JSON outputs
- [dataset/extracted_tables/](dataset/extracted_tables/) - Table images and structure JSON

---

**End of Document**
