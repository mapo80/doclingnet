# DoclingNet: Image→Markdown Implementation Plan

**Document Version:** 1.0
**Last Updated:** 2025-10-26
**Objective:** Implement complete image-to-markdown conversion workflow matching Python Docling functionality

---

## Python Docling Workflow Analysis

### High-Level Flow (Images)

```
Input: Image File (PNG, JPEG, TIFF, etc.)
    ↓
[DocumentConverter]
    ├── Format Detection → IMAGE
    ├── Pipeline Selection → StandardPdfPipeline
    └── Backend Selection → DoclingParseV4DocumentBackend
    ↓
[StandardPdfPipeline.execute()]
    │
    ├─ 1. PagePreprocessingModel
    │     └─ Load image, scale if needed
    │
    ├─ 2. BaseOcrModel (EasyOCR/Tesseract/RapidOCR)
    │     └─ Extract text cells with bounding boxes
    │
    ├─ 3. LayoutModel
    │     └─ Detect layout elements (text, table, figure, etc.)
    │     └─ Creates "clusters" (groups of related cells)
    │
    ├─ 4. TableStructureModel
    │     └─ Analyze table structure (if tables detected)
    │     └─ Extract rows, columns, cells with spans
    │
    └─ 5. PageAssembleModel
          └─ Combine OCR + Layout + TableStructure
          └─ Create AssembledUnit with:
                • elements (all doc items)
                • headers (page headers/footers)
                • body (main content)
    ↓
[ReadingOrderModel]
    └─ Sort elements in reading order
    └─ Build DoclingDocument with hierarchical structure
    ↓
[DoclingDocument.export_to_markdown()]
    └─ Traverse document tree
    └─ Render each element as markdown
    └─ Return markdown string
```

---

## Python Docling Key Components

### 1. StandardPdfPipeline (applies to images too!)

**Location:** `/docling/pipeline/standard_pdf_pipeline.py`

**Pipeline Stages:**
```python
self.build_pipe = [
    PagePreprocessingModel(options=...),      # Load & preprocess
    ocr_model,                                 # OCR extraction
    LayoutModel(artifacts_path=...),           # Layout detection
    TableStructureModel(enabled=True, ...),    # Table structure
    PageAssembleModel(options=...),            # Assembly
]
```

**Key Insight:** Images use the SAME pipeline as PDFs (StandardPdfPipeline), but the backend treats them as single-page documents.

---

### 2. PageAssembleModel

**Location:** `/docling/models/page_assemble_model.py`

**What it does:**
1. Iterates over layout clusters (from LayoutModel)
2. For each cluster:
   - **TEXT_ELEM_LABELS** → Creates `TextElement`
     - Combines OCR text from cells
     - Sanitizes text (hyphenation, normalization)
   - **TABLE_LABELS** → Gets `Table` from TableStructureModel
   - **FIGURE_LABEL** → Creates `FigureElement`
   - **CONTAINER_LABELS** → Creates `ContainerElement`
3. Separates elements into:
   - `headers` (page headers/footers)
   - `body` (main content)
   - `elements` (all items)

**Critical Code (lines 84-150):**
```python
for cluster in page.predictions.layout.clusters:
    if cluster.label in LayoutModel.TEXT_ELEM_LABELS:
        # Extract text from OCR cells
        textlines = [cell.text for cell in cluster.cells if len(cell.text.strip()) > 0]
        text = self.sanitize_text(textlines)
        text_el = TextElement(label=cluster.label, text=text, ...)

    elif cluster.label in LayoutModel.TABLE_LABELS:
        # Get table structure from predictions
        tbl = page.predictions.tablestructure.table_map.get(cluster.id)

    elif cluster.label == LayoutModel.FIGURE_LABEL:
        fig = FigureElement(label=cluster.label, ...)
```

---

### 3. ReadingOrderModel

**Location:** `/docling/models/readingorder_model.py`

**What it does:**
1. Converts assembled elements → reading order elements
2. Sorts by position (top-to-bottom, left-to-right)
3. Builds `DoclingDocument` with proper hierarchy
4. Adds provenance (page number, bounding box)

**Key Features:**
- Handles multi-column layouts
- Processes list items (numbered, bulleted)
- Creates parent-child relationships
- Maintains references between items

---

### 4. DoclingDocument.export_to_markdown()

**Location:** `docling-core` library (external)

**What it does:**
1. Traverses document tree in reading order
2. For each item:
   - **Title/Section** → Markdown headers (`#`, `##`, `###`)
   - **Paragraph** → Plain text with spacing
   - **List** → Markdown list (`-` or `1.`)
   - **Table** → Markdown table or HTML `<table>`
   - **Code** → Code blocks (` ``` `)
   - **Formula** → LaTeX or image
   - **Picture** → Markdown image (`![alt](path)`)
   - **Caption** → Italicized text
3. Adds spacing between sections
4. Returns final markdown string

---

## .NET Implementation Plan

### Phase 1: Core Pipeline (Week 1)

#### 1.1 Implement StandardImagePipeline

**New Class:** `src/Docling.Pipelines/StandardImagePipeline.cs`

```csharp
public class StandardImagePipeline
{
    private readonly PipelineOptions _options;

    public async Task<DoclingDocument> ExecuteAsync(string imagePath)
    {
        var context = new PipelineContext();
        context.Set("image_path", imagePath);

        // Stage 1: Load image
        await _preprocessingStage.ExecuteAsync(context);

        // Stage 2: OCR extraction
        await _ocrStage.ExecuteAsync(context);

        // Stage 3: Layout detection
        await _layoutStage.ExecuteAsync(context);

        // Stage 4: Table structure
        await _tableStructureStage.ExecuteAsync(context);

        // Stage 5: Assembly
        await _assemblyStage.ExecuteAsync(context);

        // Stage 6: Reading order
        await _readingOrderStage.ExecuteAsync(context);

        return context.GetRequired<DoclingDocument>("document");
    }
}
```

**Integration Points:**
- `PagePreprocessingStage` → Load SKBitmap
- `OcrStage` → Call EasyOcrNet
- `LayoutAnalysisStage` → Call LayoutSdk
- `TableStructureInferenceStage` → Call TableFormer
- `PageAssemblyStage` → Build DoclingDocument
- `ReadingOrderStage` → Sort elements

---

#### 1.2 Wire Existing Components

**Modify:** `src/Docling.Pipelines/Ocr/OcrStage.cs`

```csharp
public class OcrStage : IPipelineStage
{
    private readonly OcrEngine _ocrEngine;

    public async Task ExecuteAsync(PipelineContext context, CancellationToken ct)
    {
        var image = context.GetRequired<SKBitmap>("preprocessed_image");

        // Run OCR on full image
        var ocrResults = await _ocrEngine.ProcessImageAsync(image).ConfigureAwait(false);

        // Store OCR cells
        var cells = ocrResults.Select(r => new OcrCell
        {
            Text = r.Text,
            BoundingBox = new BoundingBox(r.BoundingBox.MinX, r.BoundingBox.MinY,
                                          r.BoundingBox.MaxX, r.BoundingBox.MaxY),
            Confidence = r.Confidence
        }).ToList();

        context.Set("ocr_cells", cells);
    }
}
```

**Modify:** `src/Docling.Pipelines/Layout/LayoutAnalysisStage.cs`

```csharp
public class LayoutAnalysisStage : IPipelineStage
{
    private readonly LayoutSdk _layoutSdk;

    public Task ExecuteAsync(PipelineContext context, CancellationToken ct)
    {
        var imagePath = context.GetRequired<string>("image_path");

        // Run layout detection
        var layoutResult = _layoutSdk.Process(imagePath, overlay: false, LayoutRuntime.Onnx);

        // Convert to clusters
        var clusters = layoutResult.Boxes.Select((box, idx) => new LayoutCluster
        {
            Id = $"cluster_{idx}",
            Label = MapLabelToDocItemKind(box.Label),
            BoundingBox = new BoundingBox(box.X, box.Y, box.X + box.Width, box.Y + box.Height),
            Confidence = box.Confidence,
            Cells = new List<OcrCell>() // Will be populated in assembly
        }).ToList();

        context.Set("layout_clusters", clusters);
        return Task.CompletedTask;
    }
}
```

---

#### 1.3 Implement PageAssemblyStage

**New Class:** `src/Docling.Pipelines/Assembly/PageAssemblyStage.cs`

```csharp
public class PageAssemblyStage : IPipelineStage
{
    public async Task ExecuteAsync(PipelineContext context, CancellationToken ct)
    {
        var ocrCells = context.GetRequired<List<OcrCell>>("ocr_cells");
        var layoutClusters = context.GetRequired<List<LayoutCluster>>("layout_clusters");
        var tableResults = context.TryGet<Dictionary<string, TableStructure>>("table_structures",
                                                                               out var tables)
            ? tables
            : new Dictionary<string, TableStructure>();

        // Step 1: Assign OCR cells to layout clusters
        AssignCellsToClusters(ocrCells, layoutClusters);

        // Step 2: Build document items
        var docBuilder = new DoclingDocumentBuilder("image", new[] { new PageReference(1, ...) });

        foreach (var cluster in layoutClusters)
        {
            switch (cluster.Label)
            {
                case DocItemKind.Text:
                case DocItemKind.Paragraph:
                case DocItemKind.SectionHeader:
                case DocItemKind.Caption:
                    // Combine text from OCR cells
                    var text = SanitizeText(cluster.Cells.Select(c => c.Text));
                    docBuilder.AddParagraph(text, cluster.BoundingBox, pageNumber: 1);
                    break;

                case DocItemKind.Table:
                    // Get table structure
                    if (tables.TryGetValue(cluster.Id, out var tableStructure))
                    {
                        docBuilder.AddTable(tableStructure, cluster.BoundingBox, pageNumber: 1);
                    }
                    break;

                case DocItemKind.Picture:
                    docBuilder.AddPicture(cluster.BoundingBox, pageNumber: 1);
                    break;
            }
        }

        var document = docBuilder.Build();
        context.Set("document", document);
    }

    private void AssignCellsToClusters(List<OcrCell> cells, List<LayoutCluster> clusters)
    {
        foreach (var cell in cells)
        {
            // Find cluster with max overlap
            var bestCluster = clusters
                .Select(c => (cluster: c, overlap: c.BoundingBox.IntersectionArea(cell.BoundingBox)))
                .OrderByDescending(x => x.overlap)
                .FirstOrDefault();

            if (bestCluster.overlap > 0.5 * cell.BoundingBox.Area)
            {
                bestCluster.cluster.Cells.Add(cell);
            }
        }
    }

    private string SanitizeText(IEnumerable<string> lines)
    {
        // Python: lines 34-65 in page_assemble_model.py
        var linesList = lines.ToList();
        if (linesList.Count <= 1)
            return string.Join(" ", linesList);

        for (int i = 1; i < linesList.Count; i++)
        {
            var prevLine = linesList[i - 1];
            if (prevLine.EndsWith("-") && IsAlphanumericWord(prevLine, linesList[i]))
            {
                linesList[i - 1] = prevLine[..^1]; // Remove hyphen
            }
            else
            {
                linesList[i - 1] += " ";
            }
        }

        var sanitized = string.Concat(linesList)
            .Replace("⁄", "/")
            .Replace("'", "'")
            .Replace("'", "'")
            .Replace(""", "\"")
            .Replace(""", "\"")
            .Replace("•", "·");

        return sanitized.Trim();
    }
}
```

---

### Phase 2: Reading Order (Week 1)

#### 2.1 Implement ReadingOrderStage

**New Class:** `src/Docling.Pipelines/ReadingOrder/ReadingOrderStage.cs`

```csharp
public class ReadingOrderStage : IPipelineStage
{
    public Task ExecuteAsync(PipelineContext context, CancellationToken ct)
    {
        var document = context.GetRequired<DoclingDocument>("document");

        // Sort items by position (top-to-bottom, left-to-right)
        var sortedItems = document.Items
            .OrderBy(item => GetReadingOrder(item))
            .ToList();

        // Rebuild document with sorted items
        var reorderedDoc = RebuildDocumentWithOrder(document, sortedItems);

        context.Set("document", reorderedDoc);
        return Task.CompletedTask;
    }

    private double GetReadingOrder(DocItem item)
    {
        // Simple heuristic: Y * 10000 + X
        // This sorts top-to-bottom, left-to-right
        var bbox = item.Provenance?.BoundingBox;
        if (bbox == null) return 0;

        return bbox.Top * 10000 + bbox.Left;
    }
}
```

**Advanced Reading Order (Future):**
- Multi-column detection
- Hierarchy preservation (parent-child)
- List item grouping

---

### Phase 3: Markdown Export (Week 1)

#### 3.1 Enhance MarkdownDocSerializer

**Modify:** `src/Docling.Export/Serialization/MarkdownDocSerializer.cs`

```csharp
public class MarkdownDocSerializer
{
    private readonly MarkdownSerializerOptions _options;
    private readonly StringBuilder _output;

    public string Serialize(DoclingDocument document, IReadOnlyList<ImageExportArtifact>? imageExports = null)
    {
        _output.Clear();

        foreach (var item in document.Items)
        {
            switch (item.Kind)
            {
                case DocItemKind.Title:
                    WriteTitle(item as TitleItem);
                    break;

                case DocItemKind.SectionHeader:
                    WriteSectionHeader(item as SectionHeaderItem);
                    break;

                case DocItemKind.Paragraph:
                case DocItemKind.Text:
                    WriteParagraph(item as ParagraphItem);
                    break;

                case DocItemKind.Table:
                    WriteTable(item as TableItem);
                    break;

                case DocItemKind.Picture:
                    WritePicture(item as PictureItem, imageExports);
                    break;

                case DocItemKind.Caption:
                    WriteCaption(item as CaptionItem);
                    break;

                case DocItemKind.Code:
                    WriteCode(item as CodeItem);
                    break;

                case DocItemKind.ListItem:
                    WriteListItem(item as ListItem);
                    break;
            }
        }

        return _output.ToString();
    }

    private void WriteSectionHeader(SectionHeaderItem item)
    {
        // Determine level (1-6)
        var level = DetermineHeaderLevel(item);
        var prefix = new string('#', level);

        _output.AppendLine();
        _output.AppendLine($"{prefix} {item.Text}");
        _output.AppendLine();
    }

    private void WriteParagraph(ParagraphItem item)
    {
        _output.AppendLine(item.Text);
        _output.AppendLine();
    }

    private void WriteTable(TableItem table)
    {
        if (_options.TableFormat == TableFormat.Markdown)
        {
            WriteMarkdownTable(table);
        }
        else
        {
            WriteHtmlTable(table);
        }
    }

    private void WriteMarkdownTable(TableItem table)
    {
        // Get table dimensions
        var rows = table.Rows;
        var cols = table.Columns;

        // Build table grid
        var grid = new string[rows, cols];
        foreach (var cell in table.Cells)
        {
            grid[cell.RowIndex, cell.ColumnIndex] = cell.Text ?? "";
        }

        // Write header row
        for (int c = 0; c < cols; c++)
        {
            _output.Append("| ");
            _output.Append(grid[0, c].Replace("|", "\\|"));
            _output.Append(" ");
        }
        _output.AppendLine("|");

        // Write separator
        for (int c = 0; c < cols; c++)
        {
            _output.Append("|---");
        }
        _output.AppendLine("|");

        // Write body rows
        for (int r = 1; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                _output.Append("| ");
                _output.Append(grid[r, c].Replace("|", "\\|"));
                _output.Append(" ");
            }
            _output.AppendLine("|");
        }

        _output.AppendLine();
    }

    private void WritePicture(PictureItem picture, IReadOnlyList<ImageExportArtifact>? exports)
    {
        var altText = picture.Caption ?? "Image";
        var imagePath = FindImagePath(picture, exports);

        _output.AppendLine($"![{altText}]({imagePath})");
        _output.AppendLine();
    }
}
```

---

### Phase 4: Integration & Testing (Days 6-7)

#### 4.1 Create End-to-End Test

**New File:** `src/Docling.Cli/Commands/ConvertImageCommand.cs`

```csharp
public class ConvertImageCommand
{
    public async Task<string> ExecuteAsync(string imagePath)
    {
        // Initialize pipeline
        var options = new PipelineOptions
        {
            DoOcr = true,
            DoTableStructure = true,
            LayoutOptions = new LayoutOptions { /* ... */ },
            OcrOptions = new OcrOptions { /* ... */ }
        };

        var pipeline = new StandardImagePipeline(options);

        // Execute pipeline
        var document = await pipeline.ExecuteAsync(imagePath);

        // Export to markdown
        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions());
        var markdown = serializer.Serialize(document);

        return markdown;
    }
}
```

**CLI Usage:**
```bash
dotnet run --project src/Docling.Cli -- convert dataset/2305.03393v1-pg9-img.png --output markdown
```

**Expected Output:**
```markdown
# 5.1 Hyper Parameter Optimization

We have chosen the Pub TabNet data set to perform HPO, since it includes highly diverse set of tables...

## Table 1

| Model | TED | mAP |
|-------|-----|-----|
| ...   | ... | ... |

# 5.2 Quantitative Results

We picked the model parameter configuration...
```

---

## Implementation Checklist

### Week 1: Core Pipeline

- [ ] **Day 1: Infrastructure**
  - [ ] Create `StandardImagePipeline` class
  - [ ] Define `LayoutCluster` model
  - [ ] Define `OcrCell` model
  - [ ] Add context keys to `PipelineContextKeys`

- [ ] **Day 2: Wire OCR & Layout**
  - [ ] Implement `OcrStage.ExecuteAsync()` with EasyOcrNet
  - [ ] Implement `LayoutAnalysisStage.ExecuteAsync()` with LayoutSdk
  - [ ] Test OCR + Layout integration

- [ ] **Day 3: Wire Table Structure**
  - [ ] Implement `TableStructureInferenceStage.ExecuteAsync()` with TableFormer
  - [ ] Create `TableStructure` mapping from TableFormer results
  - [ ] Test table extraction

- [ ] **Day 4-5: Page Assembly**
  - [ ] Implement `AssignCellsToClusters()` method
  - [ ] Implement `SanitizeText()` method (match Python logic)
  - [ ] Implement cluster → DocItem conversion
  - [ ] Handle all DocItemKind types
  - [ ] Test with real image

- [ ] **Day 6: Reading Order**
  - [ ] Implement `ReadingOrderStage`
  - [ ] Basic spatial sorting (Y then X)
  - [ ] Test ordering accuracy

- [ ] **Day 7: Markdown Export**
  - [ ] Enhance `MarkdownDocSerializer`
  - [ ] Implement table rendering (markdown + HTML)
  - [ ] Test all element types
  - [ ] Compare output to Python Docling

---

## Data Models Needed

### 1. LayoutCluster
```csharp
public class LayoutCluster
{
    public string Id { get; set; }
    public DocItemKind Label { get; set; }
    public BoundingBox BoundingBox { get; set; }
    public float Confidence { get; set; }
    public List<OcrCell> Cells { get; set; } = new();
}
```

### 2. OcrCell
```csharp
public class OcrCell
{
    public string Text { get; set; }
    public BoundingBox BoundingBox { get; set; }
    public float Confidence { get; set; }
}
```

### 3. TableStructure
```csharp
public class TableStructure
{
    public int Rows { get; set; }
    public int Columns { get; set; }
    public List<TableCellData> Cells { get; set; }
}

public class TableCellData
{
    public int RowIndex { get; set; }
    public int ColumnIndex { get; set; }
    public int RowSpan { get; set; }
    public int ColumnSpan { get; set; }
    public string Text { get; set; }
    public BoundingBox BoundingBox { get; set; }
    public bool IsHeader { get; set; }
}
```

---

## Key Differences from Python

### 1. OCR Integration

**Python:** Multiple OCR backends via factory pattern
- EasyOCR (default)
- Tesseract
- RapidOCR
- OCR-Mac (macOS Vision)

**\.NET:** Focus on EasyOcrNet
- Already integrated and working
- Can add Tesseract later if needed

### 2. Layout Model

**Python:** Uses `docling_ibm_models.layoutmodel`
- Wrapper around ONNX
- Same Heron model

**\.NET:** Uses LayoutSdk
- Direct ONNX usage
- **✅ Equivalent functionality**

### 3. Table Structure

**Python:** Uses TableFormer from `docling_ibm_models`
**\.NET:** Uses TableFormerTorchSharpSdk
- **✅ Same model, TorchSharp implementation**
- Bug fixed (SkiaSharp alpha channel)

### 4. Reading Order

**Python:** Uses `ReadingOrderPredictor` from `docling_ibm_models`
- Rule-based algorithm
- Handles complex layouts

**\.NET:** **⚠️ Need to implement**
- Start with simple spatial sorting
- Can enhance later with rules

---

## Critical Implementation Notes

### 1. Cell-to-Cluster Assignment

**Python Logic (implicit):**
- OCR runs on full image → produces cells with bboxes
- Layout detection runs → produces clusters with bboxes
- Assembly: For each cluster, find OCR cells whose bbox overlaps > 50%

**\.NET Implementation:**
```csharp
private void AssignCellsToClusters(List<OcrCell> cells, List<LayoutCluster> clusters)
{
    foreach (var cell in cells)
    {
        var bestMatch = clusters
            .Select(c => new {
                Cluster = c,
                Overlap = CalculateIOU(c.BoundingBox, cell.BoundingBox)
            })
            .OrderByDescending(x => x.Overlap)
            .FirstOrDefault();

        if (bestMatch?.Overlap > 0.5)
        {
            bestMatch.Cluster.Cells.Add(cell);
        }
    }
}

private double CalculateIOU(BoundingBox a, BoundingBox b)
{
    var intersection = a.Intersection(b);
    var union = a.Area + b.Area - intersection.Area;
    return intersection.Area / union;
}
```

### 2. Text Sanitization

**Python Rules (page_assemble_model.py:34-65):**
1. Handle hyphenation: If line ends with `-` and next word is alphanumeric, remove hyphen
2. Add space between lines (unless hyphenated)
3. Normalize special characters:
   - `⁄` → `/`
   - `'` → `'` (smart quote)
   - `"` → `"`
   - `•` → `·`
4. Strip whitespace

**\.NET:** Match exactly to ensure compatibility

### 3. Table Cell OCR

**Python:** OCR runs on full image, cells assigned by bbox overlap

**\.NET:** Two approaches:
1. **Approach A (match Python):** Same as above
2. **Approach B (enhanced):** Run OCR on individual table cells
   - Crop table cells from image
   - Run EasyOcr on each cell
   - Higher accuracy for table content

**Recommendation:** Start with Approach A (match Python), add Approach B as option

---

## Testing Strategy

### Unit Tests

1. **LayoutCluster Tests**
   - Bbox overlap calculation
   - Cell assignment accuracy

2. **Text Sanitization Tests**
   - Hyphenation removal
   - Character normalization
   - Multi-line handling

3. **Markdown Serialization Tests**
   - Table rendering (simple, with spans)
   - Special characters escaping
   - Header levels

### Integration Tests

1. **Single Image Test**
   - Input: `dataset/2305.03393v1-pg9-img.png`
   - Expected: Markdown file matching Python output
   - Compare:
     - Element count
     - Text content
     - Table structure

2. **Multi-Element Test**
   - Image with: title, text, table, figure, caption
   - Verify all elements rendered correctly

3. **Edge Cases**
   - Empty table cells
   - Overlapping bboxes
   - No OCR text found
   - Very long text

---

## Performance Targets

**Python Docling (reference):**
- Layout: ~450ms
- OCR: ~1-2s per element
- TableFormer: ~1s
- **Total: ~5-6s**

**\.NET Target:**
- Should match or beat Python
- Parallel processing where possible
- Memory efficient (dispose resources)

---

## Success Criteria

### Minimal Viable Product (Week 1)

✅ **Functional:**
- Image → markdown conversion works end-to-end
- All layout elements detected
- OCR text extracted and assigned
- Tables rendered in markdown
- Output matches Python Docling structure

✅ **Quality:**
- Text accuracy > 90% (same as Python)
- Layout detection precision > 85%
- Table structure correct (rows, cols, spans)

### Production Ready (Future)

- Multi-page support
- Batch processing
- Error recovery
- Configuration options
- Performance optimization
- Comprehensive tests

---

## Open Questions

1. **Reading Order Algorithm:**
   - Use simple spatial sort or port Python's rule-based predictor?
   - **Decision:** Start simple, enhance if needed

2. **Table Cell OCR:**
   - Run OCR on full image (Python way) or individual cells (better accuracy)?
   - **Decision:** Both! Python way first, cell-based as option

3. **DoclingDocument Builder:**
   - Extend existing `DoclingDocumentBuilder` or create new?
   - **Decision:** Extend existing, add assembly helpers

4. **Image Export:**
   - Do we need to export extracted images (tables, figures)?
   - **Decision:** Not for MVP, add later if needed

---

## Dependencies & Prerequisites

### Existing (✅ Ready)
- LayoutSdk (Heron ONNX)
- EasyOcrNet (updated to .NET 9.0)
- TableFormerTorchSharpSdk (bug fixed)
- Docling.Core (document model)
- Docling.Export (markdown serializer)
- Docling.Pipelines (architecture)

### New (⚠️ Need Implementation)
- StandardImagePipeline
- PageAssemblyStage (complete implementation)
- ReadingOrderStage
- LayoutCluster / OcrCell models
- Enhanced MarkdownDocSerializer

---

## Next Steps

**Immediate (Today):**
1. Review this plan with team
2. Clarify any ambiguities
3. Set up development branch

**Day 1 (Tomorrow):**
1. Create `StandardImagePipeline` skeleton
2. Define data models (LayoutCluster, OcrCell, TableStructure)
3. Add pipeline context keys

**Day 2:**
1. Implement OcrStage
2. Implement LayoutAnalysisStage
3. Test integration

**Continue per checklist...**

---

## References

**Python Docling:**
- Repository: https://github.com/DS4SD/docling
- Cloned locally: `/tmp/docling-python`
- Key files analyzed:
  - `docling/pipeline/standard_pdf_pipeline.py`
  - `docling/models/page_assemble_model.py`
  - `docling/models/layout_model.py`
  - `docling/models/readingorder_model.py`

**\.NET Implementation:**
- Current code: `/Users/politom/Documents/Workspace/personal/doclingnet`
- Status doc: `DOCLING_IMPLEMENTATION_STATUS.md`
- This plan: `DOCLING_IMAGE_TO_MARKDOWN_PLAN.md`

---

**End of Plan**
