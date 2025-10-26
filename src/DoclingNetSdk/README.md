# DoclingNetSdk

**Unified .NET SDK for AI-powered document processing**

Convert images and PDFs to structured documents and Markdown using state-of-the-art AI models:
- 📐 **Layout Detection** - Heron ONNX model for detecting text, tables, figures
- 📝 **OCR** - EasyOCR with CRAFT detection + CRNN recognition
- 📊 **Table Structure** - TableFormer for extracting table structure with rows, columns, and spans

## Features

- ✅ Single unified API (`DoclingConverter`)
- ✅ Image → Markdown conversion in one line
- ✅ Structured document output (`DoclingDocument`)
- ✅ Automatic model downloading
- ✅ Batch processing support
- ✅ Configurable (OCR language, table recognition, etc.)
- ✅ .NET 9.0 with nullable reference types

## Quick Start

### Installation

```bash
# Via NuGet (when published)
dotnet add package DoclingNetSdk

# Or build from source
git clone https://github.com/yourusername/doclingnet.git
cd doclingnet
dotnet build src/DoclingNetSdk/DoclingNetSdk.csproj
```

### Basic Usage

```csharp
using DoclingNetSdk;

// 1. Create configuration (auto-detects model paths)
var config = DoclingConfiguration.CreateDefault();

// 2. Create converter
using var converter = new DoclingConverter(config);

// 3. Convert image to markdown
var result = await converter.ConvertImageAsync("document.png");

// 4. Use the results
Console.WriteLine(result.Markdown);
File.WriteAllText("output.md", result.Markdown);

// Access structured document
Console.WriteLine($"Found {result.TotalItems} items:");
foreach (var item in result.Document.Items)
{
    Console.WriteLine($"  - {item.Kind}: {item.Label}");
}
```

### Advanced Configuration

```csharp
var config = new DoclingConfiguration
{
    ArtifactsPath = "path/to/artifacts",
    OcrLanguage = "en",  // or "it", "fr", "de", etc.
    EnableTableRecognition = true,
    EnableOcr = true,
    TableFormerVariant = TableFormerVariant.Fast  // Fast | Base | Accurate
};

using var converter = new DoclingConverter(config, logger: myLogger);
var result = await converter.ConvertImageAsync("document.png");

// Note: Layout model is automatically detected from standard locations
```

### Batch Processing

```csharp
var imagePaths = Directory.GetFiles("images", "*.png");
var results = await converter.ConvertImagesAsync(imagePaths);

foreach (var (path, result) in results)
{
    var mdPath = Path.ChangeExtension(path, ".md");
    File.WriteAllText(mdPath, result.Markdown);
}
```

## API Reference

### DoclingConverter

Main entry point for document conversion.

#### Constructor

```csharp
public DoclingConverter(
    DoclingConfiguration config,
    ILogger? logger = null)
```

#### Methods

```csharp
// Convert single image
Task<DoclingConversionResult> ConvertImageAsync(
    string imagePath,
    CancellationToken cancellationToken = default)

// Convert multiple images
Task<Dictionary<string, DoclingConversionResult>> ConvertImagesAsync(
    IEnumerable<string> imagePaths,
    CancellationToken cancellationToken = default)
```

### DoclingConfiguration

Configuration options for the converter.

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `ArtifactsPath` | `string` | `./artifacts` | Path for model cache and temp files |
| `OcrLanguage` | `string` | `"en"` | OCR language code |
| `EnableTableRecognition` | `bool` | `true` | Enable table structure extraction |
| `EnableOcr` | `bool` | `true` | Enable OCR text extraction |
| `TableFormerVariant` | `TableFormerVariant` | `Fast` | Model variant (Fast/Base/Accurate) |

**Note:** The layout model path is automatically detected from standard locations. No manual configuration needed.

#### Methods

```csharp
// Create with auto-detection
static DoclingConfiguration CreateDefault()

// Validate configuration
void Validate()
```

### DoclingConversionResult

Result of a conversion operation.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Document` | `DoclingDocument` | Structured document with all elements |
| `Markdown` | `string` | Document exported as Markdown |
| `LayoutElementCount` | `int` | Number of layout elements detected |
| `OcrElementCount` | `int` | Number of OCR elements processed |
| `TableCount` | `int` | Number of tables found |
| `TotalItems` | `int` | Total document items |

### DoclingDocument

Structured document representation (from `Docling.Core`).

#### Properties

```csharp
public IReadOnlyList<DocItem> Items { get; }
public IReadOnlyList<PageReference> Pages { get; }
public string Id { get; }
public string SourceId { get; }
```

#### Item Types

- `ParagraphItem` - Text paragraphs
- `TableItem` - Tables with cells
- `TableCellItem` - Individual table cells
- `PictureItem` - Images/figures
- `CaptionItem` - Captions

## Model Requirements

### Layout Detection Model

Download the Heron ONNX model:

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/doclingnet.git
```

The model should be at:
```
src/submodules/ds4sd-docling-layout-heron-onnx/models/heron-converted.onnx
```

### OCR Models

EasyOCR models are automatically downloaded on first use from GitHub releases:
- `detection.onnx` (CRAFT detector)
- `english_g2_rec.onnx` (CRNN recognizer)

### TableFormer Models

TableFormer weights are automatically downloaded on first use from Hugging Face.

## Performance

Typical performance on a modern CPU (M1/AMD EPYC):

| Operation | Time | Notes |
|-----------|------|-------|
| Layout Detection | ~400-500ms | Per page, ONNX CPU runtime |
| OCR per Element | ~1-2s | Depends on text density |
| Table Structure | ~300ms-1s | Fast/Base/Accurate variant |
| **Total** | **~5-6s** | For typical document page |

## Examples

See the `examples/` directory for complete examples:

- `BasicUsage.cs` - Simple image to markdown
- `BatchProcessing.cs` - Process multiple files
- `AdvancedConfig.cs` - Custom configuration
- `AccessStructure.cs` - Work with DoclingDocument

## Architecture

```
DoclingConverter (SDK Entry Point)
    ↓
├── LayoutSdk → Layout Detection (Heron ONNX)
├── EasyOcrNet → OCR Text Extraction (CRAFT + CRNN)
├── TableFormerTorchSharpSdk → Table Structure (Transformer)
    ↓
DoclingDocument (Structured Output)
    ↓
MarkdownSerializer → Markdown String
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md.

## Support

- **Issues**: https://github.com/yourusername/doclingnet/issues
- **Discussions**: https://github.com/yourusername/doclingnet/discussions

## Credits

Built on top of:
- [Docling](https://github.com/DS4SD/docling) (Python) - Original implementation
- [LayoutSdk](https://github.com/DS4SD/docling-layout-heron-onnx) - Layout detection
- [TableFormer](https://github.com/DS4SD/docling-tableformer-onnx) - Table structure
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine (ported to .NET)
