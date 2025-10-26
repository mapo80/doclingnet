# DoclingNet

A complete .NET port of [Docling](https://github.com/DS4SD/docling) for converting documents (images, PDFs) to structured Markdown using state-of-the-art AI/ML models.

## Features

- **Layout Detection** - Automatic document layout analysis using Heron ONNX model
- **OCR Text Extraction** - Powered by EasyOCR with CRAFT detection and CRNN recognition
- **Table Structure Recognition** - Advanced table analysis with TableFormer (TorchSharp)
- **Markdown Export** - Clean, structured Markdown output
- **Unified SDK** - Simple, single-entry-point API
- **CLI Tool** - Command-line interface for batch processing
- **Performance** - Optimized pipeline with intelligent image reuse and pre-allocation

## Quick Start

### Prerequisites

- .NET 9.0 SDK
- Operating System: Windows, macOS, or Linux

### Installation

```bash
git clone https://github.com/yourusername/doclingnet.git
cd doclingnet
git submodule update --init --recursive
dotnet build
```

### Basic Usage (SDK)

```csharp
using DoclingNetSdk;

// 1. Create configuration (auto-detects model paths)
var config = DoclingConfiguration.CreateDefault();

// 2. Initialize converter
using var converter = new DoclingConverter(config);

// 3. Convert image to markdown
var result = await converter.ConvertImageAsync("document.png");

// 4. Use the results
Console.WriteLine(result.Markdown);
File.WriteAllText("output.md", result.Markdown);

// Access statistics
Console.WriteLine($"Layout elements: {result.LayoutElementCount}");
Console.WriteLine($"OCR elements: {result.OcrElementCount}");
Console.WriteLine($"Tables: {result.TableCount}");
```

### CLI Usage

```bash
# Convert a single document
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png

# With custom output and options
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png \
  --output result.md \
  --tableformer Accurate \
  --language en \
  --verbose
```

## Architecture

DoclingNet implements a multi-stage AI/ML pipeline:

```
┌─────────────────┐
│  Input Image    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  1. Layout      │  Heron ONNX Model
│     Detection   │  Identifies regions (title, paragraph, table, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Full-Page   │  EasyOCR (CRAFT + CRNN)
│     OCR         │  Extracts all text with bounding boxes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Table       │  TableFormer (TorchSharp)
│     Structure   │  Analyzes table structure (cells, spans)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Document    │  DoclingDocument builder
│     Building    │  Assembles structured document
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Markdown    │  MarkdownDocSerializer
│     Export      │  Exports to Markdown format
└─────────────────┘
```

### Key Components

- **DoclingConverter** - Main SDK entry point
- **LayoutSdk** - Document layout detection (Heron ONNX)
- **EasyOcrNet** - OCR text extraction (CRAFT detection + CRNN recognition)
- **TableFormerTorchSharpSdk** - Table structure recognition (Transformer-based)
- **Docling.Core** - Document models (DoclingDocument, DocItem, etc.)
- **Docling.Export** - Markdown serialization

## Configuration

### DoclingConfiguration

```csharp
public sealed class DoclingConfiguration
{
    // Directory for model cache and temporary files
    // Default: ./artifacts
    public string ArtifactsPath { get; set; } = "./artifacts";

    // OCR language code (e.g., "en", "it", "fr", "de")
    // Default: "en"
    public string OcrLanguage { get; set; } = "en";

    // Enable/disable table structure recognition
    // Default: true
    public bool EnableTableRecognition { get; set; } = true;

    // Enable/disable OCR text extraction
    // Default: true
    public bool EnableOcr { get; set; } = true;

    // TableFormer model variant: Fast, Base, or Accurate
    // Default: Accurate (best quality, only +4.8% slower than Fast)
    public TableFormerVariant TableFormerVariant { get; set; } = TableFormerVariant.Accurate;

    // Factory method with default settings
    public static DoclingConfiguration CreateDefault();
}
```

### TableFormer Variants

| Variant | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| **Fast** | ~300ms/table | Good | High-volume processing |
| **Base** | ~500ms/table | Better | Balanced use cases |
| **Accurate** | ~1s/table | Best | Quality-critical applications |

Based on benchmarks, **Accurate** is the recommended default as it provides significantly better table recognition (correct headers, accurate cell detection) with only a 4.8% performance penalty over Fast mode.

## Advanced Usage

### Custom Configuration

```csharp
var config = new DoclingConfiguration
{
    ArtifactsPath = "/path/to/models",
    OcrLanguage = "it",  // Italian
    EnableTableRecognition = true,
    EnableOcr = true,
    TableFormerVariant = TableFormerVariant.Accurate
};

// Optional: Pass custom logger
using var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Debug);
});
var logger = loggerFactory.CreateLogger<DoclingConverter>();

using var converter = new DoclingConverter(config, logger);
var result = await converter.ConvertImageAsync("document.png");
```

### Batch Processing

```csharp
var imagePaths = Directory.GetFiles("documents", "*.png");
var results = await converter.ConvertImagesAsync(imagePaths);

foreach (var (path, result) in results)
{
    var mdPath = Path.ChangeExtension(path, ".md");
    await File.WriteAllTextAsync(mdPath, result.Markdown);
    Console.WriteLine($"Processed: {path} -> {mdPath}");
}
```

### Working with Structured Documents

```csharp
var result = await converter.ConvertImageAsync("document.png");

// Access structured document
var document = result.Document;

// Iterate through all items
foreach (var item in document.Items)
{
    Console.WriteLine($"{item.Kind}: {item.Label}");

    if (item is TableItem table)
    {
        Console.WriteLine($"  Rows: {table.RowCount}, Cols: {table.ColumnCount}");
        foreach (var cell in table.Cells)
        {
            Console.WriteLine($"  Cell [{cell.RowIndex},{cell.ColumnIndex}]: {cell.Text}");
        }
    }
    else if (item is ParagraphItem paragraph)
    {
        Console.WriteLine($"  Text: {paragraph.Text}");
    }
}
```

### Cancellation Support

```csharp
using var cts = new CancellationTokenSource();
cts.CancelAfter(TimeSpan.FromMinutes(5));

try
{
    var result = await converter.ConvertImageAsync("document.png", cts.Token);
}
catch (OperationCanceledException)
{
    Console.WriteLine("Conversion cancelled");
}
```

## CLI Reference

### Command-Line Options

```bash
docling-cli <image-path> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--output <path>` | Output markdown file path | Same as input with .md extension |
| `--artifacts <path>` | Artifacts directory for models | `./artifacts` |
| `--language <code>` | OCR language code (en, it, fr, de, etc.) | `en` |
| `--no-ocr` | Disable OCR text extraction | OCR enabled |
| `--no-tables` | Disable table recognition | Tables enabled |
| `--tableformer <variant>` | TableFormer variant: Fast, Base, Accurate | `Accurate` |
| `--verbose` | Enable verbose logging | Info level |

### CLI Examples

```bash
# Simple conversion
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png

# Custom output location
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png --output results/doc.md

# Italian OCR with verbose logging
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png \
  --language it \
  --verbose

# Fast mode for high-volume processing
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png \
  --tableformer Fast

# Layout-only (no OCR or tables)
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png \
  --no-ocr \
  --no-tables
```

## Models

All AI/ML models are automatically downloaded on first use and cached in the artifacts directory.

### Layout Detection Model

- **Model**: Heron ONNX
- **Size**: ~150MB
- **Location**: Auto-detected from submodule `src/submodules/ds4sd-docling-layout-heron-onnx/models/`
- **Purpose**: Identifies document regions (title, paragraph, table, figure, etc.)

### OCR Models

- **Detection Model**: CRAFT (Character Region Awareness For Text)
  - Size: ~79MB
  - Downloaded from GitHub releases
- **Recognition Model**: CRNN (Convolutional Recurrent Neural Network)
  - Size: ~14MB per language
  - Downloaded from GitHub releases
- **Purpose**: Text extraction from document regions

### TableFormer Models

- **Fast Variant**: ~30MB
- **Base Variant**: ~60MB
- **Accurate Variant**: ~120MB
- **Location**: Downloaded from Hugging Face on first use
- **Purpose**: Table structure recognition (rows, columns, cell spans)

## Performance

### Benchmark Results

Typical processing time for a single A4 page with tables (AMD EPYC 7763 @ 2.45GHz, single core):

| Operation | Time | Notes |
|-----------|------|-------|
| Layout Detection | ~400-500ms | ONNX CPU runtime |
| Full-Page OCR | ~2-3s | Depends on text density |
| Table Recognition (Accurate) | ~1s per table | Includes structure + text matching |
| **Total** | **~11-12s** | For typical document with 2 tables |

### Performance Comparison

| Mode | Time | Quality | Recommendation |
|------|------|---------|----------------|
| Fast | ~10.95s | Good | High-volume batch processing |
| Accurate | ~11.48s (+4.8%) | Excellent | Default - best quality/speed ratio |

### Performance Notes

- **First run** takes longer due to model downloads (~250MB total)
- **GPU acceleration** can reduce inference time significantly
- **90% of time** is spent in ML inference (optimizations have diminishing returns)
- **Comparison with Python Docling**: DoclingNet is ~38% slower due to .NET ML library overhead (ONNX Runtime .NET vs PyTorch native bindings)

### Optimization Tips

- Use `Fast` variant for high-volume processing where quality is less critical
- Disable OCR (`EnableOcr = false`) if you only need layout structure
- Disable tables (`EnableTableRecognition = false`) if document contains no tables
- Process multiple documents in parallel using `Task.WhenAll`
- Consider GPU acceleration for production deployments

## Project Structure

```
DoclingNet/
├── src/
│   ├── DoclingNetSdk/              # Main SDK entry point
│   │   ├── DoclingConverter.cs     # Primary conversion class
│   │   ├── DoclingConfiguration.cs # Configuration
│   │   └── DoclingConversionResult.cs
│   │
│   ├── Docling.Core/               # Core document models
│   │   ├── DoclingDocument.cs      # Document representation
│   │   ├── DocItem.cs              # Base item class
│   │   ├── ParagraphItem.cs        # Text paragraphs
│   │   ├── TableItem.cs            # Tables
│   │   └── ...
│   │
│   ├── Docling.Export/             # Export functionality
│   │   └── Serialization/
│   │       └── MarkdownDocSerializer.cs
│   │
│   ├── Docling.Backends/           # Input backends
│   │   └── ImageBackend.cs         # Image processing
│   │
│   ├── Docling.Cli/                # Command-line tool
│   │   └── Program.cs
│   │
│   └── submodules/                 # AI/ML libraries
│       ├── ds4sd-docling-layout-heron-onnx/
│       ├── easyocrnet/
│       └── ds4sd-docling-tableformer-onnx/
│
├── dataset/                        # Test datasets
│   ├── groundtruth/                # Ground truth samples
│   └── golden/                     # Golden outputs
│
└── docs/                           # Documentation
```

## API Reference

### DoclingConverter

Main converter class for document processing.

**Constructor:**
```csharp
public DoclingConverter(
    DoclingConfiguration config,
    ILogger? logger = null)
```

**Methods:**
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

### DoclingConversionResult

Result object containing converted document and statistics.

**Properties:**
```csharp
public DoclingDocument Document { get; }       // Structured document
public string Markdown { get; }                // Markdown export
public int LayoutElementCount { get; }         // Layout elements detected
public int OcrElementCount { get; }            // OCR elements processed
public int TableCount { get; }                 // Tables found
public int TotalItems { get; }                 // Total document items
```

### DoclingDocument

Structured document representation.

**Properties:**
```csharp
public IReadOnlyList<DocItem> Items { get; }   // All document items
public IReadOnlyList<PageReference> Pages { get; }  // Page references
public string Id { get; }                      // Document ID
public string SourceId { get; }                // Source file ID
```

### DocItem Types

- **ParagraphItem** - Text paragraphs
  - `string Text` - Paragraph text content
  - `string Label` - Item label (e.g., "paragraph", "title")

- **TableItem** - Tables with structure
  - `int RowCount` - Number of rows
  - `int ColumnCount` - Number of columns
  - `IReadOnlyList<TableCellItem> Cells` - Table cells

- **TableCellItem** - Individual table cells
  - `string Text` - Cell text content
  - `int RowIndex` - Row position (0-based)
  - `int ColumnIndex` - Column position (0-based)
  - `int RowSpan` - Row span (default: 1)
  - `int ColumnSpan` - Column span (default: 1)

- **PictureItem** - Images and figures
- **CaptionItem** - Image/table captions

## Troubleshooting

### Model Not Found

**Error**: "Layout model not found at path: ..."

**Solution**: Ensure submodules are initialized:
```bash
git submodule update --init --recursive
```

The layout model should exist at:
```
src/submodules/ds4sd-docling-layout-heron-onnx/models/heron-converted.onnx
```

### Out of Memory

**Error**: OutOfMemoryException during processing

**Solutions**:
- Disable table recognition: `EnableTableRecognition = false`
- Use Fast variant: `TableFormerVariant = TableFormerVariant.Fast`
- Process pages individually instead of batch
- Increase available memory

### Slow Processing

**Issue**: Conversion takes too long

**Solutions**:
- Use Fast variant for TableFormer
- Disable OCR if only layout is needed
- Consider GPU acceleration
- Process multiple documents in parallel

### Poor Table Quality

**Issue**: Table structure not recognized correctly

**Solutions**:
- Use Accurate variant (default): `TableFormerVariant = TableFormerVariant.Accurate`
- Ensure image quality is good (high resolution, clear text)
- Check that tables have visible borders
- Verify OCR is enabled (required for table text extraction)

### OCR Not Detecting Text

**Issue**: No text extracted from regions

**Solutions**:
- Verify `EnableOcr = true` in configuration
- Check language setting matches document language
- Ensure image resolution is sufficient (min 150 DPI recommended)
- Verify text is not too small or blurry

## Development

### Building from Source

```bash
# Clone repository with submodules
git clone --recursive https://github.com/yourusername/doclingnet.git
cd doclingnet

# Build entire solution
dotnet build DoclingNet.sln

# Run tests
dotnet test

# Build specific project
dotnet build src/DoclingNetSdk/DoclingNetSdk.csproj
```

### Running Tests

```bash
# Run all tests
dotnet test

# Run with verbose output
dotnet test --logger "console;verbosity=detailed"

# Run specific test project
dotnet test tests/DoclingNetSdk.Tests/
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Related Projects

- [Docling (Python)](https://github.com/DS4SD/docling) - Original Python implementation
- [docling-layout-heron-onnx](https://github.com/DS4SD/docling-layout-heron-onnx) - Layout detection model
- [docling-tableformer-onnx](https://github.com/DS4SD/docling-tableformer-onnx) - Table structure recognition
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Original Python OCR library

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

DoclingNet is built on top of excellent open-source projects:

- **Docling** - Original Python implementation by DS4SD
- **Heron** - Document layout detection model
- **TableFormer** - Table structure recognition transformer
- **EasyOCR** - OCR engine (ported to .NET as EasyOcrNet)
- **TorchSharp** - .NET bindings for PyTorch
- **ONNX Runtime** - Cross-platform ML inference

Special thanks to the DS4SD team at IBM Research for the original Docling implementation and pre-trained models.

## Citation

If you use DoclingNet in your research or project, please cite both DoclingNet and the original Docling:

```bibtex
@software{doclingnet2024,
  title = {DoclingNet: .NET Port of Docling Document Conversion},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/doclingnet}
}

@software{docling2024,
  title = {Docling Technical Report},
  author = {Deep Search Team},
  year = {2024},
  url = {https://github.com/DS4SD/docling}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/doclingnet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/doclingnet/discussions)
- **Documentation**: [docs/](docs/)

## Changelog

See [docs/progress.md](docs/progress.md) for detailed development progress and version history.
