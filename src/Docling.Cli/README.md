# Docling CLI

Command-line interface for converting documents (images, PDFs) to Markdown using AI-powered processing.

## Features

- Layout detection with Heron model
- OCR text extraction with EasyOCR
- Table structure recognition with TableFormer
- Markdown export

## Installation

Build the project:

```bash
dotnet build src/Docling.Cli/Docling.Cli.csproj
```

## Usage

Basic usage:

```bash
dotnet run --project src/Docling.Cli/Docling.Cli.csproj <image-path>
```

With options:

```bash
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png \
  --output result.md \
  --artifacts ./models \
  --language en \
  --tableformer Accurate \
  --verbose
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output <path>` | Output markdown file path | Same as input with .md extension |
| `--artifacts <path>` | Artifacts directory for models | `./artifacts` |
| `--language <code>` | OCR language code (e.g., en, it, fr) | `en` |
| `--no-ocr` | Disable OCR text extraction | OCR enabled |
| `--no-tables` | Disable table recognition | Tables enabled |
| `--tableformer <variant>` | TableFormer variant: Fast, Base, Accurate | `Fast` |
| `--verbose` | Enable verbose logging | Info level |

## Example

```bash
# Convert a document image
dotnet run --project src/Docling.Cli/Docling.Cli.csproj document.png

# Convert with custom output and verbose logging
dotnet run --project src/Docling.Cli/Docling.Cli.csproj \
  dataset/test.png \
  --output results/test.md \
  --verbose

# Use accurate table recognition
dotnet run --project src/Docling.Cli/Docling.Cli.csproj \
  document.png \
  --tableformer Accurate
```

## Output

The CLI generates:

- A markdown file with the extracted content
- Console output showing:
  - Processing statistics (layout elements, OCR elements, tables)
  - Execution time
  - Output file location

## Performance

Typical processing time for a single A4 page (on CPU):

- Layout detection: ~500ms
- OCR (10 regions): ~2-3s
- Table recognition (2 tables): ~1-2s
- **Total**: ~4-6s per page

First run may take longer due to model downloads (~150MB total).

## Models

Models are automatically downloaded on first use:

- **Heron Layout Model** (~150MB) - Detected automatically
- **EasyOCR Detection Model** (~79MB) - Downloaded to artifacts
- **EasyOCR Recognition Model** (~14MB) - Downloaded to artifacts
- **TableFormer Models** - Downloaded based on variant selection

All models are cached in the artifacts directory.

## Architecture

This CLI uses the [DoclingNetSdk](../DoclingNetSdk/) unified SDK which provides:

- Simple, clean API
- Automatic model management
- Comprehensive logging
- Error handling

## Troubleshooting

### Model not found

Ensure the layout model exists in one of these locations:

1. `src/submodules/ds4sd-docling-layout-heron-onnx/models/heron-converted.onnx`
2. `models/heron-converted.onnx`
3. `[AppDirectory]/models/heron-converted.onnx`

### Out of memory

For large documents or limited RAM:

- Disable table recognition with `--no-tables`
- Use Fast variant instead of Accurate
- Process pages individually

### Slow processing

- Use `--tableformer Fast` for faster table recognition
- Disable OCR with `--no-ocr` if only layout is needed
- Consider running on GPU for faster inference

## License

MIT License
