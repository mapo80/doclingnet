# TableFormer User Guide

## Introduction

Welcome to the TableFormer table structure recognition system! This guide will help you get started with extracting table structures from documents using the advanced Docling TableFormer models.

## Quick Start

### Basic Usage

```csharp
using Docling.Models.Tables;
using Microsoft.Extensions.Logging;

// 1. Configure the service
var options = new TableFormerStructureServiceOptions
{
    Variant = TableFormerModelVariant.Fast,  // or Accurate
    GenerateOverlay = true                   // Enable for debugging
};

// 2. Create the service
var service = new TableFormerTableStructureService(options);

// 3. Prepare your request
var request = new TableStructureRequest
{
    Page = new PageReference(1),  // Page number
    RasterizedImage = imageData,  // Your image bytes
    BoundingBox = tableBounds     // Table location in image
};

// 4. Extract table structure
var tableStructure = await service.InferStructureAsync(request);

// 5. Use the results
foreach (var cell in tableStructure.Cells)
{
    Console.WriteLine($"Cell at ({cell.BoundingBox.Left}, {cell.BoundingBox.Top})");
    Console.WriteLine($"Size: {cell.BoundingBox.Width} x {cell.BoundingBox.Height}");
}
```

### Model Selection

#### Fast Variant (Recommended for most cases)
```csharp
var options = new TableFormerStructureServiceOptions
{
    Variant = TableFormerModelVariant.Fast,
    Runtime = TableFormerRuntime.Auto  // Automatically uses best available provider
};
```

**When to use Fast:**
- ✅ Real-time applications
- ✅ Batch processing
- ✅ Memory-constrained environments
- ✅ Interactive applications

#### Accurate Variant (Best quality)
```csharp
var options = new TableFormerStructureServiceOptions
{
    Variant = TableFormerModelVariant.Accurate,
    Runtime = TableFormerRuntime.Auto
};
```

**When to use Accurate:**
- ✅ Highest accuracy required
- ✅ Complex table structures
- ✅ Production systems
- ✅ Research applications

## Installation and Setup

### Prerequisites

#### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 2GB available (4GB recommended)
- **Storage**: 500MB for models and dependencies
- **GPU**: Optional, NVIDIA GPU with CUDA 11.8+ for acceleration

#### Dependencies
The TableFormer system requires the following packages:

```xml
<!-- Core dependencies -->
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.16.0" />
<PackageReference Include="SkiaSharp" Version="2.88.6" />
<PackageReference Include="Microsoft.Extensions.Logging" Version="7.0.0" />

<!-- Optional: GPU support -->
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.16.0" />
```

### Model Installation

#### Automatic Model Discovery

The system automatically searches for models in these locations:

1. **Primary Location** (Recommended):
   ```
   src/submodules/ds4sd-docling-tableformer-onnx/models/
   ```

2. **Environment Override**:
   ```bash
   export TABLEFORMER_MODELS_ROOT="/path/to/your/models"
   ```

3. **Fallback Locations**:
   ```
   bin/Debug/net7.0/models/tableformer-onnx/
   models/tableformer-onnx/
   ```

#### Manual Model Installation

If models are not automatically discovered:

```csharp
// Specify custom model location
var options = new TableFormerStructureServiceOptions
{
    // Use environment variable or custom path
    // TABLEFORMER_MODELS_ROOT environment variable
};
```

## Configuration Options

### Service Configuration

#### Basic Configuration
```csharp
var options = new TableFormerStructureServiceOptions
{
    // Model variant selection
    Variant = TableFormerModelVariant.Fast,

    // Runtime provider selection
    Runtime = TableFormerRuntime.Auto,  // Auto, Onnx, or Pipeline

    // Language hint (optional)
    Language = TableFormerLanguage.English,

    // Debug overlay generation
    GenerateOverlay = false,

    // Working directory for temporary files
    WorkingDirectory = Path.GetTempPath()
};
```

#### Advanced Configuration
```csharp
var options = new TableFormerStructureServiceOptions
{
    // Custom SDK options for advanced users
    SdkOptions = new TableFormerSdkOptions
    {
        Onnx = new TableFormerModelPaths(
            fastModelPath: "path/to/fast/model.onnx",
            accurateModelPath: "path/to/accurate/model.onnx"
        )
    }
};
```

### Runtime Configuration

#### CPU-Only Configuration
```csharp
var options = new TableFormerStructureServiceOptions
{
    Runtime = TableFormerRuntime.Onnx  // Force CPU-only processing
};
```

#### GPU-Accelerated Configuration
```csharp
var options = new TableFormerStructureServiceOptions
{
    Runtime = TableFormerRuntime.Auto  // Automatically uses CUDA if available
};
```

## Input Formats

### Supported Image Formats

The TableFormer system accepts images in the following formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| **PNG** | `.png` | Recommended - lossless compression |
| **JPEG** | `.jpg`, `.jpeg` | Good compression, some quality loss |
| **BMP** | `.bmp` | Uncompressed, large file sizes |
| **TIFF** | `.tiff`, `.tif` | Multi-page support |

### Image Requirements

#### Resolution
- **Minimum**: 100×100 pixels
- **Recommended**: 300-600 DPI for best results
- **Maximum**: Limited by available memory (tested up to 4000×4000)

#### Quality Guidelines
- **Contrast**: Good contrast between text and background
- **Lighting**: Even lighting, no shadows on table
- **Orientation**: Tables should be right-side up
- **Borders**: Clear table borders improve accuracy

## Output Formats

### Table Structure Object

The system returns a `TableStructure` object with the following properties:

```csharp
public class TableStructure
{
    public PageReference Page { get; }           // Source page
    public IReadOnlyList<TableCell> Cells { get; } // Detected cells
    public int RowCount { get; }                 // Number of rows
    public int ColumnCount { get; }              // Number of columns
    public TableStructureDebugArtifact? DebugArtifact { get; } // Debug overlay (if enabled)
}
```

### Table Cell Information

Each `TableCell` contains:

```csharp
public class TableCell
{
    public BoundingBox BoundingBox { get; }     // Cell location and size
    public int RowSpan { get; }                 // Vertical span (1+)
    public int ColumnSpan { get; }              // Horizontal span (1+)
    public string? Text { get; }                // Extracted text (if available)
}
```

## Advanced Usage

### Batch Processing

For processing multiple tables or images:

```csharp
// Process multiple tables from the same image
var requests = new[]
{
    new TableStructureRequest { /* table 1 config */ },
    new TableStructureRequest { /* table 2 config */ },
    new TableStructureRequest { /* table 3 config */ }
};

var results = new List<TableStructure>();
foreach (var request in requests)
{
    var result = await service.InferStructureAsync(request);
    results.Add(result);
}
```

### Custom Preprocessing

For specialized preprocessing requirements:

```csharp
// Create custom preprocessing pipeline
var customPreprocessor = new CustomImagePreprocessor();
var preprocessedImage = customPreprocessor.Preprocess(inputImage);

// Use with TableFormer
var request = new TableStructureRequest
{
    RasterizedImage = preprocessedImage,
    // ... other properties
};
```

### Error Handling

Robust error handling for production use:

```csharp
try
{
    var tableStructure = await service.InferStructureAsync(request);

    if (tableStructure.Cells.Count == 0)
    {
        Console.WriteLine("No table structure detected");
        return null;
    }

    return tableStructure;
}
catch (ArgumentException ex)
{
    // Handle invalid input parameters
    Console.WriteLine($"Invalid request: {ex.Message}");
    return null;
}
catch (InvalidOperationException ex)
{
    // Handle model or processing errors
    Console.WriteLine($"Processing error: {ex.Message}");
    return null;
}
catch (Exception ex)
{
    // Handle unexpected errors
    Console.WriteLine($"Unexpected error: {ex.Message}");
    return null;
}
```

## Performance Tuning

### Latency Optimization

#### For Real-Time Applications
```csharp
// Use Fast variant with CPU optimization
var options = new TableFormerStructureServiceOptions
{
    Variant = TableFormerModelVariant.Fast,
    Runtime = TableFormerRuntime.Onnx,  // CPU-only for predictability
    GenerateOverlay = false             // Disable debug output
};
```

#### For Batch Processing
```csharp
// Reuse service instance across multiple requests
var service = new TableFormerTableStructureService(options);

// Process multiple images
foreach (var image in imageBatch)
{
    var request = new TableStructureRequest { RasterizedImage = image };
    var result = await service.InferStructureAsync(request);
    // Process result...
}
```

### Memory Optimization

#### For Memory-Constrained Environments
```csharp
// Use Fast variant and disable overlays
var options = new TableFormerStructureServiceOptions
{
    Variant = TableFormerModelVariant.Fast,
    GenerateOverlay = false,
    WorkingDirectory = "/tmp"  // Use fast storage
};
```

#### Memory Monitoring
```csharp
// Monitor memory usage
var beforeMemory = GC.GetTotalMemory(true);
var result = await service.InferStructureAsync(request);
var afterMemory = GC.GetTotalMemory(true);

Console.WriteLine($"Memory used: {afterMemory - beforeMemory} bytes");
```

## Debugging and Troubleshooting

### Debug Overlay Generation

Enable debug overlays to visualize detection results:

```csharp
var options = new TableFormerStructureServiceOptions
{
    GenerateOverlay = true  // Enable debug visualization
};

var result = await service.InferStructureAsync(request);

// Save debug overlay
if (result.DebugArtifact != null)
{
    var overlayPath = "debug_overlay.png";
    await File.WriteAllBytesAsync(overlayPath, result.DebugArtifact.Data);
    Console.WriteLine($"Debug overlay saved to: {overlayPath}");
}
```

### Logging Configuration

Configure detailed logging for troubleshooting:

```csharp
using Microsoft.Extensions.Logging;

// Configure logging
var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Debug);
});

var logger = loggerFactory.CreateLogger<TableFormerTableStructureService>();

// Pass logger to service
var service = new TableFormerTableStructureService(options, logger);
```

### Performance Monitoring

Monitor system performance:

```csharp
// Enable performance metrics collection
var benchmark = new TableFormerBenchmark(modelsDirectory);
var results = benchmark.RunComprehensiveBenchmark(sampleImage);

// Check results
Console.WriteLine($"Fast variant: {results.FastResults?.Throughput:F1} img/s");
Console.WriteLine($"Accurate variant: {results.AccurateResults?.Throughput:F1} img/s");
```

## Integration Examples

### ASP.NET Core Integration

```csharp
// In Startup.cs or Program.cs
services.AddSingleton<ITableStructureService>(sp =>
{
    var options = new TableFormerStructureServiceOptions
    {
        Variant = TableFormerModelVariant.Fast,
        GenerateOverlay = true
    };

    return new TableFormerTableStructureService(options);
});

// In controller
public class TableExtractionController : ControllerBase
{
    private readonly ITableStructureService _tableService;

    public TableExtractionController(ITableStructureService tableService)
    {
        _tableService = tableService;
    }

    [HttpPost("extract-table")]
    public async Task<IActionResult> ExtractTable([FromBody] TableExtractionRequest request)
    {
        try
        {
            var structure = await _tableService.InferStructureAsync(request.ToStructureRequest());
            return Ok(structure);
        }
        catch (Exception ex)
        {
            return BadRequest($"Extraction failed: {ex.Message}");
        }
    }
}
```

### Console Application

```csharp
class Program
{
    static async Task Main(string[] args)
    {
        // Initialize service
        var service = new TableFormerTableStructureService();

        // Load image
        var imageBytes = await File.ReadAllBytesAsync(args[0]);

        // Extract table
        var request = new TableStructureRequest
        {
            Page = new PageReference(1),
            RasterizedImage = imageBytes,
            BoundingBox = new BoundingBox(0, 0, 1000, 800)  // Example bounds
        };

        var result = await service.InferStructureAsync(request);

        // Display results
        Console.WriteLine($"Found {result.Cells.Count} cells");
        Console.WriteLine($"Table has {result.RowCount} rows and {result.ColumnCount} columns");

        foreach (var cell in result.Cells)
        {
            Console.WriteLine($"Cell: {cell.BoundingBox}");
        }
    }
}
```

### WPF Application

```csharp
public partial class MainWindow : Window
{
    private readonly ITableStructureService _tableService;

    public MainWindow()
    {
        InitializeComponent();

        var options = new TableFormerStructureServiceOptions
        {
            Variant = TableFormerModelVariant.Fast,
            GenerateOverlay = true
        };

        _tableService = new TableFormerTableStructureService(options);
    }

    private async void ExtractTableButton_Click(object sender, RoutedEventArgs e)
    {
        // Load image from file
        var openFileDialog = new OpenFileDialog();
        if (openFileDialog.ShowDialog() == true)
        {
            var imageBytes = await File.ReadAllBytesAsync(openFileDialog.FileName);

            // Extract table structure
            var request = new TableStructureRequest
            {
                Page = new PageReference(1),
                RasterizedImage = imageBytes,
                BoundingBox = GetTableBoundsFromImage()  // Your bounds detection logic
            };

            var result = await _tableService.InferStructureAsync(request);

            // Display results
            DisplayTableStructure(result);
        }
    }
}
```

## Best Practices

### Image Preparation

#### Optimal Image Settings
- **Resolution**: 300 DPI for best results
- **Format**: PNG for lossless quality
- **Size**: Keep under 2000×2000 pixels for performance
- **Tables**: Ensure tables fill most of the image

#### Table Detection
```csharp
// Automatically detect table bounds (example)
var tableBounds = await DetectTableBounds(image);

// Manual bounds specification
var manualBounds = new BoundingBox(left: 100, top: 50, right: 900, bottom: 700);
```

### Performance Optimization

#### Caching
```csharp
// Cache service instance
private static readonly Lazy<ITableStructureService> _tableService =
    new Lazy<ITableStructureService>(() =>
        new TableFormerTableStructureService());

// Reuse across requests
var result = await _tableService.Value.InferStructureAsync(request);
```

#### Async Processing
```csharp
// Use async processing for better responsiveness
var extractionTask = service.InferStructureAsync(request);

// Do other work while processing
await DoOtherWorkAsync();

// Get results
var tableStructure = await extractionTask;
```

### Error Recovery

#### Graceful Degradation
```csharp
// Fallback strategy
try
{
    var result = await service.InferStructureAsync(request);
    return result;
}
catch (ModelNotFoundException)
{
    // Fallback to simpler table detection
    return await FallbackTableDetection(request);
}
catch (Exception ex)
{
    // Log and return empty result
    Logger.LogError(ex, "Table extraction failed");
    return new TableStructure(request.Page, Array.Empty<TableCell>(), 0, 0, null);
}
```

## Common Use Cases

### 1. Document Processing Pipeline

```csharp
public async Task<List<TableStructure>> ProcessDocumentAsync(byte[] documentBytes)
{
    // Extract all pages
    var pages = await ExtractPagesAsync(documentBytes);

    var allTables = new List<TableStructure>();

    foreach (var page in pages)
    {
        // Detect tables in each page
        var tableBounds = await DetectTableBoundsAsync(page.Image);

        foreach (var bounds in tableBounds)
        {
            var request = new TableStructureRequest
            {
                Page = page.PageReference,
                RasterizedImage = page.ImageBytes,
                BoundingBox = bounds
            };

            var table = await _tableService.InferStructureAsync(request);
            allTables.Add(table);
        }
    }

    return allTables;
}
```

### 2. Real-Time Table Extraction

```csharp
public async Task<TableStructure?> ExtractTableFromCameraAsync(byte[] imageBytes)
{
    // Quick preprocessing for real-time
    var optimizedOptions = new TableFormerStructureServiceOptions
    {
        Variant = TableFormerModelVariant.Fast,
        GenerateOverlay = false  // Faster processing
    };

    using var fastService = new TableFormerTableStructureService(optimizedOptions);

    var request = new TableStructureRequest
    {
        Page = new PageReference(1),
        RasterizedImage = imageBytes,
        BoundingBox = await DetectTableBoundsAsync(imageBytes)  // Fast detection
    };

    return await fastService.InferStructureAsync(request);
}
```

### 3. Batch Table Processing

```csharp
public async Task<Dictionary<string, TableStructure>> ProcessTableDirectoryAsync(string directoryPath)
{
    var results = new Dictionary<string, TableStructure>();

    // Process all images in directory
    foreach (var file in Directory.GetFiles(directoryPath, "*.png"))
    {
        var imageBytes = await File.ReadAllBytesAsync(file);

        var request = new TableStructureRequest
        {
            Page = new PageReference(1),
            RasterizedImage = imageBytes,
            BoundingBox = new BoundingBox(0, 0, 1000, 800)  // Assume full image
        };

        var table = await _tableService.InferStructureAsync(request);
        results[Path.GetFileName(file)] = table;
    }

    return results;
}
```

## Support and Resources

### Getting Help

#### Documentation
- **API Reference**: See inline XML documentation
- **Architecture Guide**: `docs/TABLEFORMER_ARCHITECTURE.md`
- **Performance Guide**: `docs/TABLEFORMER_PERFORMANCE.md`

#### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Pull Requests**: Contribute improvements

### Version Compatibility

#### Supported Versions

| Component | Version | Status |
|-----------|---------|--------|
| **.NET Runtime** | 7.0+ | ✅ Supported |
| **ONNX Runtime** | 1.16.0+ | ✅ Supported |
| **SkiaSharp** | 2.88.0+ | ✅ Supported |
| **CUDA Runtime** | 11.8+ | ⚠️ Optional |

#### Migration Guide

Upgrading from older versions:

```csharp
// Before (legacy)
var oldService = new LegacyTableFormerService();

// After (new implementation)
var newService = new TableFormerTableStructureService();
var result = await newService.InferStructureAsync(request);
```

## Frequently Asked Questions

### Q: Why am I getting no cells detected?

**A**: Common causes and solutions:
1. **Model not found**: Verify model paths and file integrity
2. **Poor image quality**: Ensure good contrast and lighting
3. **Wrong table bounds**: Check that BoundingBox covers the actual table
4. **Small table**: Very small tables may not be detected

### Q: How do I improve accuracy?

**A**:
1. Use `TableFormerModelVariant.Accurate` for better quality
2. Ensure high-resolution images (300+ DPI)
3. Provide accurate table bounding boxes
4. Enable debug overlays to visualize detection results

### Q: Can I use GPU acceleration?

**A**:
1. Install NVIDIA GPU drivers and CUDA 11.8+
2. Use `TableFormerRuntime.Auto` for automatic detection
3. Verify CUDA installation: `nvidia-smi` should show GPU info

### Q: How do I handle large documents?

**A**:
1. Process pages individually rather than entire document
2. Use `TableFormerModelVariant.Fast` for better performance
3. Implement pagination for very large tables
4. Monitor memory usage and implement cleanup

---

Thank you for using TableFormer! For additional support, please refer to the technical documentation or open an issue on GitHub.