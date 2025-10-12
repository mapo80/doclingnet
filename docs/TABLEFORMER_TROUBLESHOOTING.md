# TableFormer Troubleshooting Guide

## Common Issues and Solutions

This guide covers the most common problems encountered when using the TableFormer table structure recognition system, along with step-by-step solutions.

## 1. Model Loading Issues

### Problem: "Model file not found" Errors

**Error Messages:**
```
FileNotFoundException: Model file not found: /path/to/model.onnx
DirectoryNotFoundException: Models directory not found
```

**Common Causes:**
1. Models not installed in the expected location
2. Incorrect file permissions
3. Corrupted model files

**Solutions:**

#### Step 1: Verify Model Location
```bash
# Check if models exist in the expected location
ls -la src/submodules/ds4sd-docling-tableformer-onnx/models/

# Expected files:
# tableformer_fast_encoder.onnx
# tableformer_fast_tag_transformer_encoder.onnx
# tableformer_fast_tag_transformer_decoder_step.onnx
# tableformer_fast_bbox_decoder.onnx
# tableformer_accurate_*.onnx (for accurate variant)
```

#### Step 2: Check File Permissions
```bash
# Verify read permissions
stat src/submodules/ds4sd-docling-tableformer-onnx/models/tableformer_fast_encoder.onnx

# Fix permissions if needed
chmod +r src/submodules/ds4sd-docling-tableformer-onnx/models/*.onnx
```

#### Step 3: Validate Model Files
```python
# Validate ONNX model integrity
import onnx

try:
    model = onnx.load('src/submodules/ds4sd-docling-tableformer-onnx/models/tableformer_fast_encoder.onnx')
    onnx.checker.check_model(model)
    print("✅ Model file is valid")
except Exception as e:
    print(f"❌ Model file is corrupted: {e}")
```

#### Step 4: Use Environment Variable Override
```bash
# Specify custom model location
export TABLEFORMER_MODELS_ROOT="/path/to/valid/models"
```

### Problem: CUDA Provider Not Available

**Error Messages:**
```
CUDA provider not available: CUDA runtime not found
Failed to load CUDA provider
```

**Solutions:**

#### Step 1: Check CUDA Installation
```bash
# Verify CUDA installation
nvidia-smi

# Check CUDA version
nvcc --version
```

#### Step 2: Install Missing Dependencies
```bash
# On Windows
# Install CUDA Toolkit 11.8+ from NVIDIA website
# Install cuDNN matching your CUDA version

# On Linux
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify ONNX Runtime GPU package
dotnet list package --include-prerelease
```

#### Step 3: Fallback to CPU
```csharp
// Force CPU-only processing
var options = new TableFormerStructureServiceOptions
{
    Runtime = TableFormerRuntime.Onnx  // CPU only
};
```

## 2. Performance Issues

### Problem: Slow Inference Speed

**Symptoms:**
- Inference takes >500ms per table
- High CPU usage
- Poor throughput in batch processing

**Solutions:**

#### Step 1: Use Fast Variant
```csharp
var options = new TableFormerStructureServiceOptions
{
    Variant = TableFormerModelVariant.Fast  // Faster than Accurate
};
```

#### Step 2: Enable CUDA Acceleration
```csharp
var options = new TableFormerStructureServiceOptions
{
    Runtime = TableFormerRuntime.Auto  // Use GPU if available
};
```

#### Step 3: Disable Debug Features
```csharp
var options = new TableFormerStructureServiceOptions
{
    GenerateOverlay = false  // Disable for production
};
```

#### Step 4: Optimize Image Size
```csharp
// Resize large images before processing
var resizedImage = ResizeImageForTableFormer(originalImage, maxSize: 1200);
```

### Problem: High Memory Usage

**Symptoms:**
- OutOfMemoryException during processing
- System becomes unresponsive
- High memory consumption reported

**Solutions:**

#### Step 1: Monitor Memory Usage
```csharp
// Check memory before processing
var beforeMemory = GC.GetTotalMemory(true);
var result = await service.InferStructureAsync(request);
var afterMemory = GC.GetTotalMemory(true);

Console.WriteLine($"Memory used: {afterMemory - beforeMemory} bytes");
```

#### Step 2: Process Images Individually
```csharp
// Instead of batch processing
foreach (var image in images)
{
    var request = new TableStructureRequest { RasterizedImage = image };
    var result = await service.InferStructureAsync(request);

    // Process each result and dispose
    ProcessResult(result);
}
```

#### Step 3: Use CPU-Only Processing
```csharp
var options = new TableFormerStructureServiceOptions
{
    Runtime = TableFormerRuntime.Onnx  // CPU only, lower memory
};
```

## 3. Accuracy Issues

### Problem: No Cells Detected

**Symptoms:**
- `TableStructure.Cells` is empty
- No table structure found in valid tables

**Solutions:**

#### Step 1: Verify Table Bounds
```csharp
// Check that BoundingBox covers the actual table
var tableBounds = DetectTableBounds(image);  // Use proper detection
Console.WriteLine($"Table bounds: {tableBounds}");

// Ensure bounds are within image dimensions
var imageWidth = image.Width;
var imageHeight = image.Height;
var validBounds = new BoundingBox(
    Math.Max(0, tableBounds.Left),
    Math.Max(0, tableBounds.Top),
    Math.Min(imageWidth, tableBounds.Right),
    Math.Min(imageHeight, tableBounds.Bottom)
);
```

#### Step 2: Improve Image Quality
```csharp
// Enhance image contrast if needed
var enhancedImage = EnhanceImageContrast(image);

// Ensure adequate resolution (300+ DPI)
if (image.Width < 300 || image.Height < 300)
{
    var highResImage = UpscaleImage(image, targetDpi: 300);
}
```

#### Step 3: Adjust Confidence Threshold
```csharp
// Lower confidence threshold for more detections (default: 0.25)
var backend = new TableFormerDetrBackend(modelPath, confidenceThreshold: 0.15f);
```

#### Step 4: Enable Debug Overlay
```csharp
var options = new TableFormerStructureServiceOptions
{
    GenerateOverlay = true  // Visualize detections
};

var result = await service.InferStructureAsync(request);

// Save and inspect overlay
if (result.DebugArtifact != null)
{
    await File.WriteAllBytesAsync("debug_overlay.png", result.DebugArtifact.Data);
    Console.WriteLine("Check debug_overlay.png to see what was detected");
}
```

### Problem: Incorrect Cell Spans

**Symptoms:**
- Cells not properly merged horizontally/vertically
- Incorrect rowspan/colspan values

**Solutions:**

#### Step 1: Check Table Structure
```csharp
// Verify detected structure
Console.WriteLine($"Rows: {result.RowCount}, Columns: {result.ColumnCount}");
Console.WriteLine($"Cells: {result.Cells.Count}");

// Check for expected patterns
var headerCells = result.Cells.Where(c => c.RowSpan > 1 || c.ColumnSpan > 1);
Console.WriteLine($"Spanning cells: {headerCells.Count()}");
```

#### Step 2: Improve Table Detection
```csharp
// Use more accurate table bounds detection
var accurateBounds = await DetectTableBoundsWithAI(image);
var request = new TableStructureRequest
{
    BoundingBox = accurateBounds,  // Better bounds
    RasterizedImage = imageBytes
};
```

### Problem: Poor Quality on Complex Tables

**Symptoms:**
- Missing cells in complex layouts
- Incorrect header detection
- Poor handling of merged cells

**Solutions:**

#### Step 1: Use Accurate Variant
```csharp
var options = new TableFormerStructureServiceOptions
{
    Variant = TableFormerModelVariant.Accurate  // Better for complex tables
};
```

#### Step 2: Preprocessing Improvements
```csharp
// Enhance table borders for better detection
var enhancedImage = EnhanceTableBorders(image);

// Remove noise and artifacts
var cleanedImage = RemoveImageNoise(image);
```

#### Step 3: Multiple Detection Attempts
```csharp
// Try different confidence thresholds
var thresholds = new[] { 0.1f, 0.25f, 0.5f };

foreach (var threshold in thresholds)
{
    var result = await TryExtractWithThreshold(request, threshold);

    if (result.Cells.Count > 0)
    {
        Console.WriteLine($"Success with threshold {threshold}: {result.Cells.Count} cells");
        return result;
    }
}
```

## 4. Integration Issues

### Problem: Dependency Conflicts

**Error Messages:**
```
Could not load file or assembly 'Microsoft.ML.OnnxRuntime'
Assembly version conflicts
```

**Solutions:**

#### Step 1: Check Package Versions
```xml
<!-- Ensure compatible versions -->
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.16.0" />
<PackageReference Include="SkiaSharp" Version="2.88.6" />
```

#### Step 2: Clear NuGet Cache
```bash
# Clear NuGet cache
dotnet nuget locals all --clear

# Restore packages
dotnet restore
```

#### Step 3: Check Runtime Compatibility
```csharp
// Verify .NET runtime version
Console.WriteLine($"Runtime: {Environment.Version}");
Console.WriteLine($"OS: {RuntimeInformation.OSDescription}");
```

### Problem: Platform-Specific Issues

#### Windows Issues
```csharp
// Enable Windows-specific optimizations
if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
{
    // Windows-specific configuration
    var options = new SessionOptions
    {
        // Optimize for Windows
        ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
    };
}
```

#### Linux Issues
```bash
# Install system dependencies
sudo apt install libgomp1 libgdiplus

# Set environment variables
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

#### macOS Issues
```bash
# Install system dependencies
brew install libomp

# Set environment variables
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

## 5. Runtime Errors

### Problem: Invalid Input Image

**Error Messages:**
```
ArgumentException: Rasterized image payload is empty
InvalidOperationException: The provided rasterized image could not be decoded
```

**Solutions:**

#### Step 1: Validate Input Image
```csharp
// Check image data
if (imageBytes == null || imageBytes.Length == 0)
{
    throw new ArgumentException("Image data is empty");
}

// Check image format
using var image = SKBitmap.Decode(imageBytes);
if (image == null)
{
    throw new InvalidOperationException("Invalid image format");
}

// Check dimensions
if (image.Width <= 0 || image.Height <= 0)
{
    throw new InvalidOperationException("Invalid image dimensions");
}
```

#### Step 2: Supported Formats
```csharp
// Ensure correct format
var supportedFormats = new[] { ".png", ".jpg", ".jpeg", ".bmp" };
var extension = Path.GetExtension(filePath).ToLower();

if (!supportedFormats.Contains(extension))
{
    // Convert to supported format
    var convertedImage = ConvertImageToPng(imageBytes);
    imageBytes = convertedImage;
}
```

### Problem: Out of Memory Errors

**Error Messages:**
```
OutOfMemoryException: Insufficient memory
System.OutOfMemoryException
```

**Solutions:**

#### Step 1: Reduce Image Size
```csharp
// Resize large images before processing
var maxDimension = 1200;
if (image.Width > maxDimension || image.Height > maxDimension)
{
    var resizedImage = ResizeImage(image, maxDimension);
    imageBytes = EncodeImage(resizedImage);
}
```

#### Step 2: Process in Batches
```csharp
// Process large documents in smaller chunks
const int batchSize = 5;
for (int i = 0; i < pages.Count; i += batchSize)
{
    var batch = pages.Skip(i).Take(batchSize);
    await ProcessBatchAsync(batch);

    // Force garbage collection between batches
    GC.Collect();
    GC.WaitForPendingFinalizers();
}
```

#### Step 3: Monitor Memory Usage
```csharp
// Monitor and control memory usage
var memoryThreshold = 1024 * 1024 * 1024; // 1GB
var currentMemory = GC.GetTotalMemory(false);

if (currentMemory > memoryThreshold)
{
    // Dispose large objects or reduce processing
    DisposeLargeObjects();
}
```

## 6. Configuration Issues

### Problem: Incorrect Model Paths

**Symptoms:**
- Models load but inference fails
- Poor accuracy or no results
- Inconsistent behavior

**Solutions:**

#### Step 1: Verify Path Resolution
```csharp
// Check which path is being used
var environmentPath = Environment.GetEnvironmentVariable("TABLEFORMER_MODELS_ROOT");
Console.WriteLine($"Environment path: {environmentPath}");

var appDirectory = AppContext.BaseDirectory;
var expectedPath = Path.Combine(appDirectory, "src", "submodules", "ds4sd-docling-tableformer-onnx", "models");
Console.WriteLine($"Expected path: {expectedPath}");
Console.WriteLine($"Path exists: {Directory.Exists(expectedPath)}");
```

#### Step 2: List Available Models
```bash
# Check model files
find src/submodules/ds4sd-docling-tableformer-onnx/models -name "*.onnx" -exec ls -lh {} \;

# Verify file sizes (should be >10MB each)
# tableformer_fast_encoder.onnx: ~11MB
# tableformer_fast_tag_transformer_encoder.onnx: ~64MB
# tableformer_fast_tag_transformer_decoder_step.onnx: ~26MB
# tableformer_fast_bbox_decoder.onnx: ~38MB
```

### Problem: Environment Variable Issues

**Error Messages:**
```
Environment variable 'TABLEFORMER_MODELS_ROOT' not set
Failed to resolve model paths
```

**Solutions:**

#### Step 1: Set Environment Variable
```bash
# Set for current session
export TABLEFORMER_MODELS_ROOT="/path/to/models"

# Set permanently (add to ~/.bashrc or ~/.profile)
echo 'export TABLEFORMER_MODELS_ROOT="/path/to/models"' >> ~/.bashrc
```

#### Step 2: Verify Environment Variable
```csharp
// Check in code
var modelsRoot = Environment.GetEnvironmentVariable("TABLEFORMER_MODELS_ROOT");
Console.WriteLine($"Models root: {modelsRoot}");

// List contents
if (Directory.Exists(modelsRoot))
{
    var files = Directory.GetFiles(modelsRoot, "*.onnx");
    Console.WriteLine($"Found {files.Length} ONNX files");
}
```

## 7. Development and Testing Issues

### Problem: Unit Test Failures

**Error Messages:**
```
Test failed: Model file not found in test environment
System.IO.FileNotFoundException
```

**Solutions:**

#### Step 1: Copy Models for Testing
```csharp
// In test setup
private static void SetupTestModels()
{
    var testModelsDir = Path.Combine(TestContext.CurrentContext.TestDirectory, "models");
    var sourceModelsDir = Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "src", "submodules", "ds4sd-docling-tableformer-onnx", "models");

    if (Directory.Exists(sourceModelsDir) && !Directory.Exists(testModelsDir))
    {
        CopyDirectory(sourceModelsDir, testModelsDir);
    }
}
```

#### Step 2: Use Test-Specific Configuration
```csharp
// Use test models
var options = new TableFormerStructureServiceOptions
{
    // Override model location for tests
    SdkOptions = CreateTestModelOptions()
};
```

### Problem: Debugging Difficulties

**Error Messages:**
```
Cannot inspect intermediate results
No visibility into model decisions
```

**Solutions:**

#### Step 1: Enable Debug Overlays
```csharp
var options = new TableFormerStructureServiceOptions
{
    GenerateOverlay = true  // Enable visualization
};

var result = await service.InferStructureAsync(request);

// Analyze overlay
if (result.DebugArtifact != null)
{
    var overlayPath = "debug_analysis.png";
    await File.WriteAllBytesAsync(overlayPath, result.DebugArtifact.Data);
    Console.WriteLine($"Debug overlay: {overlayPath}");
}
```

#### Step 2: Enable Detailed Logging
```csharp
// Configure logging
var logger = LoggerFactory.Create(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Debug);
}).CreateLogger<TableFormerTableStructureService>();

var service = new TableFormerTableStructureService(options, logger);
```

#### Step 3: Step-by-Step Debugging
```csharp
// Debug each component individually
var preprocessor = new ImagePreprocessor();
var input = preprocessor.PreprocessImage(image);

var components = new TableFormerOnnxComponents(modelsDir);
var features = components.RunEncoder(input);

// Continue debugging...
```

## 8. Production Deployment Issues

### Problem: Performance Degradation in Production

**Symptoms:**
- Good performance in development, poor in production
- Inconsistent latency
- Memory leaks

**Solutions:**

#### Step 1: Compare Environments
```csharp
// Log environment information
Console.WriteLine($"OS: {RuntimeInformation.OSDescription}");
Console.WriteLine($"Runtime: {Environment.Version}");
Console.WriteLine($"Processor count: {Environment.ProcessorCount}");
Console.WriteLine($"Memory: {GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 1024 / 1024} MB");
```

#### Step 2: Singleton Service Pattern
```csharp
// Reuse service instance in production
public class TableFormerServiceManager
{
    private static readonly Lazy<ITableStructureService> _instance =
        new Lazy<ITableStructureService>(() =>
            new TableFormerTableStructureService());

    public static ITableStructureService Instance => _instance.Value;
}
```

#### Step 3: Health Monitoring
```csharp
// Implement health checks
public async Task<HealthCheckResult> CheckTableFormerHealth()
{
    try
    {
        // Quick test with sample data
        var testResult = await _service.InferStructureAsync(_testRequest);

        if (testResult.Cells.Count == 0)
            return HealthCheckResult.Unhealthy("No cells detected in test");

        return HealthCheckResult.Healthy();
    }
    catch (Exception ex)
    {
        return HealthCheckResult.Unhealthy($"Error: {ex.Message}");
    }
}
```

### Problem: Scalability Issues

**Symptoms:**
- Performance degrades with concurrent requests
- Memory usage grows over time
- Inconsistent response times

**Solutions:**

#### Step 1: Implement Connection Pooling
```csharp
// Pool service instances for concurrent requests
public class TableFormerServicePool
{
    private readonly ConcurrentBag<ITableStructureService> _services = new();
    private readonly int _maxServices = 10;

    public async Task<TableStructure> ExtractTableAsync(TableStructureRequest request)
    {
        var service = GetServiceFromPool();

        try
        {
            return await service.InferStructureAsync(request);
        }
        finally
        {
            ReturnServiceToPool(service);
        }
    }
}
```

#### Step 2: Request Batching
```csharp
// Batch similar requests
public class TableFormerBatchProcessor
{
    private readonly List<TableStructureRequest> _pendingRequests = new();
    private readonly Timer _batchTimer;

    public TableFormerBatchProcessor()
    {
        _batchTimer = new Timer(ProcessBatch, null, 1000, 1000); // 1 second batches
    }

    public void QueueRequest(TableStructureRequest request)
    {
        lock (_pendingRequests)
        {
            _pendingRequests.Add(request);
        }
    }
}
```

## 9. Model-Specific Issues

### Problem: Fast Variant Too Fast (Missing Details)

**Symptoms:**
- Fast variant misses small cells or complex structures
- Lower accuracy than expected

**Solutions:**

#### Step 1: Use Accurate Variant for Complex Cases
```csharp
// Detect table complexity and choose variant
var complexity = AnalyzeTableComplexity(image);

var variant = complexity > 0.7f
    ? TableFormerModelVariant.Accurate
    : TableFormerModelVariant.Fast;
```

#### Step 2: Ensemble Approach
```csharp
// Use both variants and combine results
var fastResult = await fastService.InferStructureAsync(request);
var accurateResult = await accurateService.InferStructureAsync(request);

// Combine results intelligently
var combinedResult = CombineResults(fastResult, accurateResult);
```

### Problem: Accurate Variant Too Slow

**Symptoms:**
- Accurate variant takes too long for real-time use
- Not suitable for interactive applications

**Solutions:**

#### Step 1: Use Fast for Initial Results
```csharp
// Show fast results immediately
var fastResult = await fastService.InferStructureAsync(request);
DisplayInitialResults(fastResult);

// Refine with accurate in background
var accurateResult = await Task.Run(() =>
    accurateService.InferStructureAsync(request));
UpdateWithRefinedResults(accurateResult);
```

#### Step 2: Adaptive Processing
```csharp
// Start with fast, upgrade if needed
var result = await fastService.InferStructureAsync(request);

if (NeedsRefinement(result))
{
    var refinedResult = await accurateService.InferStructureAsync(request);
    result = refinedResult;
}
```

## 10. Advanced Debugging

### Problem: Intermittent Failures

**Symptoms:**
- Random failures with no clear pattern
- Inconsistent results across runs

**Solutions:**

#### Step 1: Implement Comprehensive Logging
```csharp
// Log all operations
var logger = LoggerFactory.Create(builder =>
{
    builder.AddConsole();
    builder.AddFile("tableformer.log");
    builder.SetMinimumLevel(LogLevel.Trace);
}).CreateLogger<TableFormerTableStructureService>();

// Use structured logging
logger.LogInformation("Processing table {TableId} from {Source}", tableId, source);
```

#### Step 2: Add Health Checks
```csharp
// Regular health monitoring
var healthTimer = new Timer(async _ =>
{
    var health = await CheckServiceHealth();
    if (!health.IsHealthy)
    {
        await RestartService();
    }
}, null, 30000, 30000); // Check every 30 seconds
```

#### Step 3: Implement Circuit Breaker
```csharp
// Prevent cascade failures
var circuitBreaker = new CircuitBreaker(maxFailures: 5, timeout: 30000);

try
{
    return await circuitBreaker.ExecuteAsync(() =>
        service.InferStructureAsync(request));
}
catch (CircuitBreakerOpenException)
{
    return await FallbackProcessing(request);
}
```

## Quick Reference

### Common Error Codes and Solutions

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| **ModelNotFound** | Wrong model path | Check `src/submodules/ds4sd-docling-tableformer-onnx/models/` |
| **OutOfMemory** | Large image or memory leak | Resize image, use CPU-only mode |
| **NoCellsDetected** | Poor table bounds | Improve table detection, lower confidence threshold |
| **CUDAError** | GPU not available | Use CPU-only mode, check CUDA installation |
| **InvalidImage** | Unsupported format | Convert to PNG, check image integrity |

### Performance Tuning Quick Reference

| Scenario | Recommended Settings | Expected Performance |
|----------|-------------------|-------------------|
| **Real-time** | Fast + CPU | ~50ms, 20 img/s |
| **Batch** | Fast + CUDA | ~30ms, 35 img/s |
| **High accuracy** | Accurate + CUDA | ~120ms, 8 img/s |
| **Memory limited** | Fast + CPU | ~50ms, 312MB |

### Debugging Checklist

- [ ] ✅ Verify model files exist and are readable
- [ ] ✅ Check image format and quality
- [ ] ✅ Validate table bounding box
- [ ] ✅ Enable debug overlays
- [ ] ✅ Check system resources (memory, CPU)
- [ ] ✅ Review logs for error patterns
- [ ] ✅ Test with sample data
- [ ] ✅ Compare Fast vs Accurate variants

## Getting Additional Help

### Support Channels

#### 1. Documentation
- **Architecture**: `docs/TABLEFORMER_ARCHITECTURE.md`
- **Performance**: `docs/TABLEFORMER_PERFORMANCE.md`
- **User Guide**: `docs/TABLEFORMER_USER_GUIDE.md`

#### 2. Community Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Usage questions and best practices
- **Pull Requests**: Code contributions and improvements

#### 3. Professional Support
For enterprise deployments requiring SLA guarantees:
- Priority issue handling
- Custom feature development
- On-site consulting and training

### Reporting Issues

When reporting issues, please include:

#### Required Information
- **Version**: TableFormer and .NET runtime versions
- **Platform**: OS, CPU, GPU (if applicable)
- **Error**: Complete error message and stack trace
- **Steps**: Exact steps to reproduce the issue

#### Optional but Helpful
- **Sample Data**: Anonymized sample that reproduces the issue
- **Expected vs Actual**: What should happen vs what actually happens
- **Environment**: Docker, cloud provider, deployment configuration

#### Issue Template
```markdown
## Issue Description
[Brief description of the problem]

## Environment
- OS: [Windows/Linux/macOS]
- .NET Version: [7.0/8.0/etc]
- Hardware: [CPU/GPU specs]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Error Messages
```
[Complete error output]
```

## Additional Context
[Any other relevant information]
```

## Maintenance and Updates

### Regular Maintenance Tasks

#### 1. Model Updates
```bash
# Check for model updates
git -C src/submodules/ds4sd-docling-tableformer-onnx pull origin main

# Validate updated models
python tools/validate_models.py
```

#### 2. Performance Monitoring
```csharp
// Implement performance monitoring
var metrics = new PerformanceMetricsCollector();
metrics.StartCollection();

// Monitor key metrics
var avgLatency = metrics.GetAverageLatency();
var throughput = metrics.GetThroughput();
var errorRate = metrics.GetErrorRate();
```

#### 3. Log Rotation
```csharp
// Configure log rotation for production
var logger = LoggerFactory.Create(builder =>
{
    builder.AddFile("logs/tableformer.log", fileSizeLimitBytes: 10 * 1024 * 1024); // 10MB
});
```

## Conclusion

Most issues can be resolved by following these steps:

1. **Verify model installation** in the correct location
2. **Check image quality** and table bounds
3. **Monitor system resources** (memory, CPU)
4. **Enable appropriate logging** for debugging
5. **Test with sample data** to isolate problems

For complex issues or production deployments, refer to the technical documentation or seek community support.

---

*This troubleshooting guide is continuously updated based on user feedback and common issues encountered in production deployments.*