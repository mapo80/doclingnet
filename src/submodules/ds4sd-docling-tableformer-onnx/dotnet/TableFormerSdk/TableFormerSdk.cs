using SkiaSharp;
using TableFormerSdk.Backends;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;

namespace TableFormerSdk;

/// <summary>
/// Main SDK class for TableFormer table structure recognition
/// Uses JPQD-quantized ONNX models from HuggingFace asmud/ds4sd-docling-models-onnx
/// </summary>
public sealed class TableFormer : IDisposable
{
    private readonly Dictionary<TableFormerModelVariant, TableFormerOnnxBackend> _backends;
    private bool _disposed;

    /// <summary>
    /// Initialize TableFormer SDK with model directory
    /// </summary>
    /// <param name="modelsDirectory">Directory containing tableformer_fast.onnx and tableformer_accurate.onnx</param>
    public TableFormer(string modelsDirectory)
    {
        ArgumentNullException.ThrowIfNull(modelsDirectory);

        if (!Directory.Exists(modelsDirectory))
        {
            throw new DirectoryNotFoundException($"Models directory not found: {modelsDirectory}");
        }

        _backends = new Dictionary<TableFormerModelVariant, TableFormerOnnxBackend>();

        // Load Fast variant
        var fastPath = Path.Combine(modelsDirectory, "tableformer_fast.onnx");
        if (File.Exists(fastPath))
        {
            _backends[TableFormerModelVariant.Fast] = new TableFormerOnnxBackend(fastPath, TableFormerModelVariant.Fast);
        }
        else
        {
            Console.WriteLine($"Warning: Fast model not found at {fastPath}");
        }

        // Load Accurate variant
        var accuratePath = Path.Combine(modelsDirectory, "tableformer_accurate.onnx");
        if (File.Exists(accuratePath))
        {
            _backends[TableFormerModelVariant.Accurate] = new TableFormerOnnxBackend(accuratePath, TableFormerModelVariant.Accurate);
        }
        else
        {
            Console.WriteLine($"Warning: Accurate model not found at {accuratePath}");
        }

        if (_backends.Count == 0)
        {
            throw new FileNotFoundException(
                $"No TableFormer models found in {modelsDirectory}. " +
                $"Expected: tableformer_fast.onnx or tableformer_accurate.onnx");
        }

        Console.WriteLine($"✓ TableFormerSdk initialized with {_backends.Count} model(s)");
    }

    /// <summary>
    /// Extract table structure from image
    /// </summary>
    /// <param name="image">Table region image</param>
    /// <param name="variant">Model variant to use (default: Fast)</param>
    /// <returns>Table structure result</returns>
    public TableStructureResult ExtractTableStructure(
        SKBitmap image,
        TableFormerModelVariant variant = TableFormerModelVariant.Fast)
    {
        ObjectDisposedException.ThrowIf(_disposed, nameof(TableFormer));
        ArgumentNullException.ThrowIfNull(image);

        if (!_backends.TryGetValue(variant, out var backend))
        {
            throw new InvalidOperationException(
                $"Model variant {variant} not loaded. Available: {string.Join(", ", _backends.Keys)}");
        }

        return backend.ExtractTableStructure(image);
    }

    /// <summary>
    /// Extract table structure from image file
    /// </summary>
    /// <param name="imagePath">Path to table image</param>
    /// <param name="variant">Model variant to use (default: Fast)</param>
    /// <returns>Table structure result</returns>
    public TableStructureResult ExtractTableStructure(
        string imagePath,
        TableFormerModelVariant variant = TableFormerModelVariant.Fast)
    {
        ArgumentNullException.ThrowIfNull(imagePath);

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image not found: {imagePath}", imagePath);
        }

        using var bitmap = SKBitmap.Decode(imagePath);
        if (bitmap == null)
        {
            throw new InvalidOperationException($"Failed to decode image: {imagePath}");
        }

        return ExtractTableStructure(bitmap, variant);
    }

    /// <summary>
    /// Run performance benchmark
    /// </summary>
    /// <param name="variant">Model variant to benchmark</param>
    /// <param name="iterations">Number of iterations</param>
    /// <returns>Benchmark results</returns>
    public BenchmarkResult Benchmark(
        TableFormerModelVariant variant = TableFormerModelVariant.Fast,
        int iterations = 100)
    {
        ObjectDisposedException.ThrowIf(_disposed, nameof(TableFormer));

        if (!_backends.TryGetValue(variant, out var backend))
        {
            throw new InvalidOperationException($"Model variant {variant} not loaded");
        }

        return backend.Benchmark(iterations);
    }

    /// <summary>
    /// Check if a model variant is available
    /// </summary>
    public bool IsModelLoaded(TableFormerModelVariant variant) =>
        _backends.ContainsKey(variant);

    /// <summary>
    /// Get all loaded model variants
    /// </summary>
    public IReadOnlyList<TableFormerModelVariant> LoadedVariants =>
        _backends.Keys.ToList().AsReadOnly();

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var backend in _backends.Values)
        {
            backend.Dispose();
        }

        _backends.Clear();
        _disposed = true;
    }
}
