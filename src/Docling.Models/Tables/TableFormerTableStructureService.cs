using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using TableFormerSdk;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;

namespace Docling.Models.Tables;

public sealed class TableFormerStructureServiceOptions
{
    public TableFormerModelVariant Variant { get; init; } = TableFormerModelVariant.Fast;

    public TableFormerRuntime Runtime { get; init; } = TableFormerRuntime.Auto;

    public TableFormerLanguage? Language { get; init; }

    public bool GenerateOverlay { get; init; }

    public TableFormerSdkOptions? SdkOptions { get; init; }

    public string WorkingDirectory { get; init; } = Path.GetTempPath();
}

public sealed class TableFormerTableStructureService : ITableStructureService, IDisposable
{
    private readonly ILogger<TableFormerTableStructureService> _logger;
    private readonly TableFormerModelVariant _variant;
    private readonly TableFormerRuntime _runtime;
    private readonly TableFormerLanguage? _language;
    private readonly bool _generateOverlay;
    private readonly string _workingDirectory;
    private readonly object _invokerLock = new();
    private ITableFormerInvoker _tableFormer;
    private TableFormerSdkOptions? _currentSdkOptions;
    private readonly TableFormerMetrics _metrics = new();
    private bool _disposed;

    public TableFormerTableStructureService(
        TableFormerStructureServiceOptions? options = null,
        ILogger<TableFormerTableStructureService>? logger = null)
        : this(options, logger, tableFormer: null)
    {
    }

    internal TableFormerTableStructureService(
        TableFormerStructureServiceOptions? options,
        ILogger<TableFormerTableStructureService>? logger,
        ITableFormerInvoker? tableFormer)
    {
        options ??= new TableFormerStructureServiceOptions();
        _logger = logger ?? NullLogger<TableFormerTableStructureService>.Instance;
        _variant = options.Variant;
        _runtime = options.Runtime;
        _language = options.Language;
        _generateOverlay = options.GenerateOverlay;
        _workingDirectory = PrepareWorkingDirectory(options.WorkingDirectory);

        _tableFormer = tableFormer ?? CreateTableFormerInvoker(options.SdkOptions);
    }

    public async Task<TableStructure> InferStructureAsync(TableStructureRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();

        EnsureNotDisposed();

        if (request.RasterizedImage.IsEmpty)
        {
            throw new ArgumentException("Rasterized image payload is empty.", nameof(request));
        }

        using var imageData = SKData.CreateCopy(request.RasterizedImage.ToArray());
        using var bitmap = SKBitmap.Decode(imageData);
        if (bitmap is null)
        {
            throw new InvalidOperationException("The provided rasterized image could not be decoded.");
        }

        if (bitmap.Width <= 0 || bitmap.Height <= 0)
        {
            throw new InvalidOperationException("The rasterized image has invalid dimensions.");
        }

        var tempPath = Path.Combine(_workingDirectory, $"docling-tableformer-{Guid.NewGuid():N}.png");
        using var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read, bufferSize: 4096, useAsync: true);
        await stream.WriteAsync(request.RasterizedImage, cancellationToken).ConfigureAwait(false);
        await stream.FlushAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            cancellationToken.ThrowIfCancellationRequested();

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var result = _tableFormer.Process(tempPath, _generateOverlay, _variant, _runtime, _language);
            stopwatch.Stop();

            var cells = ConvertRegions(request.BoundingBox, bitmap.Width, bitmap.Height, result.Regions);
            var rowCount = CountAxisGroups(cells, static cell => (cell.BoundingBox.Top, cell.BoundingBox.Height));
            var columnCount = CountAxisGroups(cells, static cell => (cell.BoundingBox.Left, cell.BoundingBox.Width));

            // Record metrics
            _metrics.RecordInference(
                success: true,
                inferenceTime: stopwatch.Elapsed,
                cellsFound: cells.Count,
                rowsDetected: rowCount,
                columnsDetected: columnCount,
                backendType: IsUsingOnnxBackend() ? "onnx" : "stub"
            );

            var debugArtifact = _generateOverlay ? TryCreateDebugArtifact(request.Page, result) : null;
            return new TableStructure(request.Page, cells, rowCount, columnCount, debugArtifact);
        }
        catch (Exception ex)
        {
            // Record failed inference
            _metrics.RecordInference(
                success: false,
                inferenceTime: TimeSpan.Zero,
                cellsFound: 0,
                rowsDetected: 0,
                columnsDetected: 0,
                backendType: "error"
            );

            _logger.LogError(ex, "TableFormer inference failed");
            throw;
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _tableFormer.Dispose();
        _disposed = true;
    }

    private static string PrepareWorkingDirectory(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            path = Path.GetTempPath();
        }

        Directory.CreateDirectory(path);
        return path;
    }

    private void EnsureNotDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, nameof(TableFormerTableStructureService));
    }

    private static IReadOnlyList<TableCell> ConvertRegions(BoundingBox tableBounds, int imageWidth, int imageHeight, IReadOnlyList<TableRegion> regions)
    {
        if (imageWidth <= 0 || imageHeight <= 0 || regions.Count == 0)
        {
            return Array.Empty<TableCell>();
        }

        var scaleX = tableBounds.Width / imageWidth;
        var scaleY = tableBounds.Height / imageHeight;
        if (double.IsNaN(scaleX) || double.IsInfinity(scaleX) || double.IsNaN(scaleY) || double.IsInfinity(scaleY))
        {
            return Array.Empty<TableCell>();
        }

        var cells = new List<TableCell>(regions.Count);
        foreach (var region in regions)
        {
            if (region.Width <= 0 || region.Height <= 0)
            {
                continue;
            }

            var left = tableBounds.Left + (region.X * scaleX);
            var top = tableBounds.Top + (region.Y * scaleY);
            var right = left + (region.Width * scaleX);
            var bottom = top + (region.Height * scaleY);

            left = Math.Max(tableBounds.Left, left);
            top = Math.Max(tableBounds.Top, top);
            right = Math.Min(tableBounds.Right, right);
            bottom = Math.Min(tableBounds.Bottom, bottom);

            if (!BoundingBox.TryCreate(left, top, right, bottom, out var boundingBox) || boundingBox.IsEmpty)
            {
                continue;
            }

            cells.Add(new TableCell(boundingBox, RowSpan: 1, ColumnSpan: 1, Text: null));
        }

        return cells.Count == 0
            ? Array.Empty<TableCell>()
            : cells;
    }

    private static int CountAxisGroups(IReadOnlyList<TableCell> cells, Func<TableCell, (double Origin, double Length)> selector)
    {
        if (cells.Count == 0)
        {
            return 0;
        }

        var centers = new List<double>();
        foreach (var cell in cells.OrderBy(c => selector(c).Origin))
        {
            var (origin, length) = selector(cell);
            var size = Math.Max(length, 1d);
            var center = origin + (size / 2d);
            var tolerance = Math.Max(size * 0.5d, 1d);

            var match = centers.FindIndex(existing => Math.Abs(existing - center) <= tolerance);
            if (match < 0)
            {
                centers.Add(center);
            }
        }

        return centers.Count;
    }

    private void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException ex)
        {
            _logger.LogWarning(ex, "Failed to delete temporary TableFormer image '{Path}'.", path);
        }
        catch (UnauthorizedAccessException ex)
        {
            _logger.LogWarning(ex, "Failed to delete temporary TableFormer image '{Path}'.", path);
        }
    }

    private TableStructureDebugArtifact? TryCreateDebugArtifact(PageReference page, TableStructureResult result)
    {
        using var overlay = result.OverlayImage;
        if (overlay is null)
        {
            return null;
        }

        try
        {
            using var snapshot = SKImage.FromBitmap(overlay);
            if (snapshot is null)
            {
                return null;
            }

            using var encoded = snapshot.Encode(SKEncodedImageFormat.Png, 90);
            if (encoded is null || encoded.Size == 0)
            {
                return null;
            }

            return new TableStructureDebugArtifact(page, encoded.ToArray());
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or ObjectDisposedException)
        {
            _logger.LogWarning(ex, "Failed to encode TableFormer overlay image; skipping debug artifact.");
            return null;
        }
    }
    
    internal interface ITableFormerInvoker : IDisposable
    {
        internal TableStructureResult Process(string imagePath, bool overlay, TableFormerModelVariant variant, TableFormerRuntime runtime = TableFormerRuntime.Auto, TableFormerLanguage? language = null);
    }
    
    internal sealed class TableFormerInvoker : ITableFormerInvoker
    {
        private readonly TableFormer _sdk;
    
        public TableFormerInvoker(TableFormer sdk)
        {
            _sdk = sdk ?? throw new ArgumentNullException(nameof(sdk));
        }
    
        public TableStructureResult Process(string imagePath, bool overlay, TableFormerModelVariant variant, TableFormerRuntime runtime = TableFormerRuntime.Auto, TableFormerLanguage? language = null)
            => _sdk.Process(imagePath, overlay, variant, runtime, language);
    
        public void Dispose() => _sdk.Dispose();
    }
    
    internal sealed class NullTableFormerInvoker : ITableFormerInvoker
    {
        public static NullTableFormerInvoker Instance { get; } = new();
    
        private NullTableFormerInvoker()
        {
        }
    
        public TableStructureResult Process(string imagePath, bool overlay, TableFormerModelVariant variant, TableFormerRuntime runtime = TableFormerRuntime.Auto, TableFormerLanguage? language = null)
        {
            var resolvedLanguage = language ?? TableFormerLanguage.English;
            var resolvedRuntime = runtime switch
            {
                TableFormerRuntime.Auto => TableFormerRuntime.Onnx,
                TableFormerRuntime.Pipeline => TableFormerRuntime.Onnx,
                TableFormerRuntime.OptimizedPipeline => TableFormerRuntime.Onnx,
                _ => runtime
            };
            var snapshot = new TableFormerPerformanceSnapshot(resolvedRuntime, variant, 0, 0, 0, 0, 0);
            return new TableStructureResult(Array.Empty<TableRegion>(), null, resolvedLanguage, resolvedRuntime, TimeSpan.Zero, snapshot);
        }
    
        public void Dispose()
        {
            // Nothing to dispose
        }
    }

    private ITableFormerInvoker CreateTableFormerInvoker(TableFormerSdkOptions? sdkOptions)
    {
        sdkOptions ??= TryCreateDefaultSdkOptions(_logger);

        if (sdkOptions is null)
        {
            _logger.LogInformation("TableFormer backend unavailable; falling back to stub invoker.");
            return NullTableFormerInvoker.Instance;
        }

        try
        {
            // Apply performance optimizations based on environment variables
            var useCUDA = Environment.GetEnvironmentVariable("TABLEFORMER_USE_CUDA") == "1";
            var enableQuantization = Environment.GetEnvironmentVariable("TABLEFORMER_ENABLE_QUANTIZATION") == "1";
            var enableOptimizations = Environment.GetEnvironmentVariable("TABLEFORMER_ENABLE_OPTIMIZATIONS") != "0"; // Default true

            _logger.LogInformation("TableFormer performance settings - CUDA: {UseCUDA}, Quantization: {Quantization}, Optimizations: {Optimizations}",
                useCUDA, enableQuantization, enableOptimizations);

            var newInvoker = new TableFormerInvoker(new TableFormer(sdkOptions));
            _currentSdkOptions = sdkOptions;

            _logger.LogInformation("TableFormer backend initialized successfully with models from: {Paths}",
                string.Join(", ", new[] { sdkOptions.Onnx.Fast.EncoderPath, sdkOptions.Onnx.Fast.TagEncoderPath }));

            return newInvoker;
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or FileNotFoundException or DirectoryNotFoundException)
        {
            _logger.LogWarning(ex, "Failed to initialize TableFormer backend; falling back to stub invoker.");
            return NullTableFormerInvoker.Instance;
        }
    }

    /// <summary>
    /// Reload TableFormer models from the current configuration or environment variables.
    /// This allows hot-reloading models without restarting the application.
    /// </summary>
    public void ReloadModels()
    {
        lock (_invokerLock)
        {
            _tableFormer?.Dispose();

            var newOptions = TryCreateDefaultSdkOptions(_logger);
            if (newOptions != null)
            {
                _tableFormer = CreateTableFormerInvoker(newOptions);
                _logger.LogInformation("TableFormer models reloaded successfully");
            }
            else
            {
                _tableFormer = NullTableFormerInvoker.Instance;
                _logger.LogWarning("Failed to reload TableFormer models; using stub backend");
            }
        }
    }

    /// <summary>
    /// Get current model configuration information.
    /// </summary>
    public (string FastEncoder, string? FastTagEncoder, string? AccurateEncoder) GetCurrentModelPaths()
    {
        if (_currentSdkOptions?.Onnx != null)
        {
            return (
                _currentSdkOptions.Onnx.Fast.EncoderPath,
                _currentSdkOptions.Onnx.Fast.TagEncoderPath,
                _currentSdkOptions.Onnx.Accurate?.EncoderPath
            );
        }

        return (string.Empty, null, null);
    }

    /// <summary>
    /// Check if the backend is currently using the new ONNX implementation.
    /// </summary>
    public bool IsUsingOnnxBackend()
    {
        return _tableFormer is TableFormerInvoker;
    }

    /// <summary>
    /// Get current performance metrics for monitoring and debugging.
    /// </summary>
    public TableFormerMetricsSnapshot GetMetrics()
    {
        return _metrics.GetSnapshot();
    }

    /// <summary>
    /// Reset all performance metrics.
    /// </summary>
    public void ResetMetrics()
    {
        _metrics.Reset();
    }

    /// <summary>
    /// Process multiple table structure requests in batch for improved performance.
    /// </summary>
    public async Task<IReadOnlyList<TableStructure>> InferStructureBatchAsync(
        IEnumerable<TableStructureRequest> requests,
        CancellationToken cancellationToken = default)
    {
        var results = new List<TableStructure>();
        var stopwatch = Stopwatch.StartNew();

        foreach (var request in requests)
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                var result = await InferStructureAsync(request, cancellationToken);
                results.Add(result);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to process table request in batch");
                // Continue with other requests instead of failing completely
            }
        }

        stopwatch.Stop();
        _logger.LogInformation("Processed batch of {Count} requests in {Time:F2}s",
            results.Count, stopwatch.Elapsed.TotalSeconds);

        return results;
    }

    /// <summary>
    /// Get performance recommendations based on current metrics.
    /// </summary>
    public PerformanceRecommendations GetPerformanceRecommendations()
    {
        var metrics = _metrics.GetSnapshot();
        var recommendations = new PerformanceRecommendations();

        if (metrics.TotalInferences < 10)
        {
            recommendations.Add("Insufficient data for recommendations. Run more inferences first.");
            return recommendations;
        }

        // Analyze inference time
        if (metrics.AverageInferenceTime.TotalMilliseconds > 1000)
        {
            recommendations.Add("High average inference time detected. Consider using Fast model for better performance.");
        }

        if (metrics.MaxInferenceTime.TotalMilliseconds > 5000)
        {
            recommendations.Add("Some inferences are very slow. Check for memory pressure or large images.");
        }

        // Analyze success rate
        if (metrics.SuccessRate < 0.95)
        {
            recommendations.Add($"Low success rate ({metrics.SuccessRate:P1}). Check image quality and table complexity.");
        }

        // Memory and throughput recommendations
        if (metrics.TotalInferences > 100)
        {
            var throughput = metrics.TotalInferences / metrics.TotalInferenceTime.TotalSeconds;
            if (throughput < 1.0)
            {
                recommendations.Add("Low throughput detected. Consider batch processing for better efficiency.");
            }
        }

        return recommendations;
    }

    private static TableFormerSdkOptions? TryCreateDefaultSdkOptions(ILogger logger)
    {
        // Support multiple environment variables for different model variants
        var modelsRoot = Environment.GetEnvironmentVariable("TABLEFORMER_MODELS_ROOT");
        var fastModelPath = Environment.GetEnvironmentVariable("TABLEFORMER_FAST_MODELS_PATH");
        var accurateModelPath = Environment.GetEnvironmentVariable("TABLEFORMER_ACCURATE_MODELS_PATH");
        var baseDirectory = AppContext.BaseDirectory;

        // If specific paths are provided via environment, use them directly
        if (!string.IsNullOrWhiteSpace(fastModelPath) && Directory.Exists(fastModelPath))
        {
            try
            {
                logger.LogInformation("Using TableFormer fast models from environment variable: {Path}", fastModelPath);
                var fast = TableFormerVariantModelPaths.FromDirectory(fastModelPath, "tableformer_fast");
                TableFormerVariantModelPaths? accurate = null;

                if (!string.IsNullOrWhiteSpace(accurateModelPath) && Directory.Exists(accurateModelPath))
                {
                    logger.LogInformation("Using TableFormer accurate models from environment variable: {Path}", accurateModelPath);
                    accurate = TableFormerVariantModelPaths.FromDirectory(accurateModelPath, "tableformer_accurate");
                }

                return new TableFormerSdkOptions(new TableFormerModelPaths(fast, accurate));
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "Failed to load TableFormer models from environment paths");
            }
        }

        // Fallback to search paths
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(modelsRoot))
        {
            candidates.Add(modelsRoot);
        }

        // Updated path to use the new models location in submodules
        var submoduleModelsPath = Path.GetFullPath(Path.Combine(baseDirectory, "src", "submodules", "ds4sd-docling-tableformer-onnx", "models"));
        candidates.Add(submoduleModelsPath);

        if (!string.IsNullOrWhiteSpace(baseDirectory))
        {
            candidates.Add(Path.Combine(baseDirectory, "models", "tableformer-onnx"));
            candidates.Add(Path.GetFullPath(Path.Combine(baseDirectory, "..", "..", "..", "..", "models", "tableformer-onnx")));
        }

        candidates.Add(Path.Combine(Environment.CurrentDirectory, "models", "tableformer-onnx"));

        foreach (var candidate in candidates.Select(c => Path.GetFullPath(c)))
        {
            if (!Directory.Exists(candidate))
            {
                continue;
            }

            try
            {
                logger.LogInformation("Loading TableFormer models from: {Path}", candidate);
                var fast = TableFormerVariantModelPaths.FromDirectory(candidate, "tableformer_fast");
                TableFormerVariantModelPaths? accurate = null;

                try
                {
                    accurate = TableFormerVariantModelPaths.FromDirectory(candidate, "tableformer_accurate");
                }
                catch (FileNotFoundException)
                {
                    logger.LogInformation("Accurate TableFormer models not found in '{Directory}'. Fast variant will be used.", candidate);
                }

                return new TableFormerSdkOptions(new TableFormerModelPaths(fast, accurate));
            }
            catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or DirectoryNotFoundException)
            {
                logger.LogDebug(ex, "Failed to initialize TableFormer models from '{Directory}'.", candidate);
            }
        }

        logger.LogWarning("No TableFormer models found in any of the configured paths");
        return null;
    }
}

/// <summary>
/// Collection of performance recommendations.
/// </summary>
public sealed class PerformanceRecommendations
{
    private readonly List<string> _recommendations = new();

    public IReadOnlyList<string> Recommendations => _recommendations.AsReadOnly();

    public bool HasRecommendations => _recommendations.Any();

    public void Add(string recommendation)
    {
        _recommendations.Add(recommendation);
    }

    public override string ToString()
    {
        return HasRecommendations
            ? string.Join(Environment.NewLine, _recommendations.Select((r, i) => $"{i + 1}. {r}"))
            : "No recommendations available.";
    }
}
