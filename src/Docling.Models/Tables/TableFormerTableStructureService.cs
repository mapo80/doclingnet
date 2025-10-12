using System;
using System.Collections.Generic;
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
    private readonly ITableFormerInvoker _tableFormer;
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
            var result = _tableFormer.Process(tempPath, _generateOverlay, _variant, _runtime, _language);
            var cells = ConvertRegions(request.BoundingBox, bitmap.Width, bitmap.Height, result.Regions);
            var rowCount = CountAxisGroups(cells, static cell => (cell.BoundingBox.Top, cell.BoundingBox.Height));
            var columnCount = CountAxisGroups(cells, static cell => (cell.BoundingBox.Left, cell.BoundingBox.Width));
            var debugArtifact = _generateOverlay ? TryCreateDebugArtifact(request.Page, result) : null;
            return new TableStructure(request.Page, cells, rowCount, columnCount, debugArtifact);
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
            return new TableFormerInvoker(new TableFormer(sdkOptions));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or FileNotFoundException or DirectoryNotFoundException)
        {
            _logger.LogWarning(ex, "Failed to initialize TableFormer backend; falling back to stub invoker.");
            return NullTableFormerInvoker.Instance;
        }
    }

    private static TableFormerSdkOptions? TryCreateDefaultSdkOptions(ILogger logger)
    {
        var rootOverride = Environment.GetEnvironmentVariable("TABLEFORMER_MODELS_ROOT");
        var baseDirectory = AppContext.BaseDirectory;

        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(rootOverride))
        {
            candidates.Add(rootOverride);
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
                // For now, use a simple approach - we'll need to update this for the component-based approach
                return new TableFormerSdkOptions();
            }
            catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or DirectoryNotFoundException)
            {
                logger.LogDebug(ex, "Failed to initialize TableFormer models from '{Directory}'.", candidate);
            }
        }

        return null;
    }
}
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
            LogDeleteFailed(_logger, path, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            LogDeleteFailed(_logger, path, ex);
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
            LogOverlayEncodeFailed(_logger, ex);
            return null;
        }
    }

    private ITableFormerInvoker CreateTableFormerInvoker(TableFormerSdkOptions? sdkOptions)
    {
        sdkOptions ??= TryCreateDefaultSdkOptions(_logger);

        if (sdkOptions is null)
        {
            LogBackendUnavailable(_logger, null);
            return NullTableFormerInvoker.Instance;
        }

        try
        {
            return new TableFormerInvoker(new TableFormer(sdkOptions));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or FileNotFoundException or DirectoryNotFoundException)
        {
            LogBackendInitializationFailed(_logger, ex);
            return NullTableFormerInvoker.Instance;
        }
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

private static TableFormerSdkOptions? TryCreateDefaultSdkOptions(ILogger logger)
{
    var rootOverride = Environment.GetEnvironmentVariable("TABLEFORMER_MODELS_ROOT");
    var baseDirectory = AppContext.BaseDirectory;

    var candidates = new List<string>();
    if (!string.IsNullOrWhiteSpace(rootOverride))
    {
        candidates.Add(rootOverride);
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

            // For now, use a single model path - we'll need to update this for the component-based approach
            return new TableFormerSdkOptions(new TableFormerModelPaths(fast.EncoderPath, accurate?.EncoderPath));
        }
        catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or DirectoryNotFoundException)
        {
            logger.LogDebug(ex, "Failed to initialize TableFormer models from '{Directory}'.", candidate);
        }
    }

    return null;
}
