using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using LayoutSdk;
using LayoutSdk.Configuration;
using LayoutSdk.Factories;
using LayoutSdk.Processing;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Models.Layout;

internal interface ILayoutSdkRunner : IDisposable
{
    internal Task<LayoutSdkInferenceResult> InferAsync(ReadOnlyMemory<byte> imageContent, CancellationToken cancellationToken);
}

internal readonly record struct LayoutSdkInferenceResult(
    IReadOnlyList<LayoutSdk.BoundingBox> Boxes,
    LayoutSdkNormalisationMetadata? Normalisation);

public readonly record struct LayoutSdkNormalisationMetadata(
    int OriginalWidth,
    int OriginalHeight,
    int ScaledWidth,
    int ScaledHeight,
    double Scale,
    double OffsetX,
    double OffsetY);

[ExcludeFromCodeCoverage]
internal sealed partial class LayoutSdkRunner : ILayoutSdkRunner
{

    private readonly LayoutSdkDetectionOptions _options;
    private readonly LayoutSdk.LayoutSdk _sdk;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _semaphore;
    private readonly string _workingDirectory;
    private bool _disposed;

    private LayoutSdkRunner(LayoutSdkDetectionOptions options, ILogger logger)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.EnsureValid();
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _workingDirectory = ResolveWorkingDirectory(_options.WorkingDirectory);
        Directory.CreateDirectory(_workingDirectory);

        var sdkOptions = CreateSdkOptions(_options);
        var backendFactory = CreateBackendFactory(sdkOptions);

        _sdk = new LayoutSdk.LayoutSdk(
            sdkOptions,
            backendFactory,
            new Docling.Models.Layout.PassthroughOverlayRenderer(),
            new SkiaImagePreprocessor());
        _semaphore = new SemaphoreSlim(_options.MaxDegreeOfParallelism);
        RunnerLogger.Initialized(_logger, "Onnx", _options.Language.ToString(), _workingDirectory);
    }

    public static ILayoutSdkRunner Create(LayoutSdkDetectionOptions options, ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new LayoutSdkRunner(options.Clone(), logger);
    }

    public async Task<LayoutSdkInferenceResult> InferAsync(ReadOnlyMemory<byte> imageContent, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();

        var persisted = await PersistAsync(imageContent, cancellationToken).ConfigureAwait(false);
        var path = persisted.Path;
        try
        {
            await _semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                var result = _sdk.Process(path, _options.GenerateOverlay, LayoutRuntime.Onnx);
                try
                {
                    var boxes = result.Boxes ?? Array.Empty<LayoutSdk.BoundingBox>();
                    RunnerLogger.RawDetections(_logger, boxes.Count);

                    // Throw exception if Layout SDK returned no detections
                    if (boxes.Count == 0)
                    {
                        throw new LayoutServiceException(
                            "Layout SDK failed to detect any layout elements. " +
                            "This usually indicates a problem with the ONNX model output parsing or the image preprocessing. " +
                            "Ensure the LayoutSdk backend (OnnxRuntimeBackend/OpenVinoBackend) correctly implements the ParseOutputs method.");
                    }

                    var projected = ReprojectBoundingBoxes(boxes, persisted.Normalisation);
                    if (persisted.Normalisation is LayoutSdkNormalisationMetadata normalisation)
                    {
                        RunnerLogger.DenormalisedDetections(
                            _logger,
                            normalisation.OriginalWidth,
                            normalisation.OriginalHeight,
                            normalisation.ScaledWidth,
                            normalisation.ScaledHeight,
                            normalisation.Scale,
                            normalisation.OffsetX,
                            normalisation.OffsetY,
                            projected.Count);
                    }
                    else
                    {
                        RunnerLogger.DenormalisationSkipped(_logger, projected.Count);
                    }

                    return new LayoutSdkInferenceResult(projected, persisted.Normalisation);
                }
                finally
                {
                    result.OverlayImage?.Dispose();
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new LayoutServiceException("The layout SDK failed to execute the Heron model.", ex);
            }
            finally
            {
                _semaphore.Release();
            }
        }
        finally
        {
            if (!_options.KeepTemporaryFiles)
            {
                TryDelete(path);
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _semaphore.Dispose();
        _sdk.Dispose();
    }

    private static LayoutSdkOptions CreateSdkOptions(LayoutSdkDetectionOptions options)
    {
        if (options.ValidateModelFiles)
        {
            LayoutSdkBundledModels.EnsureAllFilesExist();
        }

        var sdkOptions = LayoutSdkBundledModels.CreateOptions(options.Language);
        if (options.ValidateModelFiles)
        {
            sdkOptions.EnsureModelPaths();
        }

        return sdkOptions;
    }

    private static LayoutBackendFactory CreateBackendFactory(LayoutSdkOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new LayoutBackendFactory(options);
    }

    private static string ResolveWorkingDirectory(string? workingDirectory)
    {
        var root = string.IsNullOrWhiteSpace(workingDirectory)
            ? Path.Combine(Path.GetTempPath(), "docling-layout-sdk")
            : Path.GetFullPath(workingDirectory);
        return root;
    }

    private async Task<PersistedImage> PersistAsync(ReadOnlyMemory<byte> content, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(_workingDirectory);
        var fileName = Path.Combine(_workingDirectory, Guid.NewGuid().ToString("N")) + ".png";
        var stream = new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.Read, 4096, FileOptions.Asynchronous | FileOptions.SequentialScan);

        // Use original image - preprocessing is now handled by SkiaImagePreprocessor
        await stream.WriteAsync(content, cancellationToken).ConfigureAwait(false);

        await stream.DisposeAsync().ConfigureAwait(false);

        // Create metadata for the original image dimensions
        using var data = SKData.CreateCopy(content.Span);
        using var bitmap = SKBitmap.Decode(data);
        LayoutSdkNormalisationMetadata? metadata = null;
        if (bitmap is not null)
        {
            metadata = new LayoutSdkNormalisationMetadata(
                bitmap.Width,
                bitmap.Height,
                bitmap.Width,
                bitmap.Height,
                Scale: 1.0,
                OffsetX: 0,
                OffsetY: 0);
        }

        return new PersistedImage(fileName, metadata);
    }



    internal static IReadOnlyList<LayoutSdk.BoundingBox> ReprojectBoundingBoxes(
        IReadOnlyList<LayoutSdk.BoundingBox> boxes,
        LayoutSdkNormalisationMetadata? metadata)
    {
        if (boxes.Count == 0)
        {
            return Array.Empty<LayoutSdk.BoundingBox>();
        }

        if (metadata is null || metadata.Value.Scale <= 0)
        {
            return boxes
                .Select(b => new LayoutSdk.BoundingBox(b.X, b.Y, b.Width, b.Height, b.Label))
                .ToArray();
        }

        var meta = metadata.Value;
        var projected = new List<LayoutSdk.BoundingBox>(boxes.Count);
        var contentMaxX = meta.OffsetX + meta.ScaledWidth;
        var contentMaxY = meta.OffsetY + meta.ScaledHeight;

        for (var i = 0; i < boxes.Count; i++)
        {
            var box = boxes[i];

            var left = Clamp(box.X, meta.OffsetX, contentMaxX) - meta.OffsetX;
            var top = Clamp(box.Y, meta.OffsetY, contentMaxY) - meta.OffsetY;
            var right = Clamp(box.X + box.Width, meta.OffsetX, contentMaxX) - meta.OffsetX;
            var bottom = Clamp(box.Y + box.Height, meta.OffsetY, contentMaxY) - meta.OffsetY;

            var originalLeft = ClampToPage(left / meta.Scale, meta.OriginalWidth);
            var originalTop = ClampToPage(top / meta.Scale, meta.OriginalHeight);
            var originalRight = ClampToPage(right / meta.Scale, meta.OriginalWidth);
            var originalBottom = ClampToPage(bottom / meta.Scale, meta.OriginalHeight);

            var width = originalRight - originalLeft;
            var height = originalBottom - originalTop;
            if (width <= 0 || height <= 0)
            {
                continue;
            }

            projected.Add(new LayoutSdk.BoundingBox(
                (float)originalLeft,
                (float)originalTop,
                (float)width,
                (float)height,
                box.Label));
        }

        return projected;
    }







    private static double Clamp(double value, double min, double max)
    {
        if (min > max)
        {
            (min, max) = (max, min);
        }

        return Math.Clamp(value, min, max);
    }

    private static double ClampToPage(double value, double max)
    {
        return Math.Clamp(value, 0, max);
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
            RunnerLogger.DeletionFailed(_logger, path, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            RunnerLogger.DeletionFailed(_logger, path, ex);
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    private static partial class RunnerLogger
    {
        [LoggerMessage(EventId = 4100, Level = LogLevel.Information, Message = "Initialised layout SDK runner (runtime: {Runtime}, language: {Language}, workspace: {WorkingDirectory}).")]
        public static partial void Initialized(ILogger logger, string runtime, string language, string workingDirectory);

        [LoggerMessage(EventId = 4101, Level = LogLevel.Warning, Message = "Failed to delete temporary layout file {Path}.")]
        public static partial void DeletionFailed(ILogger logger, string path, Exception exception);

        [LoggerMessage(EventId = 4102, Level = LogLevel.Warning, Message = "Falling back to the original layout image: {Reason}")]
        public static partial void PreprocessingFallback(ILogger logger, string reason);

        [LoggerMessage(EventId = 4103, Level = LogLevel.Debug, Message = "Normalised layout input from {OriginalWidth}x{OriginalHeight} to {ResizedWidth}x{ResizedHeight}.")]
        public static partial void ImageNormalised(ILogger logger, int originalWidth, int originalHeight, int resizedWidth, int resizedHeight);

        [LoggerMessage(EventId = 4104, Level = LogLevel.Debug, Message = "Layout SDK returned {DetectionCount} raw detections before reprojection.")]
        public static partial void RawDetections(ILogger logger, int detectionCount);

        [LoggerMessage(EventId = 4109, Level = LogLevel.Debug, Message = "Reprojected {DetectionCount} layout detections to {OriginalWidth}x{OriginalHeight} (scaled {ScaledWidth}x{ScaledHeight}, scale {Scale:F3}, offsets {OffsetX:F2},{OffsetY:F2}).")]
        public static partial void DenormalisedDetections(
            ILogger logger,
            int originalWidth,
            int originalHeight,
            int scaledWidth,
            int scaledHeight,
            double scale,
            double offsetX,
            double offsetY,
            int detectionCount);

        [LoggerMessage(EventId = 4110, Level = LogLevel.Debug, Message = "Reprojection skipped because no normalisation metadata was available (kept {DetectionCount} detections as-is).")]
        public static partial void DenormalisationSkipped(ILogger logger, int detectionCount);
    }

    private readonly record struct PersistedImage(string Path, LayoutSdkNormalisationMetadata? Normalisation);

}
