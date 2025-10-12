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
        RunnerLogger.Initialized(_logger, _options.Runtime.ToString(), _options.Language.ToString(), _workingDirectory);
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
                var result = _sdk.Process(path, _options.GenerateOverlay, _options.Runtime);
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

    internal static bool ShouldAugmentWithFallback(
        IReadOnlyList<LayoutSdk.BoundingBox> boxes,
        LayoutSdkNormalisationMetadata metadata,
        out double coverageRatio)
    {
        ArgumentNullException.ThrowIfNull(boxes);
        var pageArea = (double)metadata.OriginalWidth * metadata.OriginalHeight;
        if (pageArea <= 0)
        {
            coverageRatio = 0d;
            return true;
        }

        coverageRatio = ComputeCoverage(boxes, metadata);
        coverageRatio = Math.Clamp(coverageRatio, 0d, 1d);
        return coverageRatio < 0.55d;
    }

    internal static IReadOnlyList<LayoutSdk.BoundingBox> MergeDetections(
        IReadOnlyList<LayoutSdk.BoundingBox> primary,
        IReadOnlyList<LayoutSdk.BoundingBox> fallback)
    {
        ArgumentNullException.ThrowIfNull(primary);
        ArgumentNullException.ThrowIfNull(fallback);

        var result = new List<LayoutSdk.BoundingBox>(primary.Count + fallback.Count);
        result.AddRange(primary);

        foreach (var candidate in fallback)
        {
            if (IsContained(candidate, result))
            {
                continue;
            }

            result.Add(candidate);
        }

        return result;
    }

    private static bool IsContained(LayoutSdk.BoundingBox candidate, IReadOnlyList<LayoutSdk.BoundingBox> boxes)
    {
        foreach (var box in boxes)
        {
            if (Contains(box, candidate))
            {
                return true;
            }
        }

        return false;
    }

    private static bool Contains(LayoutSdk.BoundingBox container, LayoutSdk.BoundingBox inner)
    {
        return inner.X >= container.X &&
               inner.Y >= container.Y &&
               inner.X + inner.Width <= container.X + container.Width &&
               inner.Y + inner.Height <= container.Y + container.Height;
    }

    private static double ComputeCoverage(
        IReadOnlyList<LayoutSdk.BoundingBox> boxes,
        LayoutSdkNormalisationMetadata metadata)
    {
        var clipped = new List<LayoutSdk.BoundingBox>(boxes.Count);
        foreach (var box in boxes)
        {
            var left = Math.Max(0f, Math.Min(box.X, metadata.OriginalWidth));
            var top = Math.Max(0f, Math.Min(box.Y, metadata.OriginalHeight));
            var right = Math.Max(left, Math.Min(box.X + box.Width, metadata.OriginalWidth));
            var bottom = Math.Max(top, Math.Min(box.Y + box.Height, metadata.OriginalHeight));

            var width = right - left;
            var height = bottom - top;
            if (width <= 0f || height <= 0f)
            {
                continue;
            }

            clipped.Add(new LayoutSdk.BoundingBox(left, top, width, height, box.Label));
        }

        if (clipped.Count == 0)
        {
            return 0d;
        }

        var unionArea = ComputeUnionArea(clipped);
        var pageArea = (double)metadata.OriginalWidth * metadata.OriginalHeight;
        return unionArea / Math.Max(pageArea, double.Epsilon);
    }

    private static double ComputeUnionArea(IReadOnlyList<LayoutSdk.BoundingBox> boxes)
    {
        var events = new List<SweepEvent>(boxes.Count * 2);
        foreach (var box in boxes)
        {
            var x1 = box.X;
            var x2 = box.X + box.Width;
            var y1 = box.Y;
            var y2 = box.Y + box.Height;

            if (x2 <= x1 || y2 <= y1)
            {
                continue;
            }

            events.Add(new SweepEvent(x1, y1, y2, 1));
            events.Add(new SweepEvent(x2, y1, y2, -1));
        }

        if (events.Count == 0)
        {
            return 0d;
        }

        events.Sort();

        var active = new List<(double y1, double y2)>();
        var area = 0d;
        var previousX = events[0].X;

        foreach (var sweep in events)
        {
            var deltaX = sweep.X - previousX;
            if (deltaX > 0 && active.Count > 0)
            {
                var coveredY = ComputeCoveredLength(active);
                area += deltaX * coveredY;
            }

            if (sweep.Type > 0)
            {
                active.Add((sweep.Y1, sweep.Y2));
            }
            else
            {
                var index = active.FindIndex(segment =>
                    Math.Abs(segment.y1 - sweep.Y1) < 1e-3 &&
                    Math.Abs(segment.y2 - sweep.Y2) < 1e-3);
                if (index >= 0)
                {
                    active.RemoveAt(index);
                }
            }

            previousX = sweep.X;
        }

        return area;
    }

    private static double ComputeCoveredLength(List<(double y1, double y2)> segments)
    {
        if (segments.Count == 0)
        {
            return 0d;
        }

        segments.Sort((a, b) =>
        {
            var comparison = a.y1.CompareTo(b.y1);
            return comparison != 0 ? comparison : a.y2.CompareTo(b.y2);
        });

        var total = 0d;
        var currentStart = segments[0].y1;
        var currentEnd = segments[0].y2;

        for (var i = 1; i < segments.Count; i++)
        {
            var (start, end) = segments[i];
            if (start <= currentEnd)
            {
                currentEnd = Math.Max(currentEnd, end);
            }
            else
            {
                total += currentEnd - currentStart;
                currentStart = start;
                currentEnd = end;
            }
        }

        total += currentEnd - currentStart;
        return total;
    }

    private readonly record struct SweepEvent(double X, double Y1, double Y2, int Type) : IComparable<SweepEvent>
    {
        public int CompareTo(SweepEvent other)
        {
            var comparison = X.CompareTo(other.X);
            if (comparison != 0)
            {
                return comparison;
            }

            return Type.CompareTo(other.Type);
        }
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
