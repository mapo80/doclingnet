using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Microsoft.Extensions.Logging;
using LayoutSdk;

namespace Docling.Models.Layout;

/// <summary>
/// Implementation of <see cref="ILayoutDetectionService"/> backed by the Docling Layout SDK (Heron ONNX models).
/// </summary>
public sealed partial class LayoutSdkDetectionService :
    ILayoutDetectionService,
    ILayoutNormalizationMetadataSource,
    ILayoutProfilingTelemetrySource,
    IDisposable
{
    private readonly ILogger<LayoutSdkDetectionService> _logger;
    private readonly ILayoutSdkRunner _runner;
    private bool _disposed;
    private readonly object _telemetrySync = new();
    private readonly List<LayoutNormalizationTelemetry> _normalisations = new();
    private readonly object _profilingSync = new();
    private readonly List<LayoutInferenceProfilingTelemetry> _profiling = new();

    public LayoutSdkDetectionService(LayoutSdkDetectionOptions options, ILogger<LayoutSdkDetectionService> logger)
        : this(options, logger, LayoutSdkRunner.Create(options, logger))
    {
    }

    internal LayoutSdkDetectionService(LayoutSdkDetectionOptions options, ILogger<LayoutSdkDetectionService> logger, ILayoutSdkRunner runner)
    {
        _ = options ?? throw new ArgumentNullException(nameof(options));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _runner = runner ?? throw new ArgumentNullException(nameof(runner));
    }

    public async Task<IReadOnlyList<LayoutItem>> DetectAsync(LayoutRequest request, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        if (request.Pages.Count == 0)
        {
            return Array.Empty<LayoutItem>();
        }

        lock (_telemetrySync)
        {
            _normalisations.Clear();
        }

        lock (_profilingSync)
        {
            _profiling.Clear();
        }

        var profilingSource = _runner as ILayoutSdkProfilingSource;
        var captureProfiling = profilingSource?.IsProfilingEnabled ?? false;

        var items = new List<LayoutItem>();
        foreach (var page in request.Pages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            LayoutSdkInferenceResult inferenceResult;
            try
            {
                inferenceResult = await _runner.InferAsync(page, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (LayoutServiceException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new LayoutServiceException($"Layout inference failed for page {page.Page.PageNumber}.", ex);
            }

            if (inferenceResult.Normalisation is LayoutSdkNormalisationMetadata metadata)
            {
                lock (_telemetrySync)
                {
                    _normalisations.Add(new LayoutNormalizationTelemetry(page.Page, metadata));
                }

                ServiceLogger.NormalisationApplied(
                    _logger,
                    page.Page.PageNumber,
                    metadata.OriginalWidth,
                    metadata.OriginalHeight,
                    metadata.ScaledWidth,
                    metadata.ScaledHeight,
                    metadata.Scale,
                    metadata.OffsetX,
                    metadata.OffsetY);
            }
            else
            {
                ServiceLogger.NormalisationMissing(_logger, page.Page.PageNumber);
            }

            var boxes = inferenceResult.Boxes;
            if (boxes.Count == 0)
            {
                continue;
            }

            foreach (var box in boxes)
            {
                var item = CreateLayoutItem(page.Page, box);
                items.Add(item);
            }

            ServiceLogger.PageProcessed(_logger, page.Page.PageNumber, boxes.Count);

            if (captureProfiling && profilingSource!.TryGetProfilingSnapshot(out var snapshot))
            {
                lock (_profilingSync)
                {
                    _profiling.Add(new LayoutInferenceProfilingTelemetry(page.Page, snapshot));
                }
            }
        }

        ServiceLogger.DetectionCompleted(_logger, request.Pages.Count, items.Count);
        return items;
    }

    public IReadOnlyList<LayoutNormalizationTelemetry> ConsumeNormalizationMetadata()
    {
        lock (_telemetrySync)
        {
            if (_normalisations.Count == 0)
            {
                return Array.Empty<LayoutNormalizationTelemetry>();
            }

            var snapshot = _normalisations.ToArray();
            _normalisations.Clear();
            return snapshot;
        }
    }

    public IReadOnlyList<LayoutInferenceProfilingTelemetry> ConsumeProfilingTelemetry()
    {
        lock (_profilingSync)
        {
            if (_profiling.Count == 0)
            {
                return Array.Empty<LayoutInferenceProfilingTelemetry>();
            }

            var snapshot = _profiling.ToArray();
            _profiling.Clear();
            return snapshot;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _runner.Dispose();
    }

    private LayoutItem CreateLayoutItem(PageReference pageReference, LayoutSdk.BoundingBox boundingBox)
    {
        if (boundingBox is null)
        {
            throw new LayoutServiceException("The layout SDK returned a null bounding box instance.");
        }

        if (boundingBox.Width <= 0 || boundingBox.Height <= 0)
        {
            throw new LayoutServiceException("The layout SDK produced a bounding box with non-positive dimensions.");
        }

        Docling.Core.Geometry.BoundingBox geometryBox;
        try
        {
            geometryBox = Docling.Core.Geometry.BoundingBox.FromSize(boundingBox.X, boundingBox.Y, boundingBox.Width, boundingBox.Height);
        }
        catch (Exception ex)
        {
            throw new LayoutServiceException("The layout SDK produced an invalid bounding box.", ex);
        }

        var (kind, wasUnknown) = MapKind(boundingBox.Label);
        if (wasUnknown)
        {
            ServiceLogger.UnknownLabel(_logger, boundingBox.Label ?? string.Empty);
        }

        var polygon = Polygon.FromPoints(new[]
        {
            new Point2D(geometryBox.Left, geometryBox.Top),
            new Point2D(geometryBox.Right, geometryBox.Top),
            new Point2D(geometryBox.Right, geometryBox.Bottom),
            new Point2D(geometryBox.Left, geometryBox.Bottom),
        });

        return new LayoutItem(pageReference, geometryBox, kind, new[] { polygon });
    }

    private static (LayoutItemKind Kind, bool Unknown) MapKind(string? label)
    {
        if (string.IsNullOrWhiteSpace(label))
        {
            return (LayoutItemKind.Text, true);
        }

        var normalized = label.Trim().ToUpperInvariant();
        return normalized switch
        {
            "TABLE" or "TABLE_BODY" or "TABLE_CONTENT" or "TABLE_HEADER" => (LayoutItemKind.Table, false),
            "FIGURE" or "FIG" or "IMAGE" or "PICTURE" or "GRAPHIC" => (LayoutItemKind.Figure, false),
            "TEXT" or "PARA" or "PARAGRAPH" or "TITLE" or "LIST" or "CAPTION" or "FOOTNOTE" or "HEADER" => (LayoutItemKind.Text, false),
            _ => (LayoutItemKind.Text, true),
        };
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    private static partial class ServiceLogger
    {
        [LoggerMessage(EventId = 4000, Level = LogLevel.Information, Message = "Processed page {PageNumber} with {ItemCount} layout predictions.")]
        public static partial void PageProcessed(ILogger logger, int pageNumber, int itemCount);

        [LoggerMessage(EventId = 4001, Level = LogLevel.Information, Message = "Layout SDK produced {ItemCount} items across {PageCount} pages.")]
        public static partial void DetectionCompleted(ILogger logger, int pageCount, int itemCount);

        [LoggerMessage(EventId = 4002, Level = LogLevel.Warning, Message = "Encountered unknown layout label '{Label}'. Falling back to text kind.")]
        public static partial void UnknownLabel(ILogger logger, string label);

        [LoggerMessage(EventId = 4003, Level = LogLevel.Debug, Message = "Page {PageNumber} applied layout normalisation (original {OriginalWidth}x{OriginalHeight}, scaled {ScaledWidth}x{ScaledHeight}, scale {Scale:F3}, offsets {OffsetX:F2},{OffsetY:F2}).")]
        public static partial void NormalisationApplied(
            ILogger logger,
            int pageNumber,
            int originalWidth,
            int originalHeight,
            int scaledWidth,
            int scaledHeight,
            double scale,
            double offsetX,
            double offsetY);

        [LoggerMessage(EventId = 4004, Level = LogLevel.Debug, Message = "Page {PageNumber} processed without layout normalisation metadata.")]
        public static partial void NormalisationMissing(ILogger logger, int pageNumber);
    }
}
