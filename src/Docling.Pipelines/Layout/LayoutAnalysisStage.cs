using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Options;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Pipelines.Layout;

/// <summary>
/// Pipeline stage that invokes the layout detection model and attaches predictions to the pipeline context.
/// </summary>
public sealed partial class LayoutAnalysisStage : IPipelineStage
{
    private readonly ILayoutDetectionService _layoutDetectionService;
    private readonly LayoutOptions _options;
    private readonly ILogger<LayoutAnalysisStage> _logger;
    private readonly ILayoutDebugOverlayRenderer? _debugOverlayRenderer;

    public LayoutAnalysisStage(
        ILayoutDetectionService layoutDetectionService,
        LayoutOptions options,
        ILogger<LayoutAnalysisStage> logger,
        ILayoutDebugOverlayRenderer? debugOverlayRenderer = null)
    {
        _layoutDetectionService = layoutDetectionService ?? throw new ArgumentNullException(nameof(layoutDetectionService));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = CloneOptions(options ?? throw new ArgumentNullException(nameof(options)));
        ValidateOptions(_options);
        _debugOverlayRenderer = debugOverlayRenderer;
    }

    public string Name => "layout_analysis";

    public async Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (!context.TryGet<PageImageStore>(PipelineContextKeys.PageImageStore, out var store))
        {
            throw new InvalidOperationException("Pipeline context does not contain a page image store.");
        }

        var pages = context.GetRequired<IReadOnlyList<PageReference>>(PipelineContextKeys.PageSequence);
        if (pages.Count == 0)
        {
            StageLogger.NoPages(_logger);
            context.Set(PipelineContextKeys.LayoutItems, Array.Empty<LayoutItem>());
            context.Set(PipelineContextKeys.LayoutAnalysisCompleted, true);
            return;
        }

        if (_options.GenerateDebugArtifacts && _debugOverlayRenderer is null)
        {
            throw new InvalidOperationException("Debug artifacts are enabled but no overlay renderer was supplied.");
        }

        var documentId = context.TryGet<string>(PipelineContextKeys.DocumentId, out var docId) && !string.IsNullOrWhiteSpace(docId)
            ? docId
            : Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture);

        var payloads = new List<LayoutPagePayload>(pages.Count);
        foreach (var page in pages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var pageImage = store.Rent(page);
            var payload = CreatePayload(pageImage);
            payloads.Add(payload);
            StageLogger.PagePrepared(_logger, page.PageNumber, payload.MediaType, payload.Dpi, payload.Width, payload.Height);
        }

        var request = new LayoutRequest(
            documentId,
            _options.Model.Identifier,
            new LayoutRequestOptions(_options.CreateOrphanClusters, _options.KeepEmptyClusters, _options.SkipCellAssignment),
            payloads);

        IReadOnlyList<LayoutItem> layoutItems;
        var stopwatch = Stopwatch.StartNew();
        try
        {
            layoutItems = await _layoutDetectionService.DetectAsync(request, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            StageLogger.RequestFailed(_logger, ex);
            throw;
        }
        finally
        {
            stopwatch.Stop();
        }

        StageLogger.RequestSucceeded(_logger, layoutItems.Count, stopwatch.ElapsedMilliseconds);

        context.Set(PipelineContextKeys.LayoutItems, layoutItems);
        context.Set(PipelineContextKeys.LayoutAnalysisCompleted, true);

        if (_options.GenerateDebugArtifacts && layoutItems.Count > 0 && _debugOverlayRenderer is not null)
        {
            var overlays = new List<LayoutDebugOverlay>();
            foreach (var group in layoutItems.GroupBy(item => item.Page.PageNumber))
            {
                var pageReference = pages.FirstOrDefault(p => p.PageNumber == group.Key);
                if (pageReference == default)
                {
                    continue;
                }

                using var pageImage = store.Rent(pageReference);
                overlays.Add(_debugOverlayRenderer.CreateOverlay(pageImage, group.ToList()));
            }

            context.Set(PipelineContextKeys.LayoutDebugArtifacts, overlays);
        }
    }

    private static LayoutPagePayload CreatePayload(PageImage pageImage)
    {
        using var snapshot = SKImage.FromBitmap(pageImage.Bitmap) ?? throw new InvalidOperationException("Failed to snapshot page bitmap for layout analysis.");
        using var encoded = snapshot.Encode(SKEncodedImageFormat.Png, 100);
        if (encoded is null || encoded.Size == 0)
        {
            throw new InvalidOperationException("Failed to encode page bitmap for layout analysis.");
        }

        var artifactId = !string.IsNullOrWhiteSpace(pageImage.Metadata.SourceId)
            ? pageImage.Metadata.SourceId!
            : $"page_{pageImage.Page.PageNumber:0000}";
        var mediaType = pageImage.Metadata.MediaType ?? "image/png";
        var properties = new Dictionary<string, string>(pageImage.Metadata.Properties, StringComparer.OrdinalIgnoreCase)
        {
            [PageImageMetadataKeys.NormalizedDpi] = pageImage.Page.Dpi.ToString("F2", CultureInfo.InvariantCulture),
        };

        if (!properties.ContainsKey(PageImageMetadataKeys.ScaleFactor))
        {
            properties[PageImageMetadataKeys.ScaleFactor] = "1.0000";
        }

        return new LayoutPagePayload(
            pageImage.Page,
            artifactId,
            mediaType,
            pageImage.Page.Dpi,
            pageImage.Width,
            pageImage.Height,
            properties,
            encoded.ToArray());
    }

    private static LayoutOptions CloneOptions(LayoutOptions options)
    {
        return new LayoutOptions
        {
            CreateOrphanClusters = options.CreateOrphanClusters,
            KeepEmptyClusters = options.KeepEmptyClusters,
            Model = options.Model,
            SkipCellAssignment = options.SkipCellAssignment,
            GenerateDebugArtifacts = options.GenerateDebugArtifacts,
            DebugOverlayOpacity = options.DebugOverlayOpacity,
            DebugStrokeWidth = options.DebugStrokeWidth,
        };
    }

    private static void ValidateOptions(LayoutOptions options)
    {
        if (options.DebugOverlayOpacity is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "LayoutOptions.DebugOverlayOpacity must be between 0 and 1.");
        }

        if (options.DebugStrokeWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "LayoutOptions.DebugStrokeWidth must be positive.");
        }
    }

    private static partial class StageLogger
    {
        [LoggerMessage(EventId = 3000, Level = LogLevel.Debug, Message = "No pages available for layout analysis.")]
        public static partial void NoPages(ILogger logger);

        [LoggerMessage(EventId = 3001, Level = LogLevel.Information, Message = "Prepared page {PageNumber} for layout analysis (media: {MediaType}, dpi: {Dpi}, size: {Width}x{Height}).")]
        public static partial void PagePrepared(ILogger logger, int pageNumber, string mediaType, double dpi, int width, int height);

        [LoggerMessage(EventId = 3002, Level = LogLevel.Error, Message = "Layout detection request failed.")]
        public static partial void RequestFailed(ILogger logger, Exception exception);

        [LoggerMessage(EventId = 3003, Level = LogLevel.Information, Message = "Layout detection produced {Count} items in {ElapsedMs} ms.")]
        public static partial void RequestSucceeded(ILogger logger, int count, long elapsedMs);
    }
}
