using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Models.Ocr;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Options;
using Microsoft.Extensions.Logging;

namespace Docling.Pipelines.Ocr;

/// <summary>
/// Pipeline stage responsible for running OCR over layout blocks and emitting recognised lines into the pipeline context.
/// </summary>
public sealed partial class OcrStage : IPipelineStage
{
    private readonly IOcrServiceFactory _ocrServiceFactory;
    private readonly PdfPipelineOptions _options;
    private readonly ILogger<OcrStage> _logger;

    public OcrStage(IOcrServiceFactory ocrServiceFactory, PdfPipelineOptions options, ILogger<OcrStage> logger)
    {
        _ocrServiceFactory = ocrServiceFactory ?? throw new ArgumentNullException(nameof(ocrServiceFactory));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public string Name => "ocr";

    public async Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (!_options.DoOcr)
        {
            StageLogger.StageDisabled(_logger);
            context.Set(PipelineContextKeys.OcrResults, new OcrDocumentResult(Array.Empty<OcrBlockResult>()));
            context.Set(PipelineContextKeys.OcrCompleted, true);
            return;
        }

        if (!context.TryGet<PageImageStore>(PipelineContextKeys.PageImageStore, out var store))
        {
            throw new InvalidOperationException("Pipeline context does not contain a page image store.");
        }

        var pages = context.GetRequired<IReadOnlyList<PageReference>>(PipelineContextKeys.PageSequence);
        if (pages.Count == 0)
        {
            StageLogger.NoPages(_logger);
            context.Set(PipelineContextKeys.OcrResults, new OcrDocumentResult(Array.Empty<OcrBlockResult>()));
            context.Set(PipelineContextKeys.OcrCompleted, true);
            return;
        }

        var layoutItems = context.TryGet<IReadOnlyList<LayoutItem>>(PipelineContextKeys.LayoutItems, out var layout)
            ? layout
            : Array.Empty<LayoutItem>();

        using var service = _ocrServiceFactory.Create(_options.Ocr);

        var blocks = new List<OcrBlockResult>();
        foreach (var page in pages)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var pageImage = store.Rent(page);
            var pageArea = pageImage.BoundingBox.Area;
            var pageItems = layoutItems.Where(item => item.Page.PageNumber == page.PageNumber).ToList();
            var processedBlocks = 0;

            foreach (var item in pageItems)
            {
                if (!RequiresOcr(item))
                {
                    continue;
                }

                cancellationToken.ThrowIfCancellationRequested();

                var ratio = ComputeAreaRatio(item.BoundingBox, pageArea);
                if (ratio < _options.Ocr.BitmapAreaThreshold)
                {
                    StageLogger.RegionSkipped(_logger, page.PageNumber, ratio);
                    continue;
                }

                var metadata = CreateLayoutMetadata(item, ratio);
                var request = new OcrRequest(page, pageImage.Bitmap, item.BoundingBox, metadata);
                var stopwatch = Stopwatch.StartNew();
                var lines = await CollectAsync(service, request, cancellationToken).ConfigureAwait(false);
                stopwatch.Stop();

                var block = new OcrBlockResult(page, item.BoundingBox, OcrRegionKind.LayoutBlock, metadata, lines);
                blocks.Add(block);
                processedBlocks++;

                StageLogger.RegionRecognised(
                    _logger,
                    lines.Count,
                    page.PageNumber,
                    $"layout:{item.Kind}",
                    stopwatch.ElapsedMilliseconds);
            }

            if (processedBlocks == 0 && _options.Ocr.ForceFullPageOcr)
            {
                StageLogger.FullPageFallback(_logger, page.PageNumber);
                var metadata = CreateFullPageMetadata(page);
                var request = new OcrRequest(page, pageImage.Bitmap, pageImage.BoundingBox, metadata);
                var stopwatch = Stopwatch.StartNew();
                var lines = await CollectAsync(service, request, cancellationToken).ConfigureAwait(false);
                stopwatch.Stop();

                blocks.Add(new OcrBlockResult(page, pageImage.BoundingBox, OcrRegionKind.FullPage, metadata, lines));
                StageLogger.RegionRecognised(
                    _logger,
                    lines.Count,
                    page.PageNumber,
                    "full_page",
                    stopwatch.ElapsedMilliseconds);
            }
        }

        context.Set(PipelineContextKeys.OcrResults, new OcrDocumentResult(blocks));
        context.Set(PipelineContextKeys.OcrCompleted, true);
    }

    private static bool RequiresOcr(LayoutItem item)
        => item.Kind is LayoutItemKind.Text or LayoutItemKind.Table;

    private static async Task<List<OcrLine>> CollectAsync(IOcrService service, OcrRequest request, CancellationToken cancellationToken)
    {
        var lines = new List<OcrLine>();
        await foreach (var line in service.RecognizeAsync(request, cancellationToken).ConfigureAwait(false))
        {
            lines.Add(line);
        }

        return lines;
    }

    private static double ComputeAreaRatio(BoundingBox region, double pageArea)
    {
        if (region.IsEmpty)
        {
            return 0d;
        }

        if (pageArea <= 0d)
        {
            return 1d;
        }

        var ratio = region.Area / pageArea;
        if (double.IsNaN(ratio) || double.IsInfinity(ratio) || ratio < 0d)
        {
            return 0d;
        }

        return ratio;
    }

    private static ReadOnlyDictionary<string, string> CreateLayoutMetadata(LayoutItem item, double ratio)
    {
        var dictionary = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "layout_block",
            ["docling:layout_kind"] = item.Kind.ToString(),
            ["docling:area_ratio"] = ratio.ToString("F6", CultureInfo.InvariantCulture),
        };

        return new ReadOnlyDictionary<string, string>(dictionary);
    }

    private static ReadOnlyDictionary<string, string> CreateFullPageMetadata(PageReference page)
    {
        var dictionary = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "full_page",
            ["docling:page_number"] = page.PageNumber.ToString(CultureInfo.InvariantCulture),
        };

        return new ReadOnlyDictionary<string, string>(dictionary);
    }

    private static partial class StageLogger
    {
        [LoggerMessage(EventId = 4000, Level = LogLevel.Information, Message = "OCR stage disabled by configuration.")]
        public static partial void StageDisabled(ILogger logger);

        [LoggerMessage(EventId = 4001, Level = LogLevel.Debug, Message = "No pages available for OCR.")]
        public static partial void NoPages(ILogger logger);

        [LoggerMessage(EventId = 4002, Level = LogLevel.Debug, Message = "Skipping OCR for region on page {PageNumber} due to area ratio {Ratio}.")]
        public static partial void RegionSkipped(ILogger logger, int PageNumber, double Ratio);

        [LoggerMessage(EventId = 4003, Level = LogLevel.Information, Message = "Recognised {LineCount} lines on page {PageNumber} ({Source}) in {ElapsedMs} ms.")]
        public static partial void RegionRecognised(ILogger logger, int LineCount, int PageNumber, string Source, long ElapsedMs);

        [LoggerMessage(EventId = 4004, Level = LogLevel.Information, Message = "Executing full-page OCR fallback on page {PageNumber}.")]
        public static partial void FullPageFallback(ILogger logger, int PageNumber);
    }
}
