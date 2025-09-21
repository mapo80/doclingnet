using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Geometry;
using Docling.Models.Layout;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Options;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Pipelines.Tables;

/// <summary>
/// Pipeline stage that invokes the configured table structure service for each detected table layout item.
/// </summary>
public sealed partial class TableStructureInferenceStage : IPipelineStage
{
    private readonly ITableStructureService _tableStructureService;
    private readonly PdfPipelineOptions _options;
    private readonly ILogger<TableStructureInferenceStage> _logger;

    public TableStructureInferenceStage(
        ITableStructureService tableStructureService,
        PdfPipelineOptions options,
        ILogger<TableStructureInferenceStage> logger)
    {
        _tableStructureService = tableStructureService ?? throw new ArgumentNullException(nameof(tableStructureService));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public string Name => "table_structure";

    public async Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        context.Set(PipelineContextKeys.TableStructures, Array.Empty<TableStructure>());

        if (!_options.DoTableStructure)
        {
            StageLogger.StageDisabled(_logger);
            return;
        }

        if (!context.TryGet<PageImageStore>(PipelineContextKeys.PageImageStore, out var store))
        {
            throw new InvalidOperationException("Pipeline context does not contain a page image store.");
        }

        var layoutItems = context.TryGet<IReadOnlyList<LayoutItem>>(PipelineContextKeys.LayoutItems, out var layout)
            ? layout
            : Array.Empty<LayoutItem>();

        var tables = layoutItems.Where(item => item.Kind == LayoutItemKind.Table).ToList();
        if (tables.Count == 0)
        {
            StageLogger.NoTables(_logger);
            return;
        }

        var structures = new List<TableStructure>(tables.Count);
        foreach (var table in tables)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (!store.Contains(table.Page))
            {
                StageLogger.PageImageMissing(_logger, table.Page.PageNumber);
                continue;
            }

            using var pageImage = store.Rent(table.Page);

            var rasterized = RasterizeTable(pageImage, table.BoundingBox);
            if (rasterized.IsEmpty)
            {
                StageLogger.TableSkipped(_logger, table.Page.PageNumber, table.BoundingBox);
                continue;
            }

            var request = new TableStructureRequest(table.Page, table.BoundingBox, rasterized);
            try
            {
                var structure = await _tableStructureService
                    .InferStructureAsync(request, cancellationToken)
                    .ConfigureAwait(false);
                structures.Add(structure);
                StageLogger.TableProcessed(_logger, table.Page.PageNumber, structure.Cells.Count);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                StageLogger.TableFailed(_logger, table.Page.PageNumber, ex);
                throw;
            }
        }

        if (structures.Count > 0)
        {
            context.Set(PipelineContextKeys.TableStructures, structures);
        }
    }

    private static ReadOnlyMemory<byte> RasterizeTable(PageImage pageImage, BoundingBox bounds)
    {
        ArgumentNullException.ThrowIfNull(pageImage);

        var left = Math.Max(0f, (float)bounds.Left);
        var top = Math.Max(0f, (float)bounds.Top);
        var right = Math.Min(pageImage.Width, (float)bounds.Right);
        var bottom = Math.Min(pageImage.Height, (float)bounds.Bottom);

        if (right <= left || bottom <= top)
        {
            return ReadOnlyMemory<byte>.Empty;
        }

        var width = Math.Max(1, (int)Math.Ceiling(right - left));
        var height = Math.Max(1, (int)Math.Ceiling(bottom - top));

        using var cropped = new SKBitmap(width, height, pageImage.Bitmap.ColorType, pageImage.Bitmap.AlphaType);
        using (var canvas = new SKCanvas(cropped))
        {
            var source = new SKRect(left, top, right, bottom);
            var destination = new SKRect(0, 0, width, height);
            canvas.DrawBitmap(pageImage.Bitmap, source, destination);
            canvas.Flush();
        }

        using var snapshot = SKImage.FromBitmap(cropped);
        if (snapshot is null)
        {
            return ReadOnlyMemory<byte>.Empty;
        }

        using var encoded = snapshot.Encode(SKEncodedImageFormat.Png, 100);
        if (encoded is null || encoded.Size == 0)
        {
            return ReadOnlyMemory<byte>.Empty;
        }

        return encoded.ToArray();
    }

    private static partial class StageLogger
    {
        [LoggerMessage(EventId = 3600, Level = LogLevel.Information, Message = "Table structure stage disabled; skipping inference.")]
        public static partial void StageDisabled(ILogger logger);

        [LoggerMessage(EventId = 3601, Level = LogLevel.Debug, Message = "No table layout items detected; skipping table structure inference.")]
        public static partial void NoTables(ILogger logger);

        [LoggerMessage(EventId = 3602, Level = LogLevel.Warning, Message = "Table structure inference skipped for page {Page} because the bounding box did not intersect the page image (bounds: {Bounds}).")]
        public static partial void TableSkipped(ILogger logger, int page, BoundingBox bounds);

        [LoggerMessage(EventId = 3603, Level = LogLevel.Warning, Message = "Page image for page {Page} not present in the cache; unable to infer table structure.")]
        public static partial void PageImageMissing(ILogger logger, int page);

        [LoggerMessage(EventId = 3604, Level = LogLevel.Information, Message = "Table structure inferred for page {Page} with {CellCount} cells.")]
        public static partial void TableProcessed(ILogger logger, int page, int cellCount);

        [LoggerMessage(EventId = 3605, Level = LogLevel.Error, Message = "Table structure inference failed for page {Page}.")]
        public static partial void TableFailed(ILogger logger, int page, Exception exception);
    }
}
