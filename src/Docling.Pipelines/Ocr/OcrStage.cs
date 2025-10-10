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
using Docling.Models.Tables;
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
        var tableStructures = context.TryGet<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures, out var structures)
            ? structures
            : Array.Empty<TableStructure>();

        using var service = _ocrServiceFactory.Create(_options.Ocr);

        var blocks = new List<OcrBlockResult>();
        foreach (var page in pages)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var pageImage = store.Rent(page);
            var pageArea = pageImage.BoundingBox.Area;
            var pageItems = layoutItems.Where(item => item.Page.PageNumber == page.PageNumber).ToList();
            var pageTables = tableStructures.Count > 0
                ? SelectTablesForPage(tableStructures, page.PageNumber)
                : Array.Empty<TableStructure>();
            var processedRegions = 0;

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
                processedRegions++;

                StageLogger.RegionRecognised(
                    _logger,
                    lines.Count,
                    page.PageNumber,
                    $"layout:{item.Kind}",
                    stopwatch.ElapsedMilliseconds);
            }

            if (pageTables.Count > 0)
            {
                processedRegions += await RecognizeTableCellsAsync(
                    service,
                    page,
                    pageImage,
                    pageTables,
                    pageArea,
                    blocks,
                    cancellationToken).ConfigureAwait(false);
            }

            var fallbackReason = DetermineFullPageFallbackReason(
                _options.Ocr.ForceFullPageOcr,
                pageItems,
                pageTables,
                processedRegions);

            if (fallbackReason is not null)
            {
                StageLogger.FullPageFallback(_logger, page.PageNumber, fallbackReason);
                var metadata = CreateFullPageMetadata(page, fallbackReason);
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

    private static string? DetermineFullPageFallbackReason(
        bool forceFullPageOcr,
        List<LayoutItem> layoutItems,
        IReadOnlyList<TableStructure> tableStructures,
        int processedRegions)
    {
        if (processedRegions > 0)
        {
            return forceFullPageOcr ? "force_full_page_ocr" : null;
        }

        if (forceFullPageOcr)
        {
            return "force_due_to_empty_regions";
        }

        if (layoutItems.Count == 0 && tableStructures.Count == 0)
        {
            return "no_regions_detected";
        }

        return "regions_filtered";
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

    private async Task<int> RecognizeTableCellsAsync(
        IOcrService service,
        PageReference page,
        PageImage pageImage,
        IReadOnlyList<TableStructure> tableStructures,
        double pageArea,
        List<OcrBlockResult> blocks,
        CancellationToken cancellationToken)
    {
        var processed = 0;

        for (var tableIndex = 0; tableIndex < tableStructures.Count; tableIndex++)
        {
            var table = tableStructures[tableIndex];
            foreach (var placement in EnumerateTableCells(table))
            {
                cancellationToken.ThrowIfCancellationRequested();

                var cellRegion = placement.Cell.BoundingBox;
                if (cellRegion.IsEmpty)
                {
                    continue;
                }

                var ratio = ComputeAreaRatio(cellRegion, pageArea);
                if (ratio < _options.Ocr.BitmapAreaThreshold)
                {
                    StageLogger.RegionSkipped(_logger, page.PageNumber, ratio);
                    continue;
                }

                var metadata = CreateTableCellMetadata(tableIndex, placement, ratio);
                var request = new OcrRequest(page, pageImage.Bitmap, cellRegion, metadata);
                var stopwatch = Stopwatch.StartNew();
                var lines = await CollectAsync(service, request, cancellationToken).ConfigureAwait(false);
                stopwatch.Stop();

                blocks.Add(new OcrBlockResult(page, placement.Cell.BoundingBox, OcrRegionKind.TableCell, metadata, lines));
                processed++;

                StageLogger.RegionRecognised(
                    _logger,
                    lines.Count,
                    page.PageNumber,
                    $"table_cell:r{placement.RowIndex}c{placement.ColumnIndex}",
                    stopwatch.ElapsedMilliseconds);
            }
        }

        return processed;
    }

    private static IReadOnlyList<TableStructure> SelectTablesForPage(IReadOnlyList<TableStructure> tables, int pageNumber)
    {
        if (tables.Count == 0)
        {
            return Array.Empty<TableStructure>();
        }

        var pageTables = new List<TableStructure>();
        for (var i = 0; i < tables.Count; i++)
        {
            var table = tables[i];
            if (table.Page.PageNumber == pageNumber)
            {
                pageTables.Add(table);
            }
        }

        return pageTables.Count == 0 ? Array.Empty<TableStructure>() : pageTables;
    }

    private static IEnumerable<TableCellPlacement> EnumerateTableCells(TableStructure table)
    {
        if (table.RowCount <= 0 || table.ColumnCount <= 0)
        {
            yield break;
        }

        if (table.Cells.Count == 0)
        {
            yield break;
        }

        var occupancy = CreateOccupancy(table.RowCount, table.ColumnCount);

        foreach (var cell in table.Cells)
        {
            if (!TryFindNextAvailable(occupancy, out var rowIndex, out var columnIndex))
            {
                yield break;
            }

            var rowSpan = NormalizeSpan(cell.RowSpan, table.RowCount - rowIndex);
            var columnSpan = NormalizeSpan(cell.ColumnSpan, table.ColumnCount - columnIndex);

            MarkOccupied(occupancy, rowIndex, columnIndex, rowSpan, columnSpan);

            if (rowSpan <= 0 || columnSpan <= 0)
            {
                continue;
            }

            yield return new TableCellPlacement(cell, rowIndex, columnIndex, rowSpan, columnSpan);
        }
    }

    private static int NormalizeSpan(int span, int remaining)
    {
        if (remaining <= 0)
        {
            return 0;
        }

        if (span <= 0)
        {
            return Math.Min(1, remaining);
        }

        return Math.Min(span, remaining);
    }

    private static bool TryFindNextAvailable(bool[][] occupancy, out int rowIndex, out int columnIndex)
    {
        var rows = occupancy.Length;
        var columns = rows > 0 ? occupancy[0].Length : 0;

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < columns; c++)
            {
                if (!occupancy[r][c])
                {
                    rowIndex = r;
                    columnIndex = c;
                    return true;
                }
            }
        }

        rowIndex = -1;
        columnIndex = -1;
        return false;
    }

    private static void MarkOccupied(bool[][] occupancy, int startRow, int startColumn, int rowSpan, int columnSpan)
    {
        var rows = occupancy.Length;
        var columns = rows > 0 ? occupancy[0].Length : 0;

        var endRow = Math.Min(startRow + rowSpan, rows);
        var endColumn = Math.Min(startColumn + columnSpan, columns);

        for (var r = startRow; r < endRow; r++)
        {
            for (var c = startColumn; c < endColumn; c++)
            {
                occupancy[r][c] = true;
            }
        }
    }

    private static bool[][] CreateOccupancy(int rows, int columns)
    {
        if (rows <= 0 || columns <= 0)
        {
            return Array.Empty<bool[]>();
        }

        var occupancy = new bool[rows][];
        for (var r = 0; r < rows; r++)
        {
            occupancy[r] = new bool[columns];
        }

        return occupancy;
    }

    private static ReadOnlyDictionary<string, string> CreateTableCellMetadata(
        int tableIndex,
        TableCellPlacement placement,
        double ratio)
    {
        var dictionary = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "table_cell",
            ["docling:table_index"] = tableIndex.ToString(CultureInfo.InvariantCulture),
            ["docling:table_row_index"] = placement.RowIndex.ToString(CultureInfo.InvariantCulture),
            ["docling:table_column_index"] = placement.ColumnIndex.ToString(CultureInfo.InvariantCulture),
            ["docling:table_row_span"] = placement.RowSpan.ToString(CultureInfo.InvariantCulture),
            ["docling:table_column_span"] = placement.ColumnSpan.ToString(CultureInfo.InvariantCulture),
            ["docling:area_ratio"] = ratio.ToString("F6", CultureInfo.InvariantCulture),
        };

        return new ReadOnlyDictionary<string, string>(dictionary);
    }

    private readonly record struct TableCellPlacement(TableCell Cell, int RowIndex, int ColumnIndex, int RowSpan, int ColumnSpan);

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

    private static ReadOnlyDictionary<string, string> CreateFullPageMetadata(PageReference page, string reason)
    {
        var dictionary = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "full_page",
            ["docling:page_number"] = page.PageNumber.ToString(CultureInfo.InvariantCulture),
            ["docling:fallback_reason"] = reason,
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

        [LoggerMessage(EventId = 4004, Level = LogLevel.Information, Message = "Executing full-page OCR fallback on page {PageNumber} (reason: {Reason}).")]
        public static partial void FullPageFallback(ILogger logger, int PageNumber, string Reason);
    }
}
