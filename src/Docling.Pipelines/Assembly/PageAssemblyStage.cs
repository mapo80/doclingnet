using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Models.Ocr;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Microsoft.Extensions.Logging;

namespace Docling.Pipelines.Assembly;

/// <summary>
/// Pipeline stage responsible for projecting intermediate pipeline artefacts
/// (layout blocks, OCR lines, table structures) into <see cref="DoclingDocument"/> items.
/// </summary>
public sealed partial class PageAssemblyStage : IPipelineStage
{
    private readonly ILogger<PageAssemblyStage> _logger;

    public PageAssemblyStage(ILogger<PageAssemblyStage> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public string Name => "page_assembly";

    public Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        var pages = context.TryGet<IReadOnlyList<PageReference>>(PipelineContextKeys.PageSequence, out var sequence)
            ? sequence
            : Array.Empty<PageReference>();
        if (pages.Count == 0)
        {
            StageLogger.NoPages(_logger);
            context.Set(PipelineContextKeys.DocumentAssemblyCompleted, true);
            return Task.CompletedTask;
        }

        var layoutItems = context.TryGet<IReadOnlyList<LayoutItem>>(PipelineContextKeys.LayoutItems, out var layout)
            ? layout
            : Array.Empty<LayoutItem>();
        var ocrResult = context.TryGet<OcrDocumentResult>(PipelineContextKeys.OcrResults, out var ocr)
            ? ocr
            : new OcrDocumentResult(Array.Empty<OcrBlockResult>());
        var tableStructures = context.TryGet<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures, out var tables)
            ? tables
            : Array.Empty<TableStructure>();

        var sourceId = context.TryGet<string>(PipelineContextKeys.DocumentId, out var documentId) &&
                       !string.IsNullOrWhiteSpace(documentId)
            ? documentId!
            : "docling";

        var document = new DoclingDocument(sourceId, pages, documentId: documentId);
        var builder = new DoclingDocumentBuilder(document);

        var layoutByPage = layoutItems
            .GroupBy(item => item.Page.PageNumber)
            .ToDictionary(
                group => group.Key,
                group => group
                    .OrderBy(item => item.BoundingBox.Top)
                    .ThenBy(item => item.BoundingBox.Left)
                    .ToList());

        var ocrByPage = ocrResult.Blocks
            .GroupBy(block => block.Page.PageNumber)
            .ToDictionary(group => group.Key, group => group.ToList());

        var tableStructuresByPage = BuildTableStructureIndex(tableStructures);
        var tableCellTexts = BuildTableCellTextIndex(ocrResult);

        foreach (var page in pages)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var pageItems = layoutByPage.TryGetValue(page.PageNumber, out var items)
                ? items
                : new List<LayoutItem>();
            var pageBlocks = ocrByPage.TryGetValue(page.PageNumber, out var blocks)
                ? blocks
                : new List<OcrBlockResult>();

            var layoutBlocks = pageBlocks.Where(block => block.Kind == OcrRegionKind.LayoutBlock).ToList();
            var layoutLookup = layoutBlocks.ToDictionary(block => block.Region, block => block);

            var pageTableStructures = tableStructuresByPage.TryGetValue(page.PageNumber, out var structures)
                ? structures
                : new List<TableStructurePlacement>();

            var emittedTextualContent = false;
            var captionAnchors = new List<DocItem>();

            foreach (var item in pageItems)
            {
                switch (item.Kind)
                {
                    case LayoutItemKind.Text:
                    {
                        var block = ResolveLayoutBlock(item, layoutLookup);
                        if (block is null || block.Lines.Count == 0)
                        {
                            continue;
                        }

                        var collapsedText = ComposeText(block.Lines, " ");
                        if (string.IsNullOrWhiteSpace(collapsedText))
                        {
                            continue;
                        }

                        var classification = ClassifyTextBlock(collapsedText);
                        if (classification == TextBlockClassification.Caption)
                        {
                            var target = FindCaptionTarget(item, captionAnchors);
                            var caption = BuildCaptionItem(item, block, collapsedText, target);
                            if (caption is not null)
                            {
                                builder.AddItem(caption, CreateTextProvenance(caption, caption.Text));
                                emittedTextualContent = true;
                            }
                        }
                        else
                        {
                            var paragraphText = ComposeText(block.Lines);
                            var paragraph = BuildParagraphItem(item, block, paragraphText);
                            if (paragraph is not null)
                            {
                                builder.AddItem(paragraph, CreateTextProvenance(paragraph, paragraph.Text));
                                emittedTextualContent = true;
                            }
                        }

                        break;
                    }
                    case LayoutItemKind.Table:
                    {
                        var table = BuildTableItem(
                            item,
                            page,
                            pageTableStructures,
                            tableCellTexts);

                        if (table is not null)
                        {
                            builder.AddItem(table, CreateRegionProvenance(table));
                            captionAnchors.Add(table);
                        }

                        break;
                    }
                    case LayoutItemKind.Figure:
                    {
                        var picture = BuildPictureItem(item);
                        if (picture is not null)
                        {
                            builder.AddItem(picture, CreateRegionProvenance(picture));
                            captionAnchors.Add(picture);
                        }

                        break;
                    }
                }
            }

            if (!emittedTextualContent)
            {
                var fullPage = pageBlocks.FirstOrDefault(block => block.Kind == OcrRegionKind.FullPage);
                if (fullPage is not null)
                {
                    var fallback = BuildFallbackParagraph(page, fullPage);
                    if (fallback is not null)
                    {
                        builder.AddItem(fallback, CreateTextProvenance(fallback, fallback.Text));
                    }
                }
            }
        }

        context.Set(PipelineContextKeys.Document, document);
        context.Set(PipelineContextKeys.DocumentAssemblyCompleted, true);

        StageLogger.DocumentAssembled(_logger, document.Items.Count);
        return Task.CompletedTask;
    }

    private static ParagraphItem? BuildParagraphItem(LayoutItem layoutItem, OcrBlockResult block, string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        var metadata = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "layout_block",
            ["docling:layout_kind"] = layoutItem.Kind.ToString(),
            ["docling:line_count"] = block.Lines.Count,
        };

        return new ParagraphItem(
            layoutItem.Page,
            layoutItem.BoundingBox,
            text,
            metadata: metadata);
    }

    private static CaptionItem? BuildCaptionItem(LayoutItem layoutItem, OcrBlockResult block, string text, DocItem? target)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        var metadata = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "layout_block",
            ["docling:layout_kind"] = layoutItem.Kind.ToString(),
            ["docling:line_count"] = block.Lines.Count,
            ["docling:text_role"] = "caption",
        };

        if (target is not null)
        {
            metadata["docling:target_item_id"] = target.Id;
            metadata["docling:target_kind"] = target.Kind.ToString();
        }

        return new CaptionItem(
            layoutItem.Page,
            layoutItem.BoundingBox,
            text,
            target?.Id,
            metadata: metadata);
    }

    private static ParagraphItem? BuildFallbackParagraph(PageReference page, OcrBlockResult block)
    {
        if (block.Lines.Count == 0)
        {
            return null;
        }

        var text = ComposeText(block.Lines);
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        var metadata = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "full_page",
        };

        return new ParagraphItem(page, block.Region, text, metadata: metadata);
    }

    private static PictureItem? BuildPictureItem(LayoutItem layoutItem)
    {
        if (layoutItem.BoundingBox.IsEmpty)
        {
            return null;
        }

        var metadata = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "layout_figure",
            ["docling:layout_kind"] = layoutItem.Kind.ToString(),
        };

        return new PictureItem(layoutItem.Page, layoutItem.BoundingBox, metadata: metadata);
    }

    private static TableItem? BuildTableItem(
        LayoutItem layoutItem,
        PageReference page,
        List<TableStructurePlacement> structures,
        Dictionary<int, Dictionary<int, Dictionary<(int Row, int Column), string>>> tableCellTexts)
    {
        if (structures.Count == 0)
        {
            return null;
        }

        var placement = SelectStructure(layoutItem.BoundingBox, structures);
        if (placement is null)
        {
            return null;
        }

        var selected = placement.Value;
        structures.Remove(selected);

        var built = TableBuilder.Build(selected.Structure);
        if (built.RowCount == 0 || built.ColumnCount == 0)
        {
            return null;
        }

        var textsForPage = tableCellTexts.TryGetValue(page.PageNumber, out var tableGroups)
            ? tableGroups
            : new Dictionary<int, Dictionary<(int, int), string>>();
        var cellTexts = textsForPage.TryGetValue(selected.Index, out var cells)
            ? cells
            : new Dictionary<(int, int), string>();

        var updatedCells = new List<TableCellItem>(built.Cells.Count);
        foreach (var cell in built.Cells)
        {
            if (!cellTexts.TryGetValue((cell.RowIndex, cell.ColumnIndex), out var text))
            {
                updatedCells.Add(cell);
                continue;
            }

            updatedCells.Add(cell with { Text = text });
        }

        var boundingBox = built.BoundingBox.IsEmpty
            ? layoutItem.BoundingBox
            : built.BoundingBox;

        var metadata = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
        {
            ["docling:source"] = "table",
            ["docling:table_index"] = selected.Index,
            ["docling:cell_count"] = updatedCells.Count,
        };

        if (selected.Structure.DebugArtifact is { } artifact)
        {
            metadata["docling:debug_media_type"] = artifact.MediaType;
            metadata["docling:debug_payload_length"] = artifact.ImageContent.Length;
        }

        return new TableItem(
            selected.Structure.Page,
            boundingBox,
            updatedCells,
            built.RowCount,
            built.ColumnCount,
            metadata: metadata);
    }

    private static DocItemProvenance CreateTextProvenance(DocItem item, string text)
    {
        var length = string.IsNullOrWhiteSpace(text) ? 0 : text.Length;
        return new DocItemProvenance(item.Page.PageNumber, item.BoundingBox, 0, length);
    }

    private static DocItemProvenance CreateRegionProvenance(DocItem item)
    {
        return new DocItemProvenance(item.Page.PageNumber, item.BoundingBox);
    }

    private static TableStructurePlacement? SelectStructure(BoundingBox layoutBounds, List<TableStructurePlacement> structures)
    {
        if (structures.Count == 0)
        {
            return null;
        }

        TableStructurePlacement? best = null;
        double bestScore = 0;
        foreach (var candidate in structures)
        {
            var structureBounds = candidate.BoundingBox;
            if (structureBounds.IsEmpty)
            {
                continue;
            }

            var score = layoutBounds.IntersectionOverUnion(structureBounds);
            if (score > bestScore)
            {
                bestScore = score;
                best = candidate;
            }
        }

        return best ?? structures[0];
    }

    private static TableStructurePlacement CreatePlacement(TableStructure structure, int index)
    {
        var bounds = CalculateBounds(structure);
        return new TableStructurePlacement(structure, index, bounds);
    }

    private static BoundingBox CalculateBounds(TableStructure structure)
    {
        if (structure.Cells.Count == 0)
        {
            return default;
        }

        var bounds = structure.Cells[0].BoundingBox;
        for (var i = 1; i < structure.Cells.Count; i++)
        {
            bounds = bounds.Union(structure.Cells[i].BoundingBox);
        }

        return bounds;
    }

    private static string ComposeText(IReadOnlyList<OcrLine> lines, string separator = "\n")
    {
        if (lines.Count == 0)
        {
            return string.Empty;
        }

        return string.Join(
            separator,
            lines
                .OrderBy(line => line.BoundingBox.Top)
                .ThenBy(line => line.BoundingBox.Left)
                .Select(line => line.Text?.Trim())
                .Where(text => !string.IsNullOrWhiteSpace(text))
                .Select(text => text!));
    }

    private static OcrBlockResult? ResolveLayoutBlock(LayoutItem layoutItem, IReadOnlyDictionary<BoundingBox, OcrBlockResult> layoutLookup)
    {
        if (layoutLookup.TryGetValue(layoutItem.BoundingBox, out var block))
        {
            return block;
        }

        return FindBestLayoutMatch(layoutItem.BoundingBox, layoutLookup.Values);
    }

    private static OcrBlockResult? FindBestLayoutMatch(BoundingBox layoutBounds, IEnumerable<OcrBlockResult> blocks)
    {
        OcrBlockResult? best = null;
        double bestScore = 0;
        foreach (var block in blocks)
        {
            var score = layoutBounds.IntersectionOverUnion(block.Region);
            if (score > bestScore)
            {
                bestScore = score;
                best = block;
            }
        }

        return best;
    }

    private static DocItem? FindCaptionTarget(LayoutItem caption, IReadOnlyList<DocItem> anchors)
    {
        DocItem? best = null;
        double bestDistance = double.MaxValue;

        foreach (var anchor in anchors)
        {
            if (anchor.Page.PageNumber != caption.Page.PageNumber)
            {
                continue;
            }

            var overlap = Math.Min(caption.BoundingBox.Right, anchor.BoundingBox.Right) -
                          Math.Max(caption.BoundingBox.Left, anchor.BoundingBox.Left);
            if (overlap <= 0)
            {
                continue;
            }

            var distance = CalculateVerticalDistance(caption.BoundingBox, anchor.BoundingBox);
            var allowed = Math.Max(120d, anchor.BoundingBox.Height * 0.6);
            if (distance > allowed)
            {
                continue;
            }

            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = anchor;
            }
        }

        return best;
    }

    private static double CalculateVerticalDistance(BoundingBox caption, BoundingBox anchor)
    {
        if (caption.Top >= anchor.Bottom)
        {
            return caption.Top - anchor.Bottom;
        }

        if (anchor.Top >= caption.Bottom)
        {
            return anchor.Top - caption.Bottom;
        }

        return 0d;
    }

    private static TextBlockClassification ClassifyTextBlock(string text)
    {
        if (CaptionPrefixRegex.IsMatch(text.Trim()))
        {
            return TextBlockClassification.Caption;
        }

        return TextBlockClassification.Paragraph;
    }

    private static Dictionary<int, List<TableStructurePlacement>> BuildTableStructureIndex(IReadOnlyList<TableStructure> structures)
    {
        var index = new Dictionary<int, List<TableStructurePlacement>>();
        for (var i = 0; i < structures.Count; i++)
        {
            var structure = structures[i];
            var list = index.TryGetValue(structure.Page.PageNumber, out var existing)
                ? existing
                : (index[structure.Page.PageNumber] = new List<TableStructurePlacement>());

            list.Add(CreatePlacement(structure, list.Count));
        }

        return index;
    }

    private static Dictionary<int, Dictionary<int, Dictionary<(int Row, int Column), string>>> BuildTableCellTextIndex(OcrDocumentResult ocr)
    {
        var index = new Dictionary<int, Dictionary<int, Dictionary<(int, int), string>>>();
        foreach (var block in ocr.Blocks.Where(block => block.Kind == OcrRegionKind.TableCell))
        {
            if (!TryReadMetadata(block.Metadata, "docling:table_index", out var tableIndex) ||
                !TryReadMetadata(block.Metadata, "docling:table_row_index", out var rowIndex) ||
                !TryReadMetadata(block.Metadata, "docling:table_column_index", out var columnIndex))
            {
                continue;
            }

            var text = ComposeText(block.Lines);
            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }

            var tablesForPage = index.TryGetValue(block.Page.PageNumber, out var pageTables)
                ? pageTables
                : (index[block.Page.PageNumber] = new Dictionary<int, Dictionary<(int, int), string>>());

            var cells = tablesForPage.TryGetValue(tableIndex, out var tableCells)
                ? tableCells
                : (tablesForPage[tableIndex] = new Dictionary<(int, int), string>());

            cells[(rowIndex, columnIndex)] = text;
        }

        return index;
    }

    private static bool TryReadMetadata(IReadOnlyDictionary<string, string> metadata, string key, out int value)
    {
        value = 0;
        if (!metadata.TryGetValue(key, out var raw))
        {
            return false;
        }

        return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    private static readonly Regex CaptionPrefixRegex = new(
        @"^\s*(fig(?:ure)?|fig\.|table|tab\.|tabla|tabella|abb\.|abbildung|image|img\.|plate)\b",
        RegexOptions.Compiled | RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);

    private enum TextBlockClassification
    {
        Paragraph,
        Caption,
    }

    private readonly record struct TableStructurePlacement(TableStructure Structure, int Index, BoundingBox BoundingBox);

    private static partial class StageLogger
    {
        [LoggerMessage(EventId = 6000, Level = LogLevel.Debug, Message = "No pages available for assembly.")]
        public static partial void NoPages(ILogger logger);

        [LoggerMessage(EventId = 6001, Level = LogLevel.Information, Message = "Assembled document with {ItemCount} items.")]
        public static partial void DocumentAssembled(ILogger logger, int ItemCount);
    }
}
