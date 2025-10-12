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

        // Migliora ordinamento considerando il contenuto e la struttura
        foreach (var pageGroup in layoutByPage.Values)
        {
            ImproveReadingOrder(pageGroup);
        }

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

            // Raggruppa elementi di testo adiacenti per migliorare la struttura
            var textGroups = GroupAdjacentTextBlocks(pageItems, layoutLookup);

            foreach (var group in textGroups)
            {
                if (group.Items.Count == 0)
                {
                    continue;
                }

                // Usa il primo elemento come rappresentante del gruppo per posizione e classificazione
                var representativeItem = group.Items[0];
                var representativeBlock = group.Blocks[0];
                var mergedText = group.MergedText;

                if (string.IsNullOrWhiteSpace(mergedText))
                {
                    continue;
                }

                var classification = ClassifyTextBlock(mergedText);

                if (classification == TextBlockClassification.Caption)
                {
                    var target = FindCaptionTarget(representativeItem, captionAnchors);
                    var caption = BuildCaptionItem(representativeItem, representativeBlock, mergedText, target);
                    if (caption is not null)
                    {
                        builder.AddItem(caption, CreateTextProvenance(caption, caption.Text));
                        emittedTextualContent = true;
                        // Aggiungi tutti gli elementi del gruppo come anchor per le caption
                        foreach (var groupItem in group.Items)
                        {
                            captionAnchors.Add(caption);
                        }
                    }
                }
                else
                {
                    var textItem = BuildTextItem(representativeItem, representativeBlock, mergedText, classification);
                    if (textItem is ParagraphItem paragraphItem)
                    {
                        builder.AddItem(paragraphItem, CreateTextProvenance(paragraphItem, paragraphItem.Text));
                        emittedTextualContent = true;
                        // Aggiungi tutti gli elementi del gruppo come anchor
                        foreach (var groupItem in group.Items)
                        {
                            captionAnchors.Add(paragraphItem);
                        }
                    }
                }
            }

            // Processa elementi di testo non raggruppati (se presenti)
            var processedItems = new HashSet<LayoutItem>(textGroups.SelectMany(g => g.Items));
            foreach (var item in pageItems)
            {
                if (processedItems.Contains(item) || item.Kind != LayoutItemKind.Text)
                {
                    continue;
                }

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
                        captionAnchors.Add(caption);
                    }
                }
                else
                {
                    var paragraphText = ComposeText(block.Lines);
                    var textItem = BuildTextItem(item, block, paragraphText, classification);
                    if (textItem is ParagraphItem paragraphItem)
                    {
                        builder.AddItem(paragraphItem, CreateTextProvenance(paragraphItem, paragraphItem.Text));
                        emittedTextualContent = true;
                        captionAnchors.Add(paragraphItem);
                    }
                }
            }

            // Processa tabelle e figure
            foreach (var item in pageItems)
            {
                switch (item.Kind)
                {
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

    private static DocItem? BuildTextItem(LayoutItem layoutItem, OcrBlockResult block, string text, TextBlockClassification classification)
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
            ["docling:text_classification"] = classification.ToString(),
        };

        // Usa sempre ParagraphItem ma con metadati specifici per il tipo
        if (classification == TextBlockClassification.Title)
        {
            metadata["docling:text_role"] = "title";
        }
        else if (classification == TextBlockClassification.SectionHeader)
        {
            metadata["docling:text_role"] = "section_header";
        }

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

    private static List<TextGroup> GroupAdjacentTextBlocks(List<LayoutItem> textItems, IReadOnlyDictionary<BoundingBox, OcrBlockResult> layoutLookup)
    {
        var groups = new List<TextGroup>();
        var usedItems = new HashSet<LayoutItem>();

        foreach (var item in textItems)
        {
            if (usedItems.Contains(item) || item.Kind != LayoutItemKind.Text)
            {
                continue;
            }

            var group = new TextGroup();
            var queue = new Queue<LayoutItem>();
            queue.Enqueue(item);
            usedItems.Add(item);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                var block = ResolveLayoutBlock(current, layoutLookup);
                if (block != null && block.Lines.Count > 0)
                {
                    var text = ComposeText(block.Lines, " ");
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        group.Items.Add(current);
                        group.Blocks.Add(block);
                        group.Texts.Add(text);
                        group.BoundingBoxes.Add(current.BoundingBox);

                        // Trova elementi adiacenti (stessa riga o righe vicine)
                        var adjacentItems = FindAdjacentTextItems(current, textItems, usedItems);
                        foreach (var adjacent in adjacentItems)
                        {
                            if (!usedItems.Contains(adjacent))
                            {
                                queue.Enqueue(adjacent);
                                usedItems.Add(adjacent);
                            }
                        }
                    }
                }
            }

            if (group.Items.Count > 0)
            {
                groups.Add(group);
            }
        }

        return groups;
    }

    private static List<LayoutItem> FindAdjacentTextItems(LayoutItem current, List<LayoutItem> allItems, HashSet<LayoutItem> usedItems)
    {
        var adjacent = new List<LayoutItem>();
        var currentBox = current.BoundingBox;

        foreach (var item in allItems)
        {
            if (usedItems.Contains(item) || item.Page.PageNumber != current.Page.PageNumber || item.Kind != LayoutItemKind.Text)
            {
                continue;
            }

            var itemBox = item.BoundingBox;

            // Calcola distanza verticale e orizzontale
            var verticalDistance = Math.Min(
                Math.Abs(currentBox.Top - itemBox.Bottom),
                Math.Abs(currentBox.Bottom - itemBox.Top)
            );

            var horizontalOverlap = Math.Max(0,
                Math.Min(currentBox.Right, itemBox.Right) - Math.Max(currentBox.Left, itemBox.Left)
            );

            // Elementi adiacenti se:
            // 1. Si sovrappongono orizzontalmente E distanza verticale < 2 * altezza linea media
            // 2. Oppure distanza verticale molto piccola (< 5px) e distanza orizzontale ragionevole (< 50px)
            var avgLineHeight = Math.Min(currentBox.Height, itemBox.Height);
            var maxVerticalDistance = Math.Max(5, avgLineHeight * 2);

            if ((horizontalOverlap > 0 && verticalDistance < maxVerticalDistance) ||
                (verticalDistance < 5 && Math.Abs(currentBox.Left - itemBox.Left) < 50))
            {
                adjacent.Add(item);
            }
        }

        return adjacent;
    }

    private static string MergeTextGroupTexts(List<string> texts)
    {
        if (texts.Count == 0)
        {
            return string.Empty;
        }

        if (texts.Count == 1)
        {
            return texts[0];
        }

        // Unisci testi preservando la struttura
        var merged = new List<string>();
        var currentParagraph = new List<string>();

        foreach (var text in texts)
        {
            // Se il testo finisce con punteggiatura da fine paragrafo, termina il paragrafo corrente
            var trimmed = text.TrimEnd();
            if (trimmed.EndsWith('.') || trimmed.EndsWith('!') || trimmed.EndsWith('?'))
            {
                currentParagraph.Add(text);
                merged.Add(string.Join(" ", currentParagraph));
                currentParagraph.Clear();
            }
            else
            {
                currentParagraph.Add(text);
            }
        }

        // Aggiungi eventuali testi rimanenti
        if (currentParagraph.Count > 0)
        {
            merged.Add(string.Join(" ", currentParagraph));
        }

        return string.Join("\n\n", merged.Where(t => !string.IsNullOrWhiteSpace(t)));
    }

    private static void ImproveReadingOrder(List<LayoutItem> items)
    {
        if (items.Count <= 1)
        {
            return;
        }

        // Separa per tipo per gestire priorità di lettura
        var textItems = items.Where(i => i.Kind == LayoutItemKind.Text).ToList();
        var tableItems = items.Where(i => i.Kind == LayoutItemKind.Table).ToList();
        var figureItems = items.Where(i => i.Kind == LayoutItemKind.Figure).ToList();

        // Ordinamento speciale per documenti accademici
        var orderedItems = new List<LayoutItem>();

        // 1. Testo prima (in ordine di posizione) - i titoli sono classificati nei metadati
        orderedItems.AddRange(textItems.OrderBy(i => i.BoundingBox.Top).ThenBy(i => i.BoundingBox.Left));

        // 2. Figure e tabelle (in ordine di posizione)
        var visualItems = new List<LayoutItem>();
        visualItems.AddRange(figureItems);
        visualItems.AddRange(tableItems);
        orderedItems.AddRange(visualItems.OrderBy(i => i.BoundingBox.Top).ThenBy(i => i.BoundingBox.Left));

        // 3. Testo rimanente (in ordine di lettura migliorato)
        orderedItems.AddRange(ImproveTextReadingOrder(textItems));

        // Sostituisci la lista originale
        items.Clear();
        items.AddRange(orderedItems);
    }

    private static List<LayoutItem> ImproveTextReadingOrder(List<LayoutItem> textItems)
    {
        if (textItems.Count <= 1)
        {
            return textItems;
        }

        // Crea gruppi di elementi correlati (colonne, paragrafi)
        var columns = new List<List<LayoutItem>>();
        var used = new HashSet<LayoutItem>();

        foreach (var item in textItems)
        {
            if (used.Contains(item))
            {
                continue;
            }

            var column = new List<LayoutItem> { item };
            used.Add(item);

            // Trova elementi nella stessa colonna (posizione X simile)
            var itemCenterX = item.BoundingBox.Left + (item.BoundingBox.Width / 2);

            foreach (var other in textItems)
            {
                if (used.Contains(other))
                {
                    continue;
                }

                var otherCenterX = other.BoundingBox.Left + (other.BoundingBox.Width / 2);
                var xDistance = Math.Abs(itemCenterX - otherCenterX);

                // Se elementi sono nella stessa colonna (distanza X < 100px)
                if (xDistance < 100)
                {
                    column.Add(other);
                    used.Add(other);
                }
            }

            // Ordina elementi nella colonna per posizione Y
            column.Sort((a, b) =>
            {
                var cmp = a.BoundingBox.Top.CompareTo(b.BoundingBox.Top);
                return cmp != 0 ? cmp : a.BoundingBox.Left.CompareTo(b.BoundingBox.Left);
            });

            columns.Add(column);
        }

        // Ordina colonne per posizione X
        columns.Sort((a, b) =>
        {
            var aCenter = a[0].BoundingBox.Left + (a[0].BoundingBox.Width / 2);
            var bCenter = b[0].BoundingBox.Left + (b[0].BoundingBox.Width / 2);
            return aCenter.CompareTo(bCenter);
        });

        // Appiattisci colonne mantenendo l'ordine
        return columns.SelectMany(c => c).ToList();
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
        double bestScore = 0;

        foreach (var anchor in anchors)
        {
            if (anchor.Page.PageNumber != caption.Page.PageNumber)
            {
                continue;
            }

            var overlap = Math.Min(caption.BoundingBox.Right, anchor.BoundingBox.Right) -
                          Math.Max(caption.BoundingBox.Left, anchor.BoundingBox.Left);

            // Calcola distanza verticale
            var distance = CalculateVerticalDistance(caption.BoundingBox, anchor.BoundingBox);

            // Calcola score basato su:
            // 1. Sovrapposizione orizzontale (overlap)
            // 2. Distanza verticale (inversamente proporzionale)
            // 3. Larghezza relativa (caption dovrebbe essere più stretta dell'anchor)

            if (overlap > 0)
            {
                var allowedDistance = Math.Max(120d, anchor.BoundingBox.Height * 0.8);
                if (distance <= allowedDistance)
                {
                    // Score basato su distanza e overlap
                    var distanceScore = Math.Max(0, (allowedDistance - distance) / allowedDistance);
                    var overlapScore = overlap / Math.Max(caption.BoundingBox.Width, anchor.BoundingBox.Width);
                    var widthRatio = Math.Min(caption.BoundingBox.Width, anchor.BoundingBox.Width) /
                                    Math.Max(caption.BoundingBox.Width, anchor.BoundingBox.Width);

                    var totalScore = (distanceScore * 0.4) + (overlapScore * 0.4) + (widthRatio * 0.2);

                    if (totalScore > bestScore)
                    {
                        bestScore = totalScore;
                        best = anchor;
                    }
                }
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
        Title,
        SectionHeader,
    }

    private readonly record struct TableStructurePlacement(TableStructure Structure, int Index, BoundingBox BoundingBox);

    private sealed class TextGroup
    {
        public List<LayoutItem> Items { get; } = new();
        public List<OcrBlockResult> Blocks { get; } = new();
        public List<string> Texts { get; } = new();
        public List<BoundingBox> BoundingBoxes { get; } = new();

        public BoundingBox MergedBoundingBox
        {
            get
            {
                if (BoundingBoxes.Count == 0)
                {
                    return BoundingBox.FromSize(0, 0, 0, 0);
                }

                var result = BoundingBoxes[0];
                for (int i = 1; i < BoundingBoxes.Count; i++)
                {
                    result = result.Union(BoundingBoxes[i]);
                }

                return result;
            }
        }

        public string MergedText => MergeTextGroupTexts(Texts);
    }

    private static partial class StageLogger
    {
        [LoggerMessage(EventId = 6000, Level = LogLevel.Debug, Message = "No pages available for assembly.")]
        public static partial void NoPages(ILogger logger);

        [LoggerMessage(EventId = 6001, Level = LogLevel.Information, Message = "Assembled document with {ItemCount} items.")]
        public static partial void DocumentAssembled(ILogger logger, int ItemCount);
    }
}
