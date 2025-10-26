using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using Docling.Core.Documents;
using Docling.Export.Imaging;

namespace Docling.Export.Serialization;

/// <summary>
/// Serialises a <see cref="DoclingDocument"/> into markdown text mirroring the behaviour of the Python pipeline.
/// </summary>
public sealed class MarkdownDocSerializer
{
    private readonly MarkdownSerializerOptions _options;

    public MarkdownDocSerializer(MarkdownSerializerOptions? options = null)
    {
        _options = options ?? new MarkdownSerializerOptions();
    }

    public MarkdownSerializationResult Serialize(
        DoclingDocument document,
        IReadOnlyList<ImageExportArtifact>? imageExports = null)
    {
        ArgumentNullException.ThrowIfNull(document);

        var exports = imageExports ?? Array.Empty<ImageExportArtifact>();
        var exportsByTarget = BuildExportLookup(exports);
        var captions = BuildCaptionLookup(document.Items.OfType<CaptionItem>());
        var assetPaths = new Dictionary<string, string>(StringComparer.Ordinal);
        var assets = new List<MarkdownAsset>();
        var builder = new StringBuilder();
        var needsSeparator = false;
        var figureIndex = 0;
        var tableIndex = 0;

        foreach (var item in document.Items)
        {
            switch (item)
            {
                case ParagraphItem paragraph:
                    AppendParagraph(paragraph, builder, ref needsSeparator);
                    break;

                case PictureItem picture:
                    figureIndex++;
                    AppendPicture(picture, figureIndex, builder, captions, exportsByTarget, assets, assetPaths, ref needsSeparator);
                    break;

                case TableItem table:
                    tableIndex++;
                    AppendTable(table, tableIndex, builder, captions, exportsByTarget, assets, assetPaths, ref needsSeparator);
                    break;

                case CaptionItem caption:
                    AppendLooseCaption(caption, builder, ref needsSeparator);
                    break;
            }
        }

        var markdown = builder.ToString().TrimEnd();
        if (markdown.Length > 0)
        {
            markdown += Environment.NewLine;
        }

        var metadata = document.Properties.Count > 0
            ? new Dictionary<string, string>(document.Properties, StringComparer.OrdinalIgnoreCase)
            : new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        return new MarkdownSerializationResult(markdown, assets, metadata);
    }

    private static string FormatLabel(string format, int index)
        => string.Format(CultureInfo.InvariantCulture, format, index);

    private static Dictionary<string, Queue<string>> BuildCaptionLookup(IEnumerable<CaptionItem> captions)
    {
        var lookup = new Dictionary<string, Queue<string>>(StringComparer.Ordinal);
        foreach (var caption in captions)
        {
            if (string.IsNullOrWhiteSpace(caption.TargetItemId))
            {
                continue;
            }

            if (string.IsNullOrWhiteSpace(caption.Text))
            {
                continue;
            }

            if (!lookup.TryGetValue(caption.TargetItemId, out var queue))
            {
                queue = new Queue<string>();
                lookup[caption.TargetItemId] = queue;
            }

            queue.Enqueue(caption.Text.Trim());
        }

        return lookup;
    }

    private static Dictionary<string, List<ImageExportArtifact>> BuildExportLookup(IReadOnlyList<ImageExportArtifact> exports)
    {
        var lookup = new Dictionary<string, List<ImageExportArtifact>>(StringComparer.Ordinal);
        foreach (var export in exports)
        {
            if (string.IsNullOrWhiteSpace(export.TargetItemId))
            {
                continue;
            }

            if (!lookup.TryGetValue(export.TargetItemId, out var list))
            {
                list = new List<ImageExportArtifact>();
                lookup[export.TargetItemId] = list;
            }

            list.Add(export);
        }

        return lookup;
    }

    private static void AppendParagraph(ParagraphItem paragraph, StringBuilder builder, ref bool needsSeparator)
    {
        if (string.IsNullOrWhiteSpace(paragraph.Text))
        {
            return;
        }

        StartBlock(builder, ref needsSeparator);
        builder.AppendLine(paragraph.Text.Trim());
        needsSeparator = true;
    }

    private void AppendPicture(
        PictureItem picture,
        int figureIndex,
        StringBuilder builder,
        Dictionary<string, Queue<string>> captions,
        Dictionary<string, List<ImageExportArtifact>> exportsByTarget,
        List<MarkdownAsset> assets,
        Dictionary<string, string> assetPaths,
        ref bool needsSeparator)
    {
        var captionText = ConsumeCaption(captions, picture.Id);
        var label = FormatLabel(_options.FigureLabelFormat, figureIndex);
        var image = picture.Image ?? FindExportImage(exportsByTarget, picture.Id, ImageExportKind.Picture);
        var enrichedAlt = TryGetAltText(MarkdownAltTextContext.ForPicture(picture, label, captionText, image));
        var altText = ResolveAltText(enrichedAlt, picture.Description, captionText, label);

        StartBlock(builder, ref needsSeparator);

        if (_options.ImageMode == MarkdownImageMode.Embedded && image is not null)
        {
            EmitEmbeddedImage(builder, image, altText);
        }
        else if (_options.ImageMode == MarkdownImageMode.Referenced && image is not null)
        {
            var path = RegisterAsset(image, ImageExportKind.Picture, picture.Id, assets, assetPaths);
            builder.AppendLine(string.Format(CultureInfo.InvariantCulture, "![{0}]({1})", EscapeInline(altText), path));
        }
        else
        {
            builder.AppendLine(string.Format(
                CultureInfo.InvariantCulture,
                "> {0}: {1} (image unavailable)",
                label,
                EscapeInline(captionText ?? altText)));
        }

        EmitCaption(builder, label, captionText, altText);
        needsSeparator = true;
    }

    private void AppendTable(
        TableItem table,
        int tableIndex,
        StringBuilder builder,
        Dictionary<string, Queue<string>> captions,
        Dictionary<string, List<ImageExportArtifact>> exportsByTarget,
        List<MarkdownAsset> assets,
        Dictionary<string, string> assetPaths,
        ref bool needsSeparator)
    {
        var captionText = ConsumeCaption(captions, table.Id);
        var tableLabel = FormatLabel(_options.TableLabelFormat, tableIndex);

        StartBlock(builder, ref needsSeparator);

        if (CanRenderTable(table))
        {
            EmitStructuredTable(builder, table);
        }
        else
        {
            var image = table.PreviewImage ?? FindExportImage(exportsByTarget, table.Id, ImageExportKind.Table);
            var enrichedAlt = TryGetAltText(MarkdownAltTextContext.ForTable(table, tableLabel, captionText, image));
            var altText = ResolveAltText(enrichedAlt, null, captionText, tableLabel);

            if (_options.ImageMode == MarkdownImageMode.Embedded && image is not null)
            {
                EmitEmbeddedImage(builder, image, altText);
            }
            else if (_options.ImageMode == MarkdownImageMode.Referenced && image is not null)
            {
                var path = RegisterAsset(image, ImageExportKind.Table, table.Id, assets, assetPaths);
                builder.AppendLine(string.Format(CultureInfo.InvariantCulture, "![{0}]({1})", EscapeInline(altText), path));
            }
            else
            {
                builder.AppendLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "> {0}: {1} (table preview unavailable)",
                    tableLabel,
                    EscapeInline(captionText ?? altText)));
            }
        }

        EmitCaption(builder, tableLabel, captionText, null);
        needsSeparator = true;
    }

    private static void AppendLooseCaption(CaptionItem caption, StringBuilder builder, ref bool needsSeparator)
    {
        if (!string.IsNullOrWhiteSpace(caption.TargetItemId))
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(caption.Text))
        {
            return;
        }

        StartBlock(builder, ref needsSeparator);
        builder.AppendLine(string.Format(CultureInfo.InvariantCulture, "*{0}*", EscapeInline(caption.Text.Trim())));
        needsSeparator = true;
    }

    private static void EmitCaption(StringBuilder builder, string label, string? captionText, string? fallback)
    {
        var text = string.IsNullOrWhiteSpace(captionText) ? null : captionText.Trim();
        if (text is null && !string.IsNullOrWhiteSpace(fallback))
        {
            text = fallback.Trim();
        }

        builder.AppendLine();
        if (string.IsNullOrWhiteSpace(text))
        {
            builder.AppendLine(string.Format(CultureInfo.InvariantCulture, "*{0}.*", EscapeInline(label)));
        }
        else
        {
            builder.AppendLine(string.Format(CultureInfo.InvariantCulture, "*{0}. {1}*", EscapeInline(label), EscapeInline(text)));
        }
    }

    private static void StartBlock(StringBuilder builder, ref bool needsSeparator)
    {
        if (needsSeparator)
        {
            builder.AppendLine();
        }

        needsSeparator = false;
    }

    private static string? ConsumeCaption(Dictionary<string, Queue<string>> captions, string itemId)
    {
        if (!captions.TryGetValue(itemId, out var queue))
        {
            return null;
        }

        var parts = new List<string>();
        while (queue.Count > 0)
        {
            parts.Add(queue.Dequeue());
        }

        captions.Remove(itemId);
        return parts.Count == 0 ? null : string.Join(" ", parts);
    }

    private string? TryGetAltText(MarkdownAltTextContext context)
    {
        var provider = _options.AltTextProvider;
        if (provider is null)
        {
            return null;
        }

        try
        {
            var value = provider.GetAltText(context);
            return string.IsNullOrWhiteSpace(value) ? null : value.Trim();
        }
        catch (Exception ex) when (!IsCriticalException(ex))
        {
            return null;
        }
    }

    private static string ResolveAltText(string? enriched, string? description, string? captionText, string fallback)
    {
        if (!string.IsNullOrWhiteSpace(enriched))
        {
            return enriched.Trim();
        }

        if (!string.IsNullOrWhiteSpace(description))
        {
            return description.Trim();
        }

        if (!string.IsNullOrWhiteSpace(captionText))
        {
            return captionText.Trim();
        }

        return fallback;
    }

    private static bool IsCriticalException(Exception exception)
    {
        return exception is OutOfMemoryException
            or StackOverflowException
            or AccessViolationException
            or AppDomainUnloadedException
            or ThreadAbortException;
    }

    private static ImageRef? FindExportImage(
        Dictionary<string, List<ImageExportArtifact>> exportsByTarget,
        string itemId,
        ImageExportKind preferredKind)
    {
        if (!exportsByTarget.TryGetValue(itemId, out var exports) || exports.Count == 0)
        {
            return null;
        }

        var match = exports.FirstOrDefault(e => e.Kind == preferredKind) ?? exports[0];
        return match.Image;
    }

    private static void EmitStructuredTable(StringBuilder builder, TableItem table)
    {
        var grid = BuildGrid(table);
        if (grid.Length == 0)
        {
            builder.AppendLine("| |");
            builder.AppendLine("|-|");
            return;
        }

        var hasHeader = table.RowCount > 1 && grid[0].Any(cell => !string.IsNullOrWhiteSpace(cell));
        var header = hasHeader ? grid[0] : CreateDefaultHeader(table.ColumnCount);
        builder.AppendLine(BuildRow(header));
        builder.AppendLine(BuildSeparatorRow(header.Length));

        var startRow = hasHeader ? 1 : 0;
        for (var rowIndex = startRow; rowIndex < grid.Length; rowIndex++)
        {
            builder.AppendLine(BuildRow(grid[rowIndex]));
        }
    }

    private static string[][] BuildGrid(TableItem table)
    {
        if (table.RowCount <= 0 || table.ColumnCount <= 0)
        {
            return Array.Empty<string[]>();
        }

        var grid = new string[table.RowCount][];
        for (var row = 0; row < table.RowCount; row++)
        {
            grid[row] = new string[table.ColumnCount];
            Array.Fill(grid[row], string.Empty);
        }

        // Fill grid with cells, replicating spanned cells across all spanned positions
        // This matches Python docling behavior
        foreach (var cell in table.Cells)
        {
            var cellText = cell.Text?.Trim() ?? string.Empty;

            // Fill all positions covered by this cell's row and column span
            var endRow = Math.Min(cell.RowIndex + cell.RowSpan, table.RowCount);
            var endCol = Math.Min(cell.ColumnIndex + cell.ColumnSpan, table.ColumnCount);

            for (var row = cell.RowIndex; row < endRow; row++)
            {
                for (var col = cell.ColumnIndex; col < endCol; col++)
                {
                    if (row >= 0 && row < table.RowCount && col >= 0 && col < table.ColumnCount)
                    {
                        grid[row][col] = cellText;
                    }
                }
            }
        }

        return grid;
    }

    private static string[] CreateDefaultHeader(int columnCount)
    {
        var header = new string[columnCount];
        for (var i = 0; i < columnCount; i++)
        {
            header[i] = string.Format(CultureInfo.InvariantCulture, "Column {0}", i + 1);
        }

        return header;
    }

    private static string BuildRow(IReadOnlyList<string> cells)
    {
        var escaped = cells.Select(EscapeTableCell);
        return "| " + string.Join(" | ", escaped) + " |";
    }

    private static string BuildSeparatorRow(int columnCount)
    {
        if (columnCount <= 0)
        {
            return "|-|";
        }

        var segments = Enumerable.Repeat("---", columnCount);
        return "| " + string.Join(" | ", segments) + " |";
    }

    private static bool CanRenderTable(TableItem table)
    {
        if (table.RowCount <= 0 || table.ColumnCount <= 0)
        {
            return false;
        }

        // We can now render tables with spans since BuildGrid handles them
        // by replicating cell content across spanned positions
        return true;
    }

    private static void EmitEmbeddedImage(StringBuilder builder, ImageRef image, string altText)
    {
        var base64 = Convert.ToBase64String(image.Data.Span);
        var uri = $"data:{image.MediaType};base64,{base64}";
        builder.AppendLine(string.Format(CultureInfo.InvariantCulture, "![{0}]({1})", EscapeInline(altText), uri));
    }

    private string RegisterAsset(
        ImageRef image,
        ImageExportKind kind,
        string? targetItemId,
        List<MarkdownAsset> assets,
        Dictionary<string, string> assetPaths)
    {
        if (assetPaths.TryGetValue(image.Id, out var existing))
        {
            return existing;
        }

        var extension = GetFileExtension(image.MediaType);
        var fileName = image.Id + extension;
        var path = _options.CombineAssetPath(fileName);
        assets.Add(new MarkdownAsset(path, kind, image, targetItemId));
        assetPaths[image.Id] = path;
        return path;
    }

    private static string GetFileExtension(string mediaType)
    {
        return mediaType switch
        {
            "image/png" => ".png",
            "image/jpeg" => ".jpg",
            "image/jpg" => ".jpg",
            "image/webp" => ".webp",
            "image/gif" => ".gif",
            _ => ".bin",
        };
    }

    private static string EscapeInline(string text)
    {
        return text
            .Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("*", "\\*", StringComparison.Ordinal)
            .Replace("_", "\\_", StringComparison.Ordinal)
            .Replace("[", "\\[", StringComparison.Ordinal)
            .Replace("]", "\\]", StringComparison.Ordinal)
            .Replace("`", "\\`", StringComparison.Ordinal);
    }

    private static string EscapeTableCell(string text)
    {
        var escaped = EscapeInline(text);
        escaped = escaped.Replace("|", "\\|", StringComparison.Ordinal);
        return escaped.Replace("\n", "<br />", StringComparison.Ordinal);
    }
}
