using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Core.Documents;

/// <summary>
/// Represents a structured table reconstructed from layout and OCR analysis.
/// </summary>
public sealed class TableItem : DocItem
{
    private readonly ReadOnlyCollection<TableCellItem> _cells;
    private ImageRef? _previewImage;

    public TableItem(
        PageReference page,
        BoundingBox boundingBox,
        IReadOnlyList<TableCellItem> cells,
        int rowCount,
        int columnCount,
        string? id = null,
        IEnumerable<string>? tags = null,
        IReadOnlyDictionary<string, object?>? metadata = null,
        DateTimeOffset? createdAt = null)
        : base(DocItemKind.Table, page, boundingBox, id, tags, metadata, createdAt)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rowCount, nameof(rowCount));
        ArgumentOutOfRangeException.ThrowIfNegative(columnCount, nameof(columnCount));

        RowCount = rowCount;
        ColumnCount = columnCount;

        _cells = new ReadOnlyCollection<TableCellItem>(cells?.Count > 0
            ? new List<TableCellItem>(cells)
            : new List<TableCellItem>());

        SetMetadata("row_count", RowCount);
        SetMetadata("column_count", ColumnCount);
    }

    public int RowCount { get; }

    public int ColumnCount { get; }

    public IReadOnlyList<TableCellItem> Cells => _cells;

    public ImageRef? PreviewImage => _previewImage;

    public void SetPreviewImage(ImageRef image)
    {
        ArgumentNullException.ThrowIfNull(image);
        _previewImage = image;
        SetMetadata("docling:preview_image", image.Metadata);
        SetMetadata("docling:preview_media_type", image.MediaType);
        SetMetadata("docling:preview_width", image.Width);
        SetMetadata("docling:preview_height", image.Height);
        SetMetadata("docling:preview_dpi", image.Dpi);
    }

    public void ClearPreviewImage()
    {
        _previewImage = null;
        SetMetadata(
            "docling:preview_image",
            new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase));
        SetMetadata<string?>("docling:preview_media_type", null);
        SetMetadata("docling:preview_width", 0);
        SetMetadata("docling:preview_height", 0);
        SetMetadata("docling:preview_dpi", 0d);
    }
}
