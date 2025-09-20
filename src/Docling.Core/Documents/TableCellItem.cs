using Docling.Core.Geometry;

namespace Docling.Core.Documents;

/// <summary>
/// Represents the placement of a table cell within a reconstructed grid.
/// </summary>
public sealed record TableCellItem(
    int RowIndex,
    int ColumnIndex,
    int RowSpan,
    int ColumnSpan,
    BoundingBox BoundingBox,
    string? Text);
