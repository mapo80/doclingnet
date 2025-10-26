namespace TableFormerSdk.Models;

/// <summary>
/// Represents a table cell or region detected by TableFormer
/// </summary>
public sealed class TableRegion
{
    public TableRegion(float x, float y, float width, float height, string cellType)
    {
        X = x;
        Y = y;
        Width = width;
        Height = height;
        CellType = cellType ?? "table_cell";
    }

    /// <summary>
    /// X coordinate (top-left)
    /// </summary>
    public float X { get; }

    /// <summary>
    /// Y coordinate (top-left)
    /// </summary>
    public float Y { get; }

    /// <summary>
    /// Width of the region
    /// </summary>
    public float Width { get; }

    /// <summary>
    /// Height of the region
    /// </summary>
    public float Height { get; }

    /// <summary>
    /// Type of cell (e.g., "table_cell", "header")
    /// </summary>
    public string CellType { get; }

    public override string ToString() =>
        $"TableRegion({X:F1}, {Y:F1}, {Width:F1}x{Height:F1}) [{CellType}]";
}
