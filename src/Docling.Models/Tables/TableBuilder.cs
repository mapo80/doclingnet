using System;
using System.Collections.Generic;
using System.Linq;
using Docling.Core.Documents;
using Docling.Core.Geometry;

namespace Docling.Models.Tables;

/// <summary>
/// Reconstructs a structured <see cref="TableItem"/> from the raw <see cref="TableStructure"/> output
/// of the TableFormer service.
/// Mirrors the placement and span reconciliation logic from the Python TableBuilder.
/// </summary>
public static class TableBuilder
{
    /// <summary>
    /// Builds a <see cref="TableItem"/> by projecting the supplied <paramref name="structure"/> into a dense grid
    /// and reconciling row/column spans.
    /// </summary>
    /// <param name="structure">TableFormer structure response.</param>
    /// <returns>A populated <see cref="TableItem"/> ready for pipeline consumption.</returns>
    public static TableItem Build(TableStructure structure)
    {
        ArgumentNullException.ThrowIfNull(structure);

        var cells = structure.Cells ?? Array.Empty<TableCell>();
        if (cells.Count == 0)
        {
            return new TableItem(
                structure.Page,
                default,
                Array.Empty<TableCellItem>(),
                rowCount: 0,
                columnCount: 0);
        }

        var orderedCells = cells
            .Where(static cell => !cell.BoundingBox.IsEmpty)
            .OrderBy(static cell => cell.BoundingBox.Top)
            .ThenBy(static cell => cell.BoundingBox.Left)
            .ToList();

        if (orderedCells.Count == 0)
        {
            return new TableItem(
                structure.Page,
                default,
                Array.Empty<TableCellItem>(),
                rowCount: 0,
                columnCount: 0);
        }

        var rowCount = structure.RowCount > 0
            ? structure.RowCount
            : EstimateAxisGroups(orderedCells, static cell => (cell.BoundingBox.Top, cell.BoundingBox.Height));
        var columnCount = structure.ColumnCount > 0
            ? structure.ColumnCount
            : EstimateAxisGroups(orderedCells, static cell => (cell.BoundingBox.Left, cell.BoundingBox.Width));

        if (rowCount <= 0 || columnCount <= 0)
        {
            return new TableItem(
                structure.Page,
                default,
                Array.Empty<TableCellItem>(),
                rowCount: Math.Max(rowCount, 0),
                columnCount: Math.Max(columnCount, 0));
        }

        var placements = PlaceCells(orderedCells, rowCount, columnCount);
        var boundingBox = ComputeBoundingBox(placements);

        return new TableItem(
            structure.Page,
            boundingBox,
            placements,
            rowCount,
            columnCount);
    }

    private static List<TableCellItem> PlaceCells(
        List<TableCell> cells,
        int rowCount,
        int columnCount)
    {
        var occupancy = CreateOccupancy(rowCount, columnCount);
        var placements = new List<TableCellItem>(cells.Count);

        foreach (var cell in cells)
        {
            if (!TryFindNextAvailable(occupancy, out var rowIndex, out var columnIndex))
            {
                break;
            }

            var normalizedRowSpan = NormalizeSpan(cell.RowSpan, rowCount - rowIndex);
            var normalizedColumnSpan = NormalizeSpan(cell.ColumnSpan, columnCount - columnIndex);
            MarkOccupied(occupancy, rowIndex, columnIndex, normalizedRowSpan, normalizedColumnSpan);

            if (normalizedRowSpan <= 0 || normalizedColumnSpan <= 0)
            {
                continue;
            }

            placements.Add(new TableCellItem(
                rowIndex,
                columnIndex,
                normalizedRowSpan,
                normalizedColumnSpan,
                cell.BoundingBox,
                cell.Text));
        }

        return placements;
    }

    private static BoundingBox ComputeBoundingBox(List<TableCellItem> cells)
    {
        if (cells.Count == 0)
        {
            return default;
        }

        var bounds = cells[0].BoundingBox;
        for (var i = 1; i < cells.Count; i++)
        {
            bounds = bounds.Union(cells[i].BoundingBox);
        }

        return bounds;
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

    private static bool[][] CreateOccupancy(int rows, int columns)
    {
        var occupancy = new bool[rows][];
        for (var r = 0; r < rows; r++)
        {
            occupancy[r] = new bool[columns];
        }

        return occupancy;
    }

    private static bool TryFindNextAvailable(bool[][] occupancy, out int rowIndex, out int columnIndex)
    {
        for (var r = 0; r < occupancy.Length; r++)
        {
            var row = occupancy[r];
            for (var c = 0; c < row.Length; c++)
            {
                if (!row[c])
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
        var endRow = Math.Min(startRow + Math.Max(rowSpan, 0), occupancy.Length);
        for (var r = startRow; r < endRow; r++)
        {
            var row = occupancy[r];
            var endColumn = Math.Min(startColumn + Math.Max(columnSpan, 0), row.Length);
            for (var c = startColumn; c < endColumn; c++)
            {
                row[c] = true;
            }
        }
    }

    private static int EstimateAxisGroups(
        List<TableCell> cells,
        Func<TableCell, (double Origin, double Length)> selector)
    {
        if (cells.Count == 0)
        {
            return 0;
        }

        var centers = new List<double>();
        foreach (var cell in cells.OrderBy(cell => selector(cell).Origin))
        {
            var (origin, length) = selector(cell);
            var size = Math.Max(length, 1d);
            var center = origin + (size / 2d);
            var tolerance = Math.Max(size * 0.5d, 1d);

            var match = centers.FindIndex(existing => Math.Abs(existing - center) <= tolerance);
            if (match < 0)
            {
                centers.Add(center);
            }
        }

        return centers.Count;
    }
}
