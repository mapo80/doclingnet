using System;
using System.Collections.Generic;
using System.Linq;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Advanced table structure analyzer for detecting rows, columns, and cell relationships.
/// Assigns row/column spans and detects header patterns.
/// </summary>
internal sealed class TableStructureAnalyzer
{
    private readonly float _rowTolerance = 0.05f;
    private readonly float _colTolerance = 0.05f;

    /// <summary>
    /// Analyze table structure and assign row/column information to cells.
    /// </summary>
    public TableAnalysisResult AnalyzeStructure(IReadOnlyList<TableCellGrouper.GroupedCell> cells)
    {
        if (cells.Count == 0)
        {
            return new TableAnalysisResult
            {
                Rows = 0,
                Columns = 0,
                Headers = new List<HeaderInfo>(),
                Structure = new List<List<StructuredCell>>()
            };
        }

        // Step 1: Detect row groups based on Y coordinates
        var rowGroups = DetectRowGroups(cells);

        // Step 2: Detect column groups based on X coordinates
        var colGroups = DetectColumnGroups(cells);

        // Step 3: Create structured grid
        var structuredGrid = CreateStructuredGrid(cells, rowGroups, colGroups);

        // Step 4: Detect headers and special structures
        var headers = DetectHeaders(structuredGrid);

        // Step 5: Refine spans based on structure analysis
        var refinedGrid = RefineSpans(structuredGrid);

        return new TableAnalysisResult
        {
            Rows = rowGroups.Count,
            Columns = colGroups.Count,
            Headers = headers,
            Structure = refinedGrid
        };
    }

    private List<List<TableCellGrouper.GroupedCell>> DetectRowGroups(IReadOnlyList<TableCellGrouper.GroupedCell> cells)
    {
        var sortedCells = cells.OrderBy(c => c.BoundingBox.cy).ToList();
        var rows = new List<List<TableCellGrouper.GroupedCell>>();
        var currentRow = new List<TableCellGrouper.GroupedCell>();
        var currentRowCenter = sortedCells.Count > 0 ? sortedCells[0].BoundingBox.cy : 0;

        foreach (var cell in sortedCells)
        {
            var cellCenter = cell.BoundingBox.cy;
            var distance = Math.Abs(cellCenter - currentRowCenter);

            if (distance > _rowTolerance && currentRow.Count > 0)
            {
                rows.Add(currentRow);
                currentRow = new List<TableCellGrouper.GroupedCell>();
                currentRowCenter = cellCenter;
            }

            currentRow.Add(cell);
        }

        if (currentRow.Count > 0)
        {
            rows.Add(currentRow);
        }

        return rows;
    }

    private List<List<TableCellGrouper.GroupedCell>> DetectColumnGroups(IReadOnlyList<TableCellGrouper.GroupedCell> cells)
    {
        var sortedCells = cells.OrderBy(c => c.BoundingBox.cx).ToList();
        var cols = new List<List<TableCellGrouper.GroupedCell>>();
        var currentCol = new List<TableCellGrouper.GroupedCell>();
        var currentColCenter = sortedCells.Count > 0 ? sortedCells[0].BoundingBox.cx : 0;

        foreach (var cell in sortedCells)
        {
            var cellCenter = cell.BoundingBox.cx;
            var distance = Math.Abs(cellCenter - currentColCenter);

            if (distance > _colTolerance && currentCol.Count > 0)
            {
                cols.Add(currentCol);
                currentCol = new List<TableCellGrouper.GroupedCell>();
                currentColCenter = cellCenter;
            }

            currentCol.Add(cell);
        }

        if (currentCol.Count > 0)
        {
            cols.Add(currentCol);
        }

        return cols;
    }

    private List<List<StructuredCell>> CreateStructuredGrid(
        IReadOnlyList<TableCellGrouper.GroupedCell> cells,
        List<List<TableCellGrouper.GroupedCell>> rowGroups,
        List<List<TableCellGrouper.GroupedCell>> colGroups)
    {
        var grid = new List<List<StructuredCell>>();

        for (int row = 0; row < rowGroups.Count; row++)
        {
            var gridRow = new List<StructuredCell>();

            for (int col = 0; col < colGroups.Count; col++)
            {
                var cell = FindCellAtPosition(rowGroups[row], colGroups[col], row, col);
                if (cell != null)
                {
                    gridRow.Add(new StructuredCell
                    {
                        OriginalCell = cell,
                        GridRow = row,
                        GridCol = col,
                        RowSpan = cell.RowSpan,
                        ColSpan = cell.ColSpan
                    });
                }
                else
                {
                    gridRow.Add(new StructuredCell
                    {
                        GridRow = row,
                        GridCol = col,
                        IsEmpty = true
                    });
                }
            }

            grid.Add(gridRow);
        }

        return grid;
    }

    private TableCellGrouper.GroupedCell? FindCellAtPosition(
        List<TableCellGrouper.GroupedCell> rowGroup,
        List<TableCellGrouper.GroupedCell> colGroup,
        int rowIndex,
        int colIndex)
    {
        // Find cells that belong to both row and column groups
        foreach (var cell in rowGroup)
        {
            if (colGroup.Contains(cell))
            {
                return cell;
            }
        }
        return null;
    }

    private List<HeaderInfo> DetectHeaders(List<List<StructuredCell>> grid)
    {
        var headers = new List<HeaderInfo>();

        if (grid.Count == 0) return headers;

        // Detect column headers (first row)
        var firstRow = grid[0];
        var hasColumnHeaders = firstRow.Any(cell => !cell.IsEmpty &&
            (cell.OriginalCell?.CellType == "header" || cell.OriginalCell?.Confidence > 0.8f));

        if (hasColumnHeaders)
        {
            headers.Add(new HeaderInfo
            {
                Type = HeaderType.ColumnHeader,
                StartRow = 0,
                EndRow = 0,
                StartCol = 0,
                EndCol = grid[0].Count - 1
            });
        }

        // Detect row headers (first column)
        var firstCol = grid.Select(row => row.Count > 0 ? row[0] : null)
                          .Where(cell => cell != null && !cell.IsEmpty)
                          .ToList();

        var hasRowHeaders = firstCol.Any(cell => !cell.IsEmpty &&
            (cell.OriginalCell?.CellType == "header" || cell.OriginalCell?.Confidence > 0.8f));

        if (hasRowHeaders)
        {
            headers.Add(new HeaderInfo
            {
                Type = HeaderType.RowHeader,
                StartRow = 0,
                EndRow = grid.Count - 1,
                StartCol = 0,
                EndCol = 0
            });
        }

        return headers;
    }

    private List<List<StructuredCell>> RefineSpans(List<List<StructuredCell>> grid)
    {
        // Refine row spans based on vertical alignment
        for (int col = 0; col < grid[0].Count; col++)
        {
            for (int row = 0; row < grid.Count; row++)
            {
                var cell = grid[row][col];
                if (cell.IsEmpty || cell.OriginalCell == null) continue;

                // Look for vertical alignment in subsequent rows
                var span = 1;
                for (int nextRow = row + 1; nextRow < grid.Count; nextRow++)
                {
                    if (col < grid[nextRow].Count &&
                        grid[nextRow][col].IsEmpty &&
                        CanExtendVertically(cell, grid[nextRow][col]))
                    {
                        span++;
                    }
                    else
                    {
                        break;
                    }
                }

                if (span > 1)
                {
                    cell.RowSpan = span;
                    // Mark spanned cells as empty
                    for (int spanRow = 1; spanRow < span; spanRow++)
                    {
                        grid[row + spanRow][col].IsEmpty = true;
                    }
                }
            }
        }

        // Refine column spans based on horizontal alignment
        for (int row = 0; row < grid.Count; row++)
        {
            for (int col = 0; col < grid[row].Count; col++)
            {
                var cell = grid[row][col];
                if (cell.IsEmpty || cell.OriginalCell == null) continue;

                // Look for horizontal alignment in subsequent columns
                var span = 1;
                for (int nextCol = col + 1; nextCol < grid[row].Count; nextCol++)
                {
                    if (grid[row][nextCol].IsEmpty &&
                        CanExtendHorizontally(cell, grid[row][nextCol]))
                    {
                        span++;
                    }
                    else
                    {
                        break;
                    }
                }

                if (span > 1)
                {
                    cell.ColSpan = span;
                    // Mark spanned cells as empty
                    for (int spanCol = 1; spanCol < span; spanCol++)
                    {
                        grid[row][col + spanCol].IsEmpty = true;
                    }
                }
            }
        }

        return grid;
    }

    private static bool CanExtendVertically(StructuredCell cell, StructuredCell target)
    {
        // Check if cells can be part of a vertical span
        var heightDiff = Math.Abs(cell.OriginalCell!.BoundingBox.h - target.OriginalCell?.BoundingBox.h ?? 0);
        return heightDiff < 0.1f; // Similar heights
    }

    private static bool CanExtendHorizontally(StructuredCell cell, StructuredCell target)
    {
        // Check if cells can be part of a horizontal span
        var widthDiff = Math.Abs(cell.OriginalCell!.BoundingBox.w - target.OriginalCell?.BoundingBox.w ?? 0);
        return widthDiff < 0.1f; // Similar widths
    }

    /// <summary>
    /// Result of table structure analysis.
    /// </summary>
    public sealed class TableAnalysisResult
    {
        public int Rows { get; set; }
        public int Columns { get; set; }
        public List<HeaderInfo> Headers { get; set; } = new();
        public List<List<StructuredCell>> Structure { get; set; } = new();
    }

    /// <summary>
    /// Structured cell with grid position and span information.
    /// </summary>
    public sealed class StructuredCell
    {
        public TableCellGrouper.GroupedCell? OriginalCell { get; set; }
        public int GridRow { get; set; }
        public int GridCol { get; set; }
        public int RowSpan { get; set; } = 1;
        public int ColSpan { get; set; } = 1;
        public bool IsEmpty { get; set; }
    }

    /// <summary>
    /// Information about header regions in the table.
    /// </summary>
    public sealed class HeaderInfo
    {
        public HeaderType Type { get; set; }
        public int StartRow { get; set; }
        public int EndRow { get; set; }
        public int StartCol { get; set; }
        public int EndCol { get; set; }
    }

    /// <summary>
    /// Types of headers that can be detected.
    /// </summary>
    public enum HeaderType
    {
        RowHeader,
        ColumnHeader,
        TableHeader
    }
}