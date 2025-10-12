#if false
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Parser for extracting table structure from Markdown documents.
/// Used to create ground truth data from golden Python CLI outputs.
/// </summary>
internal static class MarkdownTableParser
{
    /// <summary>
    /// Parse table structure from markdown content.
    /// </summary>
    public static IReadOnlyList<MarkdownTableStructure> ParseMarkdownTables(string markdownContent)
    {
        var tables = new List<MarkdownTableStructure>();

        // Split markdown into lines
        var lines = markdownContent.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

        var currentTable = new List<List<string>>();
        var inTable = false;

        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();

            // Check if line is a table separator (contains | and -)
            if (IsTableSeparator(trimmedLine))
            {
                inTable = true;
                continue; // Skip separator lines
            }

            // Check if line is a table row (contains | and data)
            if (IsMarkdownTableRow(trimmedLine))
            {
                if (inTable)
                {
                    var row = ParseMarkdownTableRow(trimmedLine);
                    if (row.Count > 0)
                    {
                        currentTable.Add(row);
                    }
                }
            }
            else if (inTable && currentTable.Count > 0)
            {
                // End of current table
                if (currentTable.Count > 0)
                {
                    tables.Add(CreateMarkdownTableStructure(currentTable));
                }
                currentTable.Clear();
                inTable = false;
            }
        }

        // Add final table if exists
        if (currentTable.Count > 0)
        {
            tables.Add(CreateMarkdownTableStructure(currentTable));
        }

        return tables;
    }

    /// <summary>
    /// Parse a single table from markdown content.
    /// </summary>
    public static MarkdownTableStructure ParseMarkdownTable(string markdownContent)
    {
        var tables = ParseMarkdownTables(markdownContent);

        if (tables.Count == 0)
            throw new ArgumentException("No table found in markdown content");

        if (tables.Count > 1)
            throw new ArgumentException("Multiple tables found in markdown content - use ParseMarkdownTables for multiple tables");

        return tables[0];
    }

    private static bool IsTableSeparator(string line)
    {
        // Must contain | and have at least 3 dashes or colons
        return line.Contains('|') &&
               Regex.IsMatch(line, @"(\|\s*)?[:\-\s]+\|?[:\-\s]*(\s*\|[:\-\s]+)*[:\-\s]*(\|\s*)?");
    }

    private static bool IsMarkdownTableRow(string line)
    {
        // Must contain | and not be just a separator
        return line.Contains('|') && !IsTableSeparator(line);
    }

    private static List<string> ParseMarkdownTableRow(string line)
    {
        // Split by | and clean up each cell
        var cells = line.Split('|')
                       .Select(cell => cell.Trim())
                       .Where(cell => !string.IsNullOrEmpty(cell))
                       .ToList();

        return cells;
    }

    private static MarkdownTableStructure CreateMarkdownTableStructure(List<List<string>> tableData)
    {
        if (tableData.Count == 0)
            return new MarkdownTableStructure();

        var rows = new List<MarkdownTableRow>();
        var maxColumns = tableData.Max(row => row.Count);

        for (int i = 0; i < tableData.Count; i++)
        {
            var row = new MarkdownTableRow
            {
                RowIndex = i,
                Cells = new List<MarkdownTableCell>()
            };

            // Pad row to max columns if necessary
            var paddedCells = tableData[i];
            while (paddedCells.Count < maxColumns)
            {
                paddedCells.Add("");
            }

            for (int j = 0; j < paddedCells.Count; j++)
            {
                row.Cells.Add(new MarkdownTableCell
                {
                    RowIndex = i,
                    ColumnIndex = j,
                    Content = paddedCells[j],
                    IsHeader = (i == 0) // First row is typically header
                });
            }

            rows.Add(row);
        }

        return new MarkdownTableStructure
        {
            Rows = rows,
            TotalRows = rows.Count,
            TotalColumns = maxColumns,
            Cells = rows.SelectMany(r => r.Cells).ToList()
        };
    }

    /// <summary>
    /// Convert table structure to OTSL cells for quality metrics calculation.
    /// </summary>
    public static IReadOnlyList<OtslParser.MarkdownTableCell> ConvertToOtslCells(MarkdownTableStructure structure)
    {
        var cells = new List<OtslParser.MarkdownTableCell>();

        foreach (var cell in structure.Cells)
        {
            cells.Add(new OtslParser.MarkdownTableCell
            {
                Row = cell.RowIndex,
                Col = cell.ColumnIndex,
                RowSpan = 1, // We'll calculate spans in a separate step
                ColSpan = 1, // We'll calculate spans in a separate step
                CellType = cell.IsHeader ? "ched" : "fcel",
                IsHeader = cell.IsHeader,
                Content = cell.Content
            });
        }

        return cells;
    }

    /// <summary>
    /// Calculate row and column spans for cells that appear to be merged.
    /// </summary>
    public static IReadOnlyList<OtslParser.MarkdownTableCell> CalculateSpans(IReadOnlyList<OtslParser.MarkdownTableCell> cells)
    {
        var cellsWithSpans = new List<OtslParser.MarkdownTableCell>(cells);

        // Group cells by cell type to identify potential spans
        var contentGroups = cells.GroupBy(c => c.CellType)
                                .Where(g => g.Count() > 1)
                                .ToList();

        foreach (var group in contentGroups)
        {
            var groupCells = group.ToList();

            // Check if cells are adjacent (potential horizontal span)
            var sortedByCol = groupCells.OrderBy(c => c.Col).ToList();
            for (int i = 0; i < sortedByCol.Count - 1; i++)
            {
                if (sortedByCol[i].Row == sortedByCol[i + 1].Row &&
                    sortedByCol[i + 1].Col - sortedByCol[i].Col == 1)
                {
                    // Adjacent cells with same content - mark as spanned
                    sortedByCol[i].ColSpan = 2;
                    sortedByCol[i + 1].Row = -1; // Mark as processed
                }
            }

            // Check if cells are in consecutive rows (potential vertical span)
            var sortedByRow = groupCells.OrderBy(c => c.Row).ToList();
            for (int i = 0; i < sortedByRow.Count - 1; i++)
            {
                if (sortedByRow[i].Col == sortedByRow[i + 1].Col &&
                    sortedByRow[i + 1].Row - sortedByRow[i].Row == 1)
                {
                    // Adjacent cells with same content - mark as spanned
                    sortedByRow[i].RowSpan = 2;
                    sortedByRow[i + 1].Col = -1; // Mark as processed
                }
            }
        }

        // Remove marked cells
        cellsWithSpans.RemoveAll(c => c.Row == -1 || c.Col == -1);

        return cellsWithSpans;
    }
}

/// <summary>
/// Represents a table structure parsed from markdown.
/// </summary>
internal sealed class MarkdownTableStructure
{
    public List<MarkdownTableRow> Rows { get; set; } = new();
    public List<MarkdownTableCell> Cells { get; set; } = new();
    public int TotalRows { get; set; }
    public int TotalColumns { get; set; }
}

/// <summary>
/// Represents a row in a table structure.
/// </summary>
internal sealed class MarkdownTableRow
{
    public int RowIndex { get; set; }
    public List<MarkdownTableCell> Cells { get; set; } = new();
}

/// <summary>
/// Represents a cell in a table structure.
/// </summary>
internal sealed class MarkdownTableCell
{
    public int RowIndex { get; set; }
    public int ColumnIndex { get; set; }
    public string Content { get; set; } = "";
    public bool IsHeader { get; set; }
}
#endif
