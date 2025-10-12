using System;
using System.Collections.Generic;
using System.Linq;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Parses OTSL (Ordered Table Structure Language) sequences into table structures.
/// Handles cell spans, headers, and table layout construction.
/// </summary>
internal sealed class OtslParser
{
    /// <summary>
    /// Represents a table cell with position and span information.
    /// </summary>
    public sealed class TableCell
    {
        public int Row { get; set; }
        public int Col { get; set; }
        public int RowSpan { get; set; } = 1;
        public int ColSpan { get; set; } = 1;
        public string CellType { get; set; } = "";
        public bool IsHeader { get; set; }
    }

    /// <summary>
    /// Represents a complete table structure.
    /// </summary>
    public sealed class TableStructure
    {
        public List<List<TableCell>> Rows { get; } = new();
        public int RowCount => Rows.Count;
        public int ColCount => Rows.Count > 0 ? Rows.Max(row => row.Count) : 0;
    }

    /// <summary>
    /// Parse OTSL token sequence into table structure.
    /// </summary>
    public static TableStructure ParseOtsl(IEnumerable<string> otslTokens)
    {
        var tokens = otslTokens.ToList();
        var table = new TableStructure();
        var currentRow = new List<TableCell>();
        var currentRowIndex = 0;
        var currentColIndex = 0;

        for (int i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];

            switch (token)
            {
                case "fcel": // First cell in row
                    if (currentRow.Count > 0)
                    {
                        table.Rows.Add(currentRow);
                        currentRow = new List<TableCell>();
                    }
                    currentRowIndex++;
                    currentColIndex = 0;

                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex - 1,
                        Col = currentColIndex,
                        CellType = "fcel"
                    });
                    currentColIndex++;
                    break;

                case "lcel": // Linked cell (horizontal span)
                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex - 1,
                        Col = currentColIndex,
                        CellType = "lcel"
                    });
                    currentColIndex++;
                    break;

                case "ecel": // Empty cell
                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex - 1,
                        Col = currentColIndex,
                        CellType = "ecel"
                    });
                    currentColIndex++;
                    break;

                case "ucel": // Up cell (vertical span start)
                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex - 1,
                        Col = currentColIndex,
                        CellType = "ucel"
                    });
                    currentColIndex++;
                    break;

                case "xcel": // Cross cell (vertical span continuation)
                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex - 1,
                        Col = currentColIndex,
                        CellType = "xcel"
                    });
                    currentColIndex++;
                    break;

                case "ched": // Column header
                    if (currentRow.Count > 0)
                    {
                        currentRow.Last().IsHeader = true;
                        currentRow.Last().CellType = "ched";
                    }
                    break;

                case "rhed": // Row header
                    if (currentRow.Count > 0)
                    {
                        currentRow.Last().IsHeader = true;
                        currentRow.Last().CellType = "rhed";
                    }
                    break;

                case "srow": // Spanning row
                    if (currentRow.Count > 0)
                    {
                        currentRow.Last().RowSpan = 2; // Default span, could be made configurable
                    }
                    break;

                case "nl": // New line (row separator)
                    if (currentRow.Count > 0)
                    {
                        table.Rows.Add(currentRow);
                        currentRow = new List<TableCell>();
                    }
                    currentRowIndex++;
                    currentColIndex = 0;
                    break;

                case "<end>":
                case "<pad>":
                    // End of sequence or padding, stop processing
                    goto end_of_sequence;

                default:
                    // Unknown token, skip
                    break;
            }
        }

        end_of_sequence:

        // Add the last row if it has cells
        if (currentRow.Count > 0)
        {
            table.Rows.Add(currentRow);
        }

        // Post-process to calculate spans
        CalculateSpans(table);

        return table;
    }

    private static void CalculateSpans(TableStructure table)
    {
        // Calculate horizontal spans (lcel)
        foreach (var row in table.Rows)
        {
            for (int col = 0; col < row.Count; col++)
            {
                var cell = row[col];

                // Count consecutive lcel cells for horizontal span
                if (cell.CellType == "fcel" || cell.CellType == "lcel")
                {
                    var span = 1;
                    var startCol = col;

                    // Look ahead for consecutive lcel
                    for (int nextCol = col + 1; nextCol < row.Count; nextCol++)
                    {
                        if (row[nextCol].CellType == "lcel")
                        {
                            span++;
                        }
                        else
                        {
                            break;
                        }
                    }

                    // Update the first cell in the span
                    if (span > 1)
                    {
                        row[startCol].ColSpan = span;
                        // Mark linked cells as not visible (they're part of the span)
                        for (int linkedCol = startCol + 1; linkedCol < startCol + span; linkedCol++)
                        {
                            row[linkedCol].CellType = "linked";
                        }
                    }
                }
            }
        }

        // Calculate vertical spans (ucel/xcel)
        for (int row = 0; row < table.Rows.Count; row++)
        {
            for (int col = 0; col < table.Rows[row].Count; col++)
            {
                var cell = table.Rows[row][col];

                if (cell.CellType == "ucel")
                {
                    var span = 1;

                    // Look down for xcel in the same column
                    for (int nextRow = row + 1; nextRow < table.Rows.Count; nextRow++)
                    {
                        if (nextRow < table.Rows.Count &&
                            col < table.Rows[nextRow].Count &&
                            table.Rows[nextRow][col].CellType == "xcel")
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
                        // Mark continuation cells as not visible
                        for (int spanRow = row + 1; spanRow < row + span; spanRow++)
                        {
                            if (col < table.Rows[spanRow].Count)
                            {
                                table.Rows[spanRow][col].CellType = "spanned";
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Convert table structure back to a simple grid representation for debugging.
    /// </summary>
    public static string TableToGridString(TableStructure table)
    {
        if (table.Rows.Count == 0)
        {
            return "(empty table)";
        }

        var maxCols = table.ColCount;
        var grid = new string[table.RowCount][];
        for (int i = 0; i < table.RowCount; i++)
        {
            grid[i] = new string[maxCols];
        }

        foreach (var row in table.Rows)
        {
            foreach (var cell in row)
            {
                if (cell.CellType != "linked" && cell.CellType != "spanned")
                {
                    var displayText = GetCellDisplayText(cell);
                    grid[cell.Row, cell.Col] = displayText;

                    // Fill spanned cells
                    for (int spanRow = 1; spanRow < cell.RowSpan && cell.Row + spanRow < table.RowCount; spanRow++)
                    {
                        grid[cell.Row + spanRow, cell.Col] = "↓";
                    }
                    for (int spanCol = 1; spanCol < cell.ColSpan && cell.Col + spanCol < maxCols; spanCol++)
                    {
                        grid[cell.Row, cell.Col + spanCol] = "→";
                    }
                }
            }
        }

        // Convert grid to string
        var rows = new List<string>();
        for (int row = 0; row < table.RowCount; row++)
        {
            var rowCells = new List<string>();
            for (int col = 0; col < maxCols; col++)
            {
                rowCells.Add(grid[row][col] ?? " ");
            }
            rows.Add(string.Join("|", rowCells));
        }

        return string.Join("\n", rows);
    }

    private static string GetCellDisplayText(TableCell cell)
    {
        return cell.CellType switch
        {
            "fcel" => "F",
            "ecel" => "E",
            "ucel" => "U",
            "ched" => "H",
            "rhed" => "H",
            _ => cell.CellType.Length > 0 ? char.ToUpperInvariant(cell.CellType[0]).ToString() : "?"
        };
    }
}