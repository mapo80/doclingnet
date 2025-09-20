using System.Collections.Generic;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Tables;
using Xunit;

namespace Docling.Tests.Tables;

public sealed class TableBuilderTests
{
    [Fact]
    public void BuildProducesPlacementsWithExpectedOrdering()
    {
        var page = new PageReference(1, 300);
        var cells = new List<TableCell>
        {
            new(BoundingBox.FromSize(50, 10, 40, 10), RowSpan: 1, ColumnSpan: 1, Text: "b"),
            new(BoundingBox.FromSize(10, 10, 40, 10), RowSpan: 1, ColumnSpan: 1, Text: "a"),
            new(BoundingBox.FromSize(10, 30, 40, 10), RowSpan: 1, ColumnSpan: 1, Text: "c"),
            new(BoundingBox.FromSize(50, 30, 40, 10), RowSpan: 1, ColumnSpan: 1, Text: "d"),
        };
        var structure = new TableStructure(page, cells, RowCount: 2, ColumnCount: 2);

        var table = TableBuilder.Build(structure);

        Assert.Equal(2, table.RowCount);
        Assert.Equal(2, table.ColumnCount);
        Assert.Equal(4, table.Cells.Count);

        Assert.Equal((0, 0, "a"), (table.Cells[0].RowIndex, table.Cells[0].ColumnIndex, table.Cells[0].Text));
        Assert.Equal((0, 1, "b"), (table.Cells[1].RowIndex, table.Cells[1].ColumnIndex, table.Cells[1].Text));
        Assert.Equal((1, 0, "c"), (table.Cells[2].RowIndex, table.Cells[2].ColumnIndex, table.Cells[2].Text));
        Assert.Equal((1, 1, "d"), (table.Cells[3].RowIndex, table.Cells[3].ColumnIndex, table.Cells[3].Text));

        var bounds = table.BoundingBox;
        Assert.Equal(10, bounds.Left, 4);
        Assert.Equal(10, bounds.Top, 4);
        Assert.Equal(90, bounds.Right, 4);
        Assert.Equal(40, bounds.Bottom, 4);
    }

    [Fact]
    public void BuildNormalizesSpansAndBounds()
    {
        var page = new PageReference(2, 200);
        var cells = new List<TableCell>
        {
            new(BoundingBox.FromSize(0, 0, 20, 10), RowSpan: 5, ColumnSpan: -1, Text: "merged"),
            new(BoundingBox.FromSize(20, 0, 20, 10), RowSpan: 1, ColumnSpan: 1, Text: "right"),
        };
        var structure = new TableStructure(page, cells, RowCount: 2, ColumnCount: 2);

        var table = TableBuilder.Build(structure);

        Assert.Equal((0, 0), (table.Cells[0].RowIndex, table.Cells[0].ColumnIndex));
        Assert.Equal((2, 1), (table.Cells[0].RowSpan, table.Cells[0].ColumnSpan));
        Assert.Equal((0, 1), (table.Cells[1].RowIndex, table.Cells[1].ColumnIndex));
        Assert.Equal((1, 1), (table.Cells[1].RowSpan, table.Cells[1].ColumnSpan));
    }

    [Fact]
    public void BuildDerivesCountsWhenStructureMissing()
    {
        var page = new PageReference(3, 150);
        var cells = new List<TableCell>
        {
            new(BoundingBox.FromSize(0, 0, 10, 10), RowSpan: 1, ColumnSpan: 1, Text: "a"),
            new(BoundingBox.FromSize(10, 0, 10, 10), RowSpan: 1, ColumnSpan: 1, Text: "b"),
            new(BoundingBox.FromSize(0, 10, 20, 10), RowSpan: 1, ColumnSpan: 2, Text: "c"),
        };
        var structure = new TableStructure(page, cells, RowCount: 0, ColumnCount: 0);

        var table = TableBuilder.Build(structure);

        Assert.Equal(2, table.RowCount);
        Assert.Equal(2, table.ColumnCount);
        Assert.Equal(3, table.Cells.Count);
        Assert.Equal((1, 0), (table.Cells[2].RowIndex, table.Cells[2].ColumnIndex));
        Assert.Equal((1, 2), (table.Cells[2].RowSpan, table.Cells[2].ColumnSpan));
    }
}
