using System.Collections.Generic;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Tables;
using Xunit;

namespace Docling.Tests.Tables;

public sealed class TableFormerQualityMetricsTests
{
    [Fact]
    public void CalculateMetricsForMatchingStructuresReturnsPerfectScores()
    {
        var page = new PageReference(1, 300);
        var groundTruthCells = new[]
        {
            new TableCell(BoundingBox.FromSize(0, 0, 50, 50), RowSpan: 1, ColumnSpan: 1, Text: null),
            new TableCell(BoundingBox.FromSize(50, 0, 50, 50), RowSpan: 1, ColumnSpan: 1, Text: null)
        };

        var groundTruth = new TableStructure(page, groundTruthCells, RowCount: 1, ColumnCount: 2);
        var predicted = new TableStructure(page, new List<TableCell>(groundTruthCells), RowCount: 1, ColumnCount: 2);

        var teds = TableFormerQualityMetrics.CalculateTEDS(predicted, groundTruth);
        var map = TableFormerQualityMetrics.CalculateMAP(predicted.Cells, groundTruth.Cells);
        var accuracy = TableFormerQualityMetrics.CalculateCellAccuracy(predicted.Cells, groundTruth.Cells);
        var (rowAccuracy, colAccuracy) = TableFormerQualityMetrics.CalculateRowColumnAccuracy(predicted, groundTruth);

        Assert.Equal(1.0, teds, 3);
        Assert.Equal(1.0, map, 3);
        Assert.Equal(1.0, accuracy, 3);
        Assert.Equal(1.0, rowAccuracy, 3);
        Assert.Equal(1.0, colAccuracy, 3);

        var assessment = new TableQualityAssessment
        {
            TedsScore = teds,
            MeanAveragePrecision = map,
            CellAccuracy = accuracy,
            RowAccuracy = rowAccuracy,
            ColumnAccuracy = colAccuracy
        };

        Assert.Equal("🟢 EXCELLENT", assessment.GetQualityGrade());
        Assert.Contains("Quality:", assessment.ToString());
    }

    [Fact]
    public void CalculateMetricsDetectsMismatchedStructures()
    {
        var page = new PageReference(1, 300);
        var groundTruthCells = new[]
        {
            new TableCell(BoundingBox.FromSize(0, 0, 40, 40), 1, 1, null),
            new TableCell(BoundingBox.FromSize(40, 0, 40, 40), 1, 1, null)
        };

        var predictedCells = new[]
        {
            new TableCell(BoundingBox.FromSize(0, 0, 40, 40), 1, 1, null)
        };

        var groundTruth = new TableStructure(page, groundTruthCells, RowCount: 1, ColumnCount: 2);
        var predicted = new TableStructure(page, predictedCells, RowCount: 1, ColumnCount: 1);

        var teds = TableFormerQualityMetrics.CalculateTEDS(predicted, groundTruth);
        var map = TableFormerQualityMetrics.CalculateMAP(predicted.Cells, groundTruth.Cells);
        var accuracy = TableFormerQualityMetrics.CalculateCellAccuracy(predicted.Cells, groundTruth.Cells);
        var (rowAccuracy, colAccuracy) = TableFormerQualityMetrics.CalculateRowColumnAccuracy(predicted, groundTruth);

        Assert.True(teds < 1.0);
        Assert.True(map < 1.0);
        Assert.True(accuracy < 1.0);
        Assert.Equal(1.0, rowAccuracy, 3);
        Assert.Equal(0.0, colAccuracy, 3);
    }
}
