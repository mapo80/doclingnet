using System;
using System.Collections.Generic;
using System.Linq;
using Docling.Core.Geometry;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Quality metrics calculation for table structure recognition.
/// Implements TEDS (Table Edit Distance Score) and mAP (mean Average Precision) metrics.
/// </summary>
internal static class QualityMetrics
{
    /// <summary>
    /// Calculate TEDS (Table Edit Distance Score) between predicted and ground truth tables.
    /// TEDS measures structural similarity between tables.
    /// </summary>
    public static double CalculateTEDS(
        IReadOnlyList<OtslParser.TableCell> predictedCells,
        IReadOnlyList<OtslParser.TableCell> groundTruthCells,
        int imageWidth,
        int imageHeight)
    {
        try
        {
            // Convert cells to normalized structure representation
            var predictedStructure = CellsToStructure(predictedCells, imageWidth, imageHeight);
            var groundTruthStructure = CellsToStructure(groundTruthCells, imageWidth, imageHeight);

            // Calculate TEDS score
            return CalculateTableEditDistance(predictedStructure, groundTruthStructure);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TEDS calculation error: {ex.Message}");
            return 0.0;
        }
    }

    /// <summary>
    /// Calculate mean Average Precision (mAP) for table cell detection.
    /// </summary>
    public static double CalculateMAP(
        IReadOnlyList<OtslParser.TableCell> predictedCells,
        IReadOnlyList<OtslParser.TableCell> groundTruthCells,
        double[] confidenceScores,
        double iouThreshold = 0.5)
    {
        try
        {
            if (predictedCells.Count == 0 || groundTruthCells.Count == 0)
                return 0.0;

            // Calculate IoU matrix between predicted and ground truth cells
            var iouMatrix = CalculateIoUMatrix(predictedCells, groundTruthCells);

            // Calculate precision and recall for different confidence thresholds
            var (averagePrecision, _) = CalculatePrecisionRecallCurve(
                iouMatrix, confidenceScores, iouThreshold);

            return averagePrecision;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"mAP calculation error: {ex.Message}");
            return 0.0;
        }
    }

    /// <summary>
    /// Calculate cell-level accuracy metrics.
    /// </summary>
    public static CellAccuracyMetrics CalculateCellAccuracy(
        IReadOnlyList<OtslParser.TableCell> predictedCells,
        IReadOnlyList<OtslParser.TableCell> groundTruthCells)
    {
        var metrics = new CellAccuracyMetrics();

        if (groundTruthCells.Count == 0)
        {
            metrics.TruePositives = 0;
            metrics.FalsePositives = predictedCells.Count;
            metrics.FalseNegatives = 0;
            return metrics;
        }

        if (predictedCells.Count == 0)
        {
            metrics.TruePositives = 0;
            metrics.FalsePositives = 0;
            metrics.FalseNegatives = groundTruthCells.Count;
            return metrics;
        }

        // Calculate IoU matrix
        var iouMatrix = CalculateIoUMatrix(predictedCells, groundTruthCells);

        // Count TP, FP, FN based on IoU threshold
        var iouThreshold = 0.5;
        var matchedGroundTruth = new bool[groundTruthCells.Count];
        var matchedPredicted = new bool[predictedCells.Count];

        for (int i = 0; i < predictedCells.Count; i++)
        {
            for (int j = 0; j < groundTruthCells.Count; j++)
            {
                if (iouMatrix[i, j] >= iouThreshold && !matchedGroundTruth[j])
                {
                    matchedGroundTruth[j] = true;
                    matchedPredicted[i] = true;
                    metrics.TruePositives++;
                    break;
                }
            }
        }

        metrics.FalsePositives = predictedCells.Count - metrics.TruePositives;
        metrics.FalseNegatives = groundTruthCells.Count - matchedGroundTruth.Count(x => x);

        return metrics;
    }

    private static double[,] CalculateIoUMatrix(
        IReadOnlyList<OtslParser.TableCell> predictedCells,
        IReadOnlyList<OtslParser.TableCell> groundTruthCells)
    {
        var matrix = new double[predictedCells.Count, groundTruthCells.Count];

        for (int i = 0; i < predictedCells.Count; i++)
        {
            for (int j = 0; j < groundTruthCells.Count; j++)
            {
                matrix[i, j] = CalculateIoU(predictedCells[i], groundTruthCells[j]);
            }
        }

        return matrix;
    }

    private static double CalculateIoU(OtslParser.TableCell cell1, OtslParser.TableCell cell2)
    {
        // For OtslParser.TableCell, we calculate IoU based on grid position overlap
        // Since these are logical cells, we use row/col overlap

        var rowOverlap = Math.Max(0, Math.Min(cell1.Row + cell1.RowSpan, cell2.Row + cell2.RowSpan) -
                                   Math.Max(cell1.Row, cell2.Row));
        var colOverlap = Math.Max(0, Math.Min(cell1.Col + cell1.ColSpan, cell2.Col + cell2.ColSpan) -
                                   Math.Max(cell1.Col, cell2.Col));

        if (rowOverlap == 0 || colOverlap == 0)
            return 0.0;

        var intersectionArea = rowOverlap * colOverlap;
        var unionArea = (cell1.RowSpan * cell1.ColSpan) + (cell2.RowSpan * cell2.ColSpan) - intersectionArea;

        return unionArea > 0 ? intersectionArea / (double)unionArea : 0.0;
    }

    private static List<List<string>> CellsToStructure(
        IReadOnlyList<OtslParser.TableCell> cells,
        int imageWidth,
        int imageHeight)
    {
        // Group cells into rows based on Row index
        var rowGroups = cells
            .OrderBy(c => c.Row)
            .GroupBy(c => c.Row)
            .OrderBy(g => g.Key)
            .ToList();

        var structure = new List<List<string>>();

        foreach (var rowGroup in rowGroups)
        {
            var row = rowGroup
                .OrderBy(c => c.Col)
                .Select(c => FormatCellForStructure(c))
                .ToList();

            if (row.Any(cell => !string.IsNullOrEmpty(cell)))
            {
                structure.Add(row);
            }
        }

        return structure;
    }

    private static string FormatCellForStructure(OtslParser.TableCell cell)
    {
        // Create a simple text representation of the cell
        var content = string.Empty;

        if (cell.RowSpan > 1 || cell.ColSpan > 1)
        {
            content = $"SPAN_{cell.RowSpan}_{cell.ColSpan}";
        }
        else
        {
            content = "CELL";
        }

        return content;
    }

    private static double CalculateTableEditDistance(
        List<List<string>> structure1,
        List<List<string>> structure2)
    {
        // Simple edit distance for table structures
        // In a full implementation, this would use more sophisticated algorithms

        if (structure1.Count == 0 && structure2.Count == 0)
            return 1.0; // Perfect match for empty tables

        if (structure1.Count == 0 || structure2.Count == 0)
            return 0.0; // No match if one is empty

        // Calculate row-wise similarity
        var totalSimilarity = 0.0;
        var maxRows = Math.Max(structure1.Count, structure2.Count);

        for (int i = 0; i < Math.Min(structure1.Count, structure2.Count); i++)
        {
            var rowSimilarity = CalculateRowSimilarity(structure1[i], structure2[i]);
            totalSimilarity += rowSimilarity;
        }

        // Account for different number of rows
        var rowCountPenalty = Math.Abs(structure1.Count - structure2.Count) / (double)maxRows;
        totalSimilarity -= rowCountPenalty * 0.5; // Penalty for row count mismatch

        return Math.Max(0.0, Math.Min(1.0, totalSimilarity / maxRows));
    }

    private static double CalculateRowSimilarity(List<string> row1, List<string> row2)
    {
        if (row1.Count == 0 && row2.Count == 0)
            return 1.0;

        if (row1.Count == 0 || row2.Count == 0)
            return 0.0;

        var maxCells = Math.Max(row1.Count, row2.Count);
        var cellMatches = 0;

        for (int i = 0; i < Math.Min(row1.Count, row2.Count); i++)
        {
            if (row1[i] == row2[i])
                cellMatches++;
        }

        // Account for different number of cells
        var cellCountPenalty = Math.Abs(row1.Count - row2.Count) / (double)maxCells;

        return (cellMatches / (double)maxCells) * (1.0 - cellCountPenalty * 0.3);
    }

    private static (double averagePrecision, double averageRecall) CalculatePrecisionRecallCurve(
        double[,] iouMatrix,
        double[] confidenceScores,
        double iouThreshold)
    {
        var sortedIndices = confidenceScores
            .Select((score, index) => new { Score = score, Index = index })
            .OrderByDescending(x => x.Score)
            .Select(x => x.Index)
            .ToArray();

        var tp = new int[sortedIndices.Length];
        var fp = new int[sortedIndices.Length];
        var matchedGroundTruth = new bool[iouMatrix.GetLength(1)];

        for (int i = 0; i < sortedIndices.Length; i++)
        {
            var predIdx = sortedIndices[i];
            var isTruePositive = false;

            // Find best matching ground truth
            for (int j = 0; j < iouMatrix.GetLength(1); j++)
            {
                if (iouMatrix[predIdx, j] >= iouThreshold && !matchedGroundTruth[j])
                {
                    matchedGroundTruth[j] = true;
                    isTruePositive = true;
                    break;
                }
            }

            if (isTruePositive)
                tp[i] = 1;
            else
                fp[i] = 1;
        }

        // Calculate precision and recall at each point
        var precisionValues = new double[sortedIndices.Length];
        var recallValues = new double[sortedIndices.Length];

        var totalTP = tp.Sum();
        var totalFP = fp.Sum();
        var totalGroundTruth = iouMatrix.GetLength(1);

        for (int i = 0; i < sortedIndices.Length; i++)
        {
            var tpSum = 0;
            var fpSum = 0;

            for (int j = 0; j <= i; j++)
            {
                tpSum += tp[j];
                fpSum += fp[j];
            }

            precisionValues[i] = tpSum + fpSum > 0 ? tpSum / (double)(tpSum + fpSum) : 0.0;
            recallValues[i] = totalGroundTruth > 0 ? tpSum / (double)totalGroundTruth : 0.0;
        }

        // Calculate average precision (area under curve)
        var averagePrecision = CalculateAreaUnderCurve(precisionValues, recallValues);

        return (averagePrecision, recallValues.LastOrDefault());
    }

    private static double CalculateAreaUnderCurve(double[] precision, double[] recall)
    {
        if (precision.Length == 0)
            return 0.0;

        // Simple trapezoidal integration
        var area = 0.0;
        for (int i = 1; i < precision.Length; i++)
        {
            var width = recall[i] - recall[i - 1];
            var height = (precision[i] + precision[i - 1]) / 2.0;
            area += width * height;
        }

        return area;
    }

    /// <summary>
    /// Comprehensive quality assessment results.
    /// </summary>
    public sealed class QualityAssessment
    {
        public double TEDS { get; set; }           // Table Edit Distance Score (0-1)
        public double MAP { get; set; }            // Mean Average Precision (0-1)
        public CellAccuracyMetrics CellAccuracy { get; set; } = new();
        public StructureMetrics StructureMetrics { get; set; } = new();
        public double OverallScore { get; set; }   // Weighted combination
    }

    /// <summary>
    /// Cell-level accuracy metrics.
    /// </summary>
    public sealed class CellAccuracyMetrics
    {
        public int TruePositives { get; set; }
        public int FalsePositives { get; set; }
        public int FalseNegatives { get; set; }

        public double Precision => TruePositives + FalsePositives > 0
            ? TruePositives / (double)(TruePositives + FalsePositives)
            : 0.0;

        public double Recall => TruePositives + FalseNegatives > 0
            ? TruePositives / (double)(TruePositives + FalseNegatives)
            : 0.0;

        public double F1Score => Precision + Recall > 0
            ? 2 * Precision * Recall / (Precision + Recall)
            : 0.0;
    }

    /// <summary>
    /// Structure-level accuracy metrics.
    /// </summary>
    public sealed class StructureMetrics
    {
        public int PredictedRows { get; set; }
        public int GroundTruthRows { get; set; }
        public int PredictedCols { get; set; }
        public int GroundTruthCols { get; set; }
        public double RowAccuracy { get; set; }
        public double ColAccuracy { get; set; }

        public double StructureAccuracy =>
            (RowAccuracy + ColAccuracy) / 2.0;
    }

    // Helper class for Rectangle operations
    private sealed class Rectangle
    {
        public int X { get; }
        public int Y { get; }
        public int Width { get; }
        public int Height { get; }

        public Rectangle(int x, int y, int width, int height)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
        }

        public static Rectangle Intersect(Rectangle rect1, Rectangle rect2)
        {
            var x = Math.Max(rect1.X, rect2.X);
            var y = Math.Max(rect1.Y, rect2.Y);
            var width = Math.Max(0, Math.Min(rect1.X + rect1.Width, rect2.X + rect2.Width) - x);
            var height = Math.Max(0, Math.Min(rect1.Y + rect1.Height, rect2.Y + rect2.Height) - y);

            return new Rectangle(x, y, width, height);
        }

        public bool IsEmpty => Width <= 0 || Height <= 0;
    }
}