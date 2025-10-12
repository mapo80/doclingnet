using System;
using System.Collections.Generic;
using System.Linq;
using Docling.Core.Geometry;

namespace Docling.Models.Tables;

/// <summary>
/// Quality metrics helpers for validating TableFormer predictions.
/// The implementations favour pragmatic heuristics that can run without the Python pipeline.
/// </summary>
public static class TableFormerQualityMetrics
{
    private const double DefaultIouThreshold = 0.5;

    public static double CalculateTEDS(TableStructure predicted, TableStructure groundTruth, double iouThreshold = DefaultIouThreshold)
    {
        if (predicted is null || groundTruth is null)
        {
            return 0.0;
        }

        var matches = MatchCells(predicted.Cells, groundTruth.Cells, iouThreshold);
        var maxCells = Math.Max(predicted.Cells.Count, groundTruth.Cells.Count);
        if (maxCells == 0)
        {
            return 1.0;
        }

        var unmatchedCost = (predicted.Cells.Count - matches.Count) + (groundTruth.Cells.Count - matches.Count);
        var spanMismatchCost = matches.Count(match => match.predicted.RowSpan != match.groundTruth.RowSpan ||
                                                      match.predicted.ColumnSpan != match.groundTruth.ColumnSpan);

        var editCost = unmatchedCost + spanMismatchCost;
        var normalized = 1.0 - (editCost / (double)maxCells);
        return Math.Clamp(normalized, 0.0, 1.0);
    }

    public static double CalculateMAP(
        IReadOnlyList<TableCell> predictedCells,
        IReadOnlyList<TableCell> groundTruthCells,
        double iouThreshold = DefaultIouThreshold)
    {
        if (predictedCells.Count == 0 && groundTruthCells.Count == 0)
        {
            return 1.0;
        }

        if (predictedCells.Count == 0 || groundTruthCells.Count == 0)
        {
            return 0.0;
        }

        var matches = MatchCells(predictedCells, groundTruthCells, iouThreshold);
        var truePositives = matches.Count;
        var falsePositives = predictedCells.Count - truePositives;
        var falseNegatives = groundTruthCells.Count - truePositives;

        var precision = truePositives + falsePositives > 0
            ? (double)truePositives / (truePositives + falsePositives)
            : 0.0;

        var recall = truePositives + falseNegatives > 0
            ? (double)truePositives / (truePositives + falseNegatives)
            : 0.0;

        return precision + recall > 0
            ? 2 * precision * recall / (precision + recall)
            : 0.0;
    }

    public static double CalculateCellAccuracy(
        IReadOnlyList<TableCell> predictedCells,
        IReadOnlyList<TableCell> groundTruthCells,
        double iouThreshold = DefaultIouThreshold)
    {
        if (predictedCells.Count == 0 && groundTruthCells.Count == 0)
        {
            return 1.0;
        }

        if (predictedCells.Count == 0 || groundTruthCells.Count == 0)
        {
            return 0.0;
        }

        var matches = MatchCells(predictedCells, groundTruthCells, iouThreshold);
        var denominator = Math.Max(predictedCells.Count, groundTruthCells.Count);

        return denominator > 0 ? matches.Count / (double)denominator : 0.0;
    }

    public static (double RowAccuracy, double ColumnAccuracy) CalculateRowColumnAccuracy(
        TableStructure predicted,
        TableStructure groundTruth)
    {
        if (predicted is null || groundTruth is null)
        {
            return (0.0, 0.0);
        }

        var rowAccuracy = predicted.RowCount == groundTruth.RowCount ? 1.0 : 0.0;
        var columnAccuracy = predicted.ColumnCount == groundTruth.ColumnCount ? 1.0 : 0.0;
        return (rowAccuracy, columnAccuracy);
    }

    private static List<(TableCell predicted, TableCell groundTruth, double iou)> MatchCells(
        IReadOnlyList<TableCell> predictedCells,
        IReadOnlyList<TableCell> groundTruthCells,
        double iouThreshold)
    {
        var matches = new List<(TableCell, TableCell, double)>();
        if (predictedCells.Count == 0 || groundTruthCells.Count == 0)
        {
            return matches;
        }

        var iouMatrix = CalculateIoUMatrix(predictedCells, groundTruthCells);
        var usedGroundTruth = new HashSet<int>();

        for (var predictedIndex = 0; predictedIndex < predictedCells.Count; predictedIndex++)
        {
            var bestGroundTruth = -1;
            var bestIou = 0.0;

            for (var groundTruthIndex = 0; groundTruthIndex < groundTruthCells.Count; groundTruthIndex++)
            {
                if (usedGroundTruth.Contains(groundTruthIndex))
                {
                    continue;
                }

                var currentIou = iouMatrix[predictedIndex, groundTruthIndex];
                if (currentIou > bestIou)
                {
                    bestIou = currentIou;
                    bestGroundTruth = groundTruthIndex;
                }
            }

            if (bestGroundTruth >= 0 && bestIou >= iouThreshold)
            {
                usedGroundTruth.Add(bestGroundTruth);
                matches.Add((predictedCells[predictedIndex], groundTruthCells[bestGroundTruth], bestIou));
            }
        }

        return matches;
    }

    private static double[,] CalculateIoUMatrix(
        IReadOnlyList<TableCell> predictedCells,
        IReadOnlyList<TableCell> groundTruthCells)
    {
        var matrix = new double[predictedCells.Count, groundTruthCells.Count];

        for (var i = 0; i < predictedCells.Count; i++)
        {
            for (var j = 0; j < groundTruthCells.Count; j++)
            {
                matrix[i, j] = CalculateIoU(predictedCells[i].BoundingBox, groundTruthCells[j].BoundingBox);
            }
        }

        return matrix;
    }

    private static double CalculateIoU(BoundingBox first, BoundingBox second)
    {
        var intersection = first.Intersect(second);
        if (intersection.IsEmpty)
        {
            return 0.0;
        }

        var unionArea = first.Area + second.Area - intersection.Area;
        return unionArea > 0 ? intersection.Area / unionArea : 0.0;
    }
}

/// <summary>
/// Aggregated quality assessment used by the validation suite.
/// </summary>
public sealed class TableQualityAssessment
{
    public double TedsScore { get; init; }
    public double MeanAveragePrecision { get; init; }
    public double CellAccuracy { get; init; }
    public double RowAccuracy { get; init; }
    public double ColumnAccuracy { get; init; }

    public double OverallScore =>
        (TedsScore * 0.4) +
        (MeanAveragePrecision * 0.3) +
        (CellAccuracy * 0.2) +
        ((RowAccuracy + ColumnAccuracy) / 2 * 0.1);

    public string GetQualityGrade()
    {
        if (OverallScore >= 0.9)
        {
            return "🟢 EXCELLENT";
        }

        if (OverallScore >= 0.7)
        {
            return "🟡 GOOD";
        }

        if (OverallScore >= 0.5)
        {
            return "🟠 FAIR";
        }

        return "🔴 POOR";
    }

    public override string ToString() =>
        $"Quality: {GetQualityGrade()} ({OverallScore:P1}) — TEDS {TedsScore:P1}, mAP {MeanAveragePrecision:P1}, CellAcc {CellAccuracy:P1}";
}
