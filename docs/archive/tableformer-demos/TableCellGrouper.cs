#if false
using System;
using System.Collections.Generic;
using System.Linq;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Advanced post-processing for grouping detected boxes into logical table cells.
/// Handles overlapping boxes, merging fragments, and cell relationship detection.
/// </summary>
internal sealed class TableCellGrouper
{
    private readonly float _overlapThreshold = 0.8f;
    private readonly float _mergeThreshold = 0.1f;

    /// <summary>
    /// Group raw detections into logical table cells.
    /// </summary>
    public IReadOnlyList<GroupedCell> GroupCells(IReadOnlyList<TableFormerDetrBackend.TableFormerDetection> detections)
    {
        if (detections.Count == 0)
            return Array.Empty<GroupedCell>();

        // Step 1: Merge overlapping detections
        var mergedDetections = MergeOverlappingBoxes(detections);

        // Step 2: Group into cells based on spatial relationships
        var cells = CreateCellsFromDetections(mergedDetections);

        // Step 3: Detect cell relationships (spanning, headers, etc.)
        var processedCells = DetectCellRelationships(cells);

        return processedCells;
    }

    private List<TableFormerDetrBackend.TableFormerDetection> MergeOverlappingBoxes(
        IReadOnlyList<TableFormerDetrBackend.TableFormerDetection> detections)
    {
        var merged = new List<TableFormerDetrBackend.TableFormerDetection>();
        var used = new bool[detections.Count];

        for (int i = 0; i < detections.Count; i++)
        {
            if (used[i]) continue;

            var current = detections[i];
            var mergedBox = current.BoundingBox;
            var mergedConfidence = current.Confidence;
            var mergeCount = 1;

            // Find overlapping boxes
            for (int j = i + 1; j < detections.Count; j++)
            {
                if (used[j]) continue;

                var overlap = CalculateOverlapRatio(mergedBox, detections[j].BoundingBox);
                if (overlap > _overlapThreshold)
                {
                    // Merge boxes
                    mergedBox = MergeBoxes(mergedBox, detections[j].BoundingBox);
                    mergedConfidence += detections[j].Confidence;
                    mergeCount++;
                    used[j] = true;
                }
            }

            // Create merged detection
            merged.Add(new TableFormerDetrBackend.TableFormerDetection
            {
                BoundingBox = mergedBox,
                Confidence = mergedConfidence / mergeCount,
                ClassId = current.ClassId
            });
        }

        return merged;
    }

    private float CalculateOverlapRatio(
        (float cx1, float cy1, float w1, float h1) box1,
        (float cx2, float cy2, float w2, float h2) box2)
    {
        // Convert to corner coordinates
        var x1_1 = box1.cx1 - box1.w1 / 2;
        var y1_1 = box1.cy1 - box1.h1 / 2;
        var x2_1 = box1.cx1 + box1.w1 / 2;
        var y2_1 = box1.cy1 + box1.h1 / 2;

        var x1_2 = box2.cx2 - box2.w2 / 2;
        var y1_2 = box2.cy2 - box2.h2 / 2;
        var x2_2 = box2.cx2 + box2.w2 / 2;
        var y2_2 = box2.cy2 + box2.h2 / 2;

        // Calculate intersection
        var interX1 = Math.Max(x1_1, x1_2);
        var interY1 = Math.Max(y1_1, y1_2);
        var interX2 = Math.Min(x2_1, x2_2);
        var interY2 = Math.Min(y2_1, y2_2);

        if (interX2 <= interX1 || interY2 <= interY1)
            return 0f;

        var interArea = (interX2 - interX1) * (interY2 - interY1);
        var area1 = box1.w1 * box1.h1;
        var area2 = box2.w2 * box2.h2;

        return interArea / (area1 + area2 - interArea);
    }

    private static (float cx, float cy, float w, float h) MergeBoxes(
        (float cx1, float cy1, float w1, float h1) box1,
        (float cx2, float cy2, float w2, float h2) box2)
    {
        var x1_1 = box1.cx1 - box1.w1 / 2;
        var y1_1 = box1.cy1 - box1.h1 / 2;
        var x2_1 = box1.cx1 + box1.w1 / 2;
        var y2_1 = box1.cy1 + box1.h1 / 2;

        var x1_2 = box2.cx2 - box2.w2 / 2;
        var y1_2 = box2.cy2 - box2.h2 / 2;
        var x2_2 = box2.cx2 + box2.w2 / 2;
        var y2_2 = box2.cy2 + box2.h2 / 2;

        var mergedX1 = Math.Min(x1_1, x1_2);
        var mergedY1 = Math.Min(y1_1, y1_2);
        var mergedX2 = Math.Max(x2_1, x2_2);
        var mergedY2 = Math.Max(y2_1, y2_2);

        var mergedW = mergedX2 - mergedX1;
        var mergedH = mergedY2 - mergedY1;
        var mergedCx = mergedX1 + mergedW / 2;
        var mergedCy = mergedY1 + mergedH / 2;

        return (mergedCx, mergedCy, mergedW, mergedH);
    }

    private List<GroupedCell> CreateCellsFromDetections(
        List<TableFormerDetrBackend.TableFormerDetection> detections)
    {
        var cells = new List<GroupedCell>();

        foreach (var detection in detections)
        {
            cells.Add(new GroupedCell
            {
                BoundingBox = detection.BoundingBox,
                Confidence = detection.Confidence,
                CellType = GetCellTypeFromClassId(detection.ClassId)
            });
        }

        return cells;
    }

    private static string GetCellTypeFromClassId(int classId)
    {
        // Map class IDs to cell types (this would need to be configured based on the actual model)
        return classId switch
        {
            0 => "header",
            1 => "data",
            2 => "empty",
            _ => "unknown"
        };
    }

    private List<GroupedCell> DetectCellRelationships(List<GroupedCell> cells)
    {
        // Sort cells by position for relationship detection
        var sortedCells = cells.OrderBy(c => c.BoundingBox.cy)
                              .ThenBy(c => c.BoundingBox.cx)
                              .ToList();

        // Detect row and column alignments
        var rowGroups = GroupCellsByRow(sortedCells);
        var colGroups = GroupCellsByColumn(sortedCells);

        // Assign row and column indices
        for (int i = 0; i < sortedCells.Count; i++)
        {
            var cell = sortedCells[i];
            cell.RowIndex = FindGroupIndex(rowGroups, cell);
            cell.ColIndex = FindGroupIndex(colGroups, cell);
        }

        // Detect spanning cells
        DetectSpanningCells(sortedCells);

        return sortedCells;
    }

    private List<List<GroupedCell>> GroupCellsByRow(List<GroupedCell> cells)
    {
        var rows = new List<List<GroupedCell>>();
        var currentRow = new List<GroupedCell>();
        var currentRowY = cells.Count > 0 ? cells[0].BoundingBox.cy : 0;

        foreach (var cell in cells)
        {
            if (Math.Abs(cell.BoundingBox.cy - currentRowY) > _mergeThreshold)
            {
                if (currentRow.Count > 0)
                {
                    rows.Add(currentRow);
                }
                currentRow = new List<GroupedCell>();
                currentRowY = cell.BoundingBox.cy;
            }
            currentRow.Add(cell);
        }

        if (currentRow.Count > 0)
        {
            rows.Add(currentRow);
        }

        return rows;
    }

    private List<List<GroupedCell>> GroupCellsByColumn(List<GroupedCell> cells)
    {
        var cols = new List<List<GroupedCell>>();
        var currentCol = new List<GroupedCell>();
        var currentColX = cells.Count > 0 ? cells[0].BoundingBox.cx : 0;

        foreach (var cell in cells)
        {
            if (Math.Abs(cell.BoundingBox.cx - currentColX) > _mergeThreshold)
            {
                if (currentCol.Count > 0)
                {
                    cols.Add(currentCol);
                }
                currentCol = new List<GroupedCell>();
                currentColX = cell.BoundingBox.cx;
            }
            currentCol.Add(cell);
        }

        if (currentCol.Count > 0)
        {
            cols.Add(currentCol);
        }

        return cols;
    }

    private static int FindGroupIndex(List<List<GroupedCell>> groups, GroupedCell cell)
    {
        for (int i = 0; i < groups.Count; i++)
        {
            if (groups[i].Contains(cell))
            {
                return i;
            }
        }
        return -1;
    }

    private static void DetectSpanningCells(List<GroupedCell> cells)
    {
        // Simple spanning detection based on cell sizes and positions
        foreach (var cell in cells)
        {
            var (cx, cy, w, h) = cell.BoundingBox;

            // Detect horizontal spans (wider than average)
            var avgWidth = cells.Average(c => c.BoundingBox.w);
            if (w > avgWidth * 1.5f)
            {
                cell.ColSpan = Math.Max(2, (int)Math.Round(w / avgWidth));
            }

            // Detect vertical spans (taller than average)
            var avgHeight = cells.Average(c => c.BoundingBox.h);
            if (h > avgHeight * 1.5f)
            {
                cell.RowSpan = Math.Max(2, (int)Math.Round(h / avgHeight));
            }
        }
    }

    /// <summary>
    /// Represents a grouped table cell with relationship information.
    /// </summary>
    public sealed class GroupedCell
    {
        public (float cx, float cy, float w, float h) BoundingBox { get; set; }
        public float Confidence { get; set; }
        public string CellType { get; set; } = "";
        public int RowIndex { get; set; } = -1;
        public int ColIndex { get; set; } = -1;
        public int RowSpan { get; set; } = 1;
        public int ColSpan { get; set; } = 1;
    }
}
#endif
