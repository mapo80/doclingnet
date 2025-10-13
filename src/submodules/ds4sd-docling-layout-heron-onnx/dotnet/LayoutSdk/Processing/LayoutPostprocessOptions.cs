using System;
using System.Collections.Generic;

namespace LayoutSdk.Processing;

/// <summary>
/// Configuration options for the LayoutPostprocessor.
/// </summary>
public sealed class LayoutPostprocessOptions
{
    private static readonly string[] DefaultLabels =
    {
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
        "Document Index",
        "Code",
        "Checkbox-Selected",
        "Checkbox-Unselected",
        "Form",
        "Key-Value Region"
    };

    /// <summary>
    /// Threshold for Union-Find merge operations (IoU).
    /// </summary>
    public float UnionFindMergeThreshold { get; set; } = 0.3f;

    /// <summary>
    /// Maximum relative distance for merging same-label boxes.
    /// </summary>
    public float MaxRelativeDistance { get; set; } = 0.5f;

    /// <summary>
    /// Default confidence threshold for filtering (matches Python threshold=0.25).
    /// </summary>
    public float DefaultThreshold { get; set; } = 0.25f;

    /// <summary>
    /// Label-specific confidence thresholds.
    /// </summary>
    public Dictionary<string, float> LabelThresholds { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    internal float GetThreshold(string label) =>
        LabelThresholds.TryGetValue(label, out var threshold)
            ? threshold
            : DefaultThreshold;

    /// <summary>
    /// Indicates whether the model was trained with focal loss (RT-DETR models use focal loss).
    /// </summary>
    public bool UseFocalLoss { get; set; } = true;

    /// <summary>
    /// Label map matching the RT-DETR <c>id2label</c> metadata.
    /// </summary>
    public IReadOnlyList<string> Labels { get; set; } = DefaultLabels;

    /// <summary>
    /// Default size constraints for boxes.
    /// </summary>
    public SizeConstraint DefaultSizeConstraint { get; set; } = new SizeConstraint
    {
        MinWidth = 1f,
        MinHeight = 1f,
        MaxWidth = float.MaxValue,
        MaxHeight = float.MaxValue
    };

    /// <summary>
    /// Label-specific size constraints.
    /// </summary>
    public Dictionary<string, SizeConstraint> LabelSizeConstraints { get; set; } = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Caption"] = new SizeConstraint { MinWidth = 10f, MinHeight = 5f, MaxWidth = 1000f, MaxHeight = 100f },
        ["Footnote"] = new SizeConstraint { MinWidth = 20f, MinHeight = 10f, MaxWidth = 800f, MaxHeight = 50f },
        ["Formula"] = new SizeConstraint { MinWidth = 15f, MinHeight = 10f, MaxWidth = 500f, MaxHeight = 200f },
        ["List-item"] = new SizeConstraint { MinWidth = 10f, MinHeight = 8f, MaxWidth = 600f, MaxHeight = 30f },
        ["Page-footer"] = new SizeConstraint { MinWidth = 50f, MinHeight = 10f, MaxWidth = 1000f, MaxHeight = 50f },
        ["Page-header"] = new SizeConstraint { MinWidth = 50f, MinHeight = 10f, MaxWidth = 1000f, MaxHeight = 50f },
        ["Picture"] = new SizeConstraint { MinWidth = 20f, MinHeight = 20f, MaxWidth = 2000f, MaxHeight = 2000f },
        ["Section-header"] = new SizeConstraint { MinWidth = 30f, MinHeight = 15f, MaxWidth = 800f, MaxHeight = 60f },
        ["Table"] = new SizeConstraint { MinWidth = 50f, MinHeight = 30f, MaxWidth = 1500f, MaxHeight = 1500f },
        ["Text"] = new SizeConstraint { MinWidth = 5f, MinHeight = 5f, MaxWidth = 1200f, MaxHeight = 1000f },
        ["Title"] = new SizeConstraint { MinWidth = 20f, MinHeight = 15f, MaxWidth = 1000f, MaxHeight = 80f }
    };

    /// <summary>
    /// Cell size for spatial indexing.
    /// </summary>
    public float SpatialIndexCellSize { get; set; } = 50f;

    /// <summary>
    /// Radius for spatial context analysis.
    /// </summary>
    public float SpatialContextRadius { get; set; } = 100f;

    /// <summary>
    /// Radius for relationship analysis between boxes.
    /// </summary>
    public float RelationshipAnalysisRadius { get; set; } = 150f;

    /// <summary>
    /// Enables or disables debug logging.
    /// </summary>
    public bool EnableDebugLogging { get; set; } = false;

    /// <summary>
    /// Maximum number of iterations for Union-Find operations.
    /// </summary>
    public int MaxUnionFindIterations { get; set; } = 1000;

    /// <summary>
    /// Tolerance for floating-point comparisons.
    /// </summary>
    public float FloatingPointTolerance { get; set; } = 1e-6f;

    /// <summary>
    /// Creates default options optimized for document layout processing (matches Python behavior).
    /// </summary>
    /// <returns>Default options</returns>
    public static LayoutPostprocessOptions CreateDefault()
    {
        return new LayoutPostprocessOptions
        {
            UnionFindMergeThreshold = 0.3f,
            MaxRelativeDistance = 0.5f,
            DefaultThreshold = 0.25f,  // Match Python threshold=0.25
            Labels = DefaultLabels,
            SpatialIndexCellSize = 50f,
            SpatialContextRadius = 100f,
            RelationshipAnalysisRadius = 150f,
            EnableDebugLogging = false,
            MaxUnionFindIterations = 1000,
            FloatingPointTolerance = 1e-6f
        };
    }

    /// <summary>
    /// Creates options optimized for high-precision layout processing.
    /// </summary>
    /// <returns>High-precision options</returns>
    public static LayoutPostprocessOptions CreateHighPrecision()
    {
        return new LayoutPostprocessOptions
        {
            UnionFindMergeThreshold = 0.1f,  // Lower threshold for more merging
            MaxRelativeDistance = 0.3f,      // Closer distance for merging
            DefaultThreshold = 0.5f,         // Higher confidence threshold
            Labels = DefaultLabels,
            SpatialIndexCellSize = 25f,       // Smaller cells for precision
            SpatialContextRadius = 75f,       // Smaller context radius
            RelationshipAnalysisRadius = 100f,
            EnableDebugLogging = true,
            MaxUnionFindIterations = 2000,
            FloatingPointTolerance = 1e-8f
        };
    }

    /// <summary>
    /// Creates options optimized for performance.
    /// </summary>
    /// <returns>Performance-optimized options</returns>
    public static LayoutPostprocessOptions CreatePerformanceOptimized()
    {
        return new LayoutPostprocessOptions
        {
            UnionFindMergeThreshold = 0.5f,   // Higher threshold for less merging
            MaxRelativeDistance = 0.8f,       // Larger distance for merging
            DefaultThreshold = 0.2f,          // Lower confidence threshold
            Labels = DefaultLabels,
            SpatialIndexCellSize = 100f,      // Larger cells for performance
            SpatialContextRadius = 150f,      // Larger context radius
            RelationshipAnalysisRadius = 200f,
            EnableDebugLogging = false,
            MaxUnionFindIterations = 500,
            FloatingPointTolerance = 1e-4f
        };
    }
}

/// <summary>
/// Size constraints for bounding boxes.
/// </summary>
public sealed class SizeConstraint
{
    /// <summary>
    /// Minimum width in pixels.
    /// </summary>
    public float MinWidth { get; set; }

    /// <summary>
    /// Minimum height in pixels.
    /// </summary>
    public float MinHeight { get; set; }

    /// <summary>
    /// Maximum width in pixels.
    /// </summary>
    public float MaxWidth { get; set; }

    /// <summary>
    /// Maximum height in pixels.
    /// </summary>
    public float MaxHeight { get; set; }

    /// <summary>
    /// Checks if a bounding box satisfies the size constraints.
    /// </summary>
    /// <param name="box">Bounding box to check</param>
    /// <returns>True if the box satisfies the constraints</returns>
    public bool IsSatisfiedBy(BoundingBox box)
    {
        return box.Width >= MinWidth && box.Height >= MinHeight &&
               box.Width <= MaxWidth && box.Height <= MaxHeight;
    }
}
