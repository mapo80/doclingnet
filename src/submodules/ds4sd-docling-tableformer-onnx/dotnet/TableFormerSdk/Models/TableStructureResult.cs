using TableFormerSdk.Enums;

namespace TableFormerSdk.Models;

/// <summary>
/// Result of TableFormer inference
/// </summary>
public sealed class TableStructureResult
{
    public TableStructureResult(
        IReadOnlyList<TableRegion> regions,
        TableFormerModelVariant modelVariant,
        TimeSpan inferenceTime,
        Dictionary<string, int[]>? rawOutputShapes = null)
    {
        Regions = regions ?? Array.Empty<TableRegion>();
        ModelVariant = modelVariant;
        InferenceTime = inferenceTime;
        RawOutputShapes = rawOutputShapes ?? new Dictionary<string, int[]>();
    }

    /// <summary>
    /// Detected table cells/regions
    /// </summary>
    public IReadOnlyList<TableRegion> Regions { get; }

    /// <summary>
    /// Model variant used for inference
    /// </summary>
    public TableFormerModelVariant ModelVariant { get; }

    /// <summary>
    /// Time taken for inference
    /// </summary>
    public TimeSpan InferenceTime { get; }

    /// <summary>
    /// Raw output tensor shapes (for debugging)
    /// </summary>
    public Dictionary<string, int[]> RawOutputShapes { get; }

    public override string ToString() =>
        $"TableStructureResult: {Regions.Count} regions, {InferenceTime.TotalMilliseconds:F2}ms ({ModelVariant})";
}
