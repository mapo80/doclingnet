namespace Docling.Pipelines.Options;

/// <summary>
/// Defines the TableFormer inference mode.
/// </summary>
public enum TableFormerMode
{
    Fast,
    Accurate,
}

/// <summary>
/// Options controlling the table structure reconstruction stage.
/// </summary>
public sealed class TableStructureOptions
{
    /// <summary>
    /// Whether to reconcile predicted cell structure with the originating PDF cell graph.
    /// </summary>
    public bool DoCellMatching { get; init; } = true;

    /// <summary>
    /// Desired accuracy/speed trade-off for the TableFormer model.
    /// </summary>
    public TableFormerMode Mode { get; init; } = TableFormerMode.Accurate;
}
