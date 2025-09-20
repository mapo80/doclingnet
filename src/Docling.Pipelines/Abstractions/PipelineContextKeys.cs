using Docling.Backends.Storage;
using Docling.Core.Primitives;

namespace Docling.Pipelines.Abstractions;

/// <summary>
/// Well known keys used to exchange state between pipeline stages via <see cref="PipelineContext"/>.
/// </summary>
public static class PipelineContextKeys
{
    /// <summary>
    /// Gets the key under which the shared <see cref="PageImageStore"/> instance is registered.
    /// </summary>
    public const string PageImageStore = "Docling.Pipelines.PageImageStore";

    /// <summary>
    /// Gets the key referencing the ordered list of <see cref="PageReference"/> instances that compose the current document.
    /// </summary>
    public const string PageSequence = "Docling.Pipelines.PageSequence";

    /// <summary>
    /// Flag indicating that preprocessing completed successfully.
    /// </summary>
    public const string PreprocessingCompleted = "Docling.Pipelines.PreprocessingCompleted";

    /// <summary>
    /// Gets the key identifying the logical document id flowing through the pipeline.
    /// </summary>
    public const string DocumentId = "Docling.Pipelines.DocumentId";

    /// <summary>
    /// Gets the key referencing the layout predictions produced for the current document.
    /// </summary>
    public const string LayoutItems = "Docling.Pipelines.LayoutItems";

    /// <summary>
    /// Flag indicating that layout analysis completed successfully.
    /// </summary>
    public const string LayoutAnalysisCompleted = "Docling.Pipelines.LayoutAnalysisCompleted";

    /// <summary>
    /// Gets the key referencing optional layout debug overlays.
    /// </summary>
    public const string LayoutDebugArtifacts = "Docling.Pipelines.LayoutDebugArtifacts";
}
