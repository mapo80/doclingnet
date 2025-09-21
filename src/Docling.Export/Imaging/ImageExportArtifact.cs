using Docling.Core.Documents;

namespace Docling.Export.Imaging;

/// <summary>
/// Represents an exported image artefact alongside its logical usage within the pipeline.
/// </summary>
public sealed record ImageExportArtifact(
    ImageExportKind Kind,
    ImageRef Image,
    string? TargetItemId);

/// <summary>
/// Enumerates the supported export targets for generated images.
/// </summary>
public enum ImageExportKind
{
    Page,
    Picture,
    Table,
}
