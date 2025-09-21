using System;
using Docling.Core.Documents;
using Docling.Export.Imaging;

namespace Docling.Export.Serialization;

/// <summary>
/// Represents an asset that should be emitted alongside the markdown payload.
/// </summary>
public sealed class MarkdownAsset
{
    public MarkdownAsset(string relativePath, ImageExportKind kind, ImageRef image, string? targetItemId)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
        {
            throw new ArgumentException("Relative path must be provided for markdown assets.", nameof(relativePath));
        }

        ArgumentNullException.ThrowIfNull(image);

        RelativePath = relativePath;
        Kind = kind;
        Image = image;
        TargetItemId = string.IsNullOrWhiteSpace(targetItemId) ? null : targetItemId;
    }

    /// <summary>
    /// Gets the relative path where the asset should be written.
    /// </summary>
    public string RelativePath { get; }

    /// <summary>
    /// Gets the logical usage of the asset (figure, table, page, ...).
    /// </summary>
    public ImageExportKind Kind { get; }

    /// <summary>
    /// Gets the image payload referenced by the markdown.
    /// </summary>
    public ImageRef Image { get; }

    /// <summary>
    /// Gets the optional identifier of the document item the asset belongs to.
    /// </summary>
    public string? TargetItemId { get; }
}
