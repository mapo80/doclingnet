using System;
using System.Collections.Generic;

namespace Docling.Tooling.Parity;

/// <summary>
/// Represents a normalised snapshot of a pipeline run suitable for parity comparisons against Python goldens.
/// </summary>
internal sealed class ParityExtractionResult
{
    public ParityExtractionResult(
        string documentId,
        string markdownPath,
        string markdownSha256,
        IReadOnlyDictionary<string, string> documentProperties,
        IReadOnlyDictionary<string, string> markdownMetadata,
        IReadOnlyList<ParityPageSnapshot> pages,
        IReadOnlyList<ParityAssetSnapshot> assets,
        DateTimeOffset generatedAtUtc)
    {
        DocumentId = string.IsNullOrWhiteSpace(documentId)
            ? throw new ArgumentException("Document id must be provided.", nameof(documentId))
            : documentId;
        MarkdownPath = markdownPath ?? throw new ArgumentNullException(nameof(markdownPath));
        MarkdownSha256 = markdownSha256 ?? throw new ArgumentNullException(nameof(markdownSha256));
        DocumentProperties = documentProperties ?? throw new ArgumentNullException(nameof(documentProperties));
        MarkdownMetadata = markdownMetadata ?? throw new ArgumentNullException(nameof(markdownMetadata));
        Pages = pages ?? throw new ArgumentNullException(nameof(pages));
        Assets = assets ?? throw new ArgumentNullException(nameof(assets));
        GeneratedAtUtc = generatedAtUtc;
    }

    /// <summary>
    /// Gets the identifier assigned to the processed document.
    /// </summary>
    public string DocumentId { get; }

    /// <summary>
    /// Gets the normalised markdown relative path.
    /// </summary>
    public string MarkdownPath { get; }

    /// <summary>
    /// Gets the SHA-256 checksum of the markdown payload.
    /// </summary>
    public string MarkdownSha256 { get; }

    /// <summary>
    /// Gets the document level properties captured during the run.
    /// </summary>
    public IReadOnlyDictionary<string, string> DocumentProperties { get; }

    /// <summary>
    /// Gets the metadata emitted by the markdown serializer.
    /// </summary>
    public IReadOnlyDictionary<string, string> MarkdownMetadata { get; }

    /// <summary>
    /// Gets the collection of normalised page snapshots.
    /// </summary>
    public IReadOnlyList<ParityPageSnapshot> Pages { get; }

    /// <summary>
    /// Gets the collection of image assets referenced from the markdown output.
    /// </summary>
    public IReadOnlyList<ParityAssetSnapshot> Assets { get; }

    /// <summary>
    /// Gets the timestamp when the snapshot was generated (UTC).
    /// </summary>
    public DateTimeOffset GeneratedAtUtc { get; }
}

/// <summary>
/// Represents a normalised representation of a document page.
/// </summary>
internal sealed class ParityPageSnapshot
{
    public ParityPageSnapshot(int pageNumber, double dpi)
    {
        PageNumber = pageNumber;
        Dpi = dpi;
    }

    public int PageNumber { get; }

    public double Dpi { get; }
}

/// <summary>
/// Represents a normalised image asset referenced by the markdown output.
/// </summary>
internal sealed class ParityAssetSnapshot
{
    public ParityAssetSnapshot(
        string relativePath,
        string kind,
        string? targetItemId,
        string imageId,
        int pageNumber,
        double dpi,
        int width,
        int height,
        string mediaType,
        string checksum,
        ParityBoundingBox boundingBox)
    {
        RelativePath = relativePath ?? throw new ArgumentNullException(nameof(relativePath));
        Kind = string.IsNullOrWhiteSpace(kind) ? throw new ArgumentException("Asset kind must be provided.", nameof(kind)) : kind;
        TargetItemId = string.IsNullOrWhiteSpace(targetItemId) ? null : targetItemId;
        ImageId = string.IsNullOrWhiteSpace(imageId) ? throw new ArgumentException("Image id must be provided.", nameof(imageId)) : imageId;
        PageNumber = pageNumber;
        Dpi = dpi;
        Width = width;
        Height = height;
        MediaType = mediaType ?? throw new ArgumentNullException(nameof(mediaType));
        Checksum = checksum ?? throw new ArgumentNullException(nameof(checksum));
        BoundingBox = boundingBox;
    }

    public string RelativePath { get; }

    public string Kind { get; }

    public string? TargetItemId { get; }

    public string ImageId { get; }

    public int PageNumber { get; }

    public double Dpi { get; }

    public int Width { get; }

    public int Height { get; }

    public string MediaType { get; }

    public string Checksum { get; }

    public ParityBoundingBox BoundingBox { get; }
}

/// <summary>
/// Represents a bounding box after tolerance normalisation.
/// </summary>
internal readonly record struct ParityBoundingBox(double Left, double Top, double Right, double Bottom);
