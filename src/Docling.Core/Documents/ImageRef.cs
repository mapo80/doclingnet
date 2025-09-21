using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Core.Documents;

/// <summary>
/// Represents a binary image artefact associated with a document item or page.
/// Mirrors the Python <c>ImageRef</c> metadata object storing dimensions, DPI, and media information.
/// </summary>
public sealed class ImageRef
{
    private readonly byte[] _buffer;
    private readonly ReadOnlyDictionary<string, object?> _metadata;

    public ImageRef(
        string id,
        PageReference page,
        BoundingBox sourceRegion,
        string mediaType,
        ReadOnlySpan<byte> data,
        int width,
        int height,
        double dpi,
        string? checksum = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(id);
        ArgumentNullException.ThrowIfNull(mediaType);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width, nameof(width));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height, nameof(height));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(dpi, 0, nameof(dpi));

        Id = id;
        Page = page;
        SourceRegion = sourceRegion.Normalized();
        MediaType = mediaType;
        Width = width;
        Height = height;
        Dpi = dpi;
        Checksum = checksum;

        _buffer = data.Length == 0
            ? Array.Empty<byte>()
            : data.ToArray();

        var metadata = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
        {
            ["id"] = Id,
            ["page_number"] = Page.PageNumber,
            ["dpi"] = Dpi,
            ["media_type"] = MediaType,
            ["width"] = Width,
            ["height"] = Height,
            ["bounding_box"] = new[] { SourceRegion.Left, SourceRegion.Top, SourceRegion.Right, SourceRegion.Bottom },
        };

        if (!string.IsNullOrWhiteSpace(Checksum))
        {
            metadata["checksum"] = Checksum;
        }

        _metadata = new ReadOnlyDictionary<string, object?>(metadata);
    }

    /// <summary>
    /// Gets the unique identifier assigned to the image artefact.
    /// </summary>
    public string Id { get; }

    /// <summary>
    /// Gets the page reference the image originates from.
    /// </summary>
    public PageReference Page { get; }

    /// <summary>
    /// Gets the normalized region that was cropped from the source page image.
    /// </summary>
    public BoundingBox SourceRegion { get; }

    /// <summary>
    /// Gets the MIME type of the encoded binary payload.
    /// </summary>
    public string MediaType { get; }

    /// <summary>
    /// Gets the encoded image width in pixels.
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// Gets the encoded image height in pixels.
    /// </summary>
    public int Height { get; }

    /// <summary>
    /// Gets the DPI associated with the source page.
    /// </summary>
    public double Dpi { get; }

    /// <summary>
    /// Gets the checksum associated with the encoded payload, if available.
    /// </summary>
    public string? Checksum { get; }

    /// <summary>
    /// Gets a read-only view over the encoded image bytes.
    /// </summary>
    public ReadOnlyMemory<byte> Data => _buffer;

    /// <summary>
    /// Gets a metadata snapshot mirroring the Python image ref dictionary representation.
    /// </summary>
    public IReadOnlyDictionary<string, object?> Metadata => _metadata;
}
