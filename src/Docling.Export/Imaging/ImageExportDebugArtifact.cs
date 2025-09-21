using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Export.Imaging;

/// <summary>
/// Represents the debug payload emitted for image export crops, containing both the visual overlay and a JSON manifest.
/// </summary>
public sealed class ImageExportDebugArtifact
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNamingPolicy = null,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    public ImageExportDebugArtifact(
        PageReference page,
        ReadOnlyMemory<byte> overlayImage,
        ImageExportDebugManifest manifest,
        string overlayMediaType = "image/png",
        string manifestMediaType = "application/json")
    {
        if (overlayImage.IsEmpty)
        {
            throw new ArgumentException("Overlay image content cannot be empty.", nameof(overlayImage));
        }

        Page = page;
        OverlayImage = overlayImage;
        OverlayMediaType = overlayMediaType ?? throw new ArgumentNullException(nameof(overlayMediaType));
        Manifest = manifest ?? throw new ArgumentNullException(nameof(manifest));
        ManifestMediaType = manifestMediaType ?? throw new ArgumentNullException(nameof(manifestMediaType));
        ManifestContent = JsonSerializer.SerializeToUtf8Bytes(Manifest, SerializerOptions);
    }

    /// <summary>
    /// Gets the page reference the debug artefact corresponds to.
    /// </summary>
    public PageReference Page { get; }

    /// <summary>
    /// Gets the rendered overlay image highlighting exported crops.
    /// </summary>
    public ReadOnlyMemory<byte> OverlayImage { get; }

    /// <summary>
    /// Gets the media type associated with the overlay image.
    /// </summary>
    public string OverlayMediaType { get; }

    /// <summary>
    /// Gets the strongly typed manifest describing the exported crops for the page.
    /// </summary>
    public ImageExportDebugManifest Manifest { get; }

    /// <summary>
    /// Gets the media type associated with the manifest payload.
    /// </summary>
    public string ManifestMediaType { get; }

    /// <summary>
    /// Gets the JSON-encoded manifest content.
    /// </summary>
    public ReadOnlyMemory<byte> ManifestContent { get; }
}

/// <summary>
/// Describes the exported crops for a single page in a JSON-friendly structure.
/// </summary>
public sealed class ImageExportDebugManifest
{
    public ImageExportDebugManifest(string documentId, int pageNumber, IReadOnlyList<ImageExportDebugEntry> items)
    {
        if (string.IsNullOrWhiteSpace(documentId))
        {
            throw new ArgumentException("Document identifier cannot be null or empty.", nameof(documentId));
        }

        DocumentId = documentId;
        PageNumber = pageNumber;
        Items = items ?? throw new ArgumentNullException(nameof(items));
    }

    [JsonPropertyName("document_id")]
    public string DocumentId { get; }

    [JsonPropertyName("page_number")]
    public int PageNumber { get; }

    [JsonPropertyName("items")]
    public IReadOnlyList<ImageExportDebugEntry> Items { get; }
}

/// <summary>
/// Captures the metadata for a single exported crop.
/// </summary>
public sealed class ImageExportDebugEntry
{
    public ImageExportDebugEntry(
        string targetItemId,
        string imageId,
        ImageExportKind kind,
        ImageExportDebugBounds originalBounds,
        ImageExportDebugBounds cropBounds,
        string mediaType,
        int width,
        int height,
        double dpi,
        string? checksum)
    {
        TargetItemId = targetItemId ?? throw new ArgumentNullException(nameof(targetItemId));
        ImageId = imageId ?? throw new ArgumentNullException(nameof(imageId));
        Kind = kind;
        OriginalBounds = originalBounds ?? throw new ArgumentNullException(nameof(originalBounds));
        CropBounds = cropBounds ?? throw new ArgumentNullException(nameof(cropBounds));
        MediaType = mediaType ?? throw new ArgumentNullException(nameof(mediaType));
        Width = width;
        Height = height;
        Dpi = dpi;
        Checksum = checksum;
        KindName = kind switch
        {
            ImageExportKind.Picture => "picture",
            ImageExportKind.Table => "table",
            ImageExportKind.Page => "page",
            _ => ToLowerInvariant(kind.ToString()),
        };
    }

    [JsonPropertyName("target_item_id")]
    public string TargetItemId { get; }

    [JsonPropertyName("image_id")]
    public string ImageId { get; }

    [JsonIgnore]
    public ImageExportKind Kind { get; }

    [JsonPropertyName("kind")]
    public string KindName { get; }

    [JsonPropertyName("original_bounds")]
    public ImageExportDebugBounds OriginalBounds { get; }

    [JsonPropertyName("crop_bounds")]
    public ImageExportDebugBounds CropBounds { get; }

    [JsonPropertyName("media_type")]
    public string MediaType { get; }

    [JsonPropertyName("width")]
    public int Width { get; }

    [JsonPropertyName("height")]
    public int Height { get; }

    [JsonPropertyName("dpi")]
    public double Dpi { get; }

    [JsonPropertyName("checksum")]
    public string? Checksum { get; }

    private static string ToLowerInvariant(string value)
    {
        return string.Create(value.Length, value, static (span, source) =>
        {
            for (var i = 0; i < source.Length; i++)
            {
                span[i] = char.ToLowerInvariant(source[i]);
            }
        });
    }
}

/// <summary>
/// Represents the bounding box information for an exported crop in a serialization-friendly format.
/// </summary>
public sealed record ImageExportDebugBounds(
    [property: JsonPropertyName("left")] double Left,
    [property: JsonPropertyName("top")] double Top,
    [property: JsonPropertyName("right")] double Right,
    [property: JsonPropertyName("bottom")] double Bottom)
{
    public static ImageExportDebugBounds FromBoundingBox(BoundingBox box)
    {
        var normalized = box.Normalized();
        return new ImageExportDebugBounds(normalized.Left, normalized.Top, normalized.Right, normalized.Bottom);
    }
}
