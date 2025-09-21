using System;
using System.Collections.Generic;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Core.Documents;

/// <summary>
/// Represents an image/figure item detected on a page.
/// Mirrors the Python Docling picture entity storing optional descriptions.
/// </summary>
public sealed class PictureItem : DocItem
{
    private ImageRef? _image;

    public PictureItem(
        PageReference page,
        BoundingBox box,
        string? description = null,
        string? id = null,
        IEnumerable<string>? tags = null,
        IReadOnlyDictionary<string, object?>? metadata = null,
        DateTimeOffset? createdAt = null)
        : base(DocItemKind.Picture, page, box, id, tags, metadata, createdAt)
    {
        UpdateDescription(description ?? string.Empty);
    }

    public string Description { get; private set; } = string.Empty;

    public ImageRef? Image => _image;

    public void UpdateDescription(string description)
    {
        ArgumentNullException.ThrowIfNull(description);
        Description = description;
        if (!string.IsNullOrWhiteSpace(description))
        {
            SetMetadata("description", description);
        }
    }

    public void SetImage(ImageRef image)
    {
        ArgumentNullException.ThrowIfNull(image);
        _image = image;
        SetMetadata("docling:image_ref", image.Metadata);
        SetMetadata("docling:image_media_type", image.MediaType);
        SetMetadata("docling:image_width", image.Width);
        SetMetadata("docling:image_height", image.Height);
        SetMetadata("docling:image_dpi", image.Dpi);
    }

    public void ClearImage()
    {
        _image = null;
        SetMetadata(
            "docling:image_ref",
            new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase));
        SetMetadata<string?>("docling:image_media_type", null);
        SetMetadata("docling:image_width", 0);
        SetMetadata("docling:image_height", 0);
        SetMetadata("docling:image_dpi", 0d);
    }
}
