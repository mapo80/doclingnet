using System;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using SkiaSharp;

namespace Docling.Backends.Pdf;

/// <summary>
/// Represents a rendered PDF page together with its provenance metadata.
/// </summary>
public sealed class PageImage : IDisposable
{
    public PageImage(PageReference page, SKBitmap bitmap, PageImageMetadata? metadata = null)
    {
        Page = page;
        Bitmap = bitmap ?? throw new ArgumentNullException(nameof(bitmap));
        Metadata = metadata ?? PageImageMetadata.Empty;
        BoundingBox = BoundingBox.FromSize(0, 0, bitmap.Width, bitmap.Height);
    }

    public PageReference Page { get; }

    public SKBitmap Bitmap { get; }

    public BoundingBox BoundingBox { get; }

    public PageImageMetadata Metadata { get; internal set; }

    public int Width => Bitmap.Width;

    public int Height => Bitmap.Height;

    public PageImage Clone()
    {
        var cloneBitmap = Bitmap.Copy();
        if (cloneBitmap is null)
        {
            throw new InvalidOperationException("Failed to clone bitmap for page image.");
        }

        return new PageImage(Page, cloneBitmap, Metadata);
    }

    public void Dispose() => Bitmap.Dispose();
}
