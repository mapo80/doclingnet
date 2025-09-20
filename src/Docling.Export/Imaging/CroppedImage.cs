using System;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using SkiaSharp;

namespace Docling.Export.Imaging;

/// <summary>
/// Represents a cropped portion of a page image.
/// </summary>
public sealed class CroppedImage : IDisposable
{
    public CroppedImage(PageReference page, BoundingBox sourceRegion, SKBitmap bitmap)
    {
        Page = page;
        SourceRegion = sourceRegion;
        Bitmap = bitmap ?? throw new ArgumentNullException(nameof(bitmap));
    }

    public PageReference Page { get; }

    public BoundingBox SourceRegion { get; }

    public SKBitmap Bitmap { get; }

    public void Dispose() => Bitmap.Dispose();
}
