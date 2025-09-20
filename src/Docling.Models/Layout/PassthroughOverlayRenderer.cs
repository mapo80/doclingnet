using System;
using System.Collections.Generic;
using LayoutSdk;
using LayoutSdk.Rendering;
using SkiaSharp;

namespace Docling.Models.Layout;

internal sealed class PassthroughOverlayRenderer : IImageOverlayRenderer
{
    public SKBitmap CreateOverlay(SKBitmap baseImage, IReadOnlyList<BoundingBox> boxes)
    {
        ArgumentNullException.ThrowIfNull(baseImage);
        _ = boxes;

        var clone = baseImage.Copy();
        if (clone is not null)
        {
            return clone;
        }

        var fallback = new SKBitmap(baseImage.Info);
        if (!baseImage.CopyTo(fallback))
        {
            fallback.Dispose();
            throw new InvalidOperationException("Failed to clone the layout overlay bitmap.");
        }

        return fallback;
    }
}
