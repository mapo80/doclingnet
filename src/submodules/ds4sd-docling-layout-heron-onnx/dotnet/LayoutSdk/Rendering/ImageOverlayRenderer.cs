using SkiaSharp;
using System;
using System.Collections.Generic;

namespace LayoutSdk.Rendering;

internal sealed class ImageOverlayRenderer : IImageOverlayRenderer
{
    public SKBitmap CreateOverlay(SKBitmap baseImage, IReadOnlyList<BoundingBox> boxes)
    {
        if (baseImage == null)
        {
            throw new ArgumentNullException(nameof(baseImage));
        }

        if (boxes == null)
        {
            throw new ArgumentNullException(nameof(boxes));
        }

        var copy = baseImage.Copy();
        using var canvas = new SKCanvas(copy);
        using var paint = new SKPaint
        {
            Color = LayoutDefaults.OverlayColor,
            IsStroke = true,
            StrokeWidth = LayoutDefaults.OverlayStrokeWidth
        };

        foreach (var box in boxes)
        {
            canvas.DrawRect(box.X, box.Y, box.Width, box.Height, paint);
        }

        return copy;
    }
}
