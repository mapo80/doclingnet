using SkiaSharp;
using System.Collections.Generic;

namespace LayoutSdk.Rendering;

public interface IImageOverlayRenderer
{
    SKBitmap CreateOverlay(SKBitmap baseImage, IReadOnlyList<BoundingBox> boxes);
}
