using LayoutSdk;
using LayoutSdk.Rendering;
using SkiaSharp;
using System;
using System.Collections.Generic;
using Xunit;

namespace LayoutSdk.Tests;

public class ImageOverlayRendererTests
{
    [Fact]
    public void CreateOverlay_NullImage_Throws()
    {
        var renderer = new ImageOverlayRenderer();
        Assert.Throws<ArgumentNullException>(() => renderer.CreateOverlay(null!, Array.Empty<BoundingBox>()));
    }

    [Fact]
    public void CreateOverlay_NullBoxes_Throws()
    {
        using var bitmap = new SKBitmap(4, 4);
        var renderer = new ImageOverlayRenderer();
        Assert.Throws<ArgumentNullException>(() => renderer.CreateOverlay(bitmap, null!));
    }

    [Fact]
    public void CreateOverlay_DrawsRectangles()
    {
        using var bitmap = new SKBitmap(4, 4);
        bitmap.Erase(SKColors.Black);

        var boxes = new List<BoundingBox>
        {
            new(0, 0, 4, 4, "page")
        };

        var renderer = new ImageOverlayRenderer();
        using var overlay = renderer.CreateOverlay(bitmap, boxes);

        Assert.NotSame(bitmap, overlay);
        var pixel = overlay.GetPixel(0, 0);
        Assert.Equal(LayoutDefaults.OverlayColor, pixel);
    }
}
