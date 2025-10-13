using SkiaSharp;
using System;
using System.Collections.Generic;
using TableFormerSdk.Configuration;
using TableFormerSdk.Models;

namespace TableFormerSdk.Rendering;

internal sealed class OverlayRenderer
{
    private readonly TableVisualizationOptions _options;

    public OverlayRenderer(TableVisualizationOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public SKBitmap? CreateOverlay(SKBitmap baseImage, IReadOnlyList<TableRegion> regions)
    {
        if (regions.Count == 0)
        {
            return null;
        }

        var copy = baseImage.Copy();
        using var canvas = new SKCanvas(copy);
        using var paint = new SKPaint
        {
            Color = _options.StrokeColor,
            IsStroke = true,
            StrokeWidth = _options.StrokeWidth
        };

        foreach (var region in regions)
        {
            canvas.DrawRect(region.X, region.Y, region.Width, region.Height, paint);
        }

        return copy;
    }
}
