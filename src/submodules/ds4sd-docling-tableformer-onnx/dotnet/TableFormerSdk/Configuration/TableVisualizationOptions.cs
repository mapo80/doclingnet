using SkiaSharp;
using System;
using TableFormerSdk.Constants;

namespace TableFormerSdk.Configuration;

public sealed class TableVisualizationOptions
{
    public TableVisualizationOptions(SKColor strokeColor, float strokeWidth)
    {
        if (strokeWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(strokeWidth), strokeWidth, TableFormerConstants.InvalidStrokeWidthMessage);
        }

        StrokeColor = strokeColor;
        StrokeWidth = strokeWidth;
    }

    public SKColor StrokeColor { get; }

    public float StrokeWidth { get; }

    public static TableVisualizationOptions CreateDefault() => new(TableFormerConstants.DefaultOverlayColor, TableFormerConstants.DefaultOverlayStrokeWidth);
}
