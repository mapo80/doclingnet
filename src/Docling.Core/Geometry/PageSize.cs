using System;

namespace Docling.Core.Geometry;

/// <summary>
/// Represents the pixel dimensions of a page together with its DPI metadata.
/// </summary>
public readonly record struct PageSize
{
    public PageSize(double width, double height, double dpi)
    {
        Validate(width, height, dpi);
        Width = width;
        Height = height;
        Dpi = dpi;
    }

    public double Width { get; }

    public double Height { get; }

    public double Dpi { get; }

    public bool IsLandscape => Width >= Height;

    public bool IsPortrait => Height > Width;

    public double Area => Width * Height;

    public double AspectRatio => Height <= 0 ? double.NaN : Width / Height;

    public double WidthInInches => Width / Dpi;

    public double HeightInInches => Height / Dpi;

    public PageSize Rotate() => new(Height, Width, Dpi);

    public PageSize WithDpi(double dpi)
    {
        Validate(Width, Height, dpi);
        return new PageSize(Width, Height, dpi);
    }

    public PageSize ScaleToDpi(double targetDpi)
    {
        Validate(Width, Height, targetDpi);
        var scale = targetDpi / Dpi;
        return new PageSize(Width * scale, Height * scale, targetDpi);
    }

    public PageSize Inflate(double padding)
    {
        var width = Width + (padding * 2);
        var height = Height + (padding * 2);
        Validate(width, height, Dpi);
        return new PageSize(width, height, Dpi);
    }

    public override string ToString() => $"{Width}x{Height}@{Dpi}dpi";

    private static void Validate(double width, double height, double dpi)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(width, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(height, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(dpi, 0);
    }
}
