using PDFtoImage;
using SkiaSharp;

namespace Docling.Backends.Pdf;

/// <summary>
/// Provides strongly typed configuration for PDF rasterisation.
/// </summary>
public sealed class PdfRenderSettings
{
    public int Dpi { get; init; } = 300;

    public bool WithAnnotations { get; init; } = true;

    public bool WithFormFill { get; init; } = true;

    public bool WithAspectRatio { get; init; } = true;

    public PdfRotation Rotation { get; init; } = PdfRotation.Rotate0;

    public PdfAntiAliasing AntiAliasing { get; init; } = PdfAntiAliasing.All;

    public SKColor? BackgroundColor { get; init; } = SKColors.White;

    public string? Password { get; init; }

    public RenderOptions ToRenderOptions() =>
        new(
            Dpi,
            null,
            null,
            WithAnnotations,
            WithFormFill,
            WithAspectRatio,
            Rotation,
            AntiAliasing,
            BackgroundColor,
            null,
            false,
            false);
}
