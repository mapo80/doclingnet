namespace Docling.Backends.Pdf;

/// <summary>
/// Canonical metadata property names attached to <see cref="PageImage"/> instances.
/// </summary>
public static class PageImageMetadataKeys
{
    public const string SourceHorizontalDpi = "docling.source.dpi";
    public const string NormalizedDpi = "docling.preprocess.dpi";
    public const string ScaleFactor = "docling.preprocess.scale";
    public const string ColorMode = "docling.preprocess.color_mode";
    public const string DeskewAngle = "docling.preprocess.deskew_angle";
}
