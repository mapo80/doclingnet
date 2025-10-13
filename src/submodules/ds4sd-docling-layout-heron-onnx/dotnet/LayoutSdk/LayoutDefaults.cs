using SkiaSharp;

namespace LayoutSdk;

internal static class LayoutDefaults
{
    public const float OverlayStrokeWidth = 2f;
    public static readonly SKColor OverlayColor = SKColors.Lime;

    public const string EmptyImagePathMessage = "Image path is empty";
    public const string ImageNotFoundMessage = "Image not found";
    public const string ImageDecodeFailureMessage = "Unable to decode image";
}
