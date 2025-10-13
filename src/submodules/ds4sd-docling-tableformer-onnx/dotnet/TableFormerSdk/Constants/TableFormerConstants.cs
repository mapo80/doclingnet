using SkiaSharp;

namespace TableFormerSdk.Constants;

public static class TableFormerConstants
{
    public const string AccurateModelNotConfiguredMessage = "Accurate model path is not configured";
    public const string UnsupportedModelVariantMessage = "Unsupported model variant";
    public const string UnsupportedLanguageMessage = "The requested language '{0}' is not supported by the current SDK configuration";
    public const string InvalidStrokeWidthMessage = "Stroke width must be greater than zero";

    public const float DefaultOverlayStrokeWidth = 2f;

    public static readonly SKColor DefaultOverlayColor = SKColors.Lime;
}
