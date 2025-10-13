using LayoutSdk.Configuration;
using LayoutSdk.Metrics;
using SkiaSharp;
using System.Collections.Generic;

namespace LayoutSdk;

public record BoundingBox(float X, float Y, float Width, float Height, string Label, float Confidence = 1.0f);

public class LayoutResult
{
    public LayoutResult(
        IReadOnlyList<BoundingBox> boxes,
        SKBitmap? overlay,
        DocumentLanguage language,
        LayoutExecutionMetrics metrics)
    {
        Boxes = boxes;
        OverlayImage = overlay;
        Language = language;
        Metrics = metrics;
    }

    public IReadOnlyList<BoundingBox> Boxes { get; }

    public SKBitmap? OverlayImage { get; }

    public DocumentLanguage Language { get; }

    public LayoutExecutionMetrics Metrics { get; }
}
