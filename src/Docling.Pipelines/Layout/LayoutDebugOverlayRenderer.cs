using System;
using System.Collections.Generic;
using System.Globalization;
using Docling.Backends.Pdf;
using Docling.Core.Geometry;
using Docling.Models.Layout;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Pipelines.Layout;

/// <summary>
/// Renders diagnostic overlays that visualise layout predictions on top of the processed page images.
/// </summary>
public sealed partial class LayoutDebugOverlayRenderer : ILayoutDebugOverlayRenderer
{
    private readonly LayoutDebugOverlayOptions _options;
    private readonly ILogger<LayoutDebugOverlayRenderer> _logger;

    public LayoutDebugOverlayRenderer(LayoutDebugOverlayOptions options, ILogger<LayoutDebugOverlayRenderer> logger)
    {
        _options = options?.Clone() ?? throw new ArgumentNullException(nameof(options));
        _options.EnsureValid();
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public LayoutDebugOverlay CreateOverlay(PageImage page, IReadOnlyList<LayoutItem> layoutItems)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(layoutItems);

        RendererLogger.Rendering(_logger, layoutItems.Count, page.Page.PageNumber);

        using var surface = SKSurface.Create(new SKImageInfo(page.Width, page.Height, SKColorType.Rgba8888, SKAlphaType.Premul));
        var canvas = surface.Canvas;
        canvas.Clear(SKColors.Transparent);
        canvas.DrawBitmap(page.Bitmap, 0, 0);

        if (_options.BackgroundOpacity > 0)
        {
            using var shade = new SKPaint
            {
                Style = SKPaintStyle.Fill,
                Color = new SKColor(0, 0, 0, (byte)Math.Clamp(_options.BackgroundOpacity * 255, 0, 255)),
            };
            canvas.DrawRect(SKRect.Create(page.Width, page.Height), shade);
        }

        using var fillPaint = new SKPaint { Style = SKPaintStyle.Fill, IsAntialias = true };
        using var strokePaint = new SKPaint { Style = SKPaintStyle.Stroke, IsAntialias = true, StrokeWidth = _options.StrokeWidth };
        using var textPaint = new SKPaint { IsAntialias = true, Color = SKColors.White };
        using var font = new SKFont(_options.Typeface, _options.LabelTextSize);

        foreach (var item in layoutItems)
        {
            var color = SelectColor(item.Kind);
            fillPaint.Color = color.WithAlpha((byte)Math.Clamp(_options.FillOpacity * 255, 0, 255));
            strokePaint.Color = color;

            DrawBoundingBox(canvas, strokePaint, fillPaint, item.BoundingBox);
            foreach (var polygon in item.Polygons)
            {
                DrawPolygon(canvas, strokePaint, fillPaint, polygon);
            }

            if (_options.DrawLabels)
            {
                var label = BuildLabel(item);
                var bounds = item.BoundingBox;
                var x = (float)bounds.Left + _options.LabelPadding;
                var y = (float)bounds.Top + _options.LabelPadding + font.Size;
                canvas.DrawText(label, x, y, SKTextAlign.Left, font, textPaint);
            }
        }

        using var snapshot = surface.Snapshot();
        using var encoded = snapshot.Encode(SKEncodedImageFormat.Png, 90);
        if (encoded is null || encoded.Size == 0)
        {
            throw new InvalidOperationException("Failed to encode layout debug overlay.");
        }

        return new LayoutDebugOverlay(page.Page, encoded.ToArray());
    }

    private static void DrawBoundingBox(SKCanvas canvas, SKPaint strokePaint, SKPaint fillPaint, BoundingBox box)
    {
        var rect = new SKRect((float)box.Left, (float)box.Top, (float)box.Right, (float)box.Bottom);
        if (fillPaint.Color.Alpha > 0)
        {
            canvas.DrawRect(rect, fillPaint);
        }

        canvas.DrawRect(rect, strokePaint);
    }

    private static void DrawPolygon(SKCanvas canvas, SKPaint strokePaint, SKPaint fillPaint, Polygon polygon)
    {
        using var path = new SKPath();
        path.MoveTo((float)polygon[0].X, (float)polygon[0].Y);
        for (var i = 1; i < polygon.Count; i++)
        {
            var point = polygon[i];
            path.LineTo((float)point.X, (float)point.Y);
        }

        path.Close();
        if (fillPaint.Color.Alpha > 0)
        {
            canvas.DrawPath(path, fillPaint);
        }

        canvas.DrawPath(path, strokePaint);
    }

    private static string BuildLabel(LayoutItem item)
    {
        var label = item.Kind.ToString();
        return string.Create(CultureInfo.InvariantCulture, $"{label} @ p{item.Page.PageNumber}");
    }

    private static SKColor SelectColor(LayoutItemKind kind) => kind switch
    {
        LayoutItemKind.Text => new SKColor(56, 142, 60),
        LayoutItemKind.Table => new SKColor(30, 136, 229),
        LayoutItemKind.Figure => new SKColor(251, 140, 0),
        _ => SKColors.Magenta,
    };

    private static partial class RendererLogger
    {
        [LoggerMessage(EventId = 3100, Level = LogLevel.Debug, Message = "Rendering {Count} layout predictions for page {Page}")]
        public static partial void Rendering(ILogger logger, int count, int page);
    }
}

/// <summary>
/// Configurable options for <see cref="LayoutDebugOverlayRenderer"/>.
/// </summary>
public sealed class LayoutDebugOverlayOptions
{
    public float BackgroundOpacity { get; init; } = 0.25f;

    public float FillOpacity { get; init; } = 0.18f;

    public float StrokeWidth { get; init; } = 2.5f;

    public bool DrawLabels { get; init; } = true;

    public float LabelTextSize { get; init; } = 18f;

    public float LabelPadding { get; init; } = 6f;

    public SKTypeface Typeface { get; init; } = SKTypeface.Default;

    public LayoutDebugOverlayOptions Clone() => new()
    {
        BackgroundOpacity = BackgroundOpacity,
        FillOpacity = FillOpacity,
        StrokeWidth = StrokeWidth,
        DrawLabels = DrawLabels,
        LabelTextSize = LabelTextSize,
        LabelPadding = LabelPadding,
        Typeface = Typeface,
    };

    public void EnsureValid()
    {
        if (BackgroundOpacity is < 0 or > 1)
        {
            throw new InvalidOperationException("Background opacity must be between 0 and 1.");
        }

        if (FillOpacity is < 0 or > 1)
        {
            throw new InvalidOperationException("Fill opacity must be between 0 and 1.");
        }

        if (StrokeWidth <= 0)
        {
            throw new InvalidOperationException("Stroke width must be positive.");
        }

        if (LabelTextSize <= 0)
        {
            throw new InvalidOperationException("Label text size must be positive.");
        }

        if (LabelPadding < 0)
        {
            throw new InvalidOperationException("Label padding cannot be negative.");
        }
    }
}
