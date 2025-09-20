using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Pipelines.Preprocessing;

/// <summary>
/// Default implementation of <see cref="IPagePreprocessor"/> combining DPI normalisation,
/// colour conversion, contrast enhancement and an optional deskew pass.
/// </summary>
public sealed partial class DefaultPagePreprocessor : IPagePreprocessor
{
    private readonly PreprocessingOptions _options;
    private readonly ILogger<DefaultPagePreprocessor> _logger;

    public DefaultPagePreprocessor(PreprocessingOptions options, ILogger<DefaultPagePreprocessor> logger)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.EnsureValid();
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public Task<PageImage> PreprocessAsync(PageImage pageImage, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pageImage);
        cancellationToken.ThrowIfCancellationRequested();

        var originalDpi = ExtractSourceDpi(pageImage);
        var targetDpi = _options.TargetDpi;
        var appliedScale = targetDpi / originalDpi;
        var colorMode = PageColorMode.Preserve;

        var working = EnsureBitmap(pageImage.Bitmap);
        try
        {
            if (Math.Abs(appliedScale - 1d) > 0.001d)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ReplaceBitmap(ref working, bitmap => ResizeBitmap(bitmap, appliedScale));
            }

            cancellationToken.ThrowIfCancellationRequested();
            var colorResult = ApplyColorMode(working);
            if (!ReferenceEquals(working, colorResult.Bitmap))
            {
                working.Dispose();
            }

            working = colorResult.Bitmap;
            colorMode = colorResult.Mode;

            if (_options.NormalizeContrast)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ReplaceBitmap(ref working, ApplyContrast);
            }

            float deskewAngle = 0f;
            if (_options.EnableDeskew)
            {
                cancellationToken.ThrowIfCancellationRequested();
                deskewAngle = ApplyDeskew(ref working, cancellationToken);
            }

            var metadata = pageImage.Metadata.WithAdditionalProperties(
                BuildMetadata(appliedScale, targetDpi, colorMode, deskewAngle));
            var normalisedPage = pageImage.Page.WithDpi(targetDpi);
            var result = new PageImage(normalisedPage, working, metadata);
            working = null!; // ownership transferred to PageImage
            return Task.FromResult(result);
        }
        finally
        {
            working?.Dispose();
        }
    }

    private static double ExtractSourceDpi(PageImage pageImage)
    {
        if (pageImage.Metadata.Properties.TryGetValue(PageImageMetadataKeys.SourceHorizontalDpi, out var raw)
            && double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var dpi)
            && dpi > 0d)
        {
            return dpi;
        }

        return pageImage.Page.Dpi > 0d ? pageImage.Page.Dpi : 72d;
    }

    private static IEnumerable<KeyValuePair<string, string>> BuildMetadata(
        double scale,
        double dpi,
        PageColorMode colorMode,
        float deskewAngle)
    {
        yield return new(PageImageMetadataKeys.NormalizedDpi, dpi.ToString("F2", CultureInfo.InvariantCulture));
        yield return new(PageImageMetadataKeys.ScaleFactor, scale.ToString("F4", CultureInfo.InvariantCulture));
        yield return new(PageImageMetadataKeys.ColorMode, colorMode.ToString());
        yield return new(PageImageMetadataKeys.DeskewAngle, deskewAngle.ToString("F2", CultureInfo.InvariantCulture));
    }

    private (SKBitmap Bitmap, PageColorMode Mode) ApplyColorMode(SKBitmap bitmap)
    {
        switch (_options.ColorMode)
        {
            case PageColorMode.Preserve:
                return (bitmap, PageColorMode.Preserve);
            case PageColorMode.Grayscale:
                return (CreateColorTransformedBitmap(bitmap, ColorMatrixes.Grayscale, PageColorMode.Grayscale), PageColorMode.Grayscale);
            case PageColorMode.Binary:
                return (Binarize(bitmap), PageColorMode.Binary);
            default:
                throw new InvalidOperationException($"Unsupported colour mode '{_options.ColorMode}'.");
        }
    }

    private SKBitmap ApplyContrast(SKBitmap bitmap)
    {
        var info = new SKImageInfo(bitmap.Width, bitmap.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
        var contrasted = new SKBitmap(info);
        using var surface = new SKCanvas(contrasted);
        using var paint = new SKPaint
        {
            ColorFilter = SKColorFilter.CreateHighContrast(new SKHighContrastConfig(true, SKHighContrastConfigInvertStyle.NoInvert, _options.ContrastAmount)),
            IsAntialias = true,
        };

        surface.DrawBitmap(bitmap, 0, 0, paint);
        return contrasted;
    }

    private float ApplyDeskew(ref SKBitmap bitmap, CancellationToken cancellationToken)
    {
        using var grayscale = CreateColorTransformedBitmap(bitmap, ColorMatrixes.Grayscale, PageColorMode.Grayscale);
        cancellationToken.ThrowIfCancellationRequested();
        var angle = EstimateSkewAngle(grayscale);
        angle = Math.Clamp(angle, (float)(-_options.DeskewMaxAngle), (float)_options.DeskewMaxAngle);

        if (Math.Abs(angle) < _options.DeskewMinimumAngle)
        {
            return 0f;
        }

        var rotated = RotateBitmap(bitmap, -angle, _options.BackgroundColor);
        bitmap.Dispose();
        bitmap = rotated;
        return angle;
    }

    private static float EstimateSkewAngle(SKBitmap grayscale)
    {
        using var pixmap = grayscale.PeekPixels();
        if (pixmap is null)
        {
            return 0f;
        }

        var width = pixmap.Width;
        var height = pixmap.Height;
        var rowBytes = pixmap.RowBytes;
        var ptr = pixmap.GetPixels();
        if (ptr == IntPtr.Zero)
        {
            return 0f;
        }

        double sumWeight = 0d;
        double sumX = 0d;
        double sumY = 0d;

        var buffer = new byte[rowBytes * height];
        Marshal.Copy(ptr, buffer, 0, buffer.Length);

        for (var y = 0; y < height; y++)
        {
            var rowOffset = y * rowBytes;
            for (var x = 0; x < width; x++)
            {
                var intensity = buffer[rowOffset + x];
                var weight = 1d - intensity / 255d;
                if (weight <= 0.05d)
                {
                    continue;
                }

                sumWeight += weight;
                sumX += x * weight;
                sumY += y * weight;
            }
        }

        if (sumWeight <= double.Epsilon)
        {
            return 0f;
        }

        var centroidX = sumX / sumWeight;
        var centroidY = sumY / sumWeight;
        double mu11 = 0d;
        double mu20 = 0d;
        double mu02 = 0d;

        for (var y = 0; y < height; y++)
        {
            var rowOffset = y * rowBytes;
            for (var x = 0; x < width; x++)
            {
                var intensity = buffer[rowOffset + x];
                var weight = 1d - intensity / 255d;
                if (weight <= 0.05d)
                {
                    continue;
                }

                var dx = x - centroidX;
                var dy = y - centroidY;
                mu11 += weight * dx * dy;
                mu20 += weight * dx * dx;
                mu02 += weight * dy * dy;
            }
        }

        var angleRadians = 0.5d * Math.Atan2(2d * mu11, mu20 - mu02);
        var angleDegrees = (float)(angleRadians * 180d / Math.PI);
        if (angleDegrees < -90f)
        {
            angleDegrees += 180f;
        }
        else if (angleDegrees > 90f)
        {
            angleDegrees -= 180f;
        }

        return angleDegrees;
    }

    private static SKBitmap RotateBitmap(SKBitmap bitmap, float angleDegrees, SKColor background)
    {
        if (Math.Abs(angleDegrees) < 0.01f)
        {
            return bitmap.Copy() ?? throw new InvalidOperationException("Failed to copy bitmap prior to rotation.");
        }

        var radians = Math.Abs(angleDegrees) * Math.PI / 180d;
        var cos = Math.Cos(radians);
        var sin = Math.Sin(radians);
        var newWidth = (int)Math.Ceiling(Math.Abs(bitmap.Width * cos) + Math.Abs(bitmap.Height * sin));
        var newHeight = (int)Math.Ceiling(Math.Abs(bitmap.Width * sin) + Math.Abs(bitmap.Height * cos));
        var info = new SKImageInfo(newWidth, newHeight, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var surface = SKSurface.Create(info);
        var canvas = surface.Canvas;
        canvas.Clear(background);
        canvas.Translate(newWidth / 2f, newHeight / 2f);
        canvas.RotateDegrees(angleDegrees);
        canvas.Translate(-bitmap.Width / 2f, -bitmap.Height / 2f);
        using (var paint = new SKPaint { IsAntialias = true })
        {
            canvas.DrawBitmap(bitmap, 0, 0, paint);
        }

        canvas.Flush();
        using var snapshot = surface.Snapshot();
        return SKBitmap.FromImage(snapshot);
    }

    private SKBitmap Binarize(SKBitmap bitmap)
    {
        using var grayscale = CreateColorTransformedBitmap(bitmap, ColorMatrixes.Grayscale, PageColorMode.Binary);
        var threshold = ComputeOtsuThreshold(grayscale);
        var info = new SKImageInfo(grayscale.Width, grayscale.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
        var binary = new SKBitmap(info);

        for (var y = 0; y < grayscale.Height; y++)
        {
            for (var x = 0; x < grayscale.Width; x++)
            {
                var pixel = grayscale.GetPixel(x, y);
                var value = pixel.Red > threshold ? (byte)255 : (byte)0;
                binary.SetPixel(x, y, new SKColor(value, value, value, 255));
            }
        }

        return binary;
    }

    private static byte ComputeOtsuThreshold(SKBitmap grayscale)
    {
        Span<int> histogram = stackalloc int[256];
        for (var y = 0; y < grayscale.Height; y++)
        {
            for (var x = 0; x < grayscale.Width; x++)
            {
                var intensity = grayscale.GetPixel(x, y).Red;
                histogram[intensity]++;
            }
        }

        var totalPixels = grayscale.Width * grayscale.Height;
        long sum = 0;
        for (var t = 0; t < 256; t++)
        {
            sum += t * histogram[t];
        }

        long sumBackground = 0;
        var weightBackground = 0;
        var weightForeground = 0;
        double maxVariance = 0d;
        var threshold = 0;

        for (var t = 0; t < 256; t++)
        {
            weightBackground += histogram[t];
            if (weightBackground == 0)
            {
                continue;
            }

            weightForeground = totalPixels - weightBackground;
            if (weightForeground == 0)
            {
                break;
            }

            sumBackground += t * histogram[t];
            var meanBackground = sumBackground / (double)weightBackground;
            var meanForeground = (sum - sumBackground) / (double)weightForeground;
            var betweenClass = weightBackground * (double)weightForeground * Math.Pow(meanBackground - meanForeground, 2);

            if (betweenClass > maxVariance)
            {
                maxVariance = betweenClass;
                threshold = t;
            }
        }

        return (byte)threshold;
    }

    private SKBitmap CreateColorTransformedBitmap(SKBitmap bitmap, float[] matrix, PageColorMode mode)
    {
        var info = new SKImageInfo(bitmap.Width, bitmap.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
        var transformed = new SKBitmap(info);
        using var surface = new SKCanvas(transformed);
        using var paint = new SKPaint
        {
            ColorFilter = SKColorFilter.CreateColorMatrix(matrix),
            IsAntialias = true,
        };

        surface.DrawBitmap(bitmap, 0, 0, paint);
        UpdateColorMetadata(PageColorMode.Preserve, mode);
        return transformed;
    }

    private void UpdateColorMetadata(PageColorMode previousMode, PageColorMode nextMode)
    {
        if (previousMode == nextMode)
        {
            return;
        }

        PreprocessorLogger.ColorTransition(_logger, previousMode, nextMode);
    }

    private static SKBitmap EnsureBitmap(SKBitmap bitmap)
    {
        var copy = bitmap.Copy(SKColorType.Rgba8888);
        if (copy is null)
        {
            throw new InvalidOperationException("Failed to clone input bitmap for preprocessing.");
        }

        return copy;
    }

    private static SKBitmap ResizeBitmap(SKBitmap bitmap, double scale)
    {
        var width = Math.Max(1, (int)Math.Round(bitmap.Width * scale));
        var height = Math.Max(1, (int)Math.Round(bitmap.Height * scale));
        var info = new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Premul);
        var sampling = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear);
        var resized = bitmap.Resize(info, sampling);
        if (resized is null)
        {
            throw new InvalidOperationException("Failed to resize bitmap during preprocessing.");
        }

        return resized;
    }

    private static void ReplaceBitmap(ref SKBitmap current, Func<SKBitmap, SKBitmap> transform)
    {
        ArgumentNullException.ThrowIfNull(transform);

        var original = current;
        SKBitmap? updated = null;
        try
        {
            updated = transform(original);
            if (ReferenceEquals(updated, original))
            {
                return;
            }

            current = updated;
            original.Dispose();
            updated = null;
        }
        finally
        {
            updated?.Dispose();
        }
    }

    private static class ColorMatrixes
    {
        public static readonly float[] Grayscale =
        {
            0.2126f, 0.7152f, 0.0722f, 0f, 0f,
            0.2126f, 0.7152f, 0.0722f, 0f, 0f,
            0.2126f, 0.7152f, 0.0722f, 0f, 0f,
            0f, 0f, 0f, 1f, 0f,
        };
    }

    private static partial class PreprocessorLogger
    {
        [LoggerMessage(EventId = 2100, Level = LogLevel.Debug, Message = "Transitioned page colour mode from {Previous} to {Next}.")]
        public static partial void ColorTransition(ILogger logger, PageColorMode previous, PageColorMode next);
    }
}
