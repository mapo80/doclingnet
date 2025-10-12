using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Image preprocessor for TableFormer with letterboxing and ImageNet normalization.
/// Follows the official Docling preprocessing pipeline.
/// </summary>
internal sealed class ImagePreprocessor : IDisposable
{
    private const int TargetSize = 640; // Standard size for Docling models
    private readonly float[] _imagenetMean = { 0.485f, 0.456f, 0.406f };
    private readonly float[] _imagenetStd = { 0.229f, 0.224f, 0.225f };
    private bool _disposed;

    /// <summary>
    /// Preprocess image with letterboxing and ImageNet normalization.
    /// </summary>
    public DenseTensor<float> PreprocessImage(SKBitmap bitmap)
    {
        ArgumentNullException.ThrowIfNull(bitmap);

        // Step 1: Letterboxing resize to maintain aspect ratio
        var letterboxed = ApplyLetterboxing(bitmap);
        using (letterboxed)
        {
            // Step 2: Convert to tensor and apply ImageNet normalization
            var tensor = ConvertToTensor(letterboxed);
            return tensor;
        }
    }

    private static SKBitmap ApplyLetterboxing(SKBitmap original)
    {
        var originalWidth = original.Width;
        var originalHeight = original.Height;

        // Calculate scaling to fit within target size while maintaining aspect ratio
        var scale = Math.Min((float)TargetSize / originalWidth, (float)TargetSize / originalHeight);
        var newWidth = (int)Math.Round(originalWidth * scale);
        var newHeight = (int)Math.Round(originalHeight * scale);

        // Create letterboxed image (TargetSize x TargetSize) with padding
        var letterboxed = new SKBitmap(TargetSize, TargetSize);
        using var canvas = new SKCanvas(letterboxed);

        // Fill with gray background (128, 128, 128) as per ImageNet standards
        canvas.Clear(new SKColor(128, 128, 128));

        // Calculate padding to center the image
        var offsetX = (TargetSize - newWidth) / 2;
        var offsetY = (TargetSize - newHeight) / 2;

        // Draw original image scaled and centered
        var destRect = new SKRect(offsetX, offsetY, offsetX + newWidth, offsetY + newHeight);
        canvas.DrawBitmap(original, destRect);

        return letterboxed;
    }

    private DenseTensor<float> ConvertToTensor(SKBitmap bitmap)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });

        for (int y = 0; y < bitmap.Height; y++)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                var color = bitmap.GetPixel(x, y);

                // Convert to float [0,1] range
                var r = color.Red / 255f;
                var g = color.Green / 255f;
                var b = color.Blue / 255f;

                // Apply ImageNet normalization
                // pixel = (pixel - mean) / std
                tensor[0, 0, y, x] = (r - _imagenetMean[0]) / _imagenetStd[0];
                tensor[0, 1, y, x] = (g - _imagenetMean[1]) / _imagenetStd[1];
                tensor[0, 2, y, x] = (b - _imagenetMean[2]) / _imagenetStd[2];
            }
        }

        return tensor;
    }

    /// <summary>
    /// Convert normalized coordinates back to original image coordinates.
    /// </summary>
    public static (float Left, float Top, float Right, float Bottom) TransformCoordinates(
        float x, float y, float width, float height,
        int originalWidth, int originalHeight)
    {
        // Calculate scale and padding from letterboxing
        var scale = Math.Min((float)TargetSize / originalWidth, (float)TargetSize / originalHeight);
        var scaledWidth = originalWidth * scale;
        var scaledHeight = originalHeight * scale;

        var padX = (TargetSize - scaledWidth) / 2;
        var padY = (TargetSize - scaledHeight) / 2;

        // Convert from normalized coordinates to letterboxed coordinates
        var left = x * TargetSize;
        var top = y * TargetSize;
        var right = (x + width) * TargetSize;
        var bottom = (y + height) * TargetSize;

        // Remove padding and scale back to original dimensions
        var originalLeft = (left - padX) / scale;
        var originalTop = (top - padY) / scale;
        var originalRight = (right - padX) / scale;
        var originalBottom = (bottom - padY) / scale;

        // Clamp to original image boundaries
        originalLeft = Math.Max(0, Math.Min(originalWidth, originalLeft));
        originalTop = Math.Max(0, Math.Min(originalHeight, originalTop));
        originalRight = Math.Max(0, Math.Min(originalWidth, originalRight));
        originalBottom = Math.Max(0, Math.Min(originalHeight, originalBottom));

        return (originalLeft, originalTop, originalRight, originalBottom);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}