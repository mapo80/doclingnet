using System;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Core.Geometry;
using Docling.Export.Abstractions;
using SkiaSharp;

namespace Docling.Export.Imaging;

/// <summary>
/// High-quality cropper built on top of SkiaSharp.
/// </summary>
public sealed class ImageCropService : IImageCropService
{
    public Task<CroppedImage> CropAsync(PageImage source, BoundingBox region, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);

        cancellationToken.ThrowIfCancellationRequested();

        var normalized = region.Normalized();
        var left = ClampToImage(normalized.Left, source.Width);
        var top = ClampToImage(normalized.Top, source.Height);
        var right = ClampToImage(normalized.Right, source.Width);
        var bottom = ClampToImage(normalized.Bottom, source.Height);

        var width = Math.Max(1, (int)Math.Round(right - left));
        var height = Math.Max(1, (int)Math.Round(bottom - top));

        SKBitmap? target = null;
        try
        {
            target = new SKBitmap(width, height, source.Bitmap.ColorType, source.Bitmap.AlphaType);
            using var canvas = new SKCanvas(target);
            var sourceRect = SKRect.Create((float)left, (float)top, (float)(right - left), (float)(bottom - top));
            var destinationRect = SKRect.Create(0, 0, width, height);
            canvas.DrawBitmap(source.Bitmap, sourceRect, destinationRect);
            canvas.Flush();

            var cropped = new CroppedImage(source.Page, new BoundingBox(left, top, right, bottom), target);
            target = null;
            return Task.FromResult(cropped);
        }
        finally
        {
            target?.Dispose();
        }
    }

    private static double ClampToImage(double value, int limit) => Math.Clamp(value, 0, limit);
}
