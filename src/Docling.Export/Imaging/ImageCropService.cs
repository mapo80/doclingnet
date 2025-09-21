using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Abstractions;
using SkiaSharp;

namespace Docling.Export.Imaging;

/// <summary>
/// High-quality cropper built on top of SkiaSharp.
/// </summary>
public sealed class ImageCropService : IImageCropService, IDisposable
{
    private const double DefaultPadding = 2d;

    private readonly ConcurrentDictionary<CropCacheKey, CacheEntry> _cache = new();
    private readonly double _padding;
    private bool _disposed;

    public ImageCropService()
        : this(DefaultPadding)
    {
    }

    public ImageCropService(double padding)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(padding, nameof(padding));
        _padding = padding;
    }

    public Task<CroppedImage> CropAsync(PageImage source, BoundingBox region, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(source);

        cancellationToken.ThrowIfCancellationRequested();

        var normalized = NormalizeRegion(source, region);
        var key = CropCacheKey.From(source.Page, normalized);

        if (_cache.TryGetValue(key, out var cached))
        {
            return Task.FromResult(cached.CreateCroppedImage(source.Page, normalized));
        }

        var bitmap = ExtractBitmap(source, normalized);
        var clone = bitmap.Copy();

        if (clone is null)
        {
            return Task.FromResult(new CroppedImage(source.Page, normalized, bitmap));
        }

        var entry = new CacheEntry(clone);
        var existing = _cache.GetOrAdd(key, entry);
        if (!ReferenceEquals(existing, entry))
        {
            entry.Dispose();
            bitmap.Dispose();
            return Task.FromResult(existing.CreateCroppedImage(source.Page, normalized));
        }

        return Task.FromResult(new CroppedImage(source.Page, normalized, bitmap));
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        foreach (var entry in _cache.Values)
        {
            entry.Dispose();
        }

        _cache.Clear();
        _disposed = true;
    }

    private BoundingBox NormalizeRegion(PageImage source, BoundingBox region)
    {
        var normalized = region.Normalized();

        if (_padding > 0)
        {
            normalized = normalized.Inflate(_padding, _padding);
        }

        normalized = normalized.SnapToIntegers();

        var left = Clamp(normalized.Left, 0, Math.Max(0, source.Width - 1));
        var top = Clamp(normalized.Top, 0, Math.Max(0, source.Height - 1));
        var right = Clamp(normalized.Right, left + 1, source.Width);
        var bottom = Clamp(normalized.Bottom, top + 1, source.Height);

        return new BoundingBox(left, top, right, bottom);
    }

    private static SKBitmap ExtractBitmap(PageImage source, BoundingBox region)
    {
        var left = (float)region.Left;
        var top = (float)region.Top;
        var width = Math.Max(1, (int)Math.Round(region.Right - region.Left, MidpointRounding.AwayFromZero));
        var height = Math.Max(1, (int)Math.Round(region.Bottom - region.Top, MidpointRounding.AwayFromZero));

        var target = new SKBitmap(width, height, source.Bitmap.ColorType, source.Bitmap.AlphaType);
        using var canvas = new SKCanvas(target);
        var sourceRect = SKRect.Create(left, top, width, height);
        var destinationRect = SKRect.Create(0, 0, width, height);
        canvas.DrawBitmap(source.Bitmap, sourceRect, destinationRect);
        canvas.Flush();
        return target;
    }

    private static double Clamp(double value, double min, double max)
    {
        if (max < min)
        {
            return min;
        }

        return Math.Clamp(value, min, max);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    private readonly record struct CropCacheKey(int PageNumber, int Left, int Top, int Right, int Bottom)
    {
        public static CropCacheKey From(PageReference page, BoundingBox region)
        {
            return new CropCacheKey(
                page.PageNumber,
                (int)Math.Round(region.Left, MidpointRounding.AwayFromZero),
                (int)Math.Round(region.Top, MidpointRounding.AwayFromZero),
                (int)Math.Round(region.Right, MidpointRounding.AwayFromZero),
                (int)Math.Round(region.Bottom, MidpointRounding.AwayFromZero));
        }
    }

    private sealed class CacheEntry : IDisposable
    {
        private readonly SKBitmap _bitmap;
        private bool _disposed;

        public CacheEntry(SKBitmap bitmap)
        {
            _bitmap = bitmap ?? throw new ArgumentNullException(nameof(bitmap));
        }

        public CroppedImage CreateCroppedImage(PageReference page, BoundingBox region)
        {
            ThrowIfDisposed();

            var clone = _bitmap.Copy();
            if (clone is null)
            {
                throw new InvalidOperationException("Failed to copy cached crop bitmap.");
            }

            return new CroppedImage(page, region, clone);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _bitmap.Dispose();
            _disposed = true;
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
