using System.Reflection;
using Docling.Backends.Pdf;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Imaging;
using FluentAssertions;
using SkiaSharp;
using Xunit;

namespace Docling.Tests;

public sealed class ImageCropServiceTests
{
    [Fact]
    public async Task CropAsyncRespectsBoundingBox()
    {
        using var bitmap = new SKBitmap(10, 10);
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
        }

        using var page = new PageImage(new PageReference(0, 300), bitmap);
        using var cropper = new ImageCropService();
        var region = BoundingBox.FromSize(2, 2, 4, 4);

        using var result = await cropper.CropAsync(page, region);

        result.Bitmap.Width.Should().Be(8);
        result.Bitmap.Height.Should().Be(8);
        result.SourceRegion.Left.Should().Be(0);
    }

    [Fact]
    public async Task CropAsyncAppliesPaddingAndRounding()
    {
        using var bitmap = new SKBitmap(20, 18);
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
        }

        using var page = new PageImage(new PageReference(1, 300), bitmap);
        using var cropper = new ImageCropService();
        var region = new BoundingBox(1.4, 1.9, 6.1, 5.2);

        using var result = await cropper.CropAsync(page, region);

        result.SourceRegion.Left.Should().Be(0);
        result.SourceRegion.Top.Should().Be(0);
        result.SourceRegion.Right.Should().Be(9);
        result.SourceRegion.Bottom.Should().Be(8);
        result.Bitmap.Width.Should().Be(9);
        result.Bitmap.Height.Should().Be(8);
    }

    [Fact]
    public async Task CropAsyncCachesNormalizedRegion()
    {
        using var bitmap = new SKBitmap(24, 24);
        using var page = new PageImage(new PageReference(2, 300), bitmap);
        using var cropper = new ImageCropService();
        var region = new BoundingBox(3.3, 3.3, 12.7, 15.2);

        using var first = await cropper.CropAsync(page, region);
        using var second = await cropper.CropAsync(page, region);

        first.SourceRegion.Should().Be(second.SourceRegion);
        ReferenceEquals(first.Bitmap, second.Bitmap).Should().BeFalse();

        var cacheField = typeof(ImageCropService).GetField("_cache", BindingFlags.NonPublic | BindingFlags.Instance);
        cacheField.Should().NotBeNull();
        var cache = cacheField!.GetValue(cropper);
        cache.Should().NotBeNull();
        var countProperty = cache!.GetType().GetProperty("Count");
        countProperty.Should().NotBeNull();
        var count = (int)countProperty!.GetValue(cache)!;
        count.Should().Be(1);
    }
}
