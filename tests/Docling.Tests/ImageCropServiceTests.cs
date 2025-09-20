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
        var cropper = new ImageCropService();
        var region = BoundingBox.FromSize(2, 2, 4, 4);

        using var result = await cropper.CropAsync(page, region);

        result.Bitmap.Width.Should().Be(4);
        result.Bitmap.Height.Should().Be(4);
        result.SourceRegion.Left.Should().Be(2);
    }
}
