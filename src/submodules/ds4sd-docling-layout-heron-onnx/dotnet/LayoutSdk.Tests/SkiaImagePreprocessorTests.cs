using LayoutSdk.Processing;
using SkiaSharp;
using System;
using Xunit;

namespace LayoutSdk.Tests;

public class SkiaImagePreprocessorTests
{
    [Fact]
    public void Preprocess_NullImage_Throws()
    {
        var preprocessor = new SkiaImagePreprocessor();
        Assert.Throws<ArgumentNullException>(() => preprocessor.Preprocess(null!));
    }

    [Fact]
    public void Preprocess_ResizesAndRescalesPixels()
    {
        using var bitmap = new SKBitmap(2, 2);
        bitmap.Erase(new SKColor(64, 128, 192));

        var preprocessor = new SkiaImagePreprocessor();
        using var tensor = preprocessor.Preprocess(bitmap);

        Assert.Equal(640, tensor.Width);
        Assert.Equal(640, tensor.Height);
        Assert.Equal(3, tensor.Channels);

        var plane = tensor.Width * tensor.Height;
        var span = tensor.AsSpan();

        var expectedR = 64f / 255f;
        var expectedG = 128f / 255f;
        var expectedB = 192f / 255f;

        Assert.Equal((double)expectedR, (double)span[0], 5);
        Assert.Equal((double)expectedG, (double)span[plane + 0], 5);
        Assert.Equal((double)expectedB, (double)span[2 * plane + 0], 5);
    }
}
