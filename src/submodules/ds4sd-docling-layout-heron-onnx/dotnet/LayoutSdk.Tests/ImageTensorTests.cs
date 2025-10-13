using LayoutSdk.Processing;
using System;
using Xunit;

namespace LayoutSdk.Tests;

public class ImageTensorTests
{
    [Theory]
    [InlineData(0, 1, 1)]
    [InlineData(1, 0, 1)]
    [InlineData(1, 1, 0)]
    public void Rent_InvalidDimensions_Throws(int width, int height, int channels)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageTensor.Rent(width, height, channels));
    }

    [Fact]
    public void Rent_ReturnsZeroedBuffer()
    {
        using var tensor = ImageTensor.Rent(2, 3, 4);
        Assert.Equal(2, tensor.Width);
        Assert.Equal(3, tensor.Height);
        Assert.Equal(4, tensor.Channels);
        Assert.Equal(24, tensor.Length);

        var span = tensor.AsSpan();
        for (var i = 0; i < span.Length; i++)
        {
            Assert.Equal(0f, span[i]);
        }
    }

    [Fact]
    public void Dispose_MultipleTimes_IsSafe()
    {
        var tensor = ImageTensor.Rent(1, 1, 1);
        tensor.Dispose();
        tensor.Dispose();
    }
}
