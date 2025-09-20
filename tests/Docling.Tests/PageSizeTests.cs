using Docling.Core.Geometry;
using FluentAssertions;

namespace Docling.Tests;

public sealed class PageSizeTests
{
    [Fact]
    public void RotationSwapsDimensions()
    {
        var size = new PageSize(600, 800, 300);
        var rotated = size.Rotate();

        rotated.Width.Should().Be(800);
        rotated.Height.Should().Be(600);
        rotated.Dpi.Should().Be(300);
    }

    [Fact]
    public void ScaleToDpiResizesPixels()
    {
        var size = new PageSize(600, 800, 300);
        var scaled = size.ScaleToDpi(150);

        scaled.Width.Should().Be(300);
        scaled.Height.Should().Be(400);
        scaled.Dpi.Should().Be(150);
    }

    [Fact]
    public void InflateExpandsWithPadding()
    {
        var size = new PageSize(100, 120, 200);
        var inflated = size.Inflate(10);

        inflated.Width.Should().Be(120);
        inflated.Height.Should().Be(140);
    }
}
