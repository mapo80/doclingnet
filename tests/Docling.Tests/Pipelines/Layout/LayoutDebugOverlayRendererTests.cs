using System.Collections.Generic;
using Docling.Backends.Pdf;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Pipelines.Layout;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;

namespace Docling.Tests.Pipelines.Layout;

public sealed class LayoutDebugOverlayRendererTests
{
    [Fact]
    public void CreateOverlayDrawsBoundingBoxesWithExpectedColours()
    {
        var options = new LayoutDebugOverlayOptions
        {
            BackgroundOpacity = 0,
            FillOpacity = 0,
            DrawLabels = false,
            StrokeWidth = 2,
        };

        var renderer = new LayoutDebugOverlayRenderer(options, NullLogger<LayoutDebugOverlayRenderer>.Instance);
        var pageReference = new PageReference(0, 200);
        using var bitmap = new SKBitmap(new SKImageInfo(100, 100, SKColorType.Rgba8888, SKAlphaType.Premul));
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
        }

        var metadata = new PageImageMetadata("page-0", "doc.pdf", "image/png", new Dictionary<string, string>());
        using var pageImage = new PageImage(pageReference, bitmap.Copy() ?? throw new Xunit.Sdk.XunitException("Copy failed"), metadata);

        var layoutItems = new List<LayoutItem>
        {
            new(pageReference, BoundingBox.FromSize(10, 10, 30, 30), LayoutItemKind.Text, new List<Polygon>()),
            new(pageReference, BoundingBox.FromSize(50, 50, 20, 20), LayoutItemKind.Figure, new List<Polygon>()),
        };

        var overlay = renderer.CreateOverlay(pageImage, layoutItems);
        using var overlayBitmap = SKBitmap.Decode(overlay.ImageContent.ToArray());

        overlayBitmap.GetPixel(10, 10).Should().Be(new SKColor(56, 142, 60));
        overlayBitmap.GetPixel(50, 50).Should().Be(new SKColor(251, 140, 0));
    }

    [Fact]
    public void ConstructorValidatesOptions()
    {
        var invalidOptions = new LayoutDebugOverlayOptions { BackgroundOpacity = 2f };
        var act = () => new LayoutDebugOverlayRenderer(invalidOptions, NullLogger<LayoutDebugOverlayRenderer>.Instance);
        act.Should().Throw<InvalidOperationException>();
    }
}
