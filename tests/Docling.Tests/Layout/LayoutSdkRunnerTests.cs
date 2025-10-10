using System.Collections.Generic;
using Docling.Models.Layout;
using FluentAssertions;
using LayoutSdk;
using Xunit;

namespace Docling.Tests.Layout;

public sealed class LayoutSdkRunnerTests
{
    [Fact]
    public void ReprojectBoundingBoxesWithMetadataRestoresOriginalCoordinates()
    {
        var metadata = new LayoutSdkNormalisationMetadata(
            OriginalWidth: 1000,
            OriginalHeight: 800,
            ScaledWidth: 640,
            ScaledHeight: 512,
            Scale: 0.64,
            OffsetX: 0,
            OffsetY: 64);

        var boxes = new List<LayoutSdk.BoundingBox>
        {
            new(160, 200, 320, 128, "text"),
            new(0, 64, 640, 512, "table"),
        };

        var projected = LayoutSdkRunner.ReprojectBoundingBoxes(boxes, metadata);

        projected.Should().HaveCount(2);
        projected[0].X.Should().BeApproximately(250, 1e-3);
        projected[0].Y.Should().BeApproximately(212.5, 1e-3);
        projected[0].Width.Should().BeApproximately(500, 1e-3);
        projected[0].Height.Should().BeApproximately(200, 1e-3);

        projected[1].X.Should().Be(0);
        projected[1].Y.Should().Be(0);
        projected[1].Width.Should().Be(1000);
        projected[1].Height.Should().Be(800);
    }

    [Fact]
    public void ReprojectBoundingBoxesSkipsDetectionsFullyInLetterboxPadding()
    {
        var metadata = new LayoutSdkNormalisationMetadata(
            OriginalWidth: 1000,
            OriginalHeight: 800,
            ScaledWidth: 640,
            ScaledHeight: 512,
            Scale: 0.64,
            OffsetX: 0,
            OffsetY: 64);

        var boxes = new List<LayoutSdk.BoundingBox>
        {
            new(10, 10, 30, 30, "text"), // entirely inside top padding
            new(160, 200, 320, 128, "text"),
        };

        var projected = LayoutSdkRunner.ReprojectBoundingBoxes(boxes, metadata);

        projected.Should().HaveCount(1);
        projected[0].X.Should().BeApproximately(250, 1e-3);
        projected[0].Y.Should().BeApproximately(212.5, 1e-3);
        projected[0].Width.Should().BeApproximately(500, 1e-3);
        projected[0].Height.Should().BeApproximately(200, 1e-3);
    }

    [Fact]
    public void ReprojectBoundingBoxesWithIdentityMetadataClonesOriginalValues()
    {
        var boxes = new List<LayoutSdk.BoundingBox>
        {
            new(10, 20, 30, 40, "figure"),
        };

        var metadata = new LayoutSdkNormalisationMetadata(
            OriginalWidth: 640,
            OriginalHeight: 640,
            ScaledWidth: 640,
            ScaledHeight: 640,
            Scale: 1.0,
            OffsetX: 0,
            OffsetY: 0);

        var projected = LayoutSdkRunner.ReprojectBoundingBoxes(boxes, metadata);

        projected.Should().ContainSingle();
        projected[0].X.Should().Be(10);
        projected[0].Y.Should().Be(20);
        projected[0].Width.Should().Be(30);
        projected[0].Height.Should().Be(40);
        projected[0].Label.Should().Be("figure");
    }

    [Fact]
    public void ReprojectBoundingBoxesWithoutMetadataReturnsCopy()
    {
        var boxes = new List<LayoutSdk.BoundingBox>
        {
            new(5, 6, 7, 8, "text"),
        };

        var projected = LayoutSdkRunner.ReprojectBoundingBoxes(boxes, metadata: null);

        projected.Should().ContainSingle();
        projected[0].X.Should().Be(5);
        projected[0].Y.Should().Be(6);
        projected[0].Width.Should().Be(7);
        projected[0].Height.Should().Be(8);
        projected[0].Label.Should().Be("text");
    }

    [Fact]
    public void ShouldAugmentWithFallbackReturnsTrueWhenCoverageLow()
    {
        var metadata = new LayoutSdkNormalisationMetadata(1000, 800, 640, 512, 0.64, 0, 64);
        var boxes = new List<LayoutSdk.BoundingBox>
        {
            new(200, 300, 100, 50, "text"),
        };

        var result = LayoutSdkRunner.ShouldAugmentWithFallback(boxes, metadata, out var coverage);

        result.Should().BeTrue();
        coverage.Should().BeGreaterThan(0).And.BeLessThan(0.55);
    }

    [Fact]
    public void ShouldAugmentWithFallbackReturnsFalseWhenCoverageSufficient()
    {
        var metadata = new LayoutSdkNormalisationMetadata(1000, 800, 640, 512, 0.64, 0, 64);
        var boxes = new List<LayoutSdk.BoundingBox>
        {
            new(0, 0, 1000, 400, "text"),
            new(0, 400, 1000, 400, "text"),
        };

        var result = LayoutSdkRunner.ShouldAugmentWithFallback(boxes, metadata, out var coverage);

        result.Should().BeFalse();
        coverage.Should().BeGreaterThan(0.7);
    }

    [Fact]
    public void MergeDetectionsAddsFallbackMissingBoxesAndSuppressesContainedOnes()
    {
        var primary = new List<LayoutSdk.BoundingBox>
        {
            new(100, 100, 200, 100, "text"),
        };

        var fallback = new List<LayoutSdk.BoundingBox>
        {
            new(0, 0, 800, 600, "text"),
            new(120, 120, 40, 20, "text"),
        };

        var merged = LayoutSdkRunner.MergeDetections(primary, fallback);

        merged.Should().HaveCount(2);
        merged.Should().Contain(box => box.X == 0 && box.Y == 0 && box.Width == 800 && box.Height == 600);
        merged.Should().Contain(box => box.X == 100 && box.Y == 100 && box.Width == 200 && box.Height == 100);
    }
}
