using Docling.Core.Geometry;
using FluentAssertions;

namespace Docling.Tests;

public sealed class BoundingBoxTests
{
    [Fact]
    public void FromSizeCreatesExpectedBox()
    {
        var box = BoundingBox.FromSize(10, 20, 30, 40);

        box.Left.Should().Be(10);
        box.Top.Should().Be(20);
        box.Right.Should().Be(40);
        box.Bottom.Should().Be(60);
        box.Area.Should().Be(1200);
    }

    [Fact]
    public void ScaleRespectsFactor()
    {
        var box = new BoundingBox(0, 0, 10, 10);

        var scaled = box.Scale(2);

        scaled.Width.Should().Be(20);
        scaled.Height.Should().Be(20);
    }

    [Fact]
    public void IntersectReturnsEmptyWhenNoOverlap()
    {
        var first = new BoundingBox(0, 0, 10, 10);
        var second = new BoundingBox(20, 20, 30, 30);

        var intersection = first.Intersect(second);

        intersection.Should().Be(default(BoundingBox));
        first.Intersects(second).Should().BeFalse();
    }

    [Fact]
    public void UnionEncompassesBothBoxes()
    {
        var first = new BoundingBox(0, 0, 10, 10);
        var second = new BoundingBox(5, 5, 12, 15);

        var union = first.Union(second);

        union.Left.Should().Be(0);
        union.Right.Should().Be(12);
        union.Bottom.Should().Be(15);
        union.IntersectionOverUnion(first).Should().BeApproximately(first.Area / union.Area, 1e-12);
    }

    [Fact]
    public void ContainsValidatesPointInside()
    {
        var box = new BoundingBox(0, 0, 10, 10);
        box.Contains(5, 5).Should().BeTrue();
        box.Contains(11, 5).Should().BeFalse();
    }

    [Fact]
    public void FromPointsCreatesTightBounds()
    {
        var box = BoundingBox.FromPoints(new[]
        {
            new Point2D(2, 5),
            new Point2D(-1, 10),
            new Point2D(3, -4)
        });

        box.Left.Should().Be(-1);
        box.Top.Should().Be(-4);
        box.Right.Should().Be(3);
        box.Bottom.Should().Be(10);
    }

    [Fact]
    public void ExpandToIncludePointExtendsBounds()
    {
        var box = BoundingBox.FromSize(0, 0, 1, 1);
        var expanded = box.ExpandToInclude(new Point2D(5, -2));

        expanded.Left.Should().Be(0);
        expanded.Top.Should().Be(-2);
        expanded.Right.Should().Be(5);
        expanded.Bottom.Should().Be(1);
    }

    [Fact]
    public void DistanceToReturnsZeroForInnerPoint()
    {
        var box = BoundingBox.FromSize(0, 0, 2, 2);
        box.DistanceTo(new Point2D(1, 1)).Should().Be(0);
        box.DistanceTo(new Point2D(10, 10)).Should().BeGreaterThan(10);
    }

    [Fact]
    public void ToPolygonReturnsRectangle()
    {
        var box = BoundingBox.FromSize(1, 2, 3, 4);
        var polygon = box.ToPolygon();

        polygon.Count.Should().Be(4);
        polygon[0].Should().Be(new Point2D(1, 2));
        polygon[2].Should().Be(new Point2D(4, 6));
        polygon.Area.Should().Be(box.Area);
    }
}
