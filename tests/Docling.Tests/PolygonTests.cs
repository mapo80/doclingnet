using Docling.Core.Geometry;
using FluentAssertions;

namespace Docling.Tests;

public sealed class PolygonTests
{
    private static readonly Polygon SamplePolygon = Polygon.FromPoints(new[]
    {
        new Point2D(0, 0),
        new Point2D(4, 0),
        new Point2D(4, 3),
        new Point2D(0, 3)
    });

    [Fact]
    public void AreaMatchesExpected()
    {
        SamplePolygon.Area.Should().Be(12);
        SamplePolygon.IsCounterClockwise.Should().BeTrue();
    }

    [Fact]
    public void TranslateMovesVertices()
    {
        var translated = SamplePolygon.Translate(2, -1);

        translated[0].Should().Be(new Point2D(2, -1));
        translated.BoundingBox.Left.Should().Be(2);
        translated.BoundingBox.Top.Should().Be(-1);
    }

    [Fact]
    public void RotateAroundOriginPreservesArea()
    {
        var rotated = SamplePolygon.Rotate(Math.PI / 2);

        rotated.Area.Should().BeApproximately(SamplePolygon.Area, 1e-9);
        rotated.BoundingBox.Width.Should().BeApproximately(3, 1e-9);
        rotated.BoundingBox.Height.Should().BeApproximately(4, 1e-9);
    }

    [Fact]
    public void ContainsDetectsPointInside()
    {
        SamplePolygon.Contains(new Point2D(1, 1)).Should().BeTrue();
        SamplePolygon.Contains(new Point2D(5, 1)).Should().BeFalse();
    }

    [Fact]
    public void SimplifyRemovesCollinearVertices()
    {
        var polygon = Polygon.FromPoints(new[]
        {
            new Point2D(0, 0),
            new Point2D(2, 0),
            new Point2D(4, 0),
            new Point2D(4, 2),
            new Point2D(0, 2)
        });

        var simplified = polygon.Simplify();

        simplified.Count.Should().Be(4);
        simplified.Area.Should().Be(polygon.Area);
    }

    [Fact]
    public void IntersectsBoundingBox()
    {
        var box = BoundingBox.FromSize(3, 1, 3, 3);
        SamplePolygon.Intersects(box).Should().BeTrue();
    }
}
