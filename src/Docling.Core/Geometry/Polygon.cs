using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Docling.Core.Geometry;

/// <summary>
/// Represents a closed polygon defined by an ordered set of vertices.
/// Provides helpers used by Docling when reasoning about layout polygons.
/// </summary>
public sealed class Polygon : IReadOnlyList<Point2D>, IEquatable<Polygon>
{
    private readonly ImmutableArray<Point2D> _vertices;
    private readonly BoundingBox _boundingBox;
    private readonly double _signedArea;

    private Polygon(ImmutableArray<Point2D> vertices)
    {
        if (vertices.Length < 3)
        {
            throw new ArgumentException("A polygon requires at least three vertices.", nameof(vertices));
        }

        _vertices = Normalize(vertices);
        _boundingBox = BoundingBox.FromPoints(_vertices);
        _signedArea = CalculateSignedArea(_vertices);
    }

    public static Polygon FromPoints(IEnumerable<Point2D> points)
    {
        ArgumentNullException.ThrowIfNull(points);
        var materialized = points.ToImmutableArray();
        return new Polygon(materialized);
    }

    public int Count => _vertices.Length;

    public Point2D this[int index] => _vertices[index];

    public BoundingBox BoundingBox => _boundingBox;

    public double Area => Math.Abs(_signedArea);

    public double SignedArea => _signedArea;

    public double Perimeter
    {
        get
        {
            double length = 0;
            for (var i = 0; i < _vertices.Length; i++)
            {
                var current = _vertices[i];
                var next = _vertices[(i + 1) % _vertices.Length];
                length += current.DistanceTo(next);
            }

            return length;
        }
    }

    public bool IsClockwise => _signedArea < 0;

    public bool IsCounterClockwise => _signedArea > 0;

    public Polygon EnsureCounterClockwise() =>
        IsCounterClockwise ? this : new Polygon(_vertices.Reverse().ToImmutableArray());

    public Polygon Translate(double deltaX, double deltaY) =>
        new(_vertices.Select(p => p.Offset(deltaX, deltaY)).ToImmutableArray());

    public Polygon Scale(double scaleX, double scaleY, Point2D? origin = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(scaleX, nameof(scaleX));
        ArgumentOutOfRangeException.ThrowIfNegative(scaleY, nameof(scaleY));
        var pivot = origin ?? Point2D.Origin;
        return new Polygon(_vertices.Select(p =>
        {
            var translated = p.Offset(-pivot.X, -pivot.Y);
            var scaled = translated.Scale(scaleX, scaleY);
            return scaled.Offset(pivot.X, pivot.Y);
        }).ToImmutableArray());
    }

    public Polygon Rotate(double angleRadians, Point2D? origin = null)
    {
        var pivot = origin ?? Point2D.Origin;
        return new Polygon(_vertices.Select(p => p.Rotate(angleRadians, pivot)).ToImmutableArray());
    }

    public bool Contains(Point2D point)
    {
        var inside = false;
        for (int i = 0, j = _vertices.Length - 1; i < _vertices.Length; j = i++)
        {
            var pi = _vertices[i];
            var pj = _vertices[j];
            var intersects = ((pi.Y > point.Y) != (pj.Y > point.Y)) &&
                             (point.X < (pj.X - pi.X) * (point.Y - pi.Y) / (pj.Y - pi.Y + double.Epsilon) + pi.X);
            if (intersects)
            {
                inside = !inside;
            }
        }

        return inside;
    }

    public bool Intersects(BoundingBox box)
    {
        if (!BoundingBox.Intersects(box) && !BoundingBox.Contains(box))
        {
            return false;
        }

        if (Contains(new Point2D(box.Left, box.Top)) ||
            Contains(new Point2D(box.Right, box.Top)) ||
            Contains(new Point2D(box.Right, box.Bottom)) ||
            Contains(new Point2D(box.Left, box.Bottom)))
        {
            return true;
        }

        var polygonEdges = EnumerateEdges().ToList();
        foreach (var edge in polygonEdges)
        {
            if (EdgeIntersectsBox(edge.start, edge.end, box))
            {
                return true;
            }
        }

        return false;
    }

    public Polygon Simplify(double tolerance = 0.0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tolerance);

        if (_vertices.Length <= 3)
        {
            return this;
        }

        var simplified = new List<Point2D>(_vertices.Length);
        for (var i = 0; i < _vertices.Length; i++)
        {
            var prev = _vertices[(i - 1 + _vertices.Length) % _vertices.Length];
            var current = _vertices[i];
            var next = _vertices[(i + 1) % _vertices.Length];

            var area = Math.Abs((prev.X * (current.Y - next.Y) +
                                 current.X * (next.Y - prev.Y) +
                                 next.X * (prev.Y - current.Y)) / 2);

            if (area > tolerance)
            {
                simplified.Add(current);
            }
        }

        if (simplified.Count < 3)
        {
            simplified = _vertices.ToList();
        }

        return new Polygon(simplified.ToImmutableArray());
    }

    public Polygon Transform(Func<Point2D, Point2D> transformer)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        return new Polygon(_vertices.Select(transformer).ToImmutableArray());
    }

    public IEnumerator<Point2D> GetEnumerator() => ((IEnumerable<Point2D>)_vertices).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public bool Equals(Polygon? other)
    {
        if (other is null || other.Count != Count)
        {
            return false;
        }

        for (var i = 0; i < Count; i++)
        {
            if (!this[i].Equals(other[i]))
            {
                return false;
            }
        }

        return true;
    }

    public override bool Equals(object? obj) => Equals(obj as Polygon);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (var point in _vertices)
        {
            hash.Add(point);
        }

        return hash.ToHashCode();
    }

    private static ImmutableArray<Point2D> Normalize(ImmutableArray<Point2D> vertices)
    {
        var deduped = vertices;
        if (vertices.Length > 1 && vertices[0].Equals(vertices[^1]))
        {
            deduped = vertices.RemoveAt(vertices.Length - 1);
        }

        if (deduped.Length < 3)
        {
            throw new ArgumentException("A polygon requires at least three distinct vertices.", nameof(vertices));
        }

        return deduped;
    }

    private static double CalculateSignedArea(ImmutableArray<Point2D> vertices)
    {
        double sum = 0;
        for (var i = 0; i < vertices.Length; i++)
        {
            var current = vertices[i];
            var next = vertices[(i + 1) % vertices.Length];
            sum += (current.X * next.Y) - (next.X * current.Y);
        }

        return sum / 2;
    }

    private IEnumerable<(Point2D start, Point2D end)> EnumerateEdges()
    {
        for (var i = 0; i < _vertices.Length; i++)
        {
            var start = _vertices[i];
            var end = _vertices[(i + 1) % _vertices.Length];
            yield return (start, end);
        }
    }

    private static bool EdgeIntersectsBox(Point2D start, Point2D end, BoundingBox box)
    {
        var edgeBox = BoundingBox.FromPoints(new[] { start, end });
        if (!edgeBox.Intersects(box))
        {
            return false;
        }

        var direction = new Point2D(end.X - start.X, end.Y - start.Y);
        double[] p = { -direction.X, direction.X, -direction.Y, direction.Y };
        double[] q = { start.X - box.Left, box.Right - start.X, start.Y - box.Top, box.Bottom - start.Y };

        double u1 = 0.0, u2 = 1.0;
        for (var i = 0; i < 4; i++)
        {
            if (Math.Abs(p[i]) < double.Epsilon)
            {
                if (q[i] < 0)
                {
                    return false;
                }
            }
            else
            {
                var t = q[i] / p[i];
                if (p[i] < 0)
                {
                    u1 = Math.Max(u1, t);
                }
                else
                {
                    u2 = Math.Min(u2, t);
                }

                if (u1 > u2)
                {
                    return false;
                }
            }
        }

        return true;
    }
}
