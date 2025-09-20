using System;
using System.Collections.Generic;
using System.Globalization;

namespace Docling.Core.Geometry;

/// <summary>
/// Represents an axis-aligned bounding box using a top-left origin coordinate system.
/// Provides rich helpers mirroring the Docling Python geometry semantics.
/// </summary>
public readonly record struct BoundingBox : IEquatable<BoundingBox>
{
    private const double Epsilon = 1e-9;

    public BoundingBox(double left, double top, double right, double bottom)
    {
        if (!IsValid(left, top, right, bottom))
        {
            throw new ArgumentOutOfRangeException(nameof(right), "Bounding box coordinates are invalid.");
        }

        Left = left;
        Top = top;
        Right = right;
        Bottom = bottom;
    }

    public double Left { get; }

    public double Top { get; }

    public double Right { get; }

    public double Bottom { get; }

    public bool IsEmpty => Width <= Epsilon || Height <= Epsilon;

    public double Width => Right - Left;

    public double Height => Bottom - Top;

    public double Area => Width * Height;

    public double AspectRatio => Height <= Epsilon ? double.NaN : Width / Height;

    public static BoundingBox FromSize(double left, double top, double width, double height)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(width, nameof(width));
        ArgumentOutOfRangeException.ThrowIfNegative(height, nameof(height));

        return new BoundingBox(left, top, left + width, top + height);
    }

    public static BoundingBox FromPoints(IEnumerable<Point2D> points)
    {
        ArgumentNullException.ThrowIfNull(points);
        using var enumerator = points.GetEnumerator();
        if (!enumerator.MoveNext())
        {
            throw new ArgumentException("At least one point is required to construct a bounding box.", nameof(points));
        }

        var first = enumerator.Current;
        double left = first.X, right = first.X, top = first.Y, bottom = first.Y;
        while (enumerator.MoveNext())
        {
            var point = enumerator.Current;
            left = Math.Min(left, point.X);
            right = Math.Max(right, point.X);
            top = Math.Min(top, point.Y);
            bottom = Math.Max(bottom, point.Y);
        }

        return new BoundingBox(left, top, right, bottom);
    }

    public static bool TryCreate(double left, double top, double right, double bottom, out BoundingBox box)
    {
        if (!IsValid(left, top, right, bottom))
        {
            box = default;
            return false;
        }

        box = new BoundingBox(left, top, right, bottom);
        return true;
    }

    public BoundingBox Scale(double factor)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(factor, nameof(factor));

        return Scale(factor, factor);
    }

    public BoundingBox Scale(double scaleX, double scaleY)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(scaleX, nameof(scaleX));
        ArgumentOutOfRangeException.ThrowIfNegative(scaleY, nameof(scaleY));

        return new BoundingBox(Left, Top, Left + (Width * scaleX), Top + (Height * scaleY));
    }

    public BoundingBox Translate(double deltaX, double deltaY) =>
        new(Left + deltaX, Top + deltaY, Right + deltaX, Bottom + deltaY);

    public BoundingBox Inflate(double horizontal, double vertical)
    {
        var resultingWidth = Width + (horizontal * 2);
        var resultingHeight = Height + (vertical * 2);

        ArgumentOutOfRangeException.ThrowIfNegative(resultingWidth, nameof(horizontal));
        ArgumentOutOfRangeException.ThrowIfNegative(resultingHeight, nameof(vertical));

        return new BoundingBox(Left - horizontal, Top - vertical, Right + horizontal, Bottom + vertical);
    }

    public BoundingBox ExpandToInclude(BoundingBox other)
    {
        if (other == default)
        {
            return this;
        }

        return Union(other);
    }

    public BoundingBox ExpandToInclude(Point2D point)
    {
        var left = Math.Min(Left, point.X);
        var right = Math.Max(Right, point.X);
        var top = Math.Min(Top, point.Y);
        var bottom = Math.Max(Bottom, point.Y);
        return new BoundingBox(left, top, right, bottom);
    }

    public BoundingBox Intersect(BoundingBox other)
    {
        var left = Math.Max(Left, other.Left);
        var top = Math.Max(Top, other.Top);
        var right = Math.Min(Right, other.Right);
        var bottom = Math.Min(Bottom, other.Bottom);

        if (!IsValid(left, top, right, bottom) || right - left <= Epsilon || bottom - top <= Epsilon)
        {
            return default;
        }

        return new BoundingBox(left, top, right, bottom);
    }

    public bool Intersects(BoundingBox other)
    {
        if (IsEmpty || other.IsEmpty)
        {
            return false;
        }

        return !(other.Left >= Right ||
                 other.Right <= Left ||
                 other.Top >= Bottom ||
                 other.Bottom <= Top);
    }

    public double IntersectionOverUnion(BoundingBox other)
    {
        var intersection = Intersect(other);
        if (intersection.IsEmpty)
        {
            return 0d;
        }

        var unionArea = Area + other.Area - intersection.Area;
        return unionArea <= 0 ? 0 : intersection.Area / unionArea;
    }

    public BoundingBox Union(BoundingBox other)
    {
        var left = Math.Min(Left, other.Left);
        var top = Math.Min(Top, other.Top);
        var right = Math.Max(Right, other.Right);
        var bottom = Math.Max(Bottom, other.Bottom);

        return new BoundingBox(left, top, right, bottom);
    }

    public bool Contains(BoundingBox other) =>
        !IsEmpty && !other.IsEmpty &&
        Left <= other.Left &&
        Top <= other.Top &&
        Right >= other.Right &&
        Bottom >= other.Bottom;

    public bool Contains(Point2D point) => Contains(point.X, point.Y);

    public bool Contains(double x, double y) =>
        x >= Left && x <= Right && y >= Top && y <= Bottom;

    public double DistanceTo(Point2D point)
    {
        var clampedX = Math.Max(Left, Math.Min(Right, point.X));
        var clampedY = Math.Max(Top, Math.Min(Bottom, point.Y));
        var dx = point.X - clampedX;
        var dy = point.Y - clampedY;
        return Math.Sqrt((dx * dx) + (dy * dy));
    }

    public BoundingBox Normalized()
    {
        var left = Math.Min(Left, Right);
        var right = Math.Max(Left, Right);
        var top = Math.Min(Top, Bottom);
        var bottom = Math.Max(Top, Bottom);

        return new BoundingBox(left, top, right, bottom);
    }

    public BoundingBox SnapToIntegers() =>
        new(Math.Floor(Left), Math.Floor(Top), Math.Ceiling(Right), Math.Ceiling(Bottom));

    public BoundingBox Round(int digits)
    {
        var left = Math.Round(Left, digits, MidpointRounding.AwayFromZero);
        var top = Math.Round(Top, digits, MidpointRounding.AwayFromZero);
        var right = Math.Round(Right, digits, MidpointRounding.AwayFromZero);
        var bottom = Math.Round(Bottom, digits, MidpointRounding.AwayFromZero);
        return new BoundingBox(left, top, right, bottom);
    }

    public Polygon ToPolygon()
    {
        return Polygon.FromPoints(new[]
        {
            new Point2D(Left, Top),
            new Point2D(Right, Top),
            new Point2D(Right, Bottom),
            new Point2D(Left, Bottom)
        });
    }

    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"[{Left}, {Top}, {Right}, {Bottom}]");

    internal static bool IsValid(double left, double top, double right, double bottom) =>
        !double.IsNaN(left) &&
        !double.IsNaN(top) &&
        !double.IsNaN(right) &&
        !double.IsNaN(bottom) &&
        !double.IsInfinity(left) &&
        !double.IsInfinity(top) &&
        !double.IsInfinity(right) &&
        !double.IsInfinity(bottom) &&
        right >= left &&
        bottom >= top;
}
