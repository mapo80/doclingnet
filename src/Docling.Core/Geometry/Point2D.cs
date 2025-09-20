using System;

namespace Docling.Core.Geometry;

/// <summary>
/// Represents a double precision point in the Docling coordinate space.
/// </summary>
public readonly record struct Point2D(double X, double Y)
{
    public static Point2D Origin { get; } = new(0, 0);

    public double DistanceTo(Point2D other)
    {
        var dx = other.X - X;
        var dy = other.Y - Y;
        return Math.Sqrt((dx * dx) + (dy * dy));
    }

    public Point2D Offset(double deltaX, double deltaY) => new(X + deltaX, Y + deltaY);

    public Point2D Scale(double scaleX, double scaleY) => new(X * scaleX, Y * scaleY);

    public Point2D Rotate(double angleRadians, Point2D? origin = null)
    {
        var pivot = origin ?? Origin;
        var translatedX = X - pivot.X;
        var translatedY = Y - pivot.Y;
        var cos = Math.Cos(angleRadians);
        var sin = Math.Sin(angleRadians);
        var rotatedX = (translatedX * cos) - (translatedY * sin);
        var rotatedY = (translatedX * sin) + (translatedY * cos);
        return new Point2D(rotatedX + pivot.X, rotatedY + pivot.Y);
    }
}
