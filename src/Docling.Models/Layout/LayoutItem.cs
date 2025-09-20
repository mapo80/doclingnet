using System.Collections.Generic;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Models.Layout;

/// <summary>
/// Represents a layout model prediction for a specific region.
/// </summary>
public sealed record LayoutItem(
    PageReference Page,
    BoundingBox BoundingBox,
    LayoutItemKind Kind,
    IReadOnlyList<Polygon> Polygons);

public enum LayoutItemKind
{
    Text,
    Table,
    Figure,
}
