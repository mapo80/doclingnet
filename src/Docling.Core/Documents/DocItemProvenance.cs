using System;
using System.Collections.Generic;
using Docling.Core.Geometry;

namespace Docling.Core.Documents;

/// <summary>
/// Represents the provenance of a <see cref="DocItem"/> within the source document.
/// Mirrors the Python provenance tuple storing the originating page, bounding box, and optional char span.
/// </summary>
public sealed class DocItemProvenance
{
    public DocItemProvenance(int pageNumber, BoundingBox boundingBox, int? charStart = null, int? charEnd = null)
    {
        if (pageNumber <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pageNumber), "Page numbers must be positive.");
        }

        PageNumber = pageNumber;
        BoundingBox = boundingBox;

        if (charStart.HasValue && charStart.Value < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(charStart), "Character spans cannot start before zero.");
        }

        if (charEnd.HasValue && charEnd.Value < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(charEnd), "Character spans cannot end before zero.");
        }

        if (charStart.HasValue && charEnd.HasValue && charEnd.Value < charStart.Value)
        {
            throw new ArgumentException("Character span end must be greater than or equal to its start.", nameof(charEnd));
        }

        CharStart = charStart;
        CharEnd = charEnd;
    }

    public int PageNumber { get; }

    public BoundingBox BoundingBox { get; }

    public int? CharStart { get; }

    public int? CharEnd { get; }

    internal IReadOnlyDictionary<string, object?> ToMetadata()
    {
        var dictionary = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
        {
            ["page_number"] = PageNumber,
            ["bounding_box"] = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase)
            {
                ["left"] = BoundingBox.Left,
                ["top"] = BoundingBox.Top,
                ["right"] = BoundingBox.Right,
                ["bottom"] = BoundingBox.Bottom,
            },
        };

        if (CharStart.HasValue)
        {
            dictionary["char_start"] = CharStart.Value;
        }

        if (CharEnd.HasValue)
        {
            dictionary["char_end"] = CharEnd.Value;
        }

        return dictionary;
    }
}
