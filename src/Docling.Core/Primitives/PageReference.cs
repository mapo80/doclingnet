using System;

namespace Docling.Core.Primitives;

/// <summary>
/// Represents a strongly typed reference to a document page.
/// </summary>
public readonly record struct PageReference
{
    public PageReference(int pageNumber, double dpi)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(pageNumber, nameof(pageNumber));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(dpi, 0, nameof(dpi));

        PageNumber = pageNumber;
        Dpi = dpi;
    }

    public int PageNumber { get; }

    public double Dpi { get; }

    public PageReference WithDpi(double dpi)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(dpi, 0, nameof(dpi));
        return new PageReference(PageNumber, dpi);
    }
}
