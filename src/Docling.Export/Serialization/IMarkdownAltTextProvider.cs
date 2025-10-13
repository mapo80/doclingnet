using System;

namespace Docling.Export.Serialization;

/// <summary>
/// Provides alternative text descriptions for markdown artefacts.
/// </summary>
public interface IMarkdownAltTextProvider
{
    /// <summary>
    /// Returns alternative text for the supplied <see cref="MarkdownAltTextContext"/>.
    /// Returning <see langword="null"/> or whitespace leaves the serializer's default
    /// fallback logic in place.
    /// </summary>
    /// <param name="context">The serialization context describing the current artefact.</param>
    /// <returns>An alternative text value or <see langword="null"/> to fall back to defaults.</returns>
    public string? GetAltText(MarkdownAltTextContext context);
}
