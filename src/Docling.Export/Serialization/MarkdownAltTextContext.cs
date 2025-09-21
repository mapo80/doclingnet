using System;
using Docling.Core.Documents;
using Docling.Export.Imaging;

namespace Docling.Export.Serialization;

/// <summary>
/// Describes the item currently being serialised so that custom
/// implementations can provide enriched alternative text.
/// </summary>
public sealed class MarkdownAltTextContext
{
    private MarkdownAltTextContext(DocItem item, string label, string? caption, ImageRef? image)
    {
        Item = item ?? throw new ArgumentNullException(nameof(item));
        Label = label ?? throw new ArgumentNullException(nameof(label));
        Caption = caption;
        Image = image;
    }

    /// <summary>
    /// Gets the document item referenced by the current markdown element.
    /// </summary>
    public DocItem Item { get; }

    /// <summary>
    /// Gets the formatted label assigned to the item (e.g. "Figure 1").
    /// </summary>
    public string Label { get; }

    /// <summary>
    /// Gets the concatenated caption text associated with the item, if any.
    /// </summary>
    public string? Caption { get; }

    /// <summary>
    /// Gets the image artefact attached to the item when available.
    /// </summary>
    public ImageRef? Image { get; }

    /// <summary>
    /// Attempts to cast <see cref="Item"/> to <see cref="PictureItem"/>.
    /// </summary>
    public PictureItem? Picture => Item as PictureItem;

    /// <summary>
    /// Attempts to cast <see cref="Item"/> to <see cref="TableItem"/>.
    /// </summary>
    public TableItem? Table => Item as TableItem;

    /// <summary>
    /// Creates a context instance for picture items.
    /// </summary>
    public static MarkdownAltTextContext ForPicture(PictureItem picture, string label, string? caption, ImageRef? image)
        => new(picture ?? throw new ArgumentNullException(nameof(picture)), label, caption, image);

    /// <summary>
    /// Creates a context instance for table items.
    /// </summary>
    public static MarkdownAltTextContext ForTable(TableItem table, string label, string? caption, ImageRef? image)
        => new(table ?? throw new ArgumentNullException(nameof(table)), label, caption, image);
}
