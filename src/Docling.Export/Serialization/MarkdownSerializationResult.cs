using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Docling.Export.Serialization;

/// <summary>
/// Represents the outcome of serialising a <see cref="Docling.Core.Documents.DoclingDocument"/> into markdown.
/// </summary>
public sealed class MarkdownSerializationResult
{
    private readonly ReadOnlyCollection<MarkdownAsset> _assetsView;
    private readonly ReadOnlyDictionary<string, string> _metadataView;

    public MarkdownSerializationResult(
        string markdown,
        IReadOnlyList<MarkdownAsset> assets,
        IReadOnlyDictionary<string, string> metadata)
    {
        ArgumentNullException.ThrowIfNull(markdown);
        ArgumentNullException.ThrowIfNull(assets);
        ArgumentNullException.ThrowIfNull(metadata);

        Markdown = markdown;
        _assetsView = new ReadOnlyCollection<MarkdownAsset>(assets is List<MarkdownAsset> list ? list : new List<MarkdownAsset>(assets));
        _metadataView = new ReadOnlyDictionary<string, string>(metadata is Dictionary<string, string> dict
            ? dict
            : new Dictionary<string, string>(metadata, StringComparer.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Gets the generated markdown payload.
    /// </summary>
    public string Markdown { get; }

    /// <summary>
    /// Gets the collection of assets required by the markdown document.
    /// </summary>
    public IReadOnlyList<MarkdownAsset> Assets => _assetsView;

    /// <summary>
    /// Gets a metadata dictionary mirroring <see cref="Docling.Core.Documents.DoclingDocument.Properties"/>.
    /// </summary>
    public IReadOnlyDictionary<string, string> Metadata => _metadataView;
}
