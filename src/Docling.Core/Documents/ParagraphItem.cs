using System;
using System.Collections.Generic;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Core.Documents;

/// <summary>
/// Represents a contiguous text paragraph.
/// </summary>
public sealed class ParagraphItem : DocItem
{
    public ParagraphItem(
        PageReference page,
        BoundingBox box,
        string text,
        string? id = null,
        IEnumerable<string>? tags = null,
        IReadOnlyDictionary<string, object?>? metadata = null,
        DateTimeOffset? createdAt = null)
        : base(DocItemKind.Paragraph, page, box, id, tags, metadata, createdAt)
    {
        UpdateText(text);
    }

    public string Text { get; private set; } = string.Empty;

    public void UpdateText(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        Text = text;
        SetMetadata("text", text);
    }
}
