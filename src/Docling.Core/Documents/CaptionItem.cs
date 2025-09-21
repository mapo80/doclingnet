using System;
using System.Collections.Generic;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Core.Documents;

/// <summary>
/// Represents a caption associated with a figure or table.
/// </summary>
public sealed class CaptionItem : DocItem
{
    public CaptionItem(
        PageReference page,
        BoundingBox box,
        string text,
        string? targetItemId = null,
        string? id = null,
        IEnumerable<string>? tags = null,
        IReadOnlyDictionary<string, object?>? metadata = null,
        DateTimeOffset? createdAt = null)
        : base(DocItemKind.Caption, page, box, id, tags, metadata, createdAt)
    {
        UpdateText(text);
        UpdateTarget(targetItemId);
    }

    public string Text { get; private set; } = string.Empty;

    public string? TargetItemId { get; private set; }

    public void UpdateText(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        Text = text;
        SetMetadata("text", text);
    }

    public void UpdateTarget(string? targetItemId)
    {
        TargetItemId = string.IsNullOrWhiteSpace(targetItemId) ? null : targetItemId;
        if (TargetItemId is not null)
        {
            SetMetadata("docling:target_item_id", TargetItemId);
        }
    }
}
