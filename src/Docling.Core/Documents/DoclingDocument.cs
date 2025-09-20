using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Linq;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Core.Documents;

/// <summary>
/// Represents the aggregate Docling document composed by the pipeline.
/// Provides deterministic ordering, metadata handling, and query helpers.
/// </summary>
public sealed class DoclingDocument
{
    private readonly List<DocItem> _items = new();
    private readonly Dictionary<string, DocItem> _itemsById = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _properties = new(StringComparer.OrdinalIgnoreCase);
    private readonly ReadOnlyCollection<DocItem> _itemsView;
    private readonly ReadOnlyDictionary<string, string> _propertiesView;

    public DoclingDocument(
        string sourceId,
        IReadOnlyList<PageReference> pages,
        string? documentId = null,
        DateTimeOffset? createdAt = null,
        IReadOnlyDictionary<string, string>? properties = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourceId);
        ArgumentNullException.ThrowIfNull(pages);
        if (pages.Count == 0)
        {
            throw new ArgumentException("At least one page reference must be supplied.", nameof(pages));
        }

        SourceId = sourceId;
        Id = string.IsNullOrWhiteSpace(documentId) ? $"doc-{Guid.NewGuid():N}" : documentId!;
        CreatedAt = createdAt ?? DateTimeOffset.UtcNow;
        Pages = pages.ToImmutableArray();
        if (properties is not null)
        {
            foreach (var kvp in properties)
            {
                SetProperty(kvp.Key, kvp.Value);
            }
        }

        _itemsView = _items.AsReadOnly();
        _propertiesView = new ReadOnlyDictionary<string, string>(_properties);
    }

    public string Id { get; }

    public string SourceId { get; }

    public DateTimeOffset CreatedAt { get; }

    public IReadOnlyList<PageReference> Pages { get; }

    public IReadOnlyList<DocItem> Items => _itemsView;

    public IReadOnlyDictionary<string, string> Properties => _propertiesView;

    public void SetProperty(string name, string value)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(value);
        _properties[name] = value;
    }

    public bool TryGetProperty(string name, out string value)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        return _properties.TryGetValue(name, out value!);
    }

    public void MergeProperties(IReadOnlyDictionary<string, string> properties)
    {
        ArgumentNullException.ThrowIfNull(properties);
        foreach (var (key, value) in properties)
        {
            if (string.IsNullOrWhiteSpace(key) || value is null)
            {
                continue;
            }

            _properties[key] = value;
        }
    }

    public void AddItem(DocItem item)
    {
        ArgumentNullException.ThrowIfNull(item);
        if (!_itemsById.TryAdd(item.Id, item))
        {
            throw new InvalidOperationException($"An item with id '{item.Id}' already exists in the document.");
        }

        _items.Add(item);
        _items.Sort(CompareItems);
    }

    public void AddItems(IEnumerable<DocItem> items)
    {
        ArgumentNullException.ThrowIfNull(items);
        foreach (var item in items)
        {
            AddItem(item);
        }
    }

    public bool RemoveItem(string id)
    {
        ArgumentException.ThrowIfNullOrEmpty(id);
        if (!_itemsById.TryGetValue(id, out var item))
        {
            return false;
        }

        _itemsById.Remove(id);
        _items.Remove(item);
        return true;
    }

    public bool TryGetItem(string id, out DocItem? item)
    {
        ArgumentException.ThrowIfNullOrEmpty(id);
        var result = _itemsById.TryGetValue(id, out var resolved);
        item = resolved;
        return result;
    }

    public IReadOnlyList<DocItem> GetItemsOfKind(DocItemKind kind)
    {
        return _items.Where(x => x.Kind == kind).ToImmutableArray();
    }

    public bool TryFindFirstBoundingBox(Func<DocItem, bool> predicate, out BoundingBox box)
    {
        ArgumentNullException.ThrowIfNull(predicate);

        foreach (var item in _items)
        {
            if (predicate(item))
            {
                box = item.BoundingBox;
                return true;
            }
        }

        box = default;
        return false;
    }

    public DoclingDocument Clone(Func<DocItem, DocItem>? transform = null)
    {
        var clone = new DoclingDocument(SourceId, Pages, Id, CreatedAt, _properties);
        var projector = transform ?? (static item => item);
        foreach (var item in _items)
        {
            clone.AddItem(projector(item));
        }

        return clone;
    }

    private static int CompareItems(DocItem left, DocItem right)
    {
        var pageComparison = left.Page.PageNumber.CompareTo(right.Page.PageNumber);
        if (pageComparison != 0)
        {
            return pageComparison;
        }

        var topComparison = left.BoundingBox.Top.CompareTo(right.BoundingBox.Top);
        if (topComparison != 0)
        {
            return topComparison;
        }

        var leftComparison = left.BoundingBox.Left.CompareTo(right.BoundingBox.Left);
        if (leftComparison != 0)
        {
            return leftComparison;
        }

        var createdComparison = left.CreatedAt.CompareTo(right.CreatedAt);
        if (createdComparison != 0)
        {
            return createdComparison;
        }

        return string.CompareOrdinal(left.Id, right.Id);
    }
}
