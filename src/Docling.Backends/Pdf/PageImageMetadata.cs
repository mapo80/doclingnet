using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Docling.Backends.Pdf;

/// <summary>
/// Represents auxiliary metadata attached to a <see cref="PageImage"/>.
/// </summary>
public sealed class PageImageMetadata
{
    private static readonly IReadOnlyDictionary<string, string> EmptyProperties =
        new ReadOnlyDictionary<string, string>(new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase));

    public static PageImageMetadata Empty { get; } = new(null, null, null, null);

    public PageImageMetadata(
        string? sourceId,
        string? sourceName,
        string? mediaType,
        IReadOnlyDictionary<string, string>? properties)
    {
        SourceId = sourceId;
        SourceName = sourceName;
        MediaType = mediaType;
        Properties = properties is null
            ? EmptyProperties
            : new ReadOnlyDictionary<string, string>(new Dictionary<string, string>(properties, StringComparer.OrdinalIgnoreCase));
    }

    public string? SourceId { get; }

    public string? SourceName { get; }

    public string? MediaType { get; }

    public IReadOnlyDictionary<string, string> Properties { get; }

    public PageImageMetadata WithAdditionalProperties(IEnumerable<KeyValuePair<string, string>> properties)
    {
        ArgumentNullException.ThrowIfNull(properties);

        var merged = new Dictionary<string, string>(Properties, StringComparer.OrdinalIgnoreCase);
        foreach (var (key, value) in properties)
        {
            merged[key] = value;
        }

        return new PageImageMetadata(SourceId, SourceName, MediaType, merged);
    }
}
