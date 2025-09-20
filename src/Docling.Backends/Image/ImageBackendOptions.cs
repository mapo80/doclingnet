using System;
using System.Collections.Generic;

namespace Docling.Backends.Image;

/// <summary>
/// Configuration for the <see cref="ImageBackend"/>.
/// </summary>
public sealed class ImageBackendOptions
{
    public required IReadOnlyList<ImageSourceDescriptor> Sources { get; init; }

    public int DefaultDpi { get; init; } = 300;

    public string? DocumentId { get; init; }

    public string? SourceName { get; init; }

    public IReadOnlyDictionary<string, string>? Metadata { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Sources);

        if (Sources.Count == 0)
        {
            throw new InvalidOperationException("At least one image source must be provided.");
        }

        foreach (var source in Sources)
        {
            ArgumentNullException.ThrowIfNull(source.StreamFactory);
        }
    }
}
