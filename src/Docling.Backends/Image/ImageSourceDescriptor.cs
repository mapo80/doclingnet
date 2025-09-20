using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Docling.Backends.Image;

/// <summary>
/// Represents a single image source consumed by the <see cref="ImageBackend"/>.
/// </summary>
public sealed class ImageSourceDescriptor
{
    public required Func<CancellationToken, Task<Stream>> StreamFactory { get; init; }

    public string? Identifier { get; init; }

    public string? FileName { get; init; }

    public string? MediaType { get; init; }

    public int? Dpi { get; init; }

    public IReadOnlyDictionary<string, string>? Metadata { get; init; }
}
