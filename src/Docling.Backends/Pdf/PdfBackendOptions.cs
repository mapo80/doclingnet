using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Docling.Backends.Pdf;

/// <summary>
/// Configuration container for <see cref="PdfBackend"/> instances.
/// </summary>
public sealed class PdfBackendOptions
{
    public required Func<CancellationToken, Task<Stream>> StreamFactory { get; init; }

    public IReadOnlyCollection<int>? Pages { get; init; }

    public PdfRenderSettings RenderSettings { get; init; } = new();

    public string? DocumentId { get; init; }

    public string? SourceName { get; init; }

    public IReadOnlyDictionary<string, string>? Metadata { get; init; }
}
