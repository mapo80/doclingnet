using System.Collections.Generic;
using System.Threading;
using Docling.Backends.Pdf;

namespace Docling.Backends.Abstractions;

/// <summary>
/// Represents a backend capable of providing page images for the pipeline.
/// </summary>
public interface IImageBackend
{
    IAsyncEnumerable<PageImage> LoadAsync(CancellationToken cancellationToken);
}
