using System.Threading;
using System.Threading.Tasks;

namespace Docling.Backends.Abstractions;

/// <summary>
/// Specialized backend for PDF sources with page count introspection.
/// </summary>
public interface IPdfBackend : IImageBackend
{
    public Task<int> GetPageCountAsync(CancellationToken cancellationToken);
}
