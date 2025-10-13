using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Docling.Backends.Pdf;

/// <summary>
/// Abstraction over the PDF rasterisation engine, simplifying testing.
/// </summary>
public interface IPdfPageRenderer
{
    public Task<int> GetPageCountAsync(Stream pdfStream, CancellationToken cancellationToken);

    public IAsyncEnumerable<PageImage> RenderAsync(Stream pdfStream, IReadOnlyCollection<int>? pages, PdfRenderSettings settings, CancellationToken cancellationToken);
}
