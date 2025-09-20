using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;

namespace Docling.Pipelines.Preprocessing;

/// <summary>
/// Normalises <see cref="PageImage"/> instances ahead of layout/OCR analysis.
/// </summary>
public interface IPagePreprocessor
{
    Task<PageImage> PreprocessAsync(PageImage pageImage, CancellationToken cancellationToken = default);
}
