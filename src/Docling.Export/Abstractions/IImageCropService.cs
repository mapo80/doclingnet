using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Core.Geometry;
using Docling.Export.Imaging;

namespace Docling.Export.Abstractions;

/// <summary>
/// Crops rendered pages to produce focused artefacts (figures, tables, etc.).
/// </summary>
public interface IImageCropService
{
    public Task<CroppedImage> CropAsync(PageImage source, BoundingBox region, CancellationToken cancellationToken = default);
}
