using System.Collections.Generic;
using System.Threading;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using SkiaSharp;

namespace Docling.Models.Ocr;

public sealed record OcrRequest(PageReference Page, SKBitmap Image, BoundingBox Region, IReadOnlyDictionary<string, string> Metadata);

public sealed record OcrLine(string Text, BoundingBox BoundingBox, double Confidence);

public interface IOcrService : IDisposable
{
    IAsyncEnumerable<OcrLine> RecognizeAsync(OcrRequest request, CancellationToken cancellationToken = default);
}
