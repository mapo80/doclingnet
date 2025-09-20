using System.Collections.Generic;
using System.Threading;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using SkiaSharp;

namespace Docling.Models.Ocr;

public sealed record OcrRequest(PageReference Page, SKBitmap Image, BoundingBox Region, IReadOnlyDictionary<string, string> Metadata);

public sealed record OcrLine(string Text, BoundingBox BoundingBox, double Confidence);

public enum OcrRegionKind
{
    /// <summary>
    /// OCR executed on a layout block detected by the layout analysis model.
    /// </summary>
    LayoutBlock,

    /// <summary>
    /// OCR executed on a single table cell as part of table structure recovery.
    /// </summary>
    TableCell,

    /// <summary>
    /// OCR executed on the entire page bitmap (fallback when no layout is available).
    /// </summary>
    FullPage,
}

public sealed record OcrBlockResult(
    PageReference Page,
    BoundingBox Region,
    OcrRegionKind Kind,
    IReadOnlyDictionary<string, string> Metadata,
    IReadOnlyList<OcrLine> Lines);

public sealed record OcrDocumentResult(IReadOnlyList<OcrBlockResult> Blocks);

public interface IOcrService : IDisposable
{
    IAsyncEnumerable<OcrLine> RecognizeAsync(OcrRequest request, CancellationToken cancellationToken = default);
}
