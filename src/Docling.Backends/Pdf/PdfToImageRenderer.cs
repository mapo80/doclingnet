using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends;
using Docling.Core.Primitives;
using PDFtoImage;

namespace Docling.Backends.Pdf;

/// <summary>
/// Default <see cref="IPdfPageRenderer"/> implementation backed by the <c>PdfToImage</c> NuGet package.
/// </summary>
public sealed class PdfToImageRenderer : IPdfPageRenderer
{
    [SupportedOSPlatform("windows")]
    [SupportedOSPlatform("linux")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("ios13.6")]
    [SupportedOSPlatform("maccatalyst13.5")]
    [SupportedOSPlatform("android31.0")]
    public async Task<int> GetPageCountAsync(Stream pdfStream, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(pdfStream);
        var buffer = await StreamUtilities.ReadAllBytesAsync(pdfStream, cancellationToken).ConfigureAwait(false);
        return Conversion.GetPageCount(buffer, password: null);
    }

    [SupportedOSPlatform("windows")]
    [SupportedOSPlatform("linux")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("ios13.6")]
    [SupportedOSPlatform("maccatalyst13.5")]
    [SupportedOSPlatform("android31.0")]
    public async IAsyncEnumerable<PageImage> RenderAsync(
        Stream pdfStream,
        IReadOnlyCollection<int>? pages,
        PdfRenderSettings settings,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(pdfStream);
        ArgumentNullException.ThrowIfNull(settings);

        var buffer = await StreamUtilities.ReadAllBytesAsync(pdfStream, cancellationToken).ConfigureAwait(false);
        var renderOptions = settings.ToRenderOptions();
        var zeroIndexedPages = pages?.ToArray() ?? Enumerable.Range(0, Conversion.GetPageCount(buffer, settings.Password)).ToArray();

        var orderedPages = zeroIndexedPages.OrderBy(static page => page).ToArray();

        var pageIndex = 0;
        await foreach (var bitmap in Conversion.ToImagesAsync(buffer, orderedPages, settings.Password, renderOptions, cancellationToken).ConfigureAwait(false))
        {
            yield return new PageImage(new PageReference(orderedPages[pageIndex++], settings.Dpi), bitmap);
        }
    }
}
