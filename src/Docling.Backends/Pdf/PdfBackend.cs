using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Abstractions;
using Docling.Backends;
using Docling.Core.Primitives;

namespace Docling.Backends.Pdf;

/// <summary>
/// Default PDF ingestion backend orchestrating <see cref="IPdfPageRenderer"/>.
/// </summary>
public sealed class PdfBackend : IPdfBackend
{
    private readonly IPdfPageRenderer _renderer;
    private readonly PdfBackendOptions _options;

    public PdfBackend(IPdfPageRenderer renderer, PdfBackendOptions options)
    {
        _renderer = renderer ?? throw new ArgumentNullException(nameof(renderer));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        ArgumentNullException.ThrowIfNull(options.StreamFactory);
        ArgumentNullException.ThrowIfNull(options.RenderSettings);
    }

    public async Task<int> GetPageCountAsync(CancellationToken cancellationToken)
    {
        var stream = await _options.StreamFactory(cancellationToken).ConfigureAwait(false);
        try
        {
            var buffer = await StreamUtilities.ReadAllBytesAsync(stream, cancellationToken).ConfigureAwait(false);
            using var pdfStream = StreamUtilities.AsMemoryStream(buffer);
            return await _renderer.GetPageCountAsync(pdfStream, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            await stream.DisposeAsync().ConfigureAwait(false);
        }
    }

    public async IAsyncEnumerable<PageImage> LoadAsync([EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var stream = await _options.StreamFactory(cancellationToken).ConfigureAwait(false);
        try
        {
            var buffer = await StreamUtilities.ReadAllBytesAsync(stream, cancellationToken).ConfigureAwait(false);
            var pages = _options.Pages?.ToArray();

            using var rendererStream = StreamUtilities.AsMemoryStream(buffer);
            await foreach (var pageImage in _renderer.RenderAsync(rendererStream, pages, _options.RenderSettings, cancellationToken).ConfigureAwait(false))
            {
                var metadata = CreateBaseMetadata(pageImage.Page.PageNumber).WithAdditionalProperties(RuntimeProperties(pageImage));
                pageImage.Metadata = metadata;
                yield return pageImage;
            }
        }
        finally
        {
            await stream.DisposeAsync().ConfigureAwait(false);
        }
    }

    private PageImageMetadata CreateBaseMetadata(int pageNumber)
    {
        var properties = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["pageNumber"] = (pageNumber + 1).ToString(CultureInfo.InvariantCulture),
        };

        if (_options.Metadata is not null)
        {
            foreach (var kvp in _options.Metadata)
            {
                properties[kvp.Key] = kvp.Value;
            }
        }

        return new PageImageMetadata(_options.DocumentId, _options.SourceName, "application/pdf", properties);
    }

    private static IEnumerable<KeyValuePair<string, string>> RuntimeProperties(PageImage pageImage)
    {
        yield return new KeyValuePair<string, string>("widthPixels", pageImage.Width.ToString(CultureInfo.InvariantCulture));
        yield return new KeyValuePair<string, string>("heightPixels", pageImage.Height.ToString(CultureInfo.InvariantCulture));
        yield return new KeyValuePair<string, string>("dpi", pageImage.Page.Dpi.ToString(CultureInfo.InvariantCulture));
    }
}
