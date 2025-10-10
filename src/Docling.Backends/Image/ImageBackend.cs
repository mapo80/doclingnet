using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Abstractions;
using Docling.Backends.Pdf;
using Docling.Core.Primitives;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Backends.Image;

/// <summary>
/// Image ingestion backend turning raw image streams into <see cref="PageImage"/> instances.
/// </summary>
public sealed class ImageBackend : IImageBackend
{
    private static readonly Action<ILogger, string, Exception?> DecodeFailureMessage = LoggerMessage.Define<string>(
        LogLevel.Error,
        new EventId(1, nameof(ImageBackend) + ".ImageDecodeFailure"),
        "Failed to load image source {Identifier}.");

    private readonly ImageBackendOptions _options;
    private readonly ILogger<ImageBackend>? _logger;

    public ImageBackend(ImageBackendOptions options, ILogger<ImageBackend>? logger = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _logger = logger;
        _options.Validate();
    }

    public async IAsyncEnumerable<PageImage> LoadAsync([EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var index = 0;
        foreach (var source in _options.Sources)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var pageImage = await CreatePageImageAsync(source, index, cancellationToken).ConfigureAwait(false);
            yield return pageImage;
            index++;
        }
    }

    private async Task<PageImage> CreatePageImageAsync(ImageSourceDescriptor source, int index, CancellationToken cancellationToken)
    {
        var stream = await source.StreamFactory(cancellationToken).ConfigureAwait(false);
        try
        {
            if (stream.CanSeek)
            {
                stream.Seek(0, SeekOrigin.Begin);
            }

            var bitmap = SKBitmap.Decode(stream);
            if (bitmap is null)
            {
                throw new InvalidOperationException(FormattableString.Invariant($"Unable to decode image source '{source.Identifier ?? index.ToString(CultureInfo.InvariantCulture)}'."));
            }

            var pageReference = new PageReference(index + 1, source.Dpi ?? _options.DefaultDpi);
            var metadata = BuildMetadata(index, source, bitmap, pageReference);
            return new PageImage(pageReference, bitmap, metadata);
        }
        catch (Exception ex)
        {
            LogDecodeFailure(ex, source);
            throw;
        }
        finally
        {
            await stream.DisposeAsync().ConfigureAwait(false);
        }
    }

    private PageImageMetadata BuildMetadata(int index, ImageSourceDescriptor source, SKBitmap bitmap, PageReference pageReference)
    {
        var properties = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["pageNumber"] = (index + 1).ToString(CultureInfo.InvariantCulture),
            ["widthPixels"] = bitmap.Width.ToString(CultureInfo.InvariantCulture),
            ["heightPixels"] = bitmap.Height.ToString(CultureInfo.InvariantCulture),
            ["dpi"] = pageReference.Dpi.ToString(CultureInfo.InvariantCulture),
        };

        if (_options.Metadata is not null)
        {
            foreach (var kvp in _options.Metadata)
            {
                properties[kvp.Key] = kvp.Value;
            }
        }

        if (source.Metadata is not null)
        {
            foreach (var kvp in source.Metadata)
            {
                properties[kvp.Key] = kvp.Value;
            }
        }

        properties["source"] = source.Identifier ?? _options.SourceName ?? "image";
        if (!string.IsNullOrWhiteSpace(source.MediaType))
        {
            properties["mediaType"] = source.MediaType!;
        }

        if (!string.IsNullOrEmpty(source.FileName))
        {
            properties["fileName"] = source.FileName!;
        }

        return new PageImageMetadata(_options.DocumentId, _options.SourceName ?? source.Identifier, source.MediaType, properties);
    }

    private void LogDecodeFailure(Exception exception, ImageSourceDescriptor source)
    {
        if (_logger is null)
        {
            return;
        }

        DecodeFailureMessage(_logger, source.Identifier ?? "<unknown>", exception);
    }
}
