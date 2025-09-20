using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Microsoft.Extensions.Logging;

namespace Docling.Models.Layout;

/// <summary>
/// HTTP-based implementation of <see cref="ILayoutDetectionService"/> that calls the Python Docling layout endpoint.
/// </summary>
public sealed partial class HttpLayoutDetectionService : ILayoutDetectionService
{
    private readonly HttpClient _httpClient;
    private readonly LayoutServiceOptions _options;
    private readonly JsonSerializerOptions _serializerOptions;
    private readonly ILogger<HttpLayoutDetectionService> _logger;

    public HttpLayoutDetectionService(
        HttpClient httpClient,
        LayoutServiceOptions options,
        ILogger<HttpLayoutDetectionService> logger)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        _options = (options ?? throw new ArgumentNullException(nameof(options))).Clone();
        _options.EnsureValid();
        _serializerOptions = _options.SerializerOptions;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<IReadOnlyList<LayoutItem>> DetectAsync(LayoutRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        var dto = LayoutRequestDto.From(request);
        var payload = JsonSerializer.SerializeToUtf8Bytes(dto, _serializerOptions);
        using var message = new HttpRequestMessage(HttpMethod.Post, _options.Endpoint)
        {
            Content = new ByteArrayContent(payload)
        };
        message.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json")
        {
            CharSet = Encoding.UTF8.WebName,
        };

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(_options.RequestTimeout);

        HttpResponseMessage response;
        try
        {
            ServiceLogger.Requesting(_logger, request.DocumentId, request.Pages.Count);
            response = await _httpClient.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, timeoutCts.Token)
                .ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new LayoutServiceException("The layout detection request timed out.");
        }
        catch (Exception ex) when (ex is HttpRequestException or InvalidOperationException)
        {
            throw new LayoutServiceException("Failed to send layout detection request.", ex);
        }

        var contentStream = await response.Content.ReadAsStreamAsync(timeoutCts.Token).ConfigureAwait(false);
        using var responseStream = contentStream;
        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(timeoutCts.Token).ConfigureAwait(false);
            throw new LayoutServiceException($"Layout detection failed with status {(int)response.StatusCode}: {response.ReasonPhrase}. Body: {body}");
        }

        LayoutResponseDto? resultDto;
        try
        {
            resultDto = await JsonSerializer.DeserializeAsync<LayoutResponseDto>(responseStream, _serializerOptions, timeoutCts.Token)
                .ConfigureAwait(false);
        }
        catch (JsonException ex)
        {
            throw new LayoutServiceException("Failed to deserialize layout detection response.", ex);
        }

        if (resultDto?.Items is null || resultDto.Items.Count == 0)
        {
            return Array.Empty<LayoutItem>();
        }

        var pageLookup = request.Pages.ToDictionary(p => p.Page.PageNumber, p => p.Page);
        var layoutItems = new List<LayoutItem>(resultDto.Items.Count);
        foreach (var item in resultDto.Items)
        {
            if (!pageLookup.TryGetValue(item.PageIndex, out var pageReference))
            {
                throw new LayoutServiceException($"The layout service returned an unknown page index {item.PageIndex}.");
            }

            var kind = ParseKind(item.Kind);
            var boundingBox = ParseBoundingBox(item.BoundingBox);
            var polygons = ParsePolygons(item.Polygons);
            layoutItems.Add(new LayoutItem(pageReference, boundingBox, kind, polygons));
        }

        ServiceLogger.ResponseParsed(_logger, layoutItems.Count);
        return layoutItems;
    }

    private static LayoutItemKind ParseKind(string value)
    {
        return value.ToUpperInvariant() switch
        {
            "TEXT" => LayoutItemKind.Text,
            "TABLE" => LayoutItemKind.Table,
            "FIGURE" or "IMAGE" => LayoutItemKind.Figure,
            _ => throw new LayoutServiceException($"Unsupported layout item kind '{value}'."),
        };
    }

    private static BoundingBox ParseBoundingBox(IReadOnlyList<double> coordinates)
    {
        if (coordinates.Count != 4)
        {
            throw new LayoutServiceException("Bounding box coordinates must contain four values.");
        }

        if (!BoundingBox.TryCreate(coordinates[0], coordinates[1], coordinates[2], coordinates[3], out var box))
        {
            throw new LayoutServiceException("The layout service returned invalid bounding box coordinates.");
        }

        return box;
    }

    private static IReadOnlyList<Polygon> ParsePolygons(IReadOnlyList<IReadOnlyList<IReadOnlyList<double>>> polygons)
    {
        if (polygons.Count == 0)
        {
            return Array.Empty<Polygon>();
        }

        var result = new List<Polygon>(polygons.Count);
        foreach (var polygon in polygons)
        {
            if (polygon.Count < 3)
            {
                throw new LayoutServiceException("A polygon must contain at least three points.");
            }

            var points = new Point2D[polygon.Count];
            for (var i = 0; i < polygon.Count; i++)
            {
                var point = polygon[i];
                if (point.Count != 2)
                {
                    throw new LayoutServiceException("Polygon points must contain exactly two coordinates.");
                }

                points[i] = new Point2D(point[0], point[1]);
            }

            result.Add(Polygon.FromPoints(points));
        }

        return result;
    }

    private sealed record LayoutRequestDto(
        string DocumentId,
        string ModelIdentifier,
        LayoutRequestOptionsDto Options,
        IReadOnlyList<LayoutPageDto> Pages)
    {
        public static LayoutRequestDto From(LayoutRequest request)
        {
            var pages = new LayoutPageDto[request.Pages.Count];
            for (var i = 0; i < request.Pages.Count; i++)
            {
                pages[i] = LayoutPageDto.From(request.Pages[i]);
            }

            return new LayoutRequestDto(
                request.DocumentId,
                request.ModelIdentifier,
                new LayoutRequestOptionsDto(request.Options.CreateOrphanClusters, request.Options.KeepEmptyClusters, request.Options.SkipCellAssignment),
                pages);
        }
    }

    private sealed record LayoutRequestOptionsDto(bool CreateOrphanClusters, bool KeepEmptyClusters, bool SkipCellAssignment);

    private sealed record LayoutPageDto(
        int PageIndex,
        double Dpi,
        string ArtifactId,
        string MediaType,
        int Width,
        int Height,
        IReadOnlyDictionary<string, string> Metadata,
        string ImageContent)
    {
        public static LayoutPageDto From(LayoutPagePayload payload)
        {
            return new LayoutPageDto(
                payload.Page.PageNumber,
                payload.Dpi,
                payload.ArtifactId,
                payload.MediaType,
                payload.Width,
                payload.Height,
                payload.Metadata,
                Convert.ToBase64String(payload.ImageContent.Span));
        }
    }

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Performance", "CA1812", Justification = "Instantiated by System.Text.Json deserialization.")]
    private sealed record class LayoutResponseDto(IReadOnlyList<LayoutResponseItemDto> Items);

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Performance", "CA1812", Justification = "Instantiated by System.Text.Json deserialization.")]
    private sealed record class LayoutResponseItemDto(int PageIndex, string Kind, IReadOnlyList<double> BoundingBox, IReadOnlyList<IReadOnlyList<IReadOnlyList<double>>> Polygons);

    private static partial class ServiceLogger
    {
        [LoggerMessage(EventId = 4000, Level = LogLevel.Information, Message = "Submitting layout detection request for {Pages} pages (document {DocumentId}).")]
        public static partial void Requesting(ILogger logger, string documentId, int pages);

        [LoggerMessage(EventId = 4001, Level = LogLevel.Debug, Message = "Parsed {Items} layout items from response.")]
        public static partial void ResponseParsed(ILogger logger, int items);
    }
}
