using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;

namespace Docling.Tests.Layout;

public sealed class HttpLayoutDetectionServiceTests
{
    [Fact]
    public async Task DetectAsyncSerializesRequestAndParsesResponse()
    {
        using var handler = new RecordingHandler();
        handler.Response = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent("""
            {"items": [{"pageIndex":0,"kind":"text","boundingBox":[0,0,50,20],"polygons":[[[0,0],[50,0],[50,20]]]}]}
            """),
        };

        using var client = new HttpClient(handler);
        var service = new HttpLayoutDetectionService(client, new LayoutServiceOptions
        {
            Endpoint = new Uri("https://layout.test/analyze"),
            RequestTimeout = TimeSpan.FromSeconds(5),
        }, NullLogger<HttpLayoutDetectionService>.Instance);

        var request = CreateRequest();
        var items = await service.DetectAsync(request, CancellationToken.None);

        handler.Request.Should().NotBeNull();
        handler.Request!.Method.Should().Be(HttpMethod.Post);
        handler.Request.RequestUri.Should().Be(new Uri("https://layout.test/analyze"));
        var payload = JsonDocument.Parse(handler.Body ?? throw new Xunit.Sdk.XunitException("Request body missing."));
        payload.RootElement.GetProperty("documentId").GetString().Should().Be("doc-1");
        payload.RootElement.GetProperty("pages")[0].GetProperty("imageContent").GetString().Should().NotBeNull();

        items.Should().HaveCount(1);
        var item = items[0];
        item.Page.Should().Be(new PageReference(0, 200));
        item.Kind.Should().Be(LayoutItemKind.Text);
        item.BoundingBox.Should().Be(new BoundingBox(0, 0, 50, 20));
        item.Polygons.Should().HaveCount(1);
        item.Polygons[0].Count.Should().Be(3);
    }

    [Fact]
    public async Task DetectAsyncThrowsWhenServiceReturnsError()
    {
        using var handler = new RecordingHandler
        {
            Response = new HttpResponseMessage(HttpStatusCode.BadRequest)
            {
                Content = new StringContent("invalid"),
            },
        };

        using var client = new HttpClient(handler);
        var service = new HttpLayoutDetectionService(client, new LayoutServiceOptions
        {
            Endpoint = new Uri("https://layout.test/analyze"),
            RequestTimeout = TimeSpan.FromSeconds(2),
        }, NullLogger<HttpLayoutDetectionService>.Instance);

        var request = CreateRequest();
        await Assert.ThrowsAsync<LayoutServiceException>(() => service.DetectAsync(request, CancellationToken.None));
    }

    [Fact]
    public async Task DetectAsyncThrowsWhenResponseReferencesUnknownPage()
    {
        using var handler = new RecordingHandler();
        handler.Response = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent("""
            {"items":[{"pageIndex":5,"kind":"text","boundingBox":[0,0,10,10],"polygons":[[[0,0],[1,0],[1,1]]]}]}
            """),
        };

        using var client = new HttpClient(handler);
        var service = new HttpLayoutDetectionService(client, new LayoutServiceOptions
        {
            Endpoint = new Uri("https://layout.test/analyze"),
        }, NullLogger<HttpLayoutDetectionService>.Instance);

        await Assert.ThrowsAsync<LayoutServiceException>(() => service.DetectAsync(CreateRequest(), CancellationToken.None));
    }

    [Fact]
    public async Task DetectAsyncThrowsWhenBoundingBoxInvalid()
    {
        using var handler = new RecordingHandler();
        handler.Response = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent("""
            {"items":[{"pageIndex":0,"kind":"table","boundingBox":[0,0,10],"polygons":[[[0,0],[1,0],[1,1]]]}]}
            """),
        };

        using var client = new HttpClient(handler);
        var service = new HttpLayoutDetectionService(client, new LayoutServiceOptions
        {
            Endpoint = new Uri("https://layout.test/analyze"),
        }, NullLogger<HttpLayoutDetectionService>.Instance);

        await Assert.ThrowsAsync<LayoutServiceException>(() => service.DetectAsync(CreateRequest(), CancellationToken.None));
    }

    [Fact]
    public async Task DetectAsyncRaisesTimeoutWhenRequestTakesTooLong()
    {
        using var handler = new DelayHandler();
        using var client = new HttpClient(handler);
        var service = new HttpLayoutDetectionService(client, new LayoutServiceOptions
        {
            Endpoint = new Uri("https://layout.test/analyze"),
            RequestTimeout = TimeSpan.FromMilliseconds(10),
        }, NullLogger<HttpLayoutDetectionService>.Instance);

        await Assert.ThrowsAsync<LayoutServiceException>(() => service.DetectAsync(CreateRequest(), CancellationToken.None));
    }

    private static LayoutRequest CreateRequest()
    {
        var pageReference = new PageReference(0, 200);
        var payload = new LayoutPagePayload(
            pageReference,
            "artifact-0",
            "image/png",
            200,
            100,
            100,
            new Dictionary<string, string> { ["dpi"] = "200" },
            new byte[] { 1, 2, 3 });

        return new LayoutRequest(
            "doc-1",
            "model-a",
            new LayoutRequestOptions(true, false, false),
            new List<LayoutPagePayload> { payload });
    }

    private sealed class RecordingHandler : HttpMessageHandler
    {
        public HttpRequestMessage? Request { get; private set; }

        public HttpResponseMessage Response { get; set; } = new(HttpStatusCode.OK)
        {
            Content = new StringContent("{}"),
        };

        public string? Body { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Request = request;
            if (request.Content is not null)
            {
                Body = await request.Content.ReadAsStringAsync(cancellationToken);
            }

            return Response;
        }
    }

    private sealed class DelayHandler : HttpMessageHandler
    {
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            await Task.Delay(TimeSpan.FromMilliseconds(200), cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"items":[]}"""),
            };
        }
    }
}
