using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using FluentAssertions;
using LayoutSdk;
using Microsoft.Extensions.Logging.Abstractions;

namespace Docling.Tests.Layout;

public sealed class LayoutSdkDetectionServiceTests
{
    [Fact]
    public async Task DetectAsyncProjectsBoundingBoxes()
    {
        using var runner = new FakeRunner(new List<LayoutSdk.BoundingBox>[]
        {
            new()
            {
                new LayoutSdk.BoundingBox(0, 0, 100, 40, "text"),
                new LayoutSdk.BoundingBox(10, 20, 50, 30, "table"),
            },
            new()
            {
                new LayoutSdk.BoundingBox(5, 5, 25, 25, "figure"),
            },
        });
        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        var items = await service.DetectAsync(CreateRequest(2), CancellationToken.None);

        items.Should().HaveCount(3);
        items[0].Page.Should().Be(new PageReference(0, 200));
        items[0].Kind.Should().Be(LayoutItemKind.Text);
        items[1].Kind.Should().Be(LayoutItemKind.Table);
        items[2].Kind.Should().Be(LayoutItemKind.Figure);
    }

    [Fact]
    public async Task DetectAsyncFallsBackToTextOnUnknownLabel()
    {
        using var runner = new FakeRunner(new List<LayoutSdk.BoundingBox>[]
        {
            new()
            {
                new LayoutSdk.BoundingBox(0, 0, 10, 10, "mystery"),
            },
        });
        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        var result = await service.DetectAsync(CreateRequest(1), CancellationToken.None);

        result.Should().ContainSingle();
        result[0].Kind.Should().Be(LayoutItemKind.Text);
    }

    [Fact]
    public async Task DetectAsyncThrowsWhenBoundingBoxInvalid()
    {
        using var runner = new FakeRunner(new List<LayoutSdk.BoundingBox>[]
        {
            new()
            {
                new LayoutSdk.BoundingBox(0, 0, 0, 10, "text"),
            },
        });

        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        await Assert.ThrowsAsync<LayoutServiceException>(() => service.DetectAsync(CreateRequest(1), CancellationToken.None));
    }

    [Fact]
    public async Task DetectAsyncThrowsWhenDisposed()
    {
        using var runner = new FakeRunner(Array.Empty<IReadOnlyList<LayoutSdk.BoundingBox>>());
        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        service.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(() => service.DetectAsync(CreateRequest(1), CancellationToken.None));
    }

    private static LayoutRequest CreateRequest(int pageCount)
    {
        var pages = new List<LayoutPagePayload>(pageCount);
        for (var i = 0; i < pageCount; i++)
        {
            pages.Add(new LayoutPagePayload(
                new PageReference(i, 200),
                $"artifact-{i}",
                "image/png",
                200,
                100,
                200,
                new Dictionary<string, string>(),
                new byte[] { 1, 2, 3 }));
        }

        return new LayoutRequest("doc-1", "docling-layout-heron", new LayoutRequestOptions(true, false, false), pages);
    }

    private sealed class FakeRunner : ILayoutSdkRunner
    {
        private readonly Queue<IReadOnlyList<LayoutSdk.BoundingBox>> _results;

        public FakeRunner(IReadOnlyList<LayoutSdk.BoundingBox>[] results)
        {
            _results = new Queue<IReadOnlyList<LayoutSdk.BoundingBox>>(results);
        }

        public void Dispose()
        {
        }

        public Task<IReadOnlyList<LayoutSdk.BoundingBox>> InferAsync(ReadOnlyMemory<byte> imageContent, CancellationToken cancellationToken)
        {
            return Task.FromResult(_results.Count > 0 ? _results.Dequeue() : Array.Empty<LayoutSdk.BoundingBox>());
        }
    }
}
