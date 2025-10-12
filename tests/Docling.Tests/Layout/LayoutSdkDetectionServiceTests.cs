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
        using var runner = new FakeRunner(
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(0, 0, 100, 40, "text"),
                    new LayoutSdk.BoundingBox(10, 20, 50, 30, "table"),
                },
                null),
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(5, 5, 25, 25, "figure"),
                },
                null));
        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        var items = await service.DetectAsync(CreateRequest(2), CancellationToken.None);

        items.Should().HaveCount(3);
        items[0].Page.Should().Be(new PageReference(0, 200));
        items[0].Kind.Should().Be(LayoutItemKind.Text);
        items[1].Kind.Should().Be(LayoutItemKind.Table);
        items[2].Kind.Should().Be(LayoutItemKind.Figure);
    }

    [Fact]
    public async Task DetectAsyncReprojectsBoxesUsingNormalizationMetadata()
    {
        var metadata = new LayoutSdkNormalisationMetadata(1000, 800, 640, 512, 0.64, 0, 64);
        using var runner = new FakeRunner(
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(160, 200, 320, 128, "text"),
                },
                metadata));

        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        var items = await service.DetectAsync(CreateRequest(1), CancellationToken.None);

        items.Should().ContainSingle();
        var box = items[0].BoundingBox;
        box.Left.Should().BeApproximately(250, 1e-3);
        box.Top.Should().BeApproximately(212.5, 1e-3);
        box.Right.Should().BeApproximately(750, 1e-3);
        box.Bottom.Should().BeApproximately(412.5, 1e-3);
    }

    [Fact]
    public async Task DetectAsyncKeepsBoxesWhenMetadataIdentity()
    {
        var metadata = new LayoutSdkNormalisationMetadata(640, 640, 640, 640, 1.0, 0, 0);
        using var runner = new FakeRunner(
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(10, 20, 30, 40, "text"),
                },
                metadata));

        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        var items = await service.DetectAsync(CreateRequest(1), CancellationToken.None);

        items.Should().ContainSingle();
        var box = items[0].BoundingBox;
        box.Left.Should().Be(10);
        box.Top.Should().Be(20);
        box.Right.Should().Be(40);
        box.Bottom.Should().Be(60);
    }

    [Fact]
    public async Task DetectAsyncSkipsBoxesClippedAwayByReprojection()
    {
        var metadata = new LayoutSdkNormalisationMetadata(1000, 800, 640, 512, 0.64, 0, 64);
        using var runner = new FakeRunner(
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(0, 10, 40, 40, "text"),
                    new LayoutSdk.BoundingBox(160, 200, 320, 128, "text"),
                },
                metadata));

        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        var items = await service.DetectAsync(CreateRequest(1), CancellationToken.None);

        items.Should().ContainSingle();
        var box = items[0].BoundingBox;
        box.Left.Should().BeApproximately(250, 1e-3);
        box.Top.Should().BeApproximately(212.5, 1e-3);
        box.Right.Should().BeApproximately(750, 1e-3);
        box.Bottom.Should().BeApproximately(412.5, 1e-3);
    }

    [Fact]
    public async Task DetectAsyncFallsBackToTextOnUnknownLabel()
    {
        using var runner = new FakeRunner(
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(0, 0, 10, 10, "mystery"),
                },
                null));
        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        var result = await service.DetectAsync(CreateRequest(1), CancellationToken.None);

        result.Should().ContainSingle();
        result[0].Kind.Should().Be(LayoutItemKind.Text);
    }

    [Fact]
    public async Task DetectAsyncThrowsWhenBoundingBoxInvalid()
    {
        using var runner = new FakeRunner(
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(0, 0, 0, 10, "text"),
                },
                null));

        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        await Assert.ThrowsAsync<LayoutServiceException>(() => service.DetectAsync(CreateRequest(1), CancellationToken.None));
    }

    [Fact]
    public async Task DetectAsyncThrowsWhenDisposed()
    {
        using var runner = new FakeRunner();
        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);
        service.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(() => service.DetectAsync(CreateRequest(1), CancellationToken.None));
    }

    [Fact]
    public async Task DetectAsyncCapturesNormalizationMetadata()
    {
        var metadata = new LayoutSdkNormalisationMetadata(1000, 800, 640, 512, 0.64, 0, 64);
        using var runner = new FakeRunner(
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(0, 0, 100, 40, "text"),
                },
                metadata));
        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);

        _ = await service.DetectAsync(CreateRequest(1), CancellationToken.None);

        var telemetry = ((ILayoutNormalizationMetadataSource)service).ConsumeNormalizationMetadata();
        telemetry.Should().ContainSingle();
        telemetry[0].Page.Should().Be(new PageReference(0, 200));
        telemetry[0].Metadata.Should().Be(metadata);

        ((ILayoutNormalizationMetadataSource)service).ConsumeNormalizationMetadata().Should().BeEmpty();
    }

    [Fact]
    public async Task DetectAsyncCapturesProfilingTelemetry()
    {
        var snapshot = new LayoutSdkProfilingSnapshot(12.3, 320.4, 38.9, 371.6);
        using var runner = new FakeRunner(
            new[] { snapshot },
            new LayoutSdkInferenceResult(
                new List<LayoutSdk.BoundingBox>
                {
                    new LayoutSdk.BoundingBox(0, 0, 100, 40, "text"),
                },
                null));

        using var service = new LayoutSdkDetectionService(new LayoutSdkDetectionOptions(), NullLogger<LayoutSdkDetectionService>.Instance, runner);

        _ = await service.DetectAsync(CreateRequest(1), CancellationToken.None);

        var telemetry = ((ILayoutProfilingTelemetrySource)service).ConsumeProfilingTelemetry();
        telemetry.Should().ContainSingle();
        telemetry[0].Page.Should().Be(new PageReference(0, 200));
        telemetry[0].Snapshot.Should().Be(snapshot);

        ((ILayoutProfilingTelemetrySource)service).ConsumeProfilingTelemetry().Should().BeEmpty();
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

    private sealed class FakeRunner : ILayoutSdkRunner, ILayoutSdkProfilingSource
    {
        private readonly Queue<LayoutSdkInferenceResult> _results;
        private readonly Queue<LayoutSdkProfilingSnapshot> _profiling;
        private readonly bool _profilingEnabled;

        public FakeRunner(params LayoutSdkInferenceResult[] results)
            : this(results, Array.Empty<LayoutSdkProfilingSnapshot>(), profilingEnabled: false)
        {
        }

        public FakeRunner(IEnumerable<LayoutSdkProfilingSnapshot> profilingSnapshots, params LayoutSdkInferenceResult[] results)
            : this(results, profilingSnapshots, profilingEnabled: true)
        {
        }

        private FakeRunner(
            IEnumerable<LayoutSdkInferenceResult> results,
            IEnumerable<LayoutSdkProfilingSnapshot> profilingSnapshots,
            bool profilingEnabled)
        {
            _results = new Queue<LayoutSdkInferenceResult>(results);
            _profiling = new Queue<LayoutSdkProfilingSnapshot>(profilingSnapshots);
            _profilingEnabled = profilingEnabled;
        }

        public void Dispose()
        {
        }

        public Task<LayoutSdkInferenceResult> InferAsync(LayoutPagePayload page, CancellationToken cancellationToken)
        {
            var result = _results.Count > 0
                ? _results.Dequeue()
                : new LayoutSdkInferenceResult(Array.Empty<LayoutSdk.BoundingBox>(), null);
            var projected = LayoutSdkRunner.ReprojectBoundingBoxes(result.Boxes, result.Normalisation);
            return Task.FromResult(new LayoutSdkInferenceResult(projected, result.Normalisation));
        }

        public bool IsProfilingEnabled => _profilingEnabled;

        public bool TryGetProfilingSnapshot(out LayoutSdkProfilingSnapshot snapshot)
        {
            if (!_profilingEnabled || _profiling.Count == 0)
            {
                snapshot = default;
                return false;
            }

            snapshot = _profiling.Dequeue();
            return true;
        }
    }
}
