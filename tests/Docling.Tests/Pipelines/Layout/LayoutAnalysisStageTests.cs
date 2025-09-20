using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Layout;
using Docling.Pipelines.Options;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;

namespace Docling.Tests.Pipelines.Layout;

public sealed class LayoutAnalysisStageTests
{
    [Fact]
    public async Task ExecuteAsyncInvokesServiceAndStoresResults()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(0, 200);
        using var pageImage = CreatePageImage(pageReference, sourceId: "page-0");
        store.Add(pageImage);

        var recorded = new RecordingLayoutDetectionService();
        var expectedItem = new LayoutItem(
            pageReference,
            BoundingBox.FromSize(10, 10, 40, 20),
            LayoutItemKind.Text,
            new[] { Polygon.FromPoints(new[] { new Point2D(10, 10), new Point2D(50, 10), new Point2D(50, 30) }) });
        recorded.Result = new List<LayoutItem> { expectedItem };

        var options = new LayoutOptions
        {
            CreateOrphanClusters = false,
            KeepEmptyClusters = true,
            SkipCellAssignment = true,
            Model = LayoutModelConfiguration.DoclingLayoutEgretMedium,
        };

        var stage = new LayoutAnalysisStage(recorded, options, NullLogger<LayoutAnalysisStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });
        context.Set(PipelineContextKeys.DocumentId, "doc-123");

        await stage.ExecuteAsync(context, CancellationToken.None);

        recorded.Requests.Should().HaveCount(1);
        var request = recorded.Requests.Single();
        request.DocumentId.Should().Be("doc-123");
        request.ModelIdentifier.Should().Be(options.Model.Identifier);
        request.Options.CreateOrphanClusters.Should().BeFalse();
        request.Options.KeepEmptyClusters.Should().BeTrue();
        request.Options.SkipCellAssignment.Should().BeTrue();
        request.Pages.Should().HaveCount(1);
        var payload = request.Pages[0];
        payload.ArtifactId.Should().Be("page-0");
        payload.MediaType.Should().Be("image/png");
        payload.ImageContent.Length.Should().BeGreaterThan(0);
        payload.Metadata.Should().ContainKey(PageImageMetadataKeys.NormalizedDpi);

        context.GetRequired<IReadOnlyList<LayoutItem>>(PipelineContextKeys.LayoutItems)
            .Should().BeSameAs(recorded.Result);
        context.GetRequired<bool>(PipelineContextKeys.LayoutAnalysisCompleted).Should().BeTrue();
    }

    [Fact]
    public async Task ExecuteAsyncSkipsWhenNoPages()
    {
        using var store = new PageImageStore();
        var recorded = new RecordingLayoutDetectionService();
        var stage = new LayoutAnalysisStage(recorded, new LayoutOptions(), NullLogger<LayoutAnalysisStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference>());

        await stage.ExecuteAsync(context, CancellationToken.None);

        recorded.Requests.Should().BeEmpty();
        context.GetRequired<IReadOnlyList<LayoutItem>>(PipelineContextKeys.LayoutItems).Should().BeEmpty();
        context.GetRequired<bool>(PipelineContextKeys.LayoutAnalysisCompleted).Should().BeTrue();
    }

    [Fact]
    public async Task ExecuteAsyncThrowsWhenStoreMissing()
    {
        var stage = new LayoutAnalysisStage(new RecordingLayoutDetectionService(), new LayoutOptions(), NullLogger<LayoutAnalysisStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { new PageReference(0, 150) });

        await Assert.ThrowsAsync<InvalidOperationException>(() => stage.ExecuteAsync(context, CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteAsyncThrowsWhenDebugEnabledWithoutRenderer()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(0, 150);
        using var pageImage = CreatePageImage(pageReference, sourceId: "src");
        store.Add(pageImage);

        var options = new LayoutOptions { GenerateDebugArtifacts = true };
        var stage = new LayoutAnalysisStage(new RecordingLayoutDetectionService(), options, NullLogger<LayoutAnalysisStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });

        await Assert.ThrowsAsync<InvalidOperationException>(() => stage.ExecuteAsync(context, CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteAsyncGeneratesDebugArtifactsWhenEnabled()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(0, 180);
        using var pageImage = CreatePageImage(pageReference, sourceId: "artifact-0");
        store.Add(pageImage);

        var recorded = new RecordingLayoutDetectionService();
        var layoutItem = new LayoutItem(
            pageReference,
            BoundingBox.FromSize(5, 5, 20, 20),
            LayoutItemKind.Figure,
            Array.Empty<Polygon>());
        recorded.Result = new List<LayoutItem> { layoutItem };

        var overlayRenderer = new RecordingOverlayRenderer();
        var options = new LayoutOptions { GenerateDebugArtifacts = true };
        var stage = new LayoutAnalysisStage(recorded, options, NullLogger<LayoutAnalysisStage>.Instance, overlayRenderer);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });

        await stage.ExecuteAsync(context, CancellationToken.None);

        overlayRenderer.Invocations.Should().Be(1);
        var overlays = context.GetRequired<IReadOnlyList<LayoutDebugOverlay>>(PipelineContextKeys.LayoutDebugArtifacts);
        overlays.Should().HaveCount(1);
        overlays[0].Page.Should().Be(pageReference);
        overlays[0].ImageContent.Length.Should().BeGreaterThan(0);
    }

    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "Ownership transferred to PageImage")]
    private static PageImage CreatePageImage(PageReference reference, string? sourceId)
    {
        var bitmap = new SKBitmap(new SKImageInfo(40, 40, SKColorType.Rgba8888, SKAlphaType.Premul));
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
            using var paint = new SKPaint { Color = SKColors.Black, Style = SKPaintStyle.Stroke, StrokeWidth = 2 };
            canvas.DrawRect(new SKRect(5, 5, 35, 35), paint);
        }

        var metadata = new PageImageMetadata(sourceId, "sample.pdf", "image/png", new Dictionary<string, string>
        {
            [PageImageMetadataKeys.SourceHorizontalDpi] = reference.Dpi.ToString(System.Globalization.CultureInfo.InvariantCulture),
        });

        return new PageImage(reference, bitmap, metadata);
    }

    private sealed class RecordingLayoutDetectionService : ILayoutDetectionService
    {
        public List<LayoutRequest> Requests { get; } = new();

        public IReadOnlyList<LayoutItem> Result { get; set; } = new List<LayoutItem>();

        public Task<IReadOnlyList<LayoutItem>> DetectAsync(LayoutRequest request, CancellationToken cancellationToken = default)
        {
            Requests.Add(request);
            return Task.FromResult(Result);
        }
    }

    private sealed class RecordingOverlayRenderer : ILayoutDebugOverlayRenderer
    {
        public int Invocations { get; private set; }

        public LayoutDebugOverlay CreateOverlay(PageImage page, IReadOnlyList<LayoutItem> layoutItems)
        {
            Invocations++;
            using var surface = SKSurface.Create(new SKImageInfo(page.Width, page.Height));
            surface.Canvas.Clear(SKColors.Transparent);
            using var encoded = surface.Snapshot().Encode(SKEncodedImageFormat.Png, 80);
            return new LayoutDebugOverlay(page.Page, encoded.ToArray());
        }
    }
}
