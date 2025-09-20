using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Models.Ocr;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Ocr;
using Docling.Pipelines.Options;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;

namespace Docling.Tests.Pipelines.Ocr;

public sealed class OcrStageTests
{
    [Fact]
    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "PageImageStore clones and disposes supplied images.")]
    public async Task ExecuteAsyncRecognizesLayoutBlocks()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(1, 200);
        store.Add(CreatePageImage(pageReference, width: 100, height: 100));

        var layoutItem = new LayoutItem(
            pageReference,
            BoundingBox.FromSize(10, 20, 40, 30),
            LayoutItemKind.Text,
            new[]
            {
                Polygon.FromPoints(new[]
                {
                    new Point2D(10, 20),
                    new Point2D(50, 20),
                    new Point2D(50, 50),
                    new Point2D(10, 50),
                }),
            });

        var options = new PdfPipelineOptions
        {
            Ocr = new EasyOcrOptions
            {
                Languages = new[] { "en" },
                BitmapAreaThreshold = 0.01,
                ForceFullPageOcr = false,
            },
        };

        var factory = new RecordingOcrServiceFactory(request =>
        {
            var line = new OcrLine("hello", request.Region, 0.9);
            return new List<OcrLine> { line };
        });

        var stage = new OcrStage(factory, options, NullLogger<OcrStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });
        context.Set(PipelineContextKeys.LayoutItems, new List<LayoutItem> { layoutItem });

        await stage.ExecuteAsync(context, CancellationToken.None);

        factory.Service.Requests.Should().HaveCount(1);
        factory.Service.Requests[0].Region.Should().Be(layoutItem.BoundingBox);

        var result = context.GetRequired<OcrDocumentResult>(PipelineContextKeys.OcrResults);
        result.Blocks.Should().HaveCount(1);
        var block = result.Blocks[0];
        block.Kind.Should().Be(OcrRegionKind.LayoutBlock);
        block.Lines.Should().ContainSingle(line => line.Text == "hello");
        context.GetRequired<bool>(PipelineContextKeys.OcrCompleted).Should().BeTrue();
        factory.Service.Disposed.Should().BeTrue();
    }

    [Fact]
    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "PageImageStore clones and disposes supplied images.")]
    public async Task ExecuteAsyncRecognizesTableCells()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(5, 200);
        store.Add(CreatePageImage(pageReference, width: 120, height: 120));

        var tableStructure = new TableStructure(
            pageReference,
            new List<TableCell>
            {
                new(BoundingBox.FromSize(10, 10, 40, 40), RowSpan: 1, ColumnSpan: 1, Text: null),
                new(BoundingBox.FromSize(60, 10, 40, 40), RowSpan: 1, ColumnSpan: 1, Text: null),
            },
            RowCount: 1,
            ColumnCount: 2);

        var options = new PdfPipelineOptions
        {
            Ocr = new EasyOcrOptions
            {
                Languages = new[] { "en" },
                BitmapAreaThreshold = 0.01,
                ForceFullPageOcr = false,
            },
        };

        var factory = new RecordingOcrServiceFactory(request =>
        {
            request.Metadata.Should().ContainKey("docling:source").WhoseValue.Should().Be("table_cell");
            return new List<OcrLine> { new("cell", request.Region, 0.8) };
        });

        var stage = new OcrStage(factory, options, NullLogger<OcrStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });
        context.Set(PipelineContextKeys.TableStructures, new List<TableStructure> { tableStructure });

        await stage.ExecuteAsync(context, CancellationToken.None);

        factory.Service.Requests.Should().HaveCount(2);
        factory.Service.Requests.Should().OnlyContain(request => request.Metadata.ContainsKey("docling:table_column_index"));

        var result = context.GetRequired<OcrDocumentResult>(PipelineContextKeys.OcrResults);
        result.Blocks.Should().HaveCount(2);
        result.Blocks.Should().OnlyContain(block => block.Kind == OcrRegionKind.TableCell);
        context.GetRequired<bool>(PipelineContextKeys.OcrCompleted).Should().BeTrue();
    }

    [Fact]
    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "PageImageStore clones and disposes supplied images.")]
    public async Task ExecuteAsyncSkipsSmallTableCells()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(8, 200);
        store.Add(CreatePageImage(pageReference, width: 200, height: 200));

        var tinyCell = new TableCell(BoundingBox.FromSize(0, 0, 10, 10), RowSpan: 1, ColumnSpan: 1, Text: null);
        var tableStructure = new TableStructure(pageReference, new List<TableCell> { tinyCell }, RowCount: 1, ColumnCount: 1);

        var options = new PdfPipelineOptions
        {
            Ocr = new EasyOcrOptions
            {
                Languages = new[] { "en" },
                BitmapAreaThreshold = 0.25,
                ForceFullPageOcr = false,
            },
        };

        var factory = new RecordingOcrServiceFactory(_ => new List<OcrLine>());
        var stage = new OcrStage(factory, options, NullLogger<OcrStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });
        context.Set(PipelineContextKeys.TableStructures, new List<TableStructure> { tableStructure });

        await stage.ExecuteAsync(context, CancellationToken.None);

        factory.Service.Requests.Should().BeEmpty();
        var result = context.GetRequired<OcrDocumentResult>(PipelineContextKeys.OcrResults);
        result.Blocks.Should().BeEmpty();
        context.GetRequired<bool>(PipelineContextKeys.OcrCompleted).Should().BeTrue();
    }

    [Fact]
    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "PageImageStore clones and disposes supplied images.")]
    public async Task ExecuteAsyncSkipsSmallRegions()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(2, 200);
        store.Add(CreatePageImage(pageReference, width: 400, height: 400));

        var tinyItem = new LayoutItem(
            pageReference,
            BoundingBox.FromSize(5, 5, 10, 10),
            LayoutItemKind.Text,
            new[]
            {
                Polygon.FromPoints(new[]
                {
                    new Point2D(5, 5),
                    new Point2D(15, 5),
                    new Point2D(15, 15),
                    new Point2D(5, 15),
                }),
            });

        var options = new PdfPipelineOptions
        {
            Ocr = new EasyOcrOptions
            {
                Languages = new[] { "en" },
                BitmapAreaThreshold = 0.5,
                ForceFullPageOcr = false,
            },
        };

        var factory = new RecordingOcrServiceFactory(_ => new List<OcrLine>());
        var stage = new OcrStage(factory, options, NullLogger<OcrStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });
        context.Set(PipelineContextKeys.LayoutItems, new List<LayoutItem> { tinyItem });

        await stage.ExecuteAsync(context, CancellationToken.None);

        factory.Service.Requests.Should().BeEmpty();
        var result = context.GetRequired<OcrDocumentResult>(PipelineContextKeys.OcrResults);
        result.Blocks.Should().BeEmpty();
    }

    [Fact]
    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "PageImageStore clones and disposes supplied images.")]
    public async Task ExecuteAsyncFallsBackToFullPageWhenForced()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(3, 200);
        store.Add(CreatePageImage(pageReference, width: 120, height: 160));

        var options = new PdfPipelineOptions
        {
            Ocr = new EasyOcrOptions
            {
                Languages = new[] { "en" },
                ForceFullPageOcr = true,
                BitmapAreaThreshold = 0.25,
            },
        };

        var factory = new RecordingOcrServiceFactory(request =>
        {
            request.Region.Should().Be(BoundingBox.FromSize(0, 0, 120, 160));
            return new List<OcrLine> { new("page", request.Region, 1.0) };
        });

        var stage = new OcrStage(factory, options, NullLogger<OcrStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });

        await stage.ExecuteAsync(context, CancellationToken.None);

        factory.Service.Requests.Should().HaveCount(1);
        var result = context.GetRequired<OcrDocumentResult>(PipelineContextKeys.OcrResults);
        result.Blocks.Should().ContainSingle(block => block.Kind == OcrRegionKind.FullPage);
    }

    [Fact]
    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "PageImageStore clones and disposes supplied images.")]
    public async Task ExecuteAsyncHonoursDisableSwitch()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(4, 180);
        store.Add(CreatePageImage(pageReference, width: 60, height: 60));

        var options = new PdfPipelineOptions
        {
            DoOcr = false,
        };

        var factory = new RecordingOcrServiceFactory(_ => new List<OcrLine>());
        var stage = new OcrStage(factory, options, NullLogger<OcrStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });

        await stage.ExecuteAsync(context, CancellationToken.None);

        factory.Service.Requests.Should().BeEmpty();
        context.GetRequired<OcrDocumentResult>(PipelineContextKeys.OcrResults).Blocks.Should().BeEmpty();
        context.GetRequired<bool>(PipelineContextKeys.OcrCompleted).Should().BeTrue();
    }

    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "PageImageStore takes ownership of the page image clone.")]
    private static PageImage CreatePageImage(PageReference page, int width, int height)
    {
        var bitmap = new SKBitmap(new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Premul));
        return new PageImage(page, bitmap);
    }

    private sealed class RecordingOcrServiceFactory : IOcrServiceFactory
    {
        private readonly Func<OcrRequest, IReadOnlyList<OcrLine>> _responseFactory;

        public RecordingOcrServiceFactory(Func<OcrRequest, IReadOnlyList<OcrLine>> responseFactory)
        {
            _responseFactory = responseFactory;
            Service = new RecordingOcrService(responseFactory);
        }

        public RecordingOcrService Service { get; }

        public IOcrService Create(OcrOptions options) => Service;

        internal sealed class RecordingOcrService : IOcrService
        {
            private readonly Func<OcrRequest, IReadOnlyList<OcrLine>> _responseFactory;

            public RecordingOcrService(Func<OcrRequest, IReadOnlyList<OcrLine>> responseFactory)
            {
                _responseFactory = responseFactory;
            }

            public List<OcrRequest> Requests { get; } = new();

            public bool Disposed { get; private set; }

            public async IAsyncEnumerable<OcrLine> RecognizeAsync(OcrRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
            {
                Requests.Add(request);
                foreach (var line in _responseFactory(request))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    yield return line;
                    await Task.Yield();
                }
            }

            public void Dispose()
            {
                Disposed = true;
            }
        }

    }
}
