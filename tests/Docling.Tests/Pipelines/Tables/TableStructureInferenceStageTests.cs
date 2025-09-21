using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Options;
using Docling.Pipelines.Tables;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using Xunit;

namespace Docling.Tests.Pipelines.Tables;

public sealed class TableStructureInferenceStageTests
{
    [Fact]
    public async Task ExecuteAsyncWhenTableStructureDisabledSetsEmptyResult()
    {
        using var store = new PageImageStore();
        var context = CreateContext(store);
        var options = new PdfPipelineOptions { DoTableStructure = false };
        var service = new FakeTableStructureService();
        var stage = new TableStructureInferenceStage(service, options, NullLogger<TableStructureInferenceStage>.Instance);

        await stage.ExecuteAsync(context, CancellationToken.None);

        service.InvocationCount.Should().Be(0);
        context.GetRequired<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures).Should().BeEmpty();
    }

    [Fact]
    public async Task ExecuteAsyncWhenNoTablesFillsEmptyCollection()
    {
        using var store = new PageImageStore();
        var context = CreateContext(store);
        context.Set(PipelineContextKeys.LayoutItems, new[]
        {
            new LayoutItem(new PageReference(0, 144), BoundingBox.FromSize(0, 0, 50, 50), LayoutItemKind.Text, Array.Empty<Polygon>()),
        });

        var options = new PdfPipelineOptions();
        var service = new FakeTableStructureService();
        var stage = new TableStructureInferenceStage(service, options, NullLogger<TableStructureInferenceStage>.Instance);

        await stage.ExecuteAsync(context, CancellationToken.None);

        service.InvocationCount.Should().Be(0);
        context.GetRequired<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures).Should().BeEmpty();
    }

    [Fact]
    public async Task ExecuteAsyncWithTablesInvokesServiceAndStoresResult()
    {
        using var store = new PageImageStore();
        var page = new PageReference(1, 200);
        using (var bitmap = new SKBitmap(80, 60))
        using (var pageImage = new PageImage(page, bitmap, new PageImageMetadata("doc", "source", "image/png", null)))
        {
            store.Add(pageImage);
        }

        var tableBounds = BoundingBox.FromSize(5, 10, 30, 20);
        var layoutItem = new LayoutItem(page, tableBounds, LayoutItemKind.Table, Array.Empty<Polygon>());
        var context = CreateContext(store);
        context.Set(PipelineContextKeys.LayoutItems, new[] { layoutItem });

        var options = new PdfPipelineOptions();
        var service = new FakeTableStructureService
        {
            Handler = request => new TableStructure(request.Page, new[]
            {
                new TableCell(request.BoundingBox, 1, 1, null),
            }, 1, 1),
        };
        var stage = new TableStructureInferenceStage(service, options, NullLogger<TableStructureInferenceStage>.Instance);

        await stage.ExecuteAsync(context, CancellationToken.None);

        service.InvocationCount.Should().Be(1);
        service.LastRequest.Should().NotBeNull();
        service.LastRequest!.BoundingBox.Should().Be(tableBounds);
        service.LastRequest.RasterizedImage.IsEmpty.Should().BeFalse();

        var structures = context.GetRequired<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures);
        structures.Should().HaveCount(1);
        structures[0].Cells.Should().HaveCount(1);
        structures[0].Cells[0].BoundingBox.Should().Be(tableBounds);
    }

    private static PipelineContext CreateContext(PageImageStore store)
    {
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.LayoutItems, Array.Empty<LayoutItem>());
        return context;
    }

    private sealed class FakeTableStructureService : ITableStructureService
    {
        public int InvocationCount { get; private set; }

        public TableStructureRequest? LastRequest { get; private set; }

        public Func<TableStructureRequest, TableStructure>? Handler { get; init; }

        public Task<TableStructure> InferStructureAsync(TableStructureRequest request, CancellationToken cancellationToken = default)
        {
            InvocationCount++;
            LastRequest = request;
            if (Handler is null)
            {
                throw new InvalidOperationException("Handler must be set before invoking the fake service.");
            }

            return Task.FromResult(Handler(request));
        }
    }
}
