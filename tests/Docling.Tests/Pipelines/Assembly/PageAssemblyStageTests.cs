using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Models.Ocr;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Assembly;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace Docling.Tests.Pipelines.Assembly;

public sealed class PageAssemblyStageTests
{
    [Fact]
    public async Task AssembleAsyncBuildsParagraphsFromLayoutOcr()
    {
        var page = new PageReference(1, 144);
        var layoutBox = BoundingBox.FromSize(10, 20, 200, 80);
        var layoutItem = new LayoutItem(page, layoutBox, LayoutItemKind.Text, Array.Empty<Polygon>());
        var lines = new List<OcrLine>
        {
            new("Hello", BoundingBox.FromSize(10, 20, 190, 20), 0.95),
            new("world", BoundingBox.FromSize(10, 60, 190, 20), 0.94),
        };
        var ocrBlock = new OcrBlockResult(page, layoutBox, OcrRegionKind.LayoutBlock, new Dictionary<string, string>(), lines);
        var context = CreateContext();
        context.Set(PipelineContextKeys.PageSequence, new[] { page });
        context.Set(PipelineContextKeys.LayoutItems, new[] { layoutItem });
        context.Set(PipelineContextKeys.OcrResults, new OcrDocumentResult(new[] { ocrBlock }));
        context.Set(PipelineContextKeys.DocumentId, "doc-source");

        var stage = new PageAssemblyStage(NullLogger<PageAssemblyStage>.Instance);
        await stage.ExecuteAsync(context, CancellationToken.None);

        context.GetRequired<bool>(PipelineContextKeys.DocumentAssemblyCompleted).Should().BeTrue();
        var document = context.GetRequired<DoclingDocument>(PipelineContextKeys.Document);
        document.SourceId.Should().Be("doc-source");
        document.Items.Should().HaveCount(1);
        var paragraph = document.Items[0].Should().BeOfType<ParagraphItem>().Subject;
        paragraph.Text.Should().Be("Hello world");
        paragraph.Metadata.Should().ContainKey("docling:source");
        paragraph.Provenance.Should().ContainSingle();
        var provenance = paragraph.Provenance[0];
        provenance.PageNumber.Should().Be(page.PageNumber);
        provenance.CharStart.Should().Be(0);
        provenance.CharEnd.Should().Be(paragraph.Text.Length);
        provenance.BoundingBox.Should().Be(layoutBox);
        var provenanceMetadata = paragraph.Metadata.Should().ContainKey("docling:provenance").WhoseValue
            .Should().BeAssignableTo<IReadOnlyList<IReadOnlyDictionary<string, object?>>>().Subject;
        provenanceMetadata.Should().HaveCount(1);
        provenanceMetadata[0]["page_number"].Should().Be(page.PageNumber);
    }

    [Fact]
    public async Task AssembleAsyncReconstructsTablesWithCellText()
    {
        var page = new PageReference(2, 200);
        var tableBounds = BoundingBox.FromSize(50, 40, 120, 60);
        var layoutTable = new LayoutItem(page, tableBounds, LayoutItemKind.Table, Array.Empty<Polygon>());
        var structureCells = new List<TableCell>
        {
            new(BoundingBox.FromSize(50, 40, 60, 30), 1, 1, null),
            new(BoundingBox.FromSize(110, 40, 60, 30), 1, 1, null),
            new(BoundingBox.FromSize(50, 70, 60, 30), 1, 1, null),
            new(BoundingBox.FromSize(110, 70, 60, 30), 1, 1, null),
        };
        var structure = new TableStructure(page, structureCells, 2, 2);
        var cellOneMetadata = new Dictionary<string, string>
        {
            ["docling:source"] = "table_cell",
            ["docling:table_index"] = "0",
            ["docling:table_row_index"] = "0",
            ["docling:table_column_index"] = "0",
            ["docling:table_row_span"] = "1",
            ["docling:table_column_span"] = "1",
            ["docling:area_ratio"] = "0.500000",
        };
        var cellTwoMetadata = new Dictionary<string, string>
        {
            ["docling:source"] = "table_cell",
            ["docling:table_index"] = "0",
            ["docling:table_row_index"] = "1",
            ["docling:table_column_index"] = "1",
            ["docling:table_row_span"] = "1",
            ["docling:table_column_span"] = "1",
            ["docling:area_ratio"] = "0.500000",
        };
        var ocrBlocks = new List<OcrBlockResult>
        {
            new(page, structureCells[0].BoundingBox, OcrRegionKind.TableCell, cellOneMetadata, new[]
            {
                new OcrLine("top left", structureCells[0].BoundingBox, 0.93),
            }),
            new(page, structureCells[3].BoundingBox, OcrRegionKind.TableCell, cellTwoMetadata, new[]
            {
                new OcrLine("bottom right", structureCells[3].BoundingBox, 0.92),
            }),
        };

        var context = CreateContext();
        context.Set(PipelineContextKeys.PageSequence, new[] { page });
        context.Set(PipelineContextKeys.LayoutItems, new[] { layoutTable });
        context.Set(PipelineContextKeys.TableStructures, new[] { structure });
        context.Set(PipelineContextKeys.OcrResults, new OcrDocumentResult(ocrBlocks));

        var stage = new PageAssemblyStage(NullLogger<PageAssemblyStage>.Instance);
        await stage.ExecuteAsync(context, CancellationToken.None);

        var document = context.GetRequired<DoclingDocument>(PipelineContextKeys.Document);
        document.Items.Should().HaveCount(1);
        var table = document.Items[0].Should().BeOfType<TableItem>().Subject;
        table.RowCount.Should().Be(2);
        table.ColumnCount.Should().Be(2);
        table.Cells.Should().Contain(cell => cell.Text == "top left");
        table.Cells.Should().Contain(cell => cell.Text == "bottom right");
        table.Metadata.Should().ContainKey("docling:table_index");
        table.Provenance.Should().ContainSingle();
        var tableProv = table.Provenance[0];
        tableProv.PageNumber.Should().Be(page.PageNumber);
        tableProv.CharStart.Should().BeNull();
        tableProv.CharEnd.Should().BeNull();
        tableProv.BoundingBox.Should().Be(tableBounds);
    }

    [Fact]
    public async Task AssembleAsyncRegistersFiguresWithCaptions()
    {
        var page = new PageReference(4, 180);
        var figureBounds = BoundingBox.FromSize(100, 100, 240, 180);
        var captionBounds = BoundingBox.FromSize(110, 290, 220, 60);
        var figureItem = new LayoutItem(page, figureBounds, LayoutItemKind.Figure, Array.Empty<Polygon>());
        var captionItem = new LayoutItem(page, captionBounds, LayoutItemKind.Text, Array.Empty<Polygon>());
        var captionLines = new List<OcrLine>
        {
            new("Figure 1. Example caption.", captionBounds, 0.92),
        };
        var captionBlock = new OcrBlockResult(
            page,
            captionBounds,
            OcrRegionKind.LayoutBlock,
            new Dictionary<string, string>(),
            captionLines);

        var context = CreateContext();
        context.Set(PipelineContextKeys.PageSequence, new[] { page });
        context.Set(PipelineContextKeys.LayoutItems, new[] { figureItem, captionItem });
        context.Set(PipelineContextKeys.OcrResults, new OcrDocumentResult(new[] { captionBlock }));

        var stage = new PageAssemblyStage(NullLogger<PageAssemblyStage>.Instance);
        await stage.ExecuteAsync(context, CancellationToken.None);

        var document = context.GetRequired<DoclingDocument>(PipelineContextKeys.Document);
        document.Items.Should().HaveCount(2);

        var picture = document.Items.OfType<PictureItem>().Should().ContainSingle().Subject;
        picture.BoundingBox.Should().Be(figureBounds);
        picture.Provenance.Should().ContainSingle();
        picture.Provenance[0].BoundingBox.Should().Be(figureBounds);

        var caption = document.Items.OfType<CaptionItem>().Should().ContainSingle().Subject;
        caption.Text.Should().Be("Figure 1. Example caption.");
        caption.TargetItemId.Should().Be(picture.Id);
        caption.Metadata.Should().Contain(new KeyValuePair<string, object?>("docling:target_item_id", picture.Id));
        caption.Provenance.Should().ContainSingle();
        caption.Provenance[0].CharStart.Should().Be(0);
        caption.Provenance[0].CharEnd.Should().Be(caption.Text.Length);
    }

    [Fact]
    public async Task AssembleAsyncFallsBackToFullPageOcr()
    {
        var page = new PageReference(3, 150);
        var fullBounds = BoundingBox.FromSize(0, 0, 800, 600);
        var ocrLines = new List<OcrLine>
        {
            new("full page text", BoundingBox.FromSize(0, 0, 800, 20), 0.9),
        };
        var ocrBlock = new OcrBlockResult(page, fullBounds, OcrRegionKind.FullPage, new Dictionary<string, string>(), ocrLines);

        var context = CreateContext();
        context.Set(PipelineContextKeys.PageSequence, new[] { page });
        context.Set(PipelineContextKeys.OcrResults, new OcrDocumentResult(new[] { ocrBlock }));

        var stage = new PageAssemblyStage(NullLogger<PageAssemblyStage>.Instance);
        await stage.ExecuteAsync(context, CancellationToken.None);

        var document = context.GetRequired<DoclingDocument>(PipelineContextKeys.Document);
        document.Items.Should().HaveCount(1);
        var paragraph = document.Items[0].Should().BeOfType<ParagraphItem>().Subject;
        paragraph.BoundingBox.Should().Be(fullBounds);
        paragraph.Text.Should().Be("full page text");
        paragraph.Provenance.Should().ContainSingle();
        paragraph.Provenance[0].BoundingBox.Should().Be(fullBounds);
    }

    private static PipelineContext CreateContext()
    {
        var services = new ServiceCollection().BuildServiceProvider();
        return new PipelineContext(services);
    }
}
