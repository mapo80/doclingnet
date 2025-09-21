using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Imaging;
using Docling.Export.Serialization;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Serialization;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace Docling.Tests.Pipelines.Serialization;

public sealed class MarkdownSerializationStageTests
{
    [Fact]
    public async Task ExecuteAsyncSerializesDocument()
    {
        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions { AssetsPath = "assets" });
        var stage = new MarkdownSerializationStage(serializer);

        var page = new PageReference(1, 144);
        var document = new DoclingDocument("src", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(0, 0, 40, 40), "Stage figure");
        document.AddItem(picture);

        var image = CreateImageRef("img-stage", page);
        picture.SetImage(image);
        var exports = new[] { new ImageExportArtifact(ImageExportKind.Picture, image, picture.Id) };

        var context = CreateContext(document, exports);

        await stage.ExecuteAsync(context, CancellationToken.None);

        context.GetRequired<bool>(PipelineContextKeys.MarkdownSerializationCompleted).Should().BeTrue();
        var result = context.GetRequired<MarkdownSerializationResult>(PipelineContextKeys.MarkdownSerializationResult);
        result.Markdown.Should().Contain("![Stage figure]");
        result.Markdown.Should().Contain("*Figure 1. Stage figure*");
        result.Assets.Should().ContainSingle(asset => asset.RelativePath.StartsWith("assets/", StringComparison.Ordinal));
    }

    [Fact]
    public async Task ExecuteAsyncDoesNotRequireExports()
    {
        var serializer = new MarkdownDocSerializer();
        var stage = new MarkdownSerializationStage(serializer);

        var page = new PageReference(2, 144);
        var document = new DoclingDocument("src", new[] { page });
        var paragraph = new ParagraphItem(page, BoundingBox.FromSize(0, 0, 100, 20), "Stage paragraph");
        document.AddItem(paragraph);

        var context = CreateContext(document, null);
        await stage.ExecuteAsync(context, CancellationToken.None);

        context.GetRequired<bool>(PipelineContextKeys.MarkdownSerializationCompleted).Should().BeTrue();
        var result = context.GetRequired<MarkdownSerializationResult>(PipelineContextKeys.MarkdownSerializationResult);
        result.Markdown.Should().Contain("Stage paragraph");
        result.Assets.Should().BeEmpty();
    }

    private static PipelineContext CreateContext(DoclingDocument document, IReadOnlyList<ImageExportArtifact>? exports)
    {
        var services = new ServiceCollection().BuildServiceProvider();
        var context = new PipelineContext(services);
        context.Set(PipelineContextKeys.Document, document);
        if (exports is not null)
        {
            context.Set(PipelineContextKeys.ImageExports, exports);
        }
        return context;
    }

    private static ImageRef CreateImageRef(string id, PageReference page)
    {
        var data = new byte[] { 5, 4, 3, 2, 1 };
        var region = BoundingBox.FromSize(0, 0, 10, 10);
        return new ImageRef(id, page, region, "image/png", data, 10, 10, page.Dpi, "FED321");
    }
}
