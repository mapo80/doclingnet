using System;
using System.Collections.Generic;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Imaging;
using Docling.Export.Serialization;
using FluentAssertions;
using Xunit;

namespace Docling.Tests.Export;

public sealed class MarkdownDocSerializerTests
{
    [Fact]
    public void SerializeProducesMarkdownWithAssets()
    {
        var page = new PageReference(1, 144);
        var document = new DoclingDocument("source", new[] { page });
        document.SetProperty("title", "Sample");

        var paragraph = new ParagraphItem(page, BoundingBox.FromSize(0, 0, 100, 20), "Hello world.");
        document.AddItem(paragraph);

        var picture = new PictureItem(page, BoundingBox.FromSize(0, 30, 80, 80), "A sample figure");
        document.AddItem(picture);

        var figureCaption = new CaptionItem(page, BoundingBox.FromSize(0, 115, 80, 10), "Figure caption", picture.Id);
        document.AddItem(figureCaption);

        var tableCells = new[]
        {
            new TableCellItem(0, 0, 1, 1, BoundingBox.FromSize(0, 0, 10, 10), "H1"),
            new TableCellItem(0, 1, 1, 1, BoundingBox.FromSize(10, 0, 10, 10), "H2"),
            new TableCellItem(1, 0, 1, 1, BoundingBox.FromSize(0, 10, 10, 10), "R1C1"),
            new TableCellItem(1, 1, 1, 1, BoundingBox.FromSize(10, 10, 10, 10), "R1C2"),
        };
        var table = new TableItem(page, BoundingBox.FromSize(0, 140, 80, 60), tableCells, 2, 2);
        document.AddItem(table);
        var tableCaption = new CaptionItem(page, BoundingBox.FromSize(0, 205, 80, 10), "Table caption", table.Id);
        document.AddItem(tableCaption);

        var pictureImage = CreateImageRef("img-picture", page);
        picture.SetImage(pictureImage);

        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions
        {
            AssetsPath = "assets",
            ImageMode = MarkdownImageMode.Referenced,
        });

        var exports = new List<ImageExportArtifact>
        {
            new(ImageExportKind.Picture, pictureImage, picture.Id),
        };

        var result = serializer.Serialize(document, exports);

        result.Markdown.Should().Contain("Hello world.");
        result.Markdown.Should().Contain("![A sample figure]");
        result.Markdown.Should().Contain("assets/img-picture.png");
        result.Markdown.Should().Contain("*Figure 1. Figure caption*");
        result.Markdown.Should().Contain("| H1 | H2 |");
        result.Markdown.Should().Contain("*Table 1. Table caption*");

        result.Assets.Should().ContainSingle();
        result.Assets[0].RelativePath.Should().Be("assets/img-picture.png");
        result.Assets[0].Kind.Should().Be(ImageExportKind.Picture);
        result.Metadata.Should().ContainKey("title").WhoseValue.Should().Be("Sample");
    }

    [Fact]
    public void SerializeEmbedsImagesWhenConfigured()
    {
        var page = new PageReference(2, 144);
        var document = new DoclingDocument("source", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(0, 0, 50, 50), "Inline");
        document.AddItem(picture);

        var image = CreateImageRef("img-embedded", page);
        picture.SetImage(image);

        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions
        {
            ImageMode = MarkdownImageMode.Embedded,
        });

        var result = serializer.Serialize(document, new[] { new ImageExportArtifact(ImageExportKind.Picture, image, picture.Id) });

        result.Markdown.Should().Contain("data:image/png;base64,");
        result.Markdown.Should().Contain("*Figure 1. Inline*");
        result.Assets.Should().BeEmpty();
    }

    [Fact]
    public void SerializeFallsBackToPlaceholderWhenImageMissing()
    {
        var page = new PageReference(3, 144);
        var document = new DoclingDocument("source", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(0, 0, 40, 40), "Placeholder");
        document.AddItem(picture);

        var serializer = new MarkdownDocSerializer();
        var result = serializer.Serialize(document, Array.Empty<ImageExportArtifact>());

        result.Markdown.Should().Contain("image unavailable");
        result.Markdown.Should().Contain("*Figure 1. Placeholder*");
        result.Assets.Should().BeEmpty();
    }

    [Fact]
    public void SerializeTableFallsBackToImageWhenSpansDetected()
    {
        var page = new PageReference(4, 200);
        var document = new DoclingDocument("source", new[] { page });
        var tableCells = new[]
        {
            new TableCellItem(0, 0, 1, 2, BoundingBox.FromSize(0, 0, 20, 10), "Header"),
            new TableCellItem(1, 0, 1, 1, BoundingBox.FromSize(0, 10, 10, 10), "A"),
            new TableCellItem(1, 1, 1, 1, BoundingBox.FromSize(10, 10, 10, 10), "B"),
        };
        var table = new TableItem(page, BoundingBox.FromSize(0, 0, 40, 30), tableCells, 2, 2);
        document.AddItem(table);

        var preview = CreateImageRef("img-table", page);
        table.SetPreviewImage(preview);

        var serializer = new MarkdownDocSerializer();
        var result = serializer.Serialize(document, new[] { new ImageExportArtifact(ImageExportKind.Table, preview, table.Id) });

        result.Markdown.Should().Contain("![Table 1]");
        result.Markdown.Should().Contain("*Table 1.*");
        result.Assets.Should().ContainSingle(asset => asset.Kind == ImageExportKind.Table && asset.TargetItemId == table.Id);
    }

    [Fact]
    public void SerializeHandlesLooseCaptionsAndAltFallback()
    {
        var page = new PageReference(5, 144);
        var document = new DoclingDocument("source", new[] { page });

        var picture = new PictureItem(page, BoundingBox.FromSize(0, 0, 40, 40), string.Empty);
        document.AddItem(picture);

        var descriptiveCaption = new CaptionItem(
            page,
            BoundingBox.FromSize(0, 45, 60, 10),
            "Caption with [brackets] and *stars* plus \\backslash_",
            picture.Id);
        document.AddItem(descriptiveCaption);

        var blankTargetedCaption = new CaptionItem(page, BoundingBox.FromSize(0, 55, 60, 10), string.Empty, picture.Id);
        document.AddItem(blankTargetedCaption);

        var emptyCaption = new CaptionItem(page, BoundingBox.FromSize(0, 60, 60, 10), string.Empty, null);
        document.AddItem(emptyCaption);

        var looseCaption = new CaptionItem(page, BoundingBox.FromSize(0, 70, 60, 10), "Loose summary", null);
        document.AddItem(looseCaption);

        var ignoredCaption = new CaptionItem(page, BoundingBox.FromSize(0, 80, 60, 10), "Ignored", string.Empty);
        document.AddItem(ignoredCaption);

        var emptyParagraph = new ParagraphItem(page, BoundingBox.FromSize(0, 90, 40, 10), string.Empty);
        document.AddItem(emptyParagraph);

        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions { AssetsPath = "assets" });

        var export = CreateImageRef("img-caption", page);
        var exports = new[]
        {
            new ImageExportArtifact(ImageExportKind.Picture, export, picture.Id),
            new ImageExportArtifact(ImageExportKind.Table, CreateImageRef("unused", page), null),
        };

        var result = serializer.Serialize(document, exports);

        result.Markdown.Should().Contain("![Caption with \\[brackets\\] and \\*stars\\* plus \\\\backslash\\_]");
        result.Markdown.Should().Contain("*Loose summary*");
        result.Assets.Should().ContainSingle(asset => asset.TargetItemId == picture.Id);
    }

    [Fact]
    public void SerializeTablePlaceholderWhenNoPreviewIsAvailable()
    {
        var page = new PageReference(6, 144);
        var document = new DoclingDocument("source", new[] { page });
        var tableCells = new[]
        {
            new TableCellItem(0, 0, 1, 2, BoundingBox.FromSize(0, 0, 30, 15), "Header"),
            new TableCellItem(1, 0, 1, 1, BoundingBox.FromSize(0, 15, 15, 15), "Left"),
            new TableCellItem(1, 1, 1, 1, BoundingBox.FromSize(15, 15, 15, 15), "Right"),
        };
        var table = new TableItem(page, BoundingBox.FromSize(0, 0, 40, 30), tableCells, 2, 2);
        document.AddItem(table);

        var serializer = new MarkdownDocSerializer();
        var result = serializer.Serialize(document, Array.Empty<ImageExportArtifact>());

        result.Markdown.Should().Contain("table preview unavailable");
        result.Assets.Should().BeEmpty();
    }

    [Fact]
    public void SerializeEmbeddedTableProducesDataUri()
    {
        var page = new PageReference(7, 144);
        var document = new DoclingDocument("source", new[] { page });
        var tableCells = new[]
        {
            new TableCellItem(0, 0, 1, 2, BoundingBox.FromSize(0, 0, 20, 10), "Header"),
            new TableCellItem(1, 0, 1, 1, BoundingBox.FromSize(0, 10, 10, 10), "A"),
            new TableCellItem(1, 1, 1, 1, BoundingBox.FromSize(10, 10, 10, 10), "B"),
        };
        var table = new TableItem(page, BoundingBox.FromSize(0, 0, 40, 30), tableCells, 2, 2);
        document.AddItem(table);

        var preview = CreateImageRef("img-table-embedded", page);
        table.SetPreviewImage(preview);

        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions
        {
            ImageMode = MarkdownImageMode.Embedded,
        });

        var result = serializer.Serialize(document, new[] { new ImageExportArtifact(ImageExportKind.Table, preview, table.Id) });

        result.Markdown.Should().Contain("data:image/png;base64,");
        result.Markdown.Should().Contain("*Table 1.*");
    }

    [Fact]
    public void SerializeGeneratesDefaultHeaderAndEscapesCells()
    {
        var page = new PageReference(8, 144);
        var document = new DoclingDocument("source", new[] { page });
        var tableCells = new[]
        {
            new TableCellItem(0, 0, 1, 1, BoundingBox.FromSize(0, 0, 10, 10), "Value"),
            new TableCellItem(
                0,
                1,
                1,
                1,
                BoundingBox.FromSize(10, 0, 10, 10),
                "Pipe|Break" + Environment.NewLine + "Line"),
            new TableCellItem(5, 0, 1, 1, BoundingBox.FromSize(0, 0, 10, 10), "OutOfBounds"),
        };
        var table = new TableItem(page, BoundingBox.FromSize(0, 0, 40, 20), tableCells, 1, 2);
        document.AddItem(table);

        var serializer = new MarkdownDocSerializer();
        var result = serializer.Serialize(document, Array.Empty<ImageExportArtifact>());

        result.Markdown.Should().Contain("| Column 1 | Column 2 |");
        result.Markdown.Should().Contain("| Value | Pipe\\|Break<br />Line |");
    }

    [Fact]
    public void SerializeFallsBackToLabelWhenNoDescriptionOrCaption()
    {
        var page = new PageReference(9, 144);
        var document = new DoclingDocument("source", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(0, 0, 30, 30), string.Empty);
        document.AddItem(picture);

        var serializer = new MarkdownDocSerializer();
        var result = serializer.Serialize(document, Array.Empty<ImageExportArtifact>());

        result.Markdown.Should().Contain("> Figure 1: Figure 1 (image unavailable)");
        result.Markdown.Should().Contain("*Figure 1. Figure 1*");
    }

    [Fact]
    public void SerializeUsesAltTextProviderForPictures()
    {
        var page = new PageReference(10, 200);
        var document = new DoclingDocument("source", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(0, 0, 40, 40), "Existing description");
        document.AddItem(picture);

        var caption = new CaptionItem(page, BoundingBox.FromSize(0, 45, 60, 10), "Generated caption", picture.Id);
        document.AddItem(caption);

        var image = CreateImageRef("img-alt-picture", page);
        picture.SetImage(image);

        var provider = new RecordingAltTextProvider("Enriched description");
        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions
        {
            AssetsPath = "assets",
            AltTextProvider = provider,
        });

        var result = serializer.Serialize(document, new[] { new ImageExportArtifact(ImageExportKind.Picture, image, picture.Id) });

        result.Markdown.Should().Contain("![Enriched description](assets/img-alt-picture.png)");
        provider.Requests.Should().HaveCount(1);
        var context = provider.Requests[0];
        context.Picture.Should().BeSameAs(picture);
        context.Label.Should().Be("Figure 1");
        context.Caption.Should().Be("Generated caption");
        context.Image.Should().BeSameAs(image);
    }

    [Fact]
    public void SerializeUsesAltTextProviderForTableFallback()
    {
        var page = new PageReference(11, 144);
        var document = new DoclingDocument("source", new[] { page });
        var tableCells = new[]
        {
            new TableCellItem(0, 0, 1, 2, BoundingBox.FromSize(0, 0, 20, 10), "Header"),
            new TableCellItem(1, 0, 1, 1, BoundingBox.FromSize(0, 10, 10, 10), "A"),
            new TableCellItem(1, 1, 1, 1, BoundingBox.FromSize(10, 10, 10, 10), "B"),
        };
        var table = new TableItem(page, BoundingBox.FromSize(0, 0, 40, 30), tableCells, 2, 2);
        document.AddItem(table);

        var caption = new CaptionItem(page, BoundingBox.FromSize(0, 35, 60, 10), "Table description", table.Id);
        document.AddItem(caption);

        var preview = CreateImageRef("img-alt-table", page);
        table.SetPreviewImage(preview);

        var provider = new RecordingAltTextProvider("Detailed table");
        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions
        {
            AltTextProvider = provider,
        });

        var result = serializer.Serialize(document, new[] { new ImageExportArtifact(ImageExportKind.Table, preview, table.Id) });

        result.Markdown.Should().Contain("![Detailed table]");
        provider.Requests.Should().HaveCount(1);
        var context = provider.Requests[0];
        context.Table.Should().BeSameAs(table);
        context.Label.Should().Be("Table 1");
        context.Caption.Should().Be("Table description");
        context.Image.Should().BeSameAs(preview);
    }

    [Fact]
    public void SerializeFallsBackWhenAltTextProviderThrows()
    {
        var page = new PageReference(12, 144);
        var document = new DoclingDocument("source", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(0, 0, 40, 40), string.Empty);
        document.AddItem(picture);

        var provider = new ThrowingAltTextProvider();
        var serializer = new MarkdownDocSerializer(new MarkdownSerializerOptions
        {
            AltTextProvider = provider,
        });

        var result = serializer.Serialize(document, Array.Empty<ImageExportArtifact>());

        result.Markdown.Should().Contain("> Figure 1: Figure 1 (image unavailable)");
        result.Markdown.Should().Contain("*Figure 1. Figure 1*");
    }

    private static ImageRef CreateImageRef(string id, PageReference page)
    {
        var data = new byte[] { 1, 2, 3, 4, 5 };
        var region = BoundingBox.FromSize(0, 0, 10, 10);
        return new ImageRef(id, page, region, "image/png", data, 10, 10, page.Dpi, "ABC123");
    }

    private sealed class RecordingAltTextProvider : IMarkdownAltTextProvider
    {
        private readonly Queue<string?> _responses;

        public RecordingAltTextProvider(params string?[] responses)
        {
            _responses = new Queue<string?>(responses ?? Array.Empty<string?>());
        }

        public List<MarkdownAltTextContext> Requests { get; } = new();

        public string? GetAltText(MarkdownAltTextContext context)
        {
            Requests.Add(context);
            return _responses.Count > 0 ? _responses.Dequeue() : null;
        }
    }

    private sealed class ThrowingAltTextProvider : IMarkdownAltTextProvider
    {
        public string? GetAltText(MarkdownAltTextContext context)
        {
            throw new InvalidOperationException("Alt-text enrichment failed");
        }
    }
}
