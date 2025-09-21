using System.Collections.Generic;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using FluentAssertions;
using Xunit;

namespace Docling.Tests;

public sealed class DocItemTests
{
    private static readonly string[] ParagraphTag = ["Paragraph"];

    [Fact]
    public void GeneratesDeterministicMetadata()
    {
        var page = new PageReference(1, 300);
        var item = new ParagraphItem(page, BoundingBox.FromSize(0, 0, 10, 10), "hello");

        item.TryGetMetadata<string>("text", out var value).Should().BeTrue();
        value.Should().Be("hello");
        item.HasTag("paragraph").Should().BeFalse();
    }

    [Fact]
    public void TagsSupportCaseInsensitiveLookup()
    {
        var page = new PageReference(1, 300);
        var item = new ParagraphItem(page, BoundingBox.FromSize(0, 0, 10, 10), "hello", tags: ParagraphTag);

        item.HasTag("paragraph").Should().BeTrue();
        item.RemoveTag("PARAGRAPH").Should().BeTrue();
        item.HasTag("paragraph").Should().BeFalse();
    }

    [Fact]
    public void UpdateTextRefreshesMetadata()
    {
        var page = new PageReference(1, 300);
        var item = new ParagraphItem(page, BoundingBox.FromSize(0, 0, 10, 10), "hello");

        item.UpdateText("updated");
        item.Text.Should().Be("updated");
        item.TryGetMetadata<string>("text", out var value).Should().BeTrue();
        value.Should().Be("updated");
    }

    [Fact]
    public void PictureSetImageUpdatesMetadata()
    {
        var page = new PageReference(1, 300);
        var picture = new PictureItem(page, BoundingBox.FromSize(0, 0, 10, 10));
        var image = new ImageRef(
            "img-1",
            page,
            BoundingBox.FromSize(0, 0, 5, 5),
            "image/png",
            new byte[] { 1, 2, 3 },
            5,
            5,
            page.Dpi);

        picture.SetImage(image);

        picture.Image.Should().BeSameAs(image);
        picture.Metadata.Should().ContainKey("docling:image_ref");
        picture.Metadata.Should().Contain(new KeyValuePair<string, object?>("docling:image_media_type", "image/png"));
    }

    [Fact]
    public void TablePreviewImageUpdatesMetadata()
    {
        var page = new PageReference(1, 300);
        var table = new TableItem(
            page,
            BoundingBox.FromSize(0, 0, 20, 20),
            Array.Empty<TableCellItem>(),
            0,
            0);
        var image = new ImageRef(
            "img-2",
            page,
            BoundingBox.FromSize(0, 0, 10, 10),
            "image/png",
            new byte[] { 4, 5, 6 },
            10,
            10,
            page.Dpi);

        table.SetPreviewImage(image);

        table.PreviewImage.Should().BeSameAs(image);
        table.Metadata.Should().ContainKey("docling:preview_image");
        table.Metadata.Should().Contain(new KeyValuePair<string, object?>("docling:preview_media_type", "image/png"));
    }
}
