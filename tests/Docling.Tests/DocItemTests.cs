using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using FluentAssertions;

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
}
