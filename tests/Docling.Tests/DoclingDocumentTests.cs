using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using FluentAssertions;

namespace Docling.Tests;

public sealed class DoclingDocumentTests
{
    private static DoclingDocument CreateDocument() =>
        new("doc-1", new[] { new PageReference(0, 300) });

    [Fact]
    public void AddItemMaintainsPageOrder()
    {
        var document = CreateDocument();
        var later = new ParagraphItem(new PageReference(0, 300), BoundingBox.FromSize(0, 20, 10, 10), "later");
        var earlier = new ParagraphItem(new PageReference(0, 300), BoundingBox.FromSize(0, 10, 10, 10), "earlier");

        document.AddItem(later);
        document.AddItem(earlier);

        document.Items.Should().HaveCount(2);
        document.Items[0].Should().BeSameAs(earlier);
    }

    [Fact]
    public void TryFindFirstBoundingBoxReturnsMatch()
    {
        var document = CreateDocument();
        var item = new ParagraphItem(new PageReference(0, 300), BoundingBox.FromSize(1, 2, 3, 4), "text");
        document.AddItem(item);

        var result = document.TryFindFirstBoundingBox(x => x.Kind == DocItemKind.Paragraph, out var bbox);

        result.Should().BeTrue();
        bbox.Should().Be(item.BoundingBox);
    }

    [Fact]
    public void PreventsDuplicateIds()
    {
        var document = CreateDocument();
        var page = new PageReference(0, 300);
        var box = BoundingBox.FromSize(0, 0, 10, 10);
        var id = "item-1";
        document.AddItem(new ParagraphItem(page, box, "text", id: id));

        var duplicate = () => document.AddItem(new ParagraphItem(page, box, "other", id: id));
        duplicate.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void PropertiesCanBeRetrieved()
    {
        var document = CreateDocument();
        document.SetProperty("Title", "Sample");

        document.TryGetProperty("title", out var value).Should().BeTrue();
        value.Should().Be("Sample");
    }

    [Fact]
    public void CloneCopiesItems()
    {
        var document = CreateDocument();
        var item = new ParagraphItem(new PageReference(0, 300), BoundingBox.FromSize(1, 1, 2, 2), "text");
        document.AddItem(item);

        var clone = document.Clone();

        clone.Should().NotBeSameAs(document);
        clone.Items.Should().HaveCount(1);
        clone.Items[0].Should().BeSameAs(item);
    }
}
