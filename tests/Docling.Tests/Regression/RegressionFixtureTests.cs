using System;
using FluentAssertions;
using Xunit;

namespace Docling.Tests.Regression;

public sealed class RegressionFixtureTests
{
    [Fact]
    public void LoadResolvesMarkdownAndMetadata()
    {
        var fixture = RegressionFixture.Load("amt_handbook_sample");

        fixture.Name.Should().Be("amt_handbook_sample");
        fixture.MarkdownPath.Should().EndWith("amt_handbook_sample.md");
        fixture.DocumentJsonPath.Should().EndWith("amt_handbook_sample.json");
        fixture.PagesJsonPath.Should().EndWith("amt_handbook_sample.pages.json");
        fixture.DocTagsPath.Should().EndWith("amt_handbook_sample.doctags.txt");

        fixture.Markdown.Should().NotBeNullOrWhiteSpace();
        fixture.DocTags.Should().NotBeEmpty();

        fixture.DocumentJson.Should().NotBeNull();
        fixture.DocumentJson!["name"]!.GetValue<string>().Should().Be("amt_handbook_sample");

        fixture.PagesJson.Should().NotBeNull();
        fixture.PagesJson!.Count.Should().BeGreaterThan(0);
    }

    [Fact]
    public void LoadHandlesFixturesWithoutMarkdown()
    {
        var fixture = RegressionFixture.Load("2305.03393v1");

        fixture.Markdown.Should().BeNull();
        fixture.DocumentJson.Should().BeNull();
        fixture.DocTags.Should().NotBeEmpty();
    }

    [Fact]
    public void CompareMarkdownHighlightsDifferences()
    {
        var fixture = RegressionFixture.Load("amt_handbook_sample");
        fixture.Markdown.Should().NotBeNull();

        var identical = fixture.CompareMarkdown(fixture.Markdown!);
        identical.AreEquivalent.Should().BeTrue();
        identical.Differences.Should().BeEmpty();

        var mutated = fixture.Markdown!.Replace("Boots", "Shoes", StringComparison.Ordinal);
        var comparison = fixture.CompareMarkdown(mutated);

        comparison.AreEquivalent.Should().BeFalse();
        comparison.Differences.Should().NotBeEmpty();
        comparison.Differences[0].LineNumber.Should().BeGreaterThan(0);
    }
}
