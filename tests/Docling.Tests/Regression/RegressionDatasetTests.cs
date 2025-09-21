using FluentAssertions;
using Xunit;

namespace Docling.Tests.Regression;

public sealed class RegressionDatasetTests
{
    [Fact]
    public void LoadAmtHandbookSample()
    {
        var dataset = RegressionDatasets.Load(RegressionDatasetId.AmtHandbookSample);

        dataset.Name.Should().Be("amt_handbook_sample");
        dataset.AssetPath.Should().EndWith("amt_handbook_sample.pdf");
        dataset.Fixture.Markdown.Should().NotBeNullOrWhiteSpace();
        dataset.Fixture.DocTags.Should().NotBeEmpty();
    }

    [Fact]
    public void LoadArxivPageImageSample()
    {
        var dataset = RegressionDatasets.Load(RegressionDatasetId.Arxiv230503393Page9);

        dataset.Name.Should().Be("2305.03393v1-pg9");
        dataset.AssetPath.Should().EndWith("2305.03393v1-pg9-img.png");
        dataset.Fixture.Markdown.Should().NotBeNullOrWhiteSpace();
        dataset.Fixture.PagesJson.Should().NotBeNull();
    }
}
