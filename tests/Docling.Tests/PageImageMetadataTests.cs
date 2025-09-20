using System.Collections.Generic;
using Docling.Backends.Pdf;
using FluentAssertions;
using Xunit;

namespace Docling.Tests;

public sealed class PageImageMetadataTests
{
    [Fact]
    public void WithAdditionalPropertiesMergesAndOverrides()
    {
        var metadata = new PageImageMetadata("doc", "source", "application/pdf", new Dictionary<string, string>
        {
            ["existing"] = "value",
            ["override"] = "old",
        });

        var merged = metadata.WithAdditionalProperties(new Dictionary<string, string>
        {
            ["override"] = "new",
            ["added"] = "extra",
        });

        merged.Properties.Should().ContainKey("existing");
        merged.Properties.Should().ContainKey("override").WhoseValue.Should().Be("new");
        merged.Properties.Should().ContainKey("added");
        merged.SourceId.Should().Be("doc");
        merged.SourceName.Should().Be("source");
        merged.MediaType.Should().Be("application/pdf");
    }
}
