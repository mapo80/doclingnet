#pragma warning disable CA2007
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Tests.Infrastructure;
using FluentAssertions;
using Xunit;

namespace Docling.Tests;

public sealed class PdfBackendTests
{
    [Fact]
    public async Task LoadAsyncReturnsImagesFromRenderer()
    {
        using var renderer = new FakePdfRenderer(2);
        var options = new PdfBackendOptions
        {
            StreamFactory = _ => Task.FromResult<Stream>(new MemoryStream()),
            RenderSettings = new PdfRenderSettings(),
            DocumentId = "doc-1",
            SourceName = "fixture.pdf",
            Metadata = new Dictionary<string, string> { ["ingest"] = "unit-test" },
        };
        var backend = new PdfBackend(renderer, options);
        var images = new List<PageImage>();

        await foreach (var image in backend.LoadAsync(CancellationToken.None))
        {
            images.Add(image);
        }

        images.Should().HaveCount(2);
        foreach (var image in images)
        {
            image.Metadata.SourceId.Should().Be("doc-1");
            image.Metadata.SourceName.Should().Be("fixture.pdf");
            image.Metadata.Properties.Should().ContainKey("ingest");
            image.Metadata.Properties.Should().ContainKey("widthPixels");
            image.Dispose();
        }
    }

    [Fact]
    public async Task GetPageCountAsyncDelegatesToRenderer()
    {
        using var renderer = new FakePdfRenderer(3);
        var options = new PdfBackendOptions
        {
            StreamFactory = _ => Task.FromResult<Stream>(new MemoryStream()),
            RenderSettings = new PdfRenderSettings(),
        };
        var backend = new PdfBackend(renderer, options);

        var count = await backend.GetPageCountAsync(CancellationToken.None);

        count.Should().Be(3);
    }
}
#pragma warning restore CA2007
