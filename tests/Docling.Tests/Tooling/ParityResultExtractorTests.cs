using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Imaging;
using Docling.Export.Serialization;
using Docling.Tooling.Parity;

namespace Docling.Tests.Tooling;

public sealed class ParityResultExtractorTests
{
    [Fact]
    public void Extract_ComputesNormalizedSnapshot()
    {
        var document = new DoclingDocument(
            "source-001",
            new[] { new PageReference(1, 299.6) },
            "doc-001",
            DateTimeOffset.Parse("2024-01-01T00:00:00Z"),
            new Dictionary<string, string> { ["docling:source"] = "unit" });

        var image = new ImageRef(
            "img-001",
            document.Pages[0],
            new BoundingBox(10.2, 20.7, 110.4, 220.9),
            "image/png",
            new byte[] { 1, 2, 3, 4, 5 },
            120,
            80,
            299.6);

        var asset = new MarkdownAsset("assets\\figure.png", ImageExportKind.Picture, image, "fig-001");
        var serializationResult = new MarkdownSerializationResult(
            "# Heading\n",
            new[] { asset },
            new Dictionary<string, string> { ["docling:version"] = "0.12.0" });

        var options = new ParityExtractionOptions
        {
            BaseDirectory = "/tmp/output",
            CoordinateTolerance = 0.25,
            CoordinateDecimals = 2,
        };

        var result = ParityResultExtractor.Extract("/tmp/output/docling.md", serializationResult, document, options);

        Assert.Equal("doc-001", result.DocumentId);
        Assert.Equal("docling.md", result.MarkdownPath);
        Assert.Equal(ComputeSha256("# Heading\n"), result.MarkdownSha256);
        Assert.Single(result.DocumentProperties);
        Assert.Equal("unit", result.DocumentProperties["docling:source"]);
        Assert.Single(result.MarkdownMetadata);
        Assert.Equal("0.12.0", result.MarkdownMetadata["docling:version"]);

        var page = Assert.Single(result.Pages);
        Assert.Equal(1, page.PageNumber);
        Assert.Equal(299.5, page.Dpi);

        var snapshot = Assert.Single(result.Assets);
        Assert.Equal("assets/figure.png", snapshot.RelativePath);
        Assert.Equal("PICTURE", snapshot.Kind);
        Assert.Equal("fig-001", snapshot.TargetItemId);
        Assert.Equal("img-001", snapshot.ImageId);
        Assert.Equal(1, snapshot.PageNumber);
        Assert.Equal(299.5, snapshot.Dpi);
        Assert.Equal(120, snapshot.Width);
        Assert.Equal(80, snapshot.Height);
        Assert.Equal("image/png", snapshot.MediaType);
        Assert.Equal(ComputeSha256(new byte[] { 1, 2, 3, 4, 5 }), snapshot.Checksum);
        Assert.Equal(new ParityBoundingBox(10.25, 20.75, 110.5, 221d), snapshot.BoundingBox);

        Assert.InRange(result.GeneratedAtUtc, DateTimeOffset.UtcNow.AddMinutes(-1), DateTimeOffset.UtcNow.AddSeconds(1));
    }

    [Fact]
    public void Extract_ComputesChecksumWhenMissing()
    {
        var document = new DoclingDocument("src", new[] { new PageReference(1, 300) }, "doc-xyz");
        var image = new ImageRef(
            "img-xyz",
            document.Pages[0],
            new BoundingBox(0, 0, 10, 10),
            "image/png",
            new byte[] { 0x10, 0x20, 0x30 },
            10,
            10,
            300,
            checksum: null);

        var asset = new MarkdownAsset("figure.png", ImageExportKind.Picture, image, null);
        var serializationResult = new MarkdownSerializationResult(
            "body",
            new[] { asset },
            new Dictionary<string, string>());

        var result = ParityResultExtractor.Extract("figure.md", serializationResult, document, new ParityExtractionOptions { CoordinateTolerance = 0.001, CoordinateDecimals = 3 });

        var snapshot = Assert.Single(result.Assets);
        Assert.Equal(ComputeSha256(new byte[] { 0x10, 0x20, 0x30 }), snapshot.Checksum);
    }

    private static string ComputeSha256(string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        return ComputeSha256(bytes);
    }

    private static string ComputeSha256(byte[] value)
    {
        var hash = SHA256.HashData(value);
        return Convert.ToHexString(hash);
    }
}
