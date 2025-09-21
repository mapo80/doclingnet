using System;
using System.IO;
using Docling.Export.Serialization;
using Docling.Pipelines.Options;
using Docling.Tooling.Commands;
using FluentAssertions;
using Xunit;

namespace Docling.Tests.Tooling;

public sealed class ConvertCommandOptionsTests
{
    private static readonly string[] ExpectedLanguages = ["en", "fr"];

    [Fact]
    public void ParseWhenArgumentsValidReturnsOptions()
    {
        var tempFile = Path.Combine(Path.GetTempPath(), $"docling-{Guid.NewGuid():N}.pdf");
        try
        {
            File.WriteAllBytes(tempFile, new byte[] { 1, 2, 3 });
            var args = new[]
            {
                "--input", tempFile,
                "--output", "results",
                "--markdown", "docling.md",
                "--assets", "assets/images",
                "--languages", "en,fr",
                "--dpi", "240",
                "--render-dpi", "260",
                "--table-mode", "fast",
                "--image-mode", "embedded",
                "--no-page-images",
                "--no-picture-images",
                "--full-page-ocr",
                "--layout-debug",
                "--image-debug",
                "--table-debug",
            };

            var result = ConvertCommandOptions.Parse(args);
            result.Success.Should().BeTrue(result.Error);
            result.ShowHelp.Should().BeFalse();
            result.Error.Should().BeNull();
            result.Options.Should().NotBeNull();
            var options = result.Options!;

            options.InputPath.Should().Be(Path.GetFullPath(tempFile));
            options.OutputDirectory.Should().Be(Path.GetFullPath("results"));
            options.MarkdownFileName.Should().Be("docling.md");
            options.AssetsDirectoryName.Should().Be("assets/images");
            options.OcrLanguages.Should().BeEquivalentTo(ExpectedLanguages);
            options.PreprocessingDpi.Should().Be(240);
            options.RenderDpi.Should().Be(260);
            options.TableMode.Should().Be(TableFormerMode.Fast);
            options.ImageMode.Should().Be(MarkdownImageMode.Embedded);
            options.GeneratePageImages.Should().BeFalse();
            options.GeneratePictureImages.Should().BeFalse();
            options.GenerateLayoutDebugArtifacts.Should().BeTrue();
            options.GenerateImageDebugArtifacts.Should().BeTrue();
            options.GenerateTableDebugArtifacts.Should().BeTrue();
            options.ForceFullPageOcr.Should().BeTrue();
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void ParseWhenHelpRequestedReturnsHelpFlag()
    {
        var result = ConvertCommandOptions.Parse(new[] { "--help" });
        result.Success.Should().BeFalse();
        result.ShowHelp.Should().BeTrue();
        result.Options.Should().BeNull();
    }

    [Fact]
    public void ParseWhenInputMissingReturnsFailure()
    {
        var result = ConvertCommandOptions.Parse(Array.Empty<string>());
        result.Success.Should().BeFalse();
        result.ShowHelp.Should().BeFalse();
        result.Error.Should().NotBeNull();
    }
}
