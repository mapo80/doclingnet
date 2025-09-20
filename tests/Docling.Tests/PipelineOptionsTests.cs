using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Docling.Pipelines.Options;
using Xunit;

namespace Docling.Tests;

public sealed class PipelineOptionsTests
{
    [Fact]
    public void AcceleratorOptionsNormalizesCudaDevice()
    {
        var options = new AcceleratorOptions
        {
            NumThreads = 8,
            Device = "CUDA:1",
        };

        Assert.Equal(8, options.NumThreads);
        Assert.Equal("cuda:1", options.Device);
    }

    [Fact]
    public void AcceleratorOptionsThrowsForInvalidDevice()
    {
        Assert.Throws<ArgumentException>(() => new AcceleratorOptions { Device = "quantum" });
    }

    [Fact]
    public void AcceleratorOptionsThrowsForInvalidThreadCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AcceleratorOptions { NumThreads = 0 });
    }

    [Fact]
    public void EasyOcrOptionsExposePythonParityDefaults()
    {
        var options = new EasyOcrOptions();

        Assert.Equal(OcrEngine.EasyOcr, options.Engine);
        Assert.Contains("en", options.Languages);
        Assert.Equal(0.5d, options.ConfidenceThreshold);
    }

    [Fact]
    public void EasyOcrOptionsAcceptCustomLanguages()
    {
        var expectedLanguages = new[] { "it", "en" };
        var options = new EasyOcrOptions
        {
            Languages = expectedLanguages,
        };

        Assert.Equal(expectedLanguages, options.Languages);
    }

    [Fact]
    public void EasyOcrOptionsRejectInvalidLanguages()
    {
        var invalidLanguages = new[] { string.Empty, "en" };
        Assert.Throws<ArgumentException>(() => new EasyOcrOptions { Languages = invalidLanguages });
    }

    [Theory]
    [InlineData(0d)]
    [InlineData(-0.1d)]
    [InlineData(1.1d)]
    public void OcrOptionsRejectInvalidThresholds(double threshold)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EasyOcrOptions { BitmapAreaThreshold = threshold });
    }

    [Fact]
    public void RapidOcrOptionsDefensivelyCopyParameters()
    {
        var input = new Dictionary<string, object?> { ["foo"] = "bar" };
        var options = new RapidOcrOptions
        {
            Parameters = input,
        };

        input["foo"] = "mutated";
        Assert.Equal("bar", options.Parameters["foo"]);
        var dictionary = Assert.IsAssignableFrom<ReadOnlyDictionary<string, object?>>(options.Parameters);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, object?>)dictionary).Add("extra", 1));
    }

    [Fact]
    public void PaginatedPipelineOptionsRejectNonPositiveScale()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PdfPipelineOptions { ImagesScale = 0 });
    }

    [Fact]
    public void PipelineOptionsTrimWhitespaceArtifactsPath()
    {
        var options = new PdfPipelineOptions
        {
            ArtifactsPath = "   ",
        };

        Assert.Null(options.ArtifactsPath);
    }

    [Fact]
    public void PipelineOptionsRejectNonPositiveTimeout()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PdfPipelineOptions { DocumentTimeout = TimeSpan.Zero });
    }

    [Fact]
    public void PictureDescriptionOptionsRequireEndpointForApiMode()
    {
        var options = new PictureDescriptionOptions
        {
            Mode = PictureDescriptionMode.Api,
        };

        Assert.Throws<InvalidOperationException>(options.EnsureValid);
    }

    [Fact]
    public void PictureDescriptionOptionsRequireModelForVlmMode()
    {
        var options = new PictureDescriptionOptions
        {
            Mode = PictureDescriptionMode.VisionLanguageModel,
            ModelRepoId = null,
        };

        Assert.Throws<InvalidOperationException>(options.EnsureValid);
    }

    [Fact]
    public void PictureDescriptionOptionsRejectInvalidConcurrency()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PictureDescriptionOptions { ApiConcurrency = 0 });
    }

    [Fact]
    public void ThreadedPdfPipelineOptionsRejectInvalidBatchSize()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ThreadedPdfPipelineOptions { OcrBatchSize = 0 });
    }

    [Fact]
    public void ThreadedPdfPipelineOptionsValidateSuccessfully()
    {
        var options = new ThreadedPdfPipelineOptions();

        var exception = Record.Exception(options.Validate);
        Assert.Null(exception);
    }
}
