using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Primitives;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Preprocessing;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using Xunit;

namespace Docling.Tests.Preprocessing;

public sealed class DefaultPagePreprocessorTests
{
    [Fact]
    public async Task PreprocessAsyncScalesToTargetDpi()
    {
        using var bitmap = CreateSolidBitmap(100, 50, SKColors.White);
        var metadata = new PageImageMetadata("page-1", "sample", "image/png", new Dictionary<string, string>
        {
            [PageImageMetadataKeys.SourceHorizontalDpi] = "150",
        });
        using var pageImage = new PageImage(new PageReference(0, 150), bitmap.Copy() ?? throw new Xunit.Sdk.XunitException("Copy failed."), metadata);
        var options = new PreprocessingOptions
        {
            TargetDpi = 300,
            EnableDeskew = false,
            NormalizeContrast = false,
            ColorMode = PageColorMode.Preserve,
        };

        var preprocessor = new DefaultPagePreprocessor(options, NullLogger<DefaultPagePreprocessor>.Instance);
        using var processed = await preprocessor.PreprocessAsync(pageImage);

        processed.Width.Should().Be(200);
        processed.Height.Should().Be(100);
        processed.Page.Dpi.Should().Be(300);
        processed.Metadata.Properties.Should().ContainKey(PageImageMetadataKeys.ScaleFactor);
        double.Parse(processed.Metadata.Properties[PageImageMetadataKeys.ScaleFactor], CultureInfo.InvariantCulture).Should().BeApproximately(2d, 0.01d);
        processed.Metadata.Properties[PageImageMetadataKeys.ColorMode].Should().Be(PageColorMode.Preserve.ToString());
        processed.Metadata.Properties[PageImageMetadataKeys.DeskewAngle].Should().Be("0.00");
    }

    [Fact]
    public async Task PreprocessAsyncConvertsToGrayscale()
    {
        using var bitmap = CreateSolidBitmap(1, 1, new SKColor(255, 0, 0));
        using var pageImage = new PageImage(new PageReference(0, 300), bitmap.Copy() ?? throw new Xunit.Sdk.XunitException("Copy failed."), PageImageMetadata.Empty);
        var options = new PreprocessingOptions
        {
            TargetDpi = 300,
            EnableDeskew = false,
            NormalizeContrast = false,
            ColorMode = PageColorMode.Grayscale,
        };

        var preprocessor = new DefaultPagePreprocessor(options, NullLogger<DefaultPagePreprocessor>.Instance);
        using var processed = await preprocessor.PreprocessAsync(pageImage);

        var pixel = processed.Bitmap.GetPixel(0, 0);
        pixel.Red.Should().Be(pixel.Green).And.Be(pixel.Blue);
        processed.Metadata.Properties[PageImageMetadataKeys.ColorMode].Should().Be(PageColorMode.Grayscale.ToString());
    }

    [Fact]
    public async Task PreprocessAsyncBinarizesUsingOtsu()
    {
        using var bitmap = new SKBitmap(new SKImageInfo(4, 1, SKColorType.Rgba8888, SKAlphaType.Premul));
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
            using var paint = new SKPaint { Color = new SKColor(0, 0, 0) };
            canvas.DrawRect(new SKRect(0, 0, 2, 1), paint);
        }

        using var pageImage = new PageImage(new PageReference(0, 300), bitmap.Copy() ?? throw new Xunit.Sdk.XunitException("Copy failed."), PageImageMetadata.Empty);
        var options = new PreprocessingOptions
        {
            TargetDpi = 300,
            EnableDeskew = false,
            NormalizeContrast = false,
            ColorMode = PageColorMode.Binary,
        };

        var preprocessor = new DefaultPagePreprocessor(options, NullLogger<DefaultPagePreprocessor>.Instance);
        using var processed = await preprocessor.PreprocessAsync(pageImage);

        processed.Metadata.Properties[PageImageMetadataKeys.ColorMode].Should().Be(PageColorMode.Binary.ToString());
        var left = processed.Bitmap.GetPixel(0, 0);
        var right = processed.Bitmap.GetPixel(3, 0);
        left.Red.Should().Be(0);
        right.Red.Should().Be(255);
    }

    [Fact]
    public async Task PreprocessAsyncDeskewsWhenAngleExceedsThreshold()
    {
        using var bitmap = new SKBitmap(new SKImageInfo(200, 200, SKColorType.Rgba8888, SKAlphaType.Premul));
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
            canvas.Translate(bitmap.Width / 2f, bitmap.Height / 2f);
            canvas.RotateDegrees(6f);
            using var paint = new SKPaint { Color = SKColors.Black, StrokeWidth = 6f };
            canvas.DrawLine(-80, 0, 80, 0, paint);
        }

        using var pageImage = new PageImage(new PageReference(0, 300), bitmap.Copy() ?? throw new Xunit.Sdk.XunitException("Copy failed."), PageImageMetadata.Empty);
        var options = new PreprocessingOptions
        {
            TargetDpi = 300,
            EnableDeskew = true,
            NormalizeContrast = false,
            DeskewMinimumAngle = 0.5,
            DeskewMaxAngle = 15,
        };

        var preprocessor = new DefaultPagePreprocessor(options, NullLogger<DefaultPagePreprocessor>.Instance);
        using var processed = await preprocessor.PreprocessAsync(pageImage);

        var applied = double.Parse(processed.Metadata.Properties[PageImageMetadataKeys.DeskewAngle], CultureInfo.InvariantCulture);
        applied.Should().BeGreaterThan(options.DeskewMinimumAngle);
        applied.Should().BeLessThan(options.DeskewMaxAngle + 0.1);
        processed.Width.Should().BeGreaterThan(bitmap.Width);
        processed.Metadata.Properties[PageImageMetadataKeys.ScaleFactor].Should().Be("1.0000");
    }

    [Fact]
    public void EnsureValidThrowsWhenMinimumAngleExceedsMaximum()
    {
        var options = new PreprocessingOptions
        {
            DeskewMaxAngle = 2,
            DeskewMinimumAngle = 5,
        };

        var act = () => options.EnsureValid();
        act.Should().Throw<InvalidOperationException>();
    }

    private static SKBitmap CreateSolidBitmap(int width, int height, SKColor color)
    {
        var bitmap = new SKBitmap(new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Premul));
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(color);
        return bitmap;
    }
}

public sealed class PagePreprocessingStageTests
{
    [Fact]
    public async Task ExecuteAsyncUpdatesStoreAndMarksCompletion()
    {
        using var store = new PageImageStore();
        var pageReference = new PageReference(0, 150);
        var metadata = new PageImageMetadata("page-1", "sample", "image/png", new Dictionary<string, string>());
        using var page = CreateStagePageImage(pageReference, metadata);

        store.Add(page);
        var preprocessor = new RecordingPreprocessor();
        var stage = new PagePreprocessingStage(preprocessor, NullLogger<PagePreprocessingStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { pageReference });

        await stage.ExecuteAsync(context, default);

        preprocessor.Processed.Should().Equal(0);
        context.GetRequired<bool>(PipelineContextKeys.PreprocessingCompleted).Should().BeTrue();
        using var normalized = store.Rent(pageReference);
        normalized.Page.Dpi.Should().Be(400);
        normalized.Metadata.Properties.Should().ContainKey("recording");
    }

    [Fact]
    public async Task ExecuteAsyncSkipsWhenNoPages()
    {
        using var store = new PageImageStore();
        var stage = new PagePreprocessingStage(new RecordingPreprocessor(), NullLogger<PagePreprocessingStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference>());

        await stage.ExecuteAsync(context, default);

        context.GetRequired<bool>(PipelineContextKeys.PreprocessingCompleted).Should().BeTrue();
    }

    [Fact]
    public async Task ExecuteAsyncThrowsWhenStoreMissing()
    {
        var stage = new PagePreprocessingStage(new RecordingPreprocessor(), NullLogger<PagePreprocessingStage>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageSequence, new List<PageReference> { new PageReference(0, 72) });

        await Assert.ThrowsAsync<InvalidOperationException>(() => stage.ExecuteAsync(context, default));
    }

    private sealed class RecordingPreprocessor : IPagePreprocessor
    {
        public List<int> Processed { get; } = new();

        public Task<PageImage> PreprocessAsync(PageImage pageImage, CancellationToken cancellationToken = default)
        {
            Processed.Add(pageImage.Page.PageNumber);
            using var clone = pageImage.Clone();
            var metadata = pageImage.Metadata.WithAdditionalProperties(new[]
            {
                new KeyValuePair<string, string>("recording", "true"),
            });
            var bitmapCopy = clone.Bitmap.Copy() ?? throw new Xunit.Sdk.XunitException("Clone copy failed.");
            var updated = new PageImage(pageImage.Page.WithDpi(400), bitmapCopy, metadata);
            return Task.FromResult(updated);
        }
    }

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "Ownership transferred to PageImage")]
    private static PageImage CreateStagePageImage(PageReference pageReference, PageImageMetadata metadata)
    {
        var bitmap = new SKBitmap(new SKImageInfo(10, 10, SKColorType.Rgba8888, SKAlphaType.Premul));
        try
        {
            using var canvas = new SKCanvas(bitmap);
            canvas.Clear(SKColors.White);
            return new PageImage(pageReference, bitmap, metadata);
        }
        catch
        {
            bitmap.Dispose();
            throw;
        }
    }
}
