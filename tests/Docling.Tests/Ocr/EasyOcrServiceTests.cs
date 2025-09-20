using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Ocr;
using Docling.Pipelines.Options;
using EasyOcrNet;
using EasyOcrNet.Languages;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;

namespace Docling.Tests.Ocr;

public sealed class EasyOcrServiceTests
{
    [Fact]
    public async Task RecognizeAsyncMapsResultsToPageSpace()
    {
        var options = new EasyOcrOptions { Languages = new[] { "en" } };
        using var bitmap = new SKBitmap(new SKImageInfo(100, 80));
        var region = BoundingBox.FromSize(10.2, 20.4, 40.5, 30.6);
        var request = new OcrRequest(new PageReference(1, 300), bitmap, region, new Dictionary<string, string>());

        using var engine = new StubEngine(new[]
        {
            new OcrResult("  Hello  ", new SKRect(5, 6, 25, 16)),
            new OcrResult("World", new SKRect(30, 10, 38, 20)),
        });

        var service = new EasyOcrService(
            options,
            NullLogger<EasyOcrService>.Instance,
            engine,
            AppContext.BaseDirectory,
            OcrLanguage.English,
            InferenceBackend.Onnx,
            "CPU");

        try
        {
            var lines = await CollectAsync(service, request);

            Assert.Equal(2, lines.Count);
            Assert.Equal("Hello", lines[0].Text);
            var expectedLeft = Math.Floor(region.Left) + 5;
            var expectedTop = Math.Floor(region.Top) + 6;
            Assert.Equal(expectedLeft, lines[0].BoundingBox.Left);
            Assert.Equal(expectedTop, lines[0].BoundingBox.Top);
            Assert.Equal(20, Math.Round(lines[0].BoundingBox.Width));
            Assert.Equal(10, Math.Round(lines[0].BoundingBox.Height));
            Assert.Equal(1.0d, lines[0].Confidence);

            Assert.Equal("World", lines[1].Text);
            Assert.Equal(Math.Floor(region.Left) + 30, lines[1].BoundingBox.Left);
        }
        finally
        {
            service.Dispose();
        }

        Assert.True(engine.DisposeCalled);
        Assert.Equal(41, engine.LastBitmapWidth);
        Assert.Equal(31, engine.LastBitmapHeight);
    }

    [Fact]
    public async Task RecognizeAsyncThrowsWhenRegionOutsideImage()
    {
        var options = new EasyOcrOptions { Languages = new[] { "en" } };
        using var bitmap = new SKBitmap(new SKImageInfo(50, 50));
        var request = new OcrRequest(new PageReference(1, 300), bitmap, BoundingBox.FromSize(200, 200, 10, 10), new Dictionary<string, string>());

        using var engine = new StubEngine(Array.Empty<OcrResult>());
        using var service = new EasyOcrService(
            options,
            NullLogger<EasyOcrService>.Instance,
            engine,
            AppContext.BaseDirectory,
            OcrLanguage.English,
            InferenceBackend.Onnx,
            "CPU");

        await Assert.ThrowsAsync<ArgumentException>(() => CollectAsync(service, request));
    }

    [Fact]
    public async Task RecognizeAsyncThrowsAfterDispose()
    {
        var options = new EasyOcrOptions { Languages = new[] { "en" } };
        using var bitmap = new SKBitmap(new SKImageInfo(20, 20));
        var request = new OcrRequest(new PageReference(1, 300), bitmap, BoundingBox.FromSize(0, 0, 10, 10), new Dictionary<string, string>());

        using var engine = new StubEngine(Array.Empty<OcrResult>());
        var service = new EasyOcrService(
            options,
            NullLogger<EasyOcrService>.Instance,
            engine,
            AppContext.BaseDirectory,
            OcrLanguage.English,
            InferenceBackend.Onnx,
            "CPU");

        service.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => CollectAsync(service, request));
        Assert.True(engine.DisposeCalled);
    }

    private static async Task<List<OcrLine>> CollectAsync(EasyOcrService service, OcrRequest request)
    {
        var results = new List<OcrLine>();
        await foreach (var line in service.RecognizeAsync(request))
        {
            results.Add(line);
        }

        return results;
    }

    private sealed class StubEngine : EasyOcrService.IEasyOcrEngine
    {
        private readonly IReadOnlyList<OcrResult> _results;

        public StubEngine(IReadOnlyList<OcrResult> results)
        {
            _results = results;
        }

        public bool DisposeCalled { get; private set; }

        public int LastBitmapWidth { get; private set; }

        public int LastBitmapHeight { get; private set; }

        public IReadOnlyList<OcrResult> Read(SKBitmap image)
        {
            LastBitmapWidth = image.Width;
            LastBitmapHeight = image.Height;
            return _results;
        }

        public void Dispose()
        {
            DisposeCalled = true;
        }
    }
}
