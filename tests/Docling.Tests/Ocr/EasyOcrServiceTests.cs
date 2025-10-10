using System;
using System.Collections.Generic;
using System.Reflection;
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
            ConfidenceManipulator.CreateResult("  Hello  ", new SKRect(5, 6, 25, 16), 0.95),
            ConfidenceManipulator.CreateResult("World", new SKRect(30, 10, 38, 20), 0.80),
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
            if (ConfidenceManipulator.IsSupported)
            {
                Assert.InRange(lines[0].Confidence, 0.94, 0.96);
            }
            else
            {
                Assert.Equal(1.0d, lines[0].Confidence);
            }

            Assert.Equal("World", lines[1].Text);
            Assert.Equal(Math.Floor(region.Left) + 30, lines[1].BoundingBox.Left);
            if (ConfidenceManipulator.IsSupported)
            {
                Assert.InRange(lines[1].Confidence, 0.79, 0.81);
            }
            else
            {
                Assert.Equal(1.0d, lines[1].Confidence);
            }
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

    [Fact]
    public async Task RecognizeAsyncFiltersLinesBelowConfidenceThreshold()
    {
        if (!ConfidenceManipulator.IsSupported)
        {
            return;
        }

        var options = new EasyOcrOptions { Languages = new[] { "en" }, ConfidenceThreshold = 0.75 };
        using var bitmap = new SKBitmap(new SKImageInfo(120, 90));
        var region = BoundingBox.FromSize(0, 0, 100, 80);
        var request = new OcrRequest(new PageReference(1, 300), bitmap, region, new Dictionary<string, string>());

        using var engine = new StubEngine(new[]
        {
            ConfidenceManipulator.CreateResult("Keep", new SKRect(0, 0, 20, 10), 0.90),
            ConfidenceManipulator.CreateResult("Drop", new SKRect(0, 10, 20, 20), 0.40),
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

            Assert.Single(lines);
            Assert.Equal("Keep", lines[0].Text);
            Assert.InRange(lines[0].Confidence, 0.89, 0.91);
        }
        finally
        {
            service.Dispose();
        }
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

    private static class ConfidenceManipulator
    {
        private static readonly Func<OcrResult, double, OcrResult>? Factory = CreateFactory();

        public static bool IsSupported => Factory is not null;

        public static OcrResult CreateResult(string text, SKRect boundingBox, double confidence)
        {
            var result = new OcrResult(text, boundingBox);
            if (Factory is null)
            {
                return result;
            }

            return Factory(result, confidence);
        }

        private static Func<OcrResult, double, OcrResult>? CreateFactory()
        {
            var type = typeof(OcrResult);
            var property = type.GetProperty("Confidence", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (property is not null)
            {
                if (property.CanWrite)
                {
                    return (result, value) =>
                    {
                        property.SetValue(result, Convert(value, property.PropertyType));
                        return result;
                    };
                }

                var backingField = type.GetField("<Confidence>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic);
                if (backingField is not null)
                {
                    return (result, value) =>
                    {
                        backingField.SetValue(result, Convert(value, backingField.FieldType));
                        return result;
                    };
                }
            }

            var candidates = new[] { "score", "_score", "probability", "_probability", "confidence", "_confidence" };
            foreach (var name in candidates)
            {
                var field = type.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (field is null)
                {
                    continue;
                }

                return (result, value) =>
                {
                    field.SetValue(result, Convert(value, field.FieldType));
                    return result;
                };
            }

            return null;
        }

        private static object Convert(double value, Type targetType)
        {
            if (targetType == typeof(double))
            {
                return value;
            }

            if (targetType == typeof(float))
            {
                return (float)value;
            }

            if (targetType == typeof(decimal))
            {
                return (decimal)value;
            }

            if (targetType == typeof(int))
            {
                return (int)Math.Round(value);
            }

            if (targetType == typeof(uint))
            {
                return (uint)Math.Max(0, Math.Round(value));
            }

            if (targetType == typeof(short))
            {
                return (short)Math.Round(value);
            }

            if (targetType == typeof(ushort))
            {
                return (ushort)Math.Max(0, Math.Round(value));
            }

            if (targetType == typeof(byte))
            {
                return (byte)Math.Max(byte.MinValue, Math.Min(byte.MaxValue, Math.Round(value * 255d)));
            }

            if (targetType == typeof(sbyte))
            {
                return (sbyte)Math.Clamp(Math.Round(value * 127d), sbyte.MinValue, sbyte.MaxValue);
            }

            try
            {
                return System.Convert.ChangeType(value, targetType, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch
            {
                return value;
            }
        }
    }
}
