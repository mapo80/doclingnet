using LayoutSdk.Inference;
using LayoutSdk.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using Xunit;
using LayoutSdk.Configuration;

namespace LayoutSdk.Tests;

public class LayoutPipelineTests
{
    private sealed class RecordingBackend : ILayoutBackend, IDisposable
    {
        public bool Disposed { get; private set; }
        public bool Called { get; private set; }

        public LayoutBackendResult Infer(ImageTensor tensor)
        {
            Called = true;
            var logits = new float[17];
            Array.Fill(logits, -10f);
            logits[9] = 5f; // Favor the "Text" class.

            var boxes = new[] { 0.5f, 0.5f, 0.5f, 0.5f };
            var logitsShape = new[] { 1, 1, logits.Length };
            var boxesShape = new[] { 1, 1, 4 };

            var logitsValue = NamedOnnxValue.CreateFromTensor(
                "logits",
                new DenseTensor<float>(logits, logitsShape));

            var boxesValue = NamedOnnxValue.CreateFromTensor(
                "pred_boxes",
                new DenseTensor<float>(boxes, boxesShape));

            return new LayoutBackendResult(
                TensorOwner.FromNamedValue(boxesValue),
                boxesShape,
                TensorOwner.FromNamedValue(logitsValue),
                logitsShape);
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }

    private sealed class RecordingPreprocessor : IImagePreprocessor
    {
        public bool Called { get; private set; }

        public ImageTensor Preprocess(SKBitmap image)
        {
            Called = true;
            return ImageTensor.Rent(image.Width, image.Height, 3);
        }
    }

    [Fact]
    public void Execute_NullImage_Throws()
    {
        var pipeline = new LayoutPipeline(
            new RecordingBackend(),
            new RecordingPreprocessor(),
            new LayoutPostprocessor(LayoutPostprocessOptions.CreateDefault()));
        Assert.Throws<ArgumentNullException>(() => pipeline.Execute(null!));
    }

    [Fact]
    public void Execute_RunsPreprocessAndInference()
    {
        using var image = new SKBitmap(2, 2);
        var backend = new RecordingBackend();
        var preprocessor = new RecordingPreprocessor();
        var pipeline = new LayoutPipeline(
            backend,
            preprocessor,
            new LayoutPostprocessor(LayoutPostprocessOptions.CreateDefault()));

        var result = pipeline.Execute(image);
        Assert.True(preprocessor.Called);
        Assert.True(backend.Called);
        Assert.Single(result.Boxes);
        Assert.True(result.Metrics.PreprocessDuration >= TimeSpan.Zero);
        Assert.True(result.Metrics.InferenceDuration >= TimeSpan.Zero);
        Assert.Equal(TimeSpan.Zero, result.Metrics.OverlayDuration);
        Assert.Equal(result.Metrics.PreprocessDuration + result.Metrics.InferenceDuration, result.Metrics.TotalDuration);
    }

    [Fact]
    public void Dispose_DisposesBackendOnce()
    {
        var backend = new RecordingBackend();
        var pipeline = new LayoutPipeline(
            backend,
            new RecordingPreprocessor(),
            new LayoutPostprocessor(LayoutPostprocessOptions.CreateDefault()));
        pipeline.Dispose();
        pipeline.Dispose();
        Assert.True(backend.Disposed);
    }
}
