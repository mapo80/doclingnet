using LayoutSdk;
using LayoutSdk.Configuration;
using LayoutSdk.Factories;
using LayoutSdk.Inference;
using LayoutSdk.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.IO;
using Xunit;

namespace LayoutSdk.Tests;

file static class BackendResultFactory
{
    public static LayoutBackendResult Create(float[] logits, int[] logitsShape, float[] boxes, int[] boxesShape)
    {
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
}

file class FakeBackend : ILayoutBackend
{
    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        var logits = new float[17];
        Array.Fill(logits, -10f);
        logits[9] = 5f; // "Text"

        var boxes = new[] { 0.5f, 0.5f, 1f, 1f };
        return BackendResultFactory.Create(
            logits,
            new[] { 1, 1, logits.Length },
            boxes,
            new[] { 1, 1, 4 });
    }
}

file class FakeBackendFactory : ILayoutBackendFactory
{
    private readonly ILayoutBackend _backend = new FakeBackend();

    public ILayoutBackend Create(LayoutRuntime runtime) => _backend;
}

file sealed class DisposableBackend : ILayoutBackend, IDisposable
{
    public bool Disposed { get; private set; }

    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        var logits = new float[17];
        Array.Fill(logits, -10f);
        var boxes = new[] { 0.5f, 0.5f, 0.5f, 0.5f };
        return BackendResultFactory.Create(
            logits,
            new[] { 1, 1, logits.Length },
            boxes,
            new[] { 1, 1, 4 });
    }

    public void Dispose() => Disposed = true;
}

file sealed class CountingBackendFactory : ILayoutBackendFactory
{
    private readonly Func<LayoutRuntime, ILayoutBackend> _factory;
    public int Created { get; private set; }

    public CountingBackendFactory(Func<LayoutRuntime, ILayoutBackend> factory)
    {
        _factory = factory;
    }

    public ILayoutBackend Create(LayoutRuntime runtime)
    {
        Created++;
        return _factory(runtime);
    }
}

public class LayoutSdkTests
{
    private static LayoutSdk CreateSdkWithFakeBackend()
    {
        var options = new LayoutSdkOptions(
            "onnx-path",
            DocumentLanguage.Italian);
        return new LayoutSdk(options, new FakeBackendFactory());
    }

    [Fact]
    public void Process_EmptyPath_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<ArgumentException>(() => sdk.Process("", false, LayoutRuntime.Onnx));
    }

    [Fact]
    public void Process_MissingImage_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<FileNotFoundException>(() => sdk.Process("missing.png", false, LayoutRuntime.Onnx));
    }

    private static string SampleImage =>
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..","..","..","..","..","dataset", "gazette_de_france.jpg"));

    [Fact]
    public void Process_Overlay_ReturnsBitmap()
    {
        var sdk = CreateSdkWithFakeBackend();
        var result = sdk.Process(SampleImage, true, LayoutRuntime.Onnx);
        Assert.NotNull(result.OverlayImage);
        Assert.Single(result.Boxes);
        Assert.Equal(DocumentLanguage.Italian, result.Language);
        Assert.True(result.Metrics.OverlayDuration >= TimeSpan.Zero);
        Assert.True(result.Metrics.PreprocessDuration >= TimeSpan.Zero);
        Assert.True(result.Metrics.InferenceDuration >= TimeSpan.Zero);
        Assert.Equal(result.Metrics.PreprocessDuration + result.Metrics.InferenceDuration + result.Metrics.OverlayDuration, result.Metrics.TotalDuration);
    }

    [Fact]
    public void Process_NoOverlay_ReturnsNull()
    {
        var sdk = CreateSdkWithFakeBackend();
        var result = sdk.Process(SampleImage, false, LayoutRuntime.Onnx);
        Assert.Null(result.OverlayImage);
        Assert.Equal(TimeSpan.Zero, result.Metrics.OverlayDuration);
    }

    [Fact]
    public void Process_ReusesPipelinePerRuntime()
    {
        var backend = new DisposableBackend();
        var factory = new CountingBackendFactory(_ => backend);
        var options = new LayoutSdkOptions("onnx");
        var sdk = new LayoutSdk(options, factory);

        try
        {
            var path = SampleImage;
            sdk.Process(path, false, LayoutRuntime.Onnx);
            sdk.Process(path, false, LayoutRuntime.Onnx);
            Assert.Equal(1, factory.Created);
        }
        finally
        {
            sdk.Dispose();
        }
    }

    [Fact]
    public void Process_CreatesPipelinePerRuntime()
    {
        var factory = new CountingBackendFactory(runtime => runtime switch
        {
            LayoutRuntime.Onnx => new DisposableBackend(),
            _ => throw new NotSupportedException("Only ONNX runtime is supported")
        });

        var options = new LayoutSdkOptions("onnx");
        var sdk = new LayoutSdk(options, factory);

        try
        {
            var path = SampleImage;
            sdk.Process(path, false, LayoutRuntime.Onnx);
            Assert.Equal(1, factory.Created);
        }
        finally
        {
            sdk.Dispose();
        }
    }

    [Fact]
    public void Dispose_DisposesCreatedPipelines()
    {
        var backend = new DisposableBackend();
        var factory = new CountingBackendFactory(_ => backend);
        var options = new LayoutSdkOptions("onnx");
        var sdk = new LayoutSdk(options, factory);

        sdk.Process(SampleImage, false, LayoutRuntime.Onnx);
        sdk.Dispose();

        Assert.True(backend.Disposed);
    }

    [Fact]
    public void OptionsEnsureModelPathsThrowsWhenMissing()
    {
        var options = new LayoutSdkOptions(
            "missing",
            validateModelPaths: true);
        Assert.Throws<FileNotFoundException>(() => options.EnsureModelPaths());
    }

    [Fact]
    public void OptionsEnsureModelPathsSucceedsWhenFilesExist()
    {
        var onnx = Path.GetTempFileName();
        try
        {
            var options = new LayoutSdkOptions(
                onnx,
                defaultLanguage: DocumentLanguage.English,
                validateModelPaths: true);
            options.EnsureModelPaths();
        }
        finally
        {
            File.Delete(onnx);
        }
    }

}
