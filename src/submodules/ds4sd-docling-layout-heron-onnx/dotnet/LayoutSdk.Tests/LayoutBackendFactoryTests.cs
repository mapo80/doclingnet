using System;
using LayoutSdk;
using LayoutSdk.Configuration;
using LayoutSdk.Factories;
using LayoutSdk.Inference;
using LayoutSdk.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace LayoutSdk.Tests;

public class LayoutBackendFactoryTests
{
    [Fact]
    public void Constructor_NullOptions_Throws()
    {
        Assert.Throws<System.ArgumentNullException>(() => new LayoutBackendFactory(null!));
    }

    private sealed class StubBackend : ILayoutBackend, IDisposable
    {
        public bool Disposed { get; private set; }

        public LayoutBackendResult Infer(ImageTensor tensor)
        {
            var logits = new float[17];
            Array.Fill(logits, -10f);
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

    [Fact]
    public void Create_Onnx_ReturnsBackend()
    {
        var options = new LayoutSdkOptions("onnx");
        var stub = new StubBackend();
        var factory = new LayoutBackendFactory(options, path => stub);
        var backend = factory.Create(LayoutRuntime.Onnx);
        Assert.Same(stub, backend);
    }

    [Fact]
    public void Create_InvalidRuntime_Throws()
    {
        var options = new LayoutSdkOptions("onnx");
        var factory = new LayoutBackendFactory(options);
        Assert.Throws<System.NotSupportedException>(() => factory.Create((LayoutRuntime)999));
    }
}
