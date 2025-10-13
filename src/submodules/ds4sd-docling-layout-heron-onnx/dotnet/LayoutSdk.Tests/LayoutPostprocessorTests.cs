using System;
using System.Collections.Generic;
using LayoutSdk.Inference;
using LayoutSdk.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace LayoutSdk.Tests;

public sealed class LayoutPostprocessorTests
{
    [Fact]
    public void Postprocess_MatchesRtDetrReferenceImplementation()
    {
        var options = LayoutPostprocessOptions.CreateDefault();
        options.Labels = new[] { "Zero", "One", "Two" };
        options.LabelThresholds = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);

        var postprocessor = new LayoutPostprocessor(options);

        var logits = new[]
        {
            2.0f, -1.0f, 0.5f,
            0.1f, 3.0f, -0.5f
        };

        var boxes = new[]
        {
            0.5f, 0.5f, 0.2f, 0.2f,
            0.2f, 0.3f, 0.4f, 0.4f
        };

        var logitsShape = new[] { 1, 2, 3 };
        var boxesShape = new[] { 1, 2, 4 };
        var logitsValue = NamedOnnxValue.CreateFromTensor(
            "logits",
            new DenseTensor<float>(logits, logitsShape));
        var boxesValue = NamedOnnxValue.CreateFromTensor(
            "pred_boxes",
            new DenseTensor<float>(boxes, boxesShape));

        using var backendResult = new LayoutBackendResult(
            TensorOwner.FromNamedValue(boxesValue),
            boxesShape,
            TensorOwner.FromNamedValue(logitsValue),
            logitsShape);

        var results = postprocessor.Postprocess(backendResult, targetHeight: 100, targetWidth: 200);

        Assert.Equal(2, results.Count);

        var first = results[0];
        Assert.Equal("One", first.Label);
        Assert.InRange(first.Confidence, 0.9525f, 0.9527f);
        Assert.InRange(first.X, -1e-3f, 1e-3f);
        Assert.InRange(first.Y, 9.999f, 10.001f);
        Assert.InRange(first.Width, 79.999f, 80.001f);
        Assert.InRange(first.Height, 39.999f, 40.001f);

        var second = results[1];
        Assert.Equal("Zero", second.Label);
        Assert.InRange(second.Confidence, 0.8807f, 0.8809f);
        Assert.InRange(second.X, 79.999f, 80.001f);
        Assert.InRange(second.Y, 39.999f, 40.001f);
        Assert.InRange(second.Width, 40.0f - 1e-3f, 40.0f + 1e-3f);
        Assert.InRange(second.Height, 20.0f - 1e-3f, 20.0f + 1e-3f);
    }
}
