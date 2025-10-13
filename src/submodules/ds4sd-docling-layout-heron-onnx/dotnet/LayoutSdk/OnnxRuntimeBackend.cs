using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

internal sealed class OnnxRuntimeBackend : ILayoutBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly OnnxInputBuilder _inputBuilder;

    public OnnxRuntimeBackend(string modelPath)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 0
        };

        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
        _inputBuilder = new OnnxInputBuilder(_inputName);
    }

    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        var shape = new[] { 1, tensor.Channels, tensor.Height, tensor.Width };

        using var inputOwner = _inputBuilder.CreateInput(tensor.AsSpan(), shape);
        var results = _session.Run(new[] { inputOwner.Value });

        return ParseOutputs(results);
    }

    public void Dispose() => _session.Dispose();

    private static LayoutBackendResult ParseOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        DisposableNamedOnnxValue? scoresOutput = null;
        DisposableNamedOnnxValue? boxesOutput = null;

        TensorOwner? scoresOwner = null;
        TensorOwner? boxesOwner = null;

        try
        {
            foreach (var output in results)
            {
                if (output.Name == "logits")
                {
                    scoresOutput = output;
                }
                else if (output.Name == "pred_boxes")
                {
                    boxesOutput = output;
                }
                else
                {
                    output.Dispose();
                }
            }

            if (scoresOutput is null)
            {
                throw new InvalidOperationException("ONNX outputs do not contain 'logits'.");
            }

            if (boxesOutput is null)
            {
                scoresOutput.Dispose();
                throw new InvalidOperationException("ONNX outputs do not contain 'pred_boxes'.");
            }

            var scoresTensor = scoresOutput.AsTensor<float>();
            var boxesTensor = boxesOutput.AsTensor<float>();

            boxesOwner = TensorOwner.FromNamedValue(boxesOutput);
            scoresOwner = TensorOwner.FromNamedValue(scoresOutput);

            boxesOutput = null;
            scoresOutput = null;

            return new LayoutBackendResult(
                boxesOwner,
                boxesTensor.Dimensions.ToArray(),
                scoresOwner,
                scoresTensor.Dimensions.ToArray());
        }
        catch
        {
            scoresOwner?.Dispose();
            boxesOwner?.Dispose();
            scoresOutput?.Dispose();
            boxesOutput?.Dispose();
            throw;
        }
    }
}
