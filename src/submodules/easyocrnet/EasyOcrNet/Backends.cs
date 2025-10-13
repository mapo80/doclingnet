using System;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenVinoSharp;
using OvTensor = OpenVinoSharp.Tensor;
using OvShape = OpenVinoSharp.Shape;

namespace EasyOcrNet;

internal readonly record struct ModelOutput(float[] Data, int[] Shape)
{
    public int Rank => Shape.Length;
    public int this[int index] => Shape[index];
}

internal interface IOcrBackend : IDisposable
{
    ModelOutput RunDetector(DenseTensor<float> input);
    ModelOutput RunRecognizer(DenseTensor<float> input);
    string Provider { get; }
}

internal sealed class OnnxBackend : IOcrBackend
{
    private readonly InferenceSession _detector;
    private readonly InferenceSession _recognizer;
    private readonly string _detectorInput;
    private readonly string _recognizerInput;

    public string Provider { get; }

    public OnnxBackend(string detectorPath, string recognizerPath)
    {
        if (!File.Exists(detectorPath))
            throw new FileNotFoundException($"Detector model not found at '{detectorPath}'", detectorPath);
        if (!File.Exists(recognizerPath))
            throw new FileNotFoundException($"Recognizer model not found at '{recognizerPath}'", recognizerPath);

        _detector = new InferenceSession(detectorPath);
        _recognizer = new InferenceSession(recognizerPath);

        _detectorInput = _detector.InputMetadata.Keys.Single();
        _recognizerInput = _recognizer.InputMetadata.Keys.Single();

        Provider = "ONNXRuntime";
    }

    public ModelOutput RunDetector(DenseTensor<float> input)
    {
        using var outputs = _detector.Run(new[] { NamedOnnxValue.CreateFromTensor(_detectorInput, input) });
        var tensor = outputs[0].AsTensor<float>();
        return new ModelOutput(tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    public ModelOutput RunRecognizer(DenseTensor<float> input)
    {
        using var outputs = _recognizer.Run(new[] { NamedOnnxValue.CreateFromTensor(_recognizerInput, input) });
        var tensor = outputs[0].AsTensor<float>();
        return new ModelOutput(tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    public void Dispose()
    {
        _detector.Dispose();
        _recognizer.Dispose();
    }
}

internal sealed class OpenVinoBackend : IOcrBackend
{
    private readonly Core _core;
    private readonly CompiledModel _detector;
    private readonly CompiledModel _recognizer;
    private readonly long[] _detectorInputDims = { 1, 3, 608, 800 };
    private readonly long[] _recognizerInputDims = { 1, 1, 64, 1000 };

    public string Provider { get; }

    public OpenVinoBackend(string detectorXmlPath, string recognizerXmlPath, string device)
    {
        var detectorBin = Path.ChangeExtension(detectorXmlPath, ".bin");
        var recognizerBin = Path.ChangeExtension(recognizerXmlPath, ".bin");

        if (!File.Exists(detectorXmlPath))
            throw new FileNotFoundException($"Detector XML not found at '{detectorXmlPath}'", detectorXmlPath);
        if (!File.Exists(detectorBin))
            throw new FileNotFoundException($"Detector BIN not found at '{detectorBin}'", detectorBin);
        if (!File.Exists(recognizerXmlPath))
            throw new FileNotFoundException($"Recognizer XML not found at '{recognizerXmlPath}'", recognizerXmlPath);
        if (!File.Exists(recognizerBin))
            throw new FileNotFoundException($"Recognizer BIN not found at '{recognizerBin}'", recognizerBin);

        try
        {
            _core = new Core();
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("OpenVINO native runtime could not be loaded. Ensure the OpenVINO runtime libraries are installed and available on PATH.", ex);
        }

        _detector = CompileModel(detectorXmlPath, detectorBin, _detectorInputDims, device);
        _recognizer = CompileModel(recognizerXmlPath, recognizerBin, _recognizerInputDims, device);

        Provider = $"OpenVINO:{device}";
    }

    private CompiledModel CompileModel(string xmlPath, string binPath, long[] inputShape, string device)
    {
        using var model = _core.read_model(xmlPath, binPath);
        using var shape = new OvShape(inputShape);
        var partial = new PartialShape(shape);
        model.reshape(partial);
        return _core.compile_model(model, device);
    }

    public ModelOutput RunDetector(DenseTensor<float> input)
    {
        return Run(_detector, input, _detectorInputDims);
    }

    public ModelOutput RunRecognizer(DenseTensor<float> input)
    {
        return Run(_recognizer, input, _recognizerInputDims);
    }

    private static ModelOutput Run(CompiledModel model, DenseTensor<float> tensor, long[] shape)
    {
        using var request = model.create_infer_request();
        using var ovShape = new OvShape(shape);
        using var ovTensor = new OvTensor(ovShape, tensor.ToArray());
        request.set_input_tensor(ovTensor);
        request.infer();
        using var outputTensor = request.get_output_tensor();
        var outputData = outputTensor.get_data<float>((int)outputTensor.get_size());
        var outputShape = outputTensor.get_shape().Select(dim => (int)dim).ToArray();
        return new ModelOutput(outputData, outputShape);
    }

    public void Dispose()
    {
        _detector.Dispose();
        _recognizer.Dispose();
        _core.Dispose();
    }
}
