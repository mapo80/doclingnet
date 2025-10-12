using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Docling.Core.Models.Tables;

/// <summary>
/// ONNX session wrappers for the 4-component TableFormer architecture.
/// This replaces the old single-model approach with a component-wise design.
/// </summary>
internal sealed class TableFormerOnnxComponents : IDisposable
{
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _tagTransformerEncoderSession;
    private readonly InferenceSession _tagTransformerDecoderStepSession;
    private readonly InferenceSession _bboxDecoderSession;

    private readonly string _encoderInputName;
    private readonly string _tagTransformerEncoderInputName;
    private readonly string _tagTransformerDecoderStepInputNames;
    private readonly string _bboxDecoderInputNames;

    public TableFormerOnnxComponents(string modelsDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelsDirectory))
        {
            throw new ArgumentException("Models directory cannot be empty", nameof(modelsDirectory));
        }

        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 1
        };

        // Initialize encoder session - use new path in submodules
        var encoderPath = Path.Combine(modelsDirectory, "tableformer_fast_encoder.onnx");
        _encoderSession = new InferenceSession(encoderPath, sessionOptions);
        _encoderInputName = _encoderSession.InputMetadata.Keys.First();

        // Initialize tag transformer encoder session
        var tagTransformerEncoderPath = Path.Combine(modelsDirectory, "tableformer_fast_tag_transformer_encoder.onnx");
        _tagTransformerEncoderSession = new InferenceSession(tagTransformerEncoderPath, sessionOptions);
        _tagTransformerEncoderInputName = _tagTransformerEncoderSession.InputMetadata.Keys.First();

        // Initialize tag transformer decoder step session
        var tagTransformerDecoderStepPath = Path.Combine(modelsDirectory, "tableformer_fast_tag_transformer_decoder_step.onnx");
        _tagTransformerDecoderStepSession = new InferenceSession(tagTransformerDecoderStepPath, sessionOptions);
        _tagTransformerDecoderStepInputNames = string.Join(",",
            _tagTransformerDecoderStepSession.InputMetadata.Keys);

        // Initialize bbox decoder session
        var bboxDecoderPath = Path.Combine(modelsDirectory, "tableformer_fast_bbox_decoder.onnx");
        _bboxDecoderSession = new InferenceSession(bboxDecoderPath, sessionOptions);
        _bboxDecoderInputNames = string.Join(",",
            _bboxDecoderSession.InputMetadata.Keys);

        sessionOptions.Dispose();
    }

    /// <summary>
    /// Encoder: (1,3,448,448) → (1,28,28,256)
    /// </summary>
    public DenseTensor<float> RunEncoder(DenseTensor<float> input)
    {
        var inputNamed = NamedOnnxValue.CreateFromTensor(_encoderInputName, input);
        using var results = _encoderSession.Run(new[] { inputNamed });
        (inputNamed as IDisposable)?.Dispose();

        var outputName = _encoderSession.OutputMetadata.Keys.First();
        var outputTensor = results.First(x => x.Name == outputName).AsTensor<float>();
        return outputTensor.ToDenseTensor();
    }

    /// <summary>
    /// Tag Transformer Encoder: (1,28,28,256) → (784,1,512)
    /// </summary>
    public DenseTensor<float> RunTagTransformerEncoder(DenseTensor<float> encoderOutput)
    {
        var inputNamed = NamedOnnxValue.CreateFromTensor(_tagTransformerEncoderInputName, encoderOutput);
        using var results = _tagTransformerEncoderSession.Run(new[] { inputNamed });
        (inputNamed as IDisposable)?.Dispose();

        var outputName = _tagTransformerEncoderSession.OutputMetadata.Keys.First();
        var outputTensor = results.First(x => x.Name == outputName).AsTensor<float>();
        return outputTensor.ToDenseTensor();
    }

    /// <summary>
    /// Tag Transformer Decoder Step: tags + memory → logits + hidden state
    /// </summary>
    public (DenseTensor<float> logits, DenseTensor<float> hiddenState) RunTagTransformerDecoderStep(
        DenseTensor<long> decodedTags,
        DenseTensor<float> memory,
        DenseTensor<bool> encoderMask)
    {
        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("decoded_tags", decodedTags),
            NamedOnnxValue.CreateFromTensor("memory", memory),
            NamedOnnxValue.CreateFromTensor("encoder_mask", encoderMask)
        };

        using var results = _tagTransformerDecoderStepSession.Run(inputs);
        foreach (var input in inputs)
        {
            (input as IDisposable)?.Dispose();
        }

        var logitsTensor = results.First(x => x.Name == "logits").AsTensor<float>();
        var hiddenStateTensor = results.First(x => x.Name == "tag_hidden").AsTensor<float>();

        return (logitsTensor.ToDenseTensor(), hiddenStateTensor.ToDenseTensor());
    }

    /// <summary>
    /// BBox Decoder: encoder features + tag hidden states → bbox classes + coordinates
    /// </summary>
    public (DenseTensor<float> bboxClasses, DenseTensor<float> bboxCoords) RunBboxDecoder(
        DenseTensor<float> encoderOutput,
        DenseTensor<float> tagHiddenStates)
    {
        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encoderOutput),
            NamedOnnxValue.CreateFromTensor("tag_hiddens", tagHiddenStates)
        };

        using var results = _bboxDecoderSession.Run(inputs);
        foreach (var input in inputs)
        {
            (input as IDisposable)?.Dispose();
        }

        var bboxClassesTensor = results.First(x => x.Name == "bbox_classes").AsTensor<float>();
        var bboxCoordsTensor = results.First(x => x.Name == "bbox_coords").AsTensor<float>();

        return (bboxClassesTensor.ToDenseTensor(), bboxCoordsTensor.ToDenseTensor());
    }

    public void Dispose()
    {
        _encoderSession.Dispose();
        _tagTransformerEncoderSession.Dispose();
        _tagTransformerDecoderStepSession.Dispose();
        _bboxDecoderSession.Dispose();
    }
}