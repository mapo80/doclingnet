//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TableFormerSdk.Backends;

/// <summary>
/// TableFormer inference using 4 ONNX component models with autoregressive loop.
/// Matches the Python implementation in tools/test_tableformer_onnx_components.py
/// </summary>
public sealed class TableFormerOnnxInference : IDisposable
{
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _tagEncoderSession;
    private readonly InferenceSession _tagDecoderSession;
    private readonly InferenceSession _bboxDecoderSession;

    private readonly Dictionary<string, long> _wordMapTag;
    private readonly long _startToken;
    private readonly long _endToken;
    private readonly long _padToken;

    private bool _disposed;

    public TableFormerOnnxInference(string modelDir, Dictionary<string, long> wordMapTag)
    {
        ArgumentNullException.ThrowIfNull(modelDir);
        ArgumentNullException.ThrowIfNull(wordMapTag);

        _wordMapTag = wordMapTag;
        _startToken = wordMapTag["<start>"];
        _endToken = wordMapTag["<end>"];
        _padToken = wordMapTag["<pad>"];

        // Session options
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };

        try
        {
            Console.WriteLine("Loading ONNX models...");

            // Load all 4 component models
            var encoderPath = Path.Combine(modelDir, "tableformer_fast_encoder.onnx");
            _encoderSession = new InferenceSession(encoderPath, options);
            Console.WriteLine("  ✅ Encoder loaded");

            var tagEncoderPath = Path.Combine(modelDir, "tableformer_fast_tag_transformer_encoder.onnx");
            _tagEncoderSession = new InferenceSession(tagEncoderPath, options);
            Console.WriteLine("  ✅ Tag Transformer Encoder loaded");

            var tagDecoderPath = Path.Combine(modelDir, "tableformer_fast_tag_transformer_decoder_step.onnx");
            _tagDecoderSession = new InferenceSession(tagDecoderPath, options);
            Console.WriteLine("  ✅ Tag Transformer Decoder Step loaded");

            var bboxDecoderPath = Path.Combine(modelDir, "tableformer_fast_bbox_decoder.onnx");
            _bboxDecoderSession = new InferenceSession(bboxDecoderPath, options);
            Console.WriteLine("  ✅ BBox Decoder loaded");

            Console.WriteLine("✅ All models loaded successfully");
            Console.WriteLine($"   Vocabulary size: {_wordMapTag.Count}");
            Console.WriteLine($"   Special tokens: <start>={_startToken}, <end>={_endToken}, <pad>={_padToken}\n");
        }
        finally
        {
            options.Dispose();
        }
    }

    /// <summary>
    /// Run complete inference on an image.
    /// </summary>
    /// <param name="imageData">Image tensor of shape (1, 3, 448, 448) in float32.</param>
    /// <param name="maxSteps">Maximum autoregressive steps.</param>
    /// <returns>Tuple of (tag_sequence, bbox_classes, bbox_coords).</returns>
    public (List<long> Tags, float[,] BBoxClasses, float[,] BBoxCoords) Predict(
        float[,,,] imageData,
        int maxSteps = 1024)
    {
        // 1. Run encoder
        var encoderOut = RunEncoder(imageData);
        Console.WriteLine($"[1/4] Encoder output shape: [{encoderOut.Dimensions[0]}, {encoderOut.Dimensions[1]}, {encoderOut.Dimensions[2]}, {encoderOut.Dimensions[3]}]");

        // 2. Run tag transformer encoder to get memory
        var memory = RunTagTransformerEncoder(encoderOut);
        Console.WriteLine($"[2/4] Memory shape: [{memory.Dimensions[0]}, {memory.Dimensions[1]}, {memory.Dimensions[2]}]");

        // 3. Autoregressive tag generation
        var (tags, tagHiddens) = GenerateTagsAutoregressive(memory, maxSteps);
        Console.WriteLine($"[3/4] Generated {tags.Count} tags, {tagHiddens.Count} hiddens");

        // 4. Predict bounding boxes
        var (bboxClasses, bboxCoords) = RunBBoxDecoder(encoderOut, tagHiddens);
        Console.WriteLine($"[4/4] Predicted {bboxCoords.GetLength(0)} bounding boxes");

        return (tags, bboxClasses, bboxCoords);
    }

    private DenseTensor<float> RunEncoder(float[,,,] imageData)
    {
        // Create input tensor - flatten and copy data
        var imageTensor = new DenseTensor<float>(new[] { 1, 3, 448, 448 });
        for (int b = 0; b < 1; b++)
            for (int c = 0; c < 3; c++)
                for (int h = 0; h < 448; h++)
                    for (int w = 0; w < 448; w++)
                        imageTensor[b, c, h, w] = imageData[b, c, h, w];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("images", imageTensor)
        };

        using var results = _encoderSession.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // Reshape to (1, 28, 28, 256)
        var outputTensor = new DenseTensor<float>(new[] { 1, 28, 28, 256 });
        int idx = 0;
        for (int b = 0; b < 1; b++)
            for (int h = 0; h < 28; h++)
                for (int w = 0; w < 28; w++)
                    for (int c = 0; c < 256; c++)
                        outputTensor[b, h, w, c] = output[idx++];

        return outputTensor;
    }

    private DenseTensor<float> RunTagTransformerEncoder(DenseTensor<float> encoderOut)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encoderOut)
        };

        using var results = _tagEncoderSession.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // Memory shape: (784, 1, 512)
        var memoryTensor = new DenseTensor<float>(new[] { 784, 1, 512 });
        Buffer.BlockCopy(output, 0, memoryTensor.Buffer.ToArray(), 0, output.Length * sizeof(float));

        return memoryTensor;
    }

    private (List<long> Tags, List<float[]> TagHiddens) GenerateTagsAutoregressive(
        DenseTensor<float> memory,
        int maxSteps)
    {
        var tags = new List<long> { _startToken };
        var tagHiddens = new List<float[]>();

        var decodedTags = new List<long> { _startToken };

        // Tokens that generate bboxes
        var bboxTokens = new HashSet<long>
        {
            _wordMapTag["fcel"],
            _wordMapTag["ecel"],
            _wordMapTag["ched"],
            _wordMapTag["rhed"],
            _wordMapTag["srow"],
            _wordMapTag["nl"],
            _wordMapTag.GetValueOrDefault("ucel", -1)
        };

        for (int step = 0; step < maxSteps; step++)
        {
            // Create decoded_tags tensor: (1, seq_len)
            var decodedTagsTensor = new DenseTensor<long>(new[] { 1, decodedTags.Count });
            for (int i = 0; i < decodedTags.Count; i++)
            {
                decodedTagsTensor[0, i] = decodedTags[i];
            }

            // Run decoder step
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("decoded_tags", decodedTagsTensor),
                NamedOnnxValue.CreateFromTensor("memory", memory)
            };

            using var results = _tagDecoderSession.Run(inputs);
            var resultsArray = results.ToArray();

            var logits = resultsArray[0].AsEnumerable<float>().ToArray();
            var hiddens = resultsArray[1].AsEnumerable<float>().ToArray();

            // Get predicted token (argmax of logits for last position)
            var vocabSize = 13;
            var lastLogits = logits.Skip(logits.Length - vocabSize).Take(vocabSize).ToArray();
            var newTag = Array.IndexOf(lastLogits, lastLogits.Max());

            // Check for end token
            if (newTag == _endToken)
            {
                tags.Add(newTag);
                break;
            }

            tags.Add(newTag);
            decodedTags.Add(newTag);

            // Save hidden state if this token generates a bbox
            if (bboxTokens.Contains(newTag))
            {
                // Extract last hidden state (last 512 values)
                var lastHidden = hiddens.Skip(hiddens.Length - 512).Take(512).ToArray();
                tagHiddens.Add(lastHidden);
            }
        }

        return (tags, tagHiddens);
    }

    private (float[,] BBoxClasses, float[,] BBoxCoords) RunBBoxDecoder(
        DenseTensor<float> encoderOut,
        List<float[]> tagHiddens)
    {
        if (tagHiddens.Count == 0)
        {
            return (new float[0, 3], new float[0, 4]);
        }

        // Create tag_hiddens tensor: (num_boxes, 1, 512)
        var numBoxes = tagHiddens.Count;
        var tagHiddensTensor = new DenseTensor<float>(new[] { numBoxes, 1, 512 });
        for (int i = 0; i < numBoxes; i++)
        {
            for (int j = 0; j < 512; j++)
            {
                tagHiddensTensor[i, 0, j] = tagHiddens[i][j];
            }
        }

        // Expand encoder_out to match batch size
        var encoderOutExpanded = new DenseTensor<float>(new[] { numBoxes, 28, 28, 256 });
        for (int b = 0; b < numBoxes; b++)
        {
            for (int h = 0; h < 28; h++)
            {
                for (int w = 0; w < 28; w++)
                {
                    for (int c = 0; c < 256; c++)
                    {
                        encoderOutExpanded[b, h, w, c] = encoderOut[0, h, w, c];
                    }
                }
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encoderOutExpanded),
            NamedOnnxValue.CreateFromTensor("tag_hiddens", tagHiddensTensor)
        };

        using var results = _bboxDecoderSession.Run(inputs);
        var resultsArray = results.ToArray();

        var bboxClassesFlat = resultsArray[0].AsEnumerable<float>().ToArray();
        var bboxCoordsFlat = resultsArray[1].AsEnumerable<float>().ToArray();

        // Reshape outputs
        var bboxClasses = new float[numBoxes, 3];
        var bboxCoords = new float[numBoxes, 4];

        for (int i = 0; i < numBoxes; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                bboxClasses[i, j] = bboxClassesFlat[i * 3 + j];
            }
            for (int j = 0; j < 4; j++)
            {
                bboxCoords[i, j] = bboxCoordsFlat[i * 4 + j];
            }
        }

        return (bboxClasses, bboxCoords);
    }

    public void Dispose()
    {
        if (_disposed) return;

        _encoderSession?.Dispose();
        _tagEncoderSession?.Dispose();
        _tagDecoderSession?.Dispose();
        _bboxDecoderSession?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
