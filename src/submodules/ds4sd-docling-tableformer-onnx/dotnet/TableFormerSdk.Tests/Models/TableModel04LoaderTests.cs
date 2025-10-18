//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using TableFormerSdk.Models;
using Xunit;

namespace TableFormerSdk.Tests.Models;

public sealed class TableModel04LoaderTests : IDisposable
{
    private readonly TableModel04 _model;

    public TableModel04LoaderTests()
    {
        var wordMapTag = new Dictionary<string, long>
        {
            ["<pad>"] = 0,
            ["<unk>"] = 1,
            ["<start>"] = 2,
            ["<end>"] = 3,
            ["ecel"] = 4,
            ["fcel"] = 5,
            ["lcel"] = 6,
            ["ucel"] = 7,
            ["xcel"] = 8,
            ["nl"] = 9,
            ["ched"] = 10,
            ["rhed"] = 11,
            ["srow"] = 12
        };

        var config = new TableModel04Config
        {
            WordMapTag = wordMapTag,
            EncImageSize = 14,
            EncoderDim = 256,
            TagAttentionDim = 256,
            TagEmbedDim = 128,
            TagDecoderDim = 256,
            BBoxAttentionDim = 256,
            BBoxEmbedDim = 128,
            BBoxDecoderDim = 256,
            EncLayers = 2,
            DecLayers = 1,
            NumHeads = 4,
            Dropout = 0.1,
            NumClasses = 2,
            MaxSteps = 20
        };

        _model = new TableModel04(config);
        _model.eval();
    }

    [Fact]
    public void LoadWeights_LoadsConvWeightsFromSafeTensors()
    {
        var targetParam = _model
            .named_parameters()
            .First(p => p.name == "_encoder.resnet.0.weight")
            .parameter!;

        var shape = targetParam.shape;
        var elementCount = shape.Aggregate(1L, (acc, dim) => acc * dim);
        var values = Enumerable
            .Range(0, checked((int)elementCount))
            .Select(i => (float)((i % 97) - 48) / 97f)
            .ToArray();

        using (no_grad())
        {
            targetParam.zero_();
        }

        var tempRoot = Path.Combine(Path.GetTempPath(), $"tableformer-loader-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);
        var safetensorsPath = Path.Combine(tempRoot, "conv_weights.safetensors");
        var debugDir = Path.Combine(tempRoot, "debug");
        var previousDebug = Environment.GetEnvironmentVariable("TABLEFORMER_DEBUG_DIR");
        Environment.SetEnvironmentVariable("TABLEFORMER_DEBUG_DIR", debugDir);

        try
        {
            WriteSafeTensorsFile(safetensorsPath, "_encoder._resnet.0.weight", shape, values);

            TableModel04Loader.LoadWeights(_model, safetensorsPath);

            var actual = targetParam.data<float>().ToArray();
            Assert.Equal(values.Length, actual.Length);

            for (int i = 0; i < actual.Length; i++)
            {
                Assert.Equal(values[i], actual[i], 6);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("TABLEFORMER_DEBUG_DIR", previousDebug);

            if (Directory.Exists(tempRoot))
            {
                try
                {
                    Directory.Delete(tempRoot, recursive: true);
                }
                catch
                {
                    // Ignore cleanup failures in tests
                }
            }
        }
    }

    private static void WriteSafeTensorsFile(string filePath, string tensorName, IReadOnlyList<long> shape, float[] values)
    {
        var dataSize = checked((long)values.Length * sizeof(float));

        var header = new Dictionary<string, object?>
        {
            [tensorName] = new Dictionary<string, object?>
            {
                ["dtype"] = "F32",
                ["shape"] = shape,
                ["data_offsets"] = new long[] { 0, dataSize }
            }
        };

        var headerJson = JsonSerializer.Serialize(header);
        var headerBytes = Encoding.UTF8.GetBytes(headerJson);

        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var writer = new BinaryWriter(stream);

        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);

        var dataBytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, dataBytes, 0, dataBytes.Length);
        writer.Write(dataBytes);
    }

    public void Dispose()
    {
        _model.Dispose();
        GC.SuppressFinalize(this);
    }
}
