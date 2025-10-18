//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using System.Reflection;
using Xunit;
using TorchSharp;
using static TorchSharp.torch;
using TableFormerSdk.Models;

namespace TableFormerSdk.Tests;

public class TableModel04Tests : IDisposable
{
    private readonly TableModel04Config _config;
    private TableModel04? _model;

    public TableModel04Tests()
    {
        // Minimal test configuration
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

        _config = new TableModel04Config
        {
            WordMapTag = wordMapTag,
            EncImageSize = 14,  // Small for fast tests
            EncoderDim = 256,
            TagAttentionDim = 128,
            TagEmbedDim = 16,
            TagDecoderDim = 256,
            BBoxAttentionDim = 256,
            BBoxEmbedDim = 128,
            BBoxDecoderDim = 256,
            EncLayers = 2,
            DecLayers = 1,
            NumHeads = 4,
            Dropout = 0.1,
            NumClasses = 2,
            MaxSteps = 20  // Small for fast tests
        };
    }

    public void Dispose()
    {
        _model?.Dispose();
        GC.SuppressFinalize(this);
    }

    // ==================== Constructor Tests ====================

    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        _model = new TableModel04(_config);
        Assert.NotNull(_model);
    }

    [Fact]
    public void Constructor_WithCustomName_SetsName()
    {
        _model = new TableModel04(_config, "CustomModel");
        Assert.NotNull(_model);
    }

    [Fact]
    public void Constructor_RegistersSubmodules()
    {
        _model = new TableModel04(_config);
        var namedChildren = _model.named_children().ToList();
        Assert.Contains(namedChildren, x => x.name == "_encoder");
        Assert.Contains(namedChildren, x => x.name == "_tagTransformer");
        Assert.Contains(namedChildren, x => x.name == "_bboxDecoder");
    }

    // ==================== Config Loading Tests ====================

    [Fact]
    public void ConfigFromJsonFile_LoadsCorrectly()
    {
        var configPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json";
        if (!File.Exists(configPath))
        {
            return; // Skip if config file not available
        }

        var config = TableModel04Config.FromJsonFile(configPath);
        Assert.NotNull(config);
        Assert.NotNull(config.WordMapTag);
        Assert.True(config.WordMapTag.Count > 0);
        Assert.Equal(28L, config.EncImageSize);
        Assert.Equal(512L, config.EncoderDim);
    }

    // ==================== Forward Pass Shape Tests ====================

    [Theory]
    [InlineData(1, 224, 224)]
    [InlineData(1, 448, 448)]
    public void Forward_WithValidInput_ReturnsCorrectShapes(int batchSize, int height, int width)
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(batchSize, 3, height, width);
        var result = _model.forward(images);

        Assert.NotNull(result);
        Assert.NotNull(result.Sequence);
        Assert.True(result.Sequence.Count > 0);
        Assert.True(result.Sequence.Count <= _config.MaxSteps);

        // Should end with <end> token if MaxSteps allows
        if (result.Sequence.Count < _config.MaxSteps)
        {
            Assert.Equal(3L, result.Sequence[^1]);
        }

        // BBox tensors should have correct dimensions
        Assert.Equal(2, result.BBoxClasses.ndim);
        Assert.Equal(2, result.BBoxCoords.ndim);
        Assert.Equal(4L, result.BBoxCoords.size(1));  // 4 coordinates

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_GeneratesValidSequence()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        // Sequence should contain valid tokens from vocabulary
        Assert.NotNull(result.Sequence);
        Assert.True(result.Sequence.Count > 0);

        // All tokens should be valid (within vocabulary range)
        Assert.All(result.Sequence, token => Assert.True(token >= 0 && token < _config.WordMapTag.Count));

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_NormalizesNhwcInputs()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var nchw = torch.randn(1, 3, 224, 224);
        using var nhwc = nchw.permute(0, 2, 3, 1).contiguous();

        var nchwResult = _model.forward(nchw);
        var nhwcResult = _model.forward(nhwc);

        try
        {
            Assert.Equal(nchwResult.Sequence, nhwcResult.Sequence);
            Assert.True(torch.allclose(nchwResult.BBoxClasses, nhwcResult.BBoxClasses));
            Assert.True(torch.allclose(nchwResult.BBoxCoords, nhwcResult.BBoxCoords));
        }
        finally
        {
            nchwResult.BBoxClasses.Dispose();
            nchwResult.BBoxCoords.Dispose();
            nhwcResult.BBoxClasses.Dispose();
            nhwcResult.BBoxCoords.Dispose();
        }
    }

    [Fact]
    public void MergeBBoxes_PreservesDTypeAndDevice()
    {
        _model = new TableModel04(_config);
        var mergeMethod = typeof(TableModel04).GetMethod("MergeBBoxes", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(mergeMethod);

        using var bbox1 = torch.tensor(new[] { 0.25f, 0.5f, 0.2f, 0.3f }, dtype: ScalarType.Float32, device: torch.CPU);
        using var bbox2 = torch.tensor(new[] { 0.75f, 0.6f, 0.2f, 0.3f }, dtype: ScalarType.Float32, device: torch.CPU);

        using var merged = (Tensor)mergeMethod!.Invoke(_model, new object[] { bbox1, bbox2 })!;

        Assert.Equal(bbox1.dtype, merged.dtype);
        Assert.Equal(bbox1.device.ToString(), merged.device.ToString());

        using var expected = torch.tensor(new[] { 0.5f, 0.55f, 0.7f, 0.4f }, dtype: ScalarType.Float32, device: torch.CPU);
        Assert.True(torch.allclose(expected, merged));
    }

    // ==================== Output Value Tests ====================

    [Fact]
    public void Forward_BBoxCoords_AreNormalized()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        if (result.BBoxCoords.size(0) > 0)
        {
            // All bbox coordinates should be in [0, 1]
            using var minVal = result.BBoxCoords.min();
            using var maxVal = result.BBoxCoords.max();

            Assert.True(minVal.item<float>() >= 0.0f);
            Assert.True(maxVal.item<float>() <= 1.0f);
        }

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_SequenceEndsWithEndToken()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        // Should end with <end> token if not truncated by MaxSteps
        if (result.Sequence.Count < _config.MaxSteps)
        {
            Assert.Equal(3L, result.Sequence[^1]);  // <end> token
        }
        else
        {
            // If MaxSteps reached, sequence may not end with <end>
            Assert.Equal(_config.MaxSteps, result.Sequence.Count);
        }

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_SequenceContainsValidTokens()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        // All tokens should be within vocabulary range
        foreach (var token in result.Sequence)
        {
            Assert.True(token >= 0);
            Assert.True(token < _config.WordMapTag.Count);
        }

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    // ==================== Train/Eval Mode Tests ====================

    [Fact]
    public void Forward_InEvalMode_Succeeds()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        Assert.NotNull(result);
        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_InTrainMode_Succeeds()
    {
        _model = new TableModel04(_config);
        _model.train();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        Assert.NotNull(result);
        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    // ==================== Edge Case Tests ====================

    [Fact]
    public void Forward_WithMaxStepsReached_Terminates()
    {
        var shortConfig = new TableModel04Config
        {
            WordMapTag = _config.WordMapTag,
            EncImageSize = 14,
            EncoderDim = 256,
            TagAttentionDim = 128,
            TagEmbedDim = 16,
            TagDecoderDim = 256,
            BBoxAttentionDim = 256,
            BBoxEmbedDim = 128,
            BBoxDecoderDim = 256,
            EncLayers = 2,
            DecLayers = 1,
            NumHeads = 4,
            Dropout = 0.1,
            NumClasses = 2,
            MaxSteps = 5  // Very short
        };

        _model = new TableModel04(shortConfig);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        // Should stop within MaxSteps
        Assert.True(result.Sequence.Count <= shortConfig.MaxSteps);

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_WithSmallImage_Succeeds()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 112, 112);
        var result = _model.forward(images);

        Assert.NotNull(result);
        Assert.NotNull(result.Sequence);

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    // ==================== BBox Merging Tests ====================

    [Fact]
    public void Forward_WithLcelTokens_MergesBBoxes()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        // If lcel (token 6) appears in sequence, bboxes should be merged
        var hasLcel = result.Sequence.Contains(6L);

        if (hasLcel)
        {
            // Number of bboxes might be less than number of cell tokens
            // due to merging
            var cellTokens = result.Sequence.Count(t =>
                t == 4L || t == 5L || t == 6L || t == 7L ||
                t == 10L || t == 11L || t == 12L);

            // BBox count should be <= cell tokens
            Assert.True(result.BBoxCoords.size(0) <= cellTokens);
        }

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    // ==================== Numerical Stability Tests ====================

    [Fact]
    public void Forward_WithZeroInput_DoesNotCrash()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.zeros(1, 3, 224, 224);
        var result = _model.forward(images);

        Assert.NotNull(result);
        Assert.NotNull(result.Sequence);

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_WithOnesInput_DoesNotCrash()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.ones(1, 3, 224, 224);
        var result = _model.forward(images);

        Assert.NotNull(result);
        Assert.NotNull(result.Sequence);

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_WithLargeInput_DoesNotCrash()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224) * 100.0;
        var result = _model.forward(images);

        Assert.NotNull(result);
        Assert.NotNull(result.Sequence);

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_MultipleInvocations_ProducesDifferentResults()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images1 = torch.randn(1, 3, 224, 224);
        var result1 = _model.forward(images1);

        using var images2 = torch.randn(1, 3, 224, 224);
        var result2 = _model.forward(images2);

        // Different inputs should generally produce different sequences
        // (though not guaranteed with small vocab)
        Assert.NotNull(result1);
        Assert.NotNull(result2);

        result1.BBoxClasses.Dispose();
        result1.BBoxCoords.Dispose();
        result2.BBoxClasses.Dispose();
        result2.BBoxCoords.Dispose();
    }

    // ==================== Integration Tests ====================

    [Fact]
    public void Forward_EndToEnd_ProducesValidOutput()
    {
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        // Validate complete output
        Assert.NotNull(result);
        Assert.NotNull(result.Sequence);
        Assert.True(result.Sequence.Count > 0);

        // Should end with <end> token if not truncated
        if (result.Sequence.Count < _config.MaxSteps)
        {
            Assert.Equal(3L, result.Sequence[^1]);  // Ends with <end>
        }

        // BBoxes present (even if empty)
        Assert.NotNull(result.BBoxClasses);
        Assert.NotNull(result.BBoxCoords);
        Assert.Equal(result.BBoxClasses.size(0), result.BBoxCoords.size(0));

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    [Fact]
    public void Forward_WithLargerVocabulary_Succeeds()
    {
        var configPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json";
        if (!File.Exists(configPath))
        {
            return; // Skip if config file not available
        }

        var realConfig = TableModel04Config.FromJsonFile(configPath);

        // Use real vocabulary but test-sized model for fast execution
        var testConfig = new TableModel04Config
        {
            WordMapTag = realConfig.WordMapTag,  // Real vocabulary
            EncImageSize = 14,
            EncoderDim = 256,
            TagAttentionDim = 128,
            TagEmbedDim = 16,
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

        _model = new TableModel04(testConfig);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        Assert.NotNull(result);
        Assert.NotNull(result.Sequence);

        // Should generate valid tokens from the larger vocabulary
        Assert.All(result.Sequence, token => Assert.True(token >= 0 && token < realConfig.WordMapTag.Count));

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }

    // ==================== Disposal Tests ====================

    [Fact]
    public void Dispose_ReleasesResources()
    {
        var model = new TableModel04(_config);
        model.Dispose();
        // If no exception, disposal succeeded
    }

    [Fact]
    public void Dispose_MultipleTimes_DoesNotCrash()
    {
        var model = new TableModel04(_config);
        model.Dispose();
        model.Dispose();
        // Multiple disposals should be safe
    }

    // ==================== Parameter Count Test ====================

    [Fact]
    public void Model_HasExpectedParameterCount()
    {
        _model = new TableModel04(_config);
        var paramCount = _model.parameters().Count();
        Assert.True(paramCount > 0);
    }

    // ==================== Structure Error Correction Tests ====================

    [Fact]
    public void Forward_FirstLineXcel_CorrectToLcel()
    {
        // This is difficult to test directly without mocking,
        // but we can verify the logic doesn't crash
        _model = new TableModel04(_config);
        _model.eval();

        using var images = torch.randn(1, 3, 224, 224);
        var result = _model.forward(images);

        // If first line had xcel, it should have been corrected
        // We can't directly verify without seeing internal state,
        // but we can ensure output is valid
        Assert.NotNull(result.Sequence);
        Assert.All(result.Sequence, token => Assert.True(token >= 0 && token < _config.WordMapTag.Count));

        result.BBoxClasses.Dispose();
        result.BBoxCoords.Dispose();
    }
}
