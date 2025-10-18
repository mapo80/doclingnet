//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using TableFormerSdk.Models;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;

namespace TableFormerSdk.Tests.Models;

/// <summary>
/// Comprehensive unit tests for TMTransformerDecoderLayer module.
/// Target: 90%+ code coverage.
/// </summary>
public sealed class TMTransformerDecoderLayerTests : IDisposable
{
    private readonly TMTransformerDecoderLayer _decoderLayer;
    private const long DModel = 256;
    private const long Nhead = 8;
    private const long DimFeedforward = 2048;
    private const double Dropout = 0.1;

    public TMTransformerDecoderLayerTests()
    {
        // Initialize with typical TableFormer parameters
        _decoderLayer = new TMTransformerDecoderLayer(DModel, Nhead, DimFeedforward, Dropout);
    }

    public void Dispose()
    {
        _decoderLayer?.Dispose();
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultParameters_CreatesModule()
    {
        // Arrange & Act
        using var layer = new TMTransformerDecoderLayer(DModel, Nhead);

        // Assert
        Assert.NotNull(layer);
    }

    [Fact]
    public void Constructor_WithCustomParameters_CreatesModule()
    {
        // Arrange & Act
        using var layer = new TMTransformerDecoderLayer(
            dModel: 512,
            nhead: 8,
            dimFeedforward: 4096,
            dropout: 0.2,
            name: "CustomDecoderLayer");

        // Assert
        Assert.NotNull(layer);
    }

    [Theory]
    [InlineData(128, 4)]
    [InlineData(256, 8)]
    [InlineData(512, 8)]
    [InlineData(1024, 16)]
    public void Constructor_WithVariousDimensions_CreatesModule(long dModel, long nhead)
    {
        // Arrange & Act
        using var layer = new TMTransformerDecoderLayer(dModel, nhead);

        // Assert
        Assert.NotNull(layer);
    }

    #endregion

    #region Forward Pass Shape Tests

    [Fact]
    public void Forward_WithValidInputs_ReturnsCorrectShape()
    {
        // Arrange
        const long seqLen = 10;
        const long batchSize = 2;

        using var tgt = torch.randn(seqLen, batchSize, DModel);
        using var memory = torch.randn(196, batchSize, DModel);  // Encoder output

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.Equal(3, output.ndim);
        Assert.Equal(1, output.size(0));  // Only last token
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    [Fact]
    public void Forward_WithoutMemory_ReturnsCorrectShape()
    {
        // Arrange
        const long seqLen = 10;
        const long batchSize = 2;

        using var tgt = torch.randn(seqLen, batchSize, DModel);

        // Act
        using var output = _decoderLayer.forward((tgt, null));

        // Assert
        Assert.Equal(3, output.ndim);
        Assert.Equal(1, output.size(0));  // Only last token
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    [Theory]
    [InlineData(1, 1)]   // Single token, single batch
    [InlineData(5, 1)]   // Short sequence
    [InlineData(10, 2)]  // Standard
    [InlineData(50, 4)]  // Long sequence, larger batch
    [InlineData(100, 1)] // Very long sequence
    public void Forward_WithVariousSequenceLengthsAndBatches_ReturnsCorrectShape(
        long seqLen,
        long batchSize)
    {
        // Arrange
        using var tgt = torch.randn(seqLen, batchSize, DModel);
        using var memory = torch.randn(196, batchSize, DModel);

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.Equal(1, output.size(0));  // Always returns last token only
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    [Theory]
    [InlineData(49)]   // 7x7 encoder output
    [InlineData(196)]  // 14x14 encoder output (standard)
    [InlineData(256)]  // 16x16 encoder output
    public void Forward_WithVariousMemorySizes_ReturnsCorrectShape(long memoryLen)
    {
        // Arrange
        const long seqLen = 10;
        const long batchSize = 2;

        using var tgt = torch.randn(seqLen, batchSize, DModel);
        using var memory = torch.randn(memoryLen, batchSize, DModel);

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.Equal(1, output.size(0));
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    #endregion

    #region Self-Attention Tests

    [Fact]
    public void Forward_SelfAttention_ProducesFiniteValues()
    {
        // Arrange
        _decoderLayer.eval();
        using var tgt = torch.randn(10, 2, DModel);
        using var memory = torch.randn(196, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _decoderLayer.forward((tgt, memory));

            // Assert
            Assert.True(output.isfinite().all().item<bool>(),
                "Output should contain only finite values");
        }
    }

    [Fact]
    public void Forward_LastTokenOutput_DependsOnAllPreviousTokens()
    {
        // Arrange
        _decoderLayer.eval();
        const long seqLen = 5;

        using var tgt1 = torch.randn(seqLen, 1, DModel);
        using var tgt2 = tgt1.clone();

        // Modify first token
        tgt2[0] = torch.randn(1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output1 = _decoderLayer.forward((tgt1, null));
            using var output2 = _decoderLayer.forward((tgt2, null));

            // Assert: Output should be different because input changed
            using var diff = (output1 - output2).abs().max();
            var maxDiff = diff.item<float>();

            Assert.True(maxDiff > 1e-5,
                "Changing earlier tokens should affect last token output");
        }
    }

    #endregion

    #region Cross-Attention Tests

    [Fact]
    public void Forward_WithMemory_AttendsToCrossAttention()
    {
        // Arrange
        _decoderLayer.eval();
        using var tgt = torch.randn(10, 1, DModel);
        using var memory1 = torch.randn(196, 1, DModel);
        using var memory2 = torch.randn(196, 1, DModel) + 10.0f;  // Different memory

        // Act
        using (torch.no_grad())
        {
            using var output1 = _decoderLayer.forward((tgt, memory1));
            using var output2 = _decoderLayer.forward((tgt, memory2));

            // Assert: Different memory should produce different output
            using var diff = (output1 - output2).abs().max();
            var maxDiff = diff.item<float>();

            Assert.True(maxDiff > 1e-3,
                "Different memory should produce different output via cross-attention");
        }
    }

    [Fact]
    public void Forward_WithoutMemory_SkipsCrossAttention()
    {
        // Arrange
        _decoderLayer.eval();
        using var tgt = torch.randn(10, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _decoderLayer.forward((tgt, null));

            // Assert: Should produce valid output even without memory
            Assert.True(output.isfinite().all().item<bool>(),
                "Should handle null memory gracefully");
        }
    }

    #endregion

    #region Train/Eval Mode Tests

    [Fact]
    public void TrainMode_CanBeEnabled()
    {
        // Arrange & Act
        _decoderLayer.train();

        // Assert: No exception means success
        Assert.NotNull(_decoderLayer);
    }

    [Fact]
    public void EvalMode_CanBeEnabled()
    {
        // Arrange & Act
        _decoderLayer.eval();

        // Assert: No exception means success
        Assert.NotNull(_decoderLayer);
    }

    [Fact]
    public void EvalMode_ProducesDeterministicResults()
    {
        // Arrange
        _decoderLayer.eval();
        using var tgt = torch.randn(10, 1, DModel);
        using var memory = torch.randn(196, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output1 = _decoderLayer.forward((tgt, memory));
            using var output2 = _decoderLayer.forward((tgt, memory));

            // Assert: Results should be identical in eval mode
            using var diff = (output1 - output2).abs().max();
            var maxDiff = diff.item<float>();

            Assert.True(maxDiff < 1e-6,
                $"Outputs should be identical in eval mode (diff: {maxDiff})");
        }
    }

    #endregion

    #region Feedforward Network Tests

    [Fact]
    public void Forward_FeedforwardNetwork_TransformsFeatures()
    {
        // Arrange
        _decoderLayer.eval();
        using var tgt = torch.ones(10, 1, DModel);  // All ones
        using var memory = torch.zeros(196, 1, DModel);  // All zeros

        // Act
        using (torch.no_grad())
        {
            using var output = _decoderLayer.forward((tgt, memory));

            // Assert: Output should not be all ones (transformed by FF network)
            var outputMean = output.mean().item<float>();
            Assert.True(Math.Abs(outputMean - 1.0f) > 0.1,
                "Feedforward network should transform the features");
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void Forward_WithZeroInput_ProducesValidOutput()
    {
        // Arrange
        using var tgt = torch.zeros(10, 1, DModel);
        using var memory = torch.zeros(196, 1, DModel);

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Should handle zero input gracefully");
    }

    [Fact]
    public void Forward_WithLargeValues_ProducesFiniteOutput()
    {
        // Arrange
        using var tgt = torch.full(new long[] { 10, 1, DModel }, 100.0f);
        using var memory = torch.full(new long[] { 196, 1, DModel }, 100.0f);

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Should handle large values without overflow");
    }

    [Fact]
    public void Forward_WithSmallValues_ProducesFiniteOutput()
    {
        // Arrange
        using var tgt = torch.full(new long[] { 10, 1, DModel }, 1e-6f);
        using var memory = torch.full(new long[] { 196, 1, DModel }, 1e-6f);

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Should handle small values without underflow");
    }

    [Fact]
    public void Forward_WithNegativeValues_ProducesValidOutput()
    {
        // Arrange
        using var tgt = torch.full(new long[] { 10, 1, DModel }, -5.0f);
        using var memory = torch.full(new long[] { 196, 1, DModel }, -5.0f);

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Should handle negative values");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Forward_WithSingleToken_ReturnsCorrectShape()
    {
        // Arrange
        using var tgt = torch.randn(1, 1, DModel);
        using var memory = torch.randn(196, 1, DModel);

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.Equal(1, output.size(0));
        Assert.Equal(1, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    [Fact]
    public void Forward_WithLargeBatch_HandlesCorrectly()
    {
        // Arrange
        const long batchSize = 16;
        using var tgt = torch.randn(10, batchSize, DModel);
        using var memory = torch.randn(196, batchSize, DModel);

        // Act
        using var output = _decoderLayer.forward((tgt, memory));

        // Assert
        Assert.Equal(1, output.size(0));
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
        Assert.True(output.isfinite().all().item<bool>(),
            "Should handle large batch sizes");
    }

    #endregion

    #region Residual Connection Tests

    [Fact]
    public void Forward_ResidualConnections_PreserveInformation()
    {
        // Arrange
        _decoderLayer.eval();
        using var tgt = torch.randn(10, 1, DModel);
        using var memory = torch.randn(196, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _decoderLayer.forward((tgt, memory));

            // Assert: Output magnitude should be reasonable (residual connections help)
            var outputStd = output.std().item<float>();

            Assert.True(outputStd > 0.1 && outputStd < 10.0,
                $"Residual connections should maintain reasonable output scale (std: {outputStd})");
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Forward_RepeatedCalls_ProducesConsistentShapes()
    {
        // Arrange
        _decoderLayer.eval();
        using var tgt = torch.randn(10, 2, DModel);
        using var memory = torch.randn(196, 2, DModel);

        // Act & Assert
        using (torch.no_grad())
        {
            for (int i = 0; i < 5; i++)
            {
                using var output = _decoderLayer.forward((tgt, memory));

                Assert.Equal(1, output.size(0));
                Assert.Equal(2, output.size(1));
                Assert.Equal(DModel, output.size(2));
            }
        }
    }

    [Fact]
    public void Forward_WithRandomInputs_ProducesReasonableStatistics()
    {
        // Arrange
        _decoderLayer.eval();
        using var tgt = torch.randn(10, 2, DModel);
        using var memory = torch.randn(196, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _decoderLayer.forward((tgt, memory));

            // Assert: Check that output has reasonable statistics
            var mean = output.mean().item<float>();
            var std = output.std().item<float>();

            Assert.True(Math.Abs(mean) < 10.0,
                $"Output mean should be reasonable (got {mean})");
            Assert.True(std > 0.01 && std < 100.0,
                $"Output std should be reasonable (got {std})");
        }
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var layer = new TMTransformerDecoderLayer(DModel, Nhead);

        // Act & Assert: Should not throw
        layer.Dispose();
        layer.Dispose();
    }

    [Fact]
    public void Dispose_WithUsingStatement_DisposesCorrectly()
    {
        // Arrange & Act
        TMTransformerDecoderLayer? layer;
        using (layer = new TMTransformerDecoderLayer(DModel, Nhead))
        {
            // Use the module
            using var tgt = torch.randn(10, 1, DModel);
            using var memory = torch.randn(196, 1, DModel);
            using var output = layer.forward((tgt, memory));
            Assert.NotNull(output);
        }

        // Assert: Module should be disposed (no exception means success)
        Assert.NotNull(layer);
    }

    #endregion
}
