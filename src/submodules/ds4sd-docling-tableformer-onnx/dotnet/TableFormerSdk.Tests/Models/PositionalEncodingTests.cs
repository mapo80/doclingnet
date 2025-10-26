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
/// Comprehensive unit tests for PositionalEncoding module.
/// Target: 90%+ code coverage.
/// </summary>
public sealed class PositionalEncodingTests : IDisposable
{
    private readonly PositionalEncoding _positionalEncoding;
    private const long DModel = 256;
    private const double Dropout = 0.1;
    private const long MaxLen = 1024;

    public PositionalEncodingTests()
    {
        // Initialize with default TableFormer parameters
        _positionalEncoding = new PositionalEncoding(DModel, Dropout, MaxLen);
    }

    public void Dispose()
    {
        _positionalEncoding?.Dispose();
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultParameters_CreatesModule()
    {
        // Arrange & Act
        using var pe = new PositionalEncoding(DModel);

        // Assert
        Assert.NotNull(pe);
    }

    [Fact]
    public void Constructor_WithCustomParameters_CreatesModule()
    {
        // Arrange & Act
        using var pe = new PositionalEncoding(
            dModel: 512,
            dropout: 0.2,
            maxLen: 2048,
            name: "CustomPE");

        // Assert
        Assert.NotNull(pe);
    }

    [Theory]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(768)]
    [InlineData(1024)]
    public void Constructor_WithVariousDModel_CreatesModule(long dModel)
    {
        // Arrange & Act
        using var pe = new PositionalEncoding(dModel);

        // Assert
        Assert.NotNull(pe);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.1)]
    [InlineData(0.3)]
    [InlineData(0.5)]
    public void Constructor_WithVariousDropout_CreatesModule(double dropout)
    {
        // Arrange & Act
        using var pe = new PositionalEncoding(DModel, dropout);

        // Assert
        Assert.NotNull(pe);
    }

    [Theory]
    [InlineData(128)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    public void Constructor_WithVariousMaxLen_CreatesModule(long maxLen)
    {
        // Arrange & Act
        using var pe = new PositionalEncoding(DModel, maxLen: maxLen);

        // Assert
        Assert.NotNull(pe);
    }

    #endregion

    #region Forward Pass Shape Tests

    [Fact]
    public void Forward_WithShortSequence_ReturnsCorrectShape()
    {
        // Arrange
        const long seqLen = 10;
        const long batchSize = 1;
        using var input = torch.randn(seqLen, batchSize, DModel);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.Equal(3, output.ndim);
        Assert.Equal(seqLen, output.size(0));
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    [Fact]
    public void Forward_WithLongSequence_ReturnsCorrectShape()
    {
        // Arrange
        const long seqLen = 500;
        const long batchSize = 2;
        using var input = torch.randn(seqLen, batchSize, DModel);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.Equal(seqLen, output.size(0));
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(10, 1)]
    [InlineData(50, 4)]
    [InlineData(100, 2)]
    [InlineData(500, 1)]
    [InlineData(1024, 1)]
    public void Forward_WithVariousSequenceLengthsAndBatchSizes_ReturnsCorrectShape(
        long seqLen,
        long batchSize)
    {
        // Arrange
        using var input = torch.randn(seqLen, batchSize, DModel);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.Equal(seqLen, output.size(0));
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    #endregion

    #region Value Range Tests

    [Fact]
    public void Forward_InEvalMode_ProducesReasonableValues()
    {
        // Arrange
        _positionalEncoding.eval();
        using var zeros = torch.zeros(10, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _positionalEncoding.forward(zeros);

            // Assert: Positional encodings based on sin/cos should be bounded
            var minVal = output.min().item<float>();
            var maxVal = output.max().item<float>();

            Assert.True(minVal >= -2.0, $"Min value {minVal} should be >= -2.0");
            Assert.True(maxVal <= 2.0, $"Max value {maxVal} should be <= 2.0");
        }
    }

    [Fact]
    public void Forward_WithZeroInput_AddsPositionalEncoding()
    {
        // Arrange
        _positionalEncoding.eval();
        using var zeros = torch.zeros(5, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _positionalEncoding.forward(zeros);

            // Assert: Output should not be all zeros (positional encoding was added)
            var sumAbs = output.abs().sum().item<float>();
            Assert.True(sumAbs > 0, "Output should not be all zeros");
        }
    }

    #endregion

    #region Position Uniqueness Tests

    [Fact]
    public void Forward_DifferentPositions_HaveDifferentEncodings()
    {
        // Arrange
        _positionalEncoding.eval();
        using var zeros = torch.zeros(5, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _positionalEncoding.forward(zeros);

            // Assert: Each position should have a unique encoding
            for (long i = 0; i < 4; i++)
            {
                using var pos1 = output[i];
                using var pos2 = output[i + 1];
                using var diff = (pos1 - pos2).abs().sum();
                var diffVal = diff.item<float>();

                Assert.True(diffVal > 0.001,
                    $"Position {i} and {i + 1} should have different encodings");
            }
        }
    }

    [Fact]
    public void Forward_SamePosition_HasSameEncoding()
    {
        // Arrange
        _positionalEncoding.eval();
        using var zeros1 = torch.zeros(10, 1, DModel);
        using var zeros2 = torch.zeros(10, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output1 = _positionalEncoding.forward(zeros1);
            using var output2 = _positionalEncoding.forward(zeros2);

            // Assert: Same positions should have identical encodings
            using var diff = (output1 - output2).abs().max();
            var maxDiff = diff.item<float>();

            Assert.True(maxDiff < 1e-6,
                $"Same positions should have identical encodings (diff: {maxDiff})");
        }
    }

    #endregion

    #region Train/Eval Mode Tests

    [Fact]
    public void TrainMode_EnablesDropout()
    {
        // Arrange
        _positionalEncoding.train();
        using var input = torch.ones(10, 1, DModel);

        // Act & Assert: In train mode, multiple passes should give different results due to dropout
        using var output1 = _positionalEncoding.forward(input);
        using var output2 = _positionalEncoding.forward(input);

        using var diff = (output1 - output2).abs().max();
        var maxDiff = diff.item<float>();

        // Note: This test might occasionally fail if dropout randomly doesn't change any values
        // But with 10x256=2560 values and 10% dropout, the probability is extremely low
        // If dropout is > 0, we expect different outputs in train mode
        var expectedDifferentOutputs = maxDiff > 0;
        Assert.True(expectedDifferentOutputs,
            "Train mode should apply dropout, causing different outputs");
    }

    [Fact]
    public void EvalMode_DisablesDropout()
    {
        // Arrange
        _positionalEncoding.eval();
        using var input = torch.ones(10, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output1 = _positionalEncoding.forward(input);
            using var output2 = _positionalEncoding.forward(input);

            // Assert: In eval mode, passes should give identical results
            using var diff = (output1 - output2).abs().max();
            var maxDiff = diff.item<float>();

            Assert.True(maxDiff < 1e-6,
                $"Eval mode should be deterministic (diff: {maxDiff})");
        }
    }

    #endregion

    #region Sequence Length Edge Cases

    [Fact]
    public void Forward_WithSequenceLengthOne_ReturnsCorrectShape()
    {
        // Arrange
        using var input = torch.randn(1, 1, DModel);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.Equal(1, output.size(0));
        Assert.Equal(1, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    [Fact]
    public void Forward_WithMaxSequenceLength_ReturnsCorrectShape()
    {
        // Arrange
        using var input = torch.randn(MaxLen, 1, DModel);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.Equal(MaxLen, output.size(0));
        Assert.Equal(1, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    #endregion

    #region Batch Size Tests

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void Forward_WithVariousBatchSizes_ReturnsCorrectShape(long batchSize)
    {
        // Arrange
        const long seqLen = 50;
        using var input = torch.randn(seqLen, batchSize, DModel);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.Equal(seqLen, output.size(0));
        Assert.Equal(batchSize, output.size(1));
        Assert.Equal(DModel, output.size(2));
    }

    [Fact]
    public void Forward_WithLargeBatch_ProducesSameEncodingForAllBatchItems()
    {
        // Arrange
        _positionalEncoding.eval();
        const long seqLen = 10;
        const long batchSize = 4;
        using var zeros = torch.zeros(seqLen, batchSize, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _positionalEncoding.forward(zeros);

            // Assert: All batch items should have the same positional encoding
            for (long b = 1; b < batchSize; b++)
            {
                using var batch0 = output[TensorIndex.Colon, 0, TensorIndex.Colon];
                using var batchB = output[TensorIndex.Colon, b, TensorIndex.Colon];
                using var diff = (batch0 - batchB).abs().max();
                var maxDiff = diff.item<float>();

                Assert.True(maxDiff < 1e-6,
                    $"Batch item {b} should have same encoding as batch 0 (diff: {maxDiff})");
            }
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void Forward_WithLargeInput_ProducesFiniteOutput()
    {
        // Arrange
        using var input = torch.full(new long[] { 10, 1, DModel }, 1000.0f);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Output should contain only finite values");
    }

    [Fact]
    public void Forward_WithSmallInput_ProducesFiniteOutput()
    {
        // Arrange
        using var input = torch.full(new long[] { 10, 1, DModel }, 1e-6f);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Output should contain only finite values");
    }

    [Fact]
    public void Forward_WithNegativeInput_ProducesValidOutput()
    {
        // Arrange
        using var input = torch.full(new long[] { 10, 1, DModel }, -5.0f);

        // Act
        using var output = _positionalEncoding.forward(input);

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Output should contain only finite values");
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Forward_RepeatedCalls_ProducesConsistentShapes()
    {
        // Arrange
        _positionalEncoding.eval();
        const long seqLen = 100;
        using var input = torch.randn(seqLen, 1, DModel);

        // Act & Assert
        using (torch.no_grad())
        {
            for (int i = 0; i < 5; i++)
            {
                using var output = _positionalEncoding.forward(input);
                Assert.Equal(seqLen, output.size(0));
                Assert.Equal(1, output.size(1));
                Assert.Equal(DModel, output.size(2));
            }
        }
    }

    [Fact]
    public void Forward_WithRandomInputs_ProducesReasonableStatistics()
    {
        // Arrange
        _positionalEncoding.eval();
        using var input = torch.randn(100, 1, DModel);

        // Act
        using (torch.no_grad())
        {
            using var output = _positionalEncoding.forward(input);

            // Assert: Check that output has reasonable statistics
            var mean = output.mean().item<float>();
            var std = output.std().item<float>();

            // Mean should be close to 0 (input is random with mean~0, positional encoding oscillates around 0)
            Assert.True(Math.Abs(mean) < 1.0, $"Mean {mean} should be close to 0");

            // Std should be reasonable (not too small, not too large)
            Assert.True(std > 0.1 && std < 10.0, $"Std {std} should be reasonable");
        }
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var pe = new PositionalEncoding(DModel);

        // Act & Assert: Should not throw
        pe.Dispose();
        pe.Dispose();
    }

    [Fact]
    public void Dispose_WithUsingStatement_DisposesCorrectly()
    {
        // Arrange & Act
        PositionalEncoding? pe;
        using (pe = new PositionalEncoding(DModel))
        {
            // Use the module
            using var input = torch.randn(10, 1, DModel);
            using var output = pe.forward(input);
            Assert.NotNull(output);
        }

        // Assert: Module should be disposed (no exception means success)
        Assert.NotNull(pe);
    }

    #endregion
}
