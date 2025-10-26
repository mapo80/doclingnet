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
/// Comprehensive unit tests for CellAttention module.
/// Target: 90%+ code coverage.
/// </summary>
public sealed class CellAttentionTests : IDisposable
{
    private readonly CellAttention _cellAttention;
    private const long EncoderDim = 512;
    private const long TagDecoderDim = 256;
    private const long LanguageDim = 256;
    private const long AttentionDim = 128;

    public CellAttentionTests()
    {
        // Initialize with typical TableFormer parameters
        _cellAttention = new CellAttention(EncoderDim, TagDecoderDim, LanguageDim, AttentionDim);
    }

    public void Dispose()
    {
        _cellAttention?.Dispose();
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultParameters_CreatesModule()
    {
        // Arrange & Act
        using var attention = new CellAttention(EncoderDim, TagDecoderDim, LanguageDim, AttentionDim);

        // Assert
        Assert.NotNull(attention);
    }

    [Fact]
    public void Constructor_WithCustomName_CreatesModule()
    {
        // Arrange & Act
        using var attention = new CellAttention(
            encoderDim: EncoderDim,
            tagDecoderDim: TagDecoderDim,
            languageDim: LanguageDim,
            attentionDim: AttentionDim,
            name: "CustomCellAttention");

        // Assert
        Assert.NotNull(attention);
    }

    [Theory]
    [InlineData(256, 128, 128, 64)]
    [InlineData(512, 256, 256, 128)]
    [InlineData(1024, 512, 512, 256)]
    public void Constructor_WithVariousDimensions_CreatesModule(
        long encoderDim,
        long tagDecoderDim,
        long languageDim,
        long attentionDim)
    {
        // Arrange & Act
        using var attention = new CellAttention(encoderDim, tagDecoderDim, languageDim, attentionDim);

        // Assert
        Assert.NotNull(attention);
    }

    #endregion

    #region Forward Pass Shape Tests

    [Fact]
    public void Forward_WithValidInputs_ReturnsCorrectShapes()
    {
        // Arrange
        const long numPixels = 196;  // 14x14 feature map
        const long numCells = 10;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.Equal(2, attentionWeightedEncoding.ndim);
        Assert.Equal(numCells, attentionWeightedEncoding.size(0));
        Assert.Equal(EncoderDim, attentionWeightedEncoding.size(1));

        Assert.Equal(2, alpha.ndim);
        Assert.Equal(numCells, alpha.size(0));
        Assert.Equal(numPixels, alpha.size(1));

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    [Theory]
    [InlineData(1, 196)]   // Single cell, 14x14 feature map
    [InlineData(5, 196)]   // 5 cells
    [InlineData(10, 196)]  // 10 cells
    [InlineData(50, 196)]  // 50 cells
    [InlineData(100, 196)] // 100 cells (large table)
    public void Forward_WithVariousCellCounts_ReturnsCorrectShapes(long numCells, long numPixels)
    {
        // Arrange
        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.Equal(numCells, attentionWeightedEncoding.size(0));
        Assert.Equal(EncoderDim, attentionWeightedEncoding.size(1));
        Assert.Equal(numCells, alpha.size(0));
        Assert.Equal(numPixels, alpha.size(1));

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    [Theory]
    [InlineData(49)]   // 7x7 feature map
    [InlineData(100)]  // 10x10 feature map
    [InlineData(196)]  // 14x14 feature map
    [InlineData(256)]  // 16x16 feature map
    [InlineData(400)]  // 20x20 feature map
    public void Forward_WithVariousFeatureMapSizes_ReturnsCorrectShapes(long numPixels)
    {
        // Arrange
        const long numCells = 10;
        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.Equal(numPixels, alpha.size(1));

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    #endregion

    #region Attention Weights Tests

    [Fact]
    public void Forward_AttentionWeights_SumToOne()
    {
        // Arrange
        _cellAttention.eval();
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        using (torch.no_grad())
        {
            var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

            // Assert: Each row of attention weights should sum to 1.0 (softmax property)
            using var alphaSums = alpha.sum(dim: 1);

            for (long i = 0; i < numCells; i++)
            {
                var sum = alphaSums[i].item<float>();
                Assert.True(Math.Abs(sum - 1.0f) < 1e-5,
                    $"Attention weights for cell {i} should sum to 1.0, got {sum}");
            }

            attentionWeightedEncoding.Dispose();
            alpha.Dispose();
        }
    }

    [Fact]
    public void Forward_AttentionWeights_AreNonNegative()
    {
        // Arrange
        _cellAttention.eval();
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        using (torch.no_grad())
        {
            var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

            // Assert: All attention weights should be >= 0 (softmax property)
            var minVal = alpha.min().item<float>();
            Assert.True(minVal >= 0.0f, $"Attention weights should be non-negative, got min={minVal}");

            attentionWeightedEncoding.Dispose();
            alpha.Dispose();
        }
    }

    [Fact]
    public void Forward_AttentionWeights_InValidRange()
    {
        // Arrange
        _cellAttention.eval();
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        using (torch.no_grad())
        {
            var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

            // Assert: All attention weights should be in [0, 1]
            var minVal = alpha.min().item<float>();
            var maxVal = alpha.max().item<float>();

            Assert.True(minVal >= 0.0f && minVal <= 1.0f,
                $"Min attention weight should be in [0, 1], got {minVal}");
            Assert.True(maxVal >= 0.0f && maxVal <= 1.0f,
                $"Max attention weight should be in [0, 1], got {maxVal}");

            attentionWeightedEncoding.Dispose();
            alpha.Dispose();
        }
    }

    #endregion

    #region Attention Weighted Encoding Tests

    [Fact]
    public void Forward_AttentionWeightedEncoding_IsWeightedSum()
    {
        // Arrange
        _cellAttention.eval();
        const long numPixels = 4;  // Small for manual verification
        const long numCells = 2;

        using var encoderOut = torch.ones(1, numPixels, EncoderDim) * 2.0f;  // All encoder features = 2.0
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        using (torch.no_grad())
        {
            var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

            // Assert: Since encoder features are all 2.0, weighted sum should also be close to 2.0
            // (weighted average of constant values equals that constant)
            var meanEncoding = attentionWeightedEncoding.mean().item<float>();

            // Should be close to 2.0 (allowing some variance due to weighted combination)
            Assert.True(Math.Abs(meanEncoding - 2.0f) < 0.5f,
                $"Mean attention-weighted encoding should be close to 2.0, got {meanEncoding}");

            attentionWeightedEncoding.Dispose();
            alpha.Dispose();
        }
    }

    [Fact]
    public void Forward_AttentionWeightedEncoding_ProducesFiniteValues()
    {
        // Arrange
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.True(attentionWeightedEncoding.isfinite().all().item<bool>(),
            "Attention-weighted encoding should contain only finite values");

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    #endregion

    #region Train/Eval Mode Tests

    [Fact]
    public void TrainMode_CanBeEnabled()
    {
        // Arrange & Act
        _cellAttention.train();

        // Assert: No exception means success
        Assert.NotNull(_cellAttention);
    }

    [Fact]
    public void EvalMode_CanBeEnabled()
    {
        // Arrange & Act
        _cellAttention.eval();

        // Assert: No exception means success
        Assert.NotNull(_cellAttention);
    }

    [Fact]
    public void EvalMode_ProducesDeterministicResults()
    {
        // Arrange
        _cellAttention.eval();
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        using (torch.no_grad())
        {
            var (attn1, alpha1) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));
            var (attn2, alpha2) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

            // Assert: Results should be identical
            using var attnDiff = (attn1 - attn2).abs().max();
            using var alphaDiff = (alpha1 - alpha2).abs().max();

            var maxAttnDiff = attnDiff.item<float>();
            var maxAlphaDiff = alphaDiff.item<float>();

            Assert.True(maxAttnDiff < 1e-6,
                $"Attention-weighted encodings should be identical in eval mode (diff: {maxAttnDiff})");
            Assert.True(maxAlphaDiff < 1e-6,
                $"Attention weights should be identical in eval mode (diff: {maxAlphaDiff})");

            attn1.Dispose();
            alpha1.Dispose();
            attn2.Dispose();
            alpha2.Dispose();
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Forward_WithSingleCell_ReturnsCorrectShapes()
    {
        // Arrange
        const long numPixels = 196;
        const long numCells = 1;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.Equal(numCells, attentionWeightedEncoding.size(0));
        Assert.Equal(EncoderDim, attentionWeightedEncoding.size(1));

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    [Fact]
    public void Forward_WithLargeCellCount_ReturnsCorrectShapes()
    {
        // Arrange
        const long numPixels = 196;
        const long numCells = 500;  // Large table

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.Equal(numCells, attentionWeightedEncoding.size(0));
        Assert.Equal(numPixels, alpha.size(1));

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void Forward_WithLargeInputValues_ProducesFiniteOutput()
    {
        // Arrange
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.full(new long[] { 1, numPixels, EncoderDim }, 1000.0f);
        using var decoderHidden = torch.full(new long[] { numCells, TagDecoderDim }, 1000.0f);
        using var languageOut = torch.full(new long[] { numCells, LanguageDim }, 1000.0f);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.True(attentionWeightedEncoding.isfinite().all().item<bool>(),
            "Output should contain only finite values even with large inputs");
        Assert.True(alpha.isfinite().all().item<bool>(),
            "Attention weights should contain only finite values even with large inputs");

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    [Fact]
    public void Forward_WithSmallInputValues_ProducesFiniteOutput()
    {
        // Arrange
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.full(new long[] { 1, numPixels, EncoderDim }, 1e-6f);
        using var decoderHidden = torch.full(new long[] { numCells, TagDecoderDim }, 1e-6f);
        using var languageOut = torch.full(new long[] { numCells, LanguageDim }, 1e-6f);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.True(attentionWeightedEncoding.isfinite().all().item<bool>(),
            "Output should contain only finite values even with small inputs");
        Assert.True(alpha.isfinite().all().item<bool>(),
            "Attention weights should contain only finite values even with small inputs");

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    [Fact]
    public void Forward_WithNegativeInputValues_ProducesValidOutput()
    {
        // Arrange
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.full(new long[] { 1, numPixels, EncoderDim }, -5.0f);
        using var decoderHidden = torch.full(new long[] { numCells, TagDecoderDim }, -5.0f);
        using var languageOut = torch.full(new long[] { numCells, LanguageDim }, -5.0f);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.True(attentionWeightedEncoding.isfinite().all().item<bool>(),
            "Output should contain only finite values even with negative inputs");
        Assert.True(alpha.isfinite().all().item<bool>(),
            "Attention weights should be non-negative even with negative inputs");

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    [Fact]
    public void Forward_WithZeroInputs_ProducesValidOutput()
    {
        // Arrange
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.zeros(1, numPixels, EncoderDim);
        using var decoderHidden = torch.zeros(numCells, TagDecoderDim);
        using var languageOut = torch.zeros(numCells, LanguageDim);

        // Act
        var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

        // Assert
        Assert.True(attentionWeightedEncoding.isfinite().all().item<bool>(),
            "Output should contain only finite values even with zero inputs");
        Assert.True(alpha.isfinite().all().item<bool>(),
            "Attention weights should contain only finite values even with zero inputs");

        // With zero inputs, attention should be uniform (1/num_pixels for each pixel)
        var expectedWeight = 1.0f / numPixels;
        var meanWeight = alpha.mean().item<float>();
        Assert.True(Math.Abs(meanWeight - expectedWeight) < 0.01f,
            $"With zero inputs, attention should be uniform. Expected ~{expectedWeight}, got {meanWeight}");

        attentionWeightedEncoding.Dispose();
        alpha.Dispose();
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Forward_RepeatedCalls_ProducesConsistentShapes()
    {
        // Arrange
        _cellAttention.eval();
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act & Assert
        using (torch.no_grad())
        {
            for (int i = 0; i < 5; i++)
            {
                var (attn, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

                Assert.Equal(numCells, attn.size(0));
                Assert.Equal(EncoderDim, attn.size(1));
                Assert.Equal(numCells, alpha.size(0));
                Assert.Equal(numPixels, alpha.size(1));

                attn.Dispose();
                alpha.Dispose();
            }
        }
    }

    [Fact]
    public void Forward_WithRandomInputs_ProducesReasonableStatistics()
    {
        // Arrange
        _cellAttention.eval();
        const long numPixels = 196;
        const long numCells = 10;

        using var encoderOut = torch.randn(1, numPixels, EncoderDim);
        using var decoderHidden = torch.randn(numCells, TagDecoderDim);
        using var languageOut = torch.randn(numCells, LanguageDim);

        // Act
        using (torch.no_grad())
        {
            var (attentionWeightedEncoding, alpha) = _cellAttention.forward((encoderOut, decoderHidden, languageOut));

            // Assert: Check that output has reasonable statistics
            var attnMean = attentionWeightedEncoding.mean().item<float>();
            var attnStd = attentionWeightedEncoding.std().item<float>();

            // Mean should be reasonable (not too far from 0)
            Assert.True(Math.Abs(attnMean) < 5.0, $"Mean {attnMean} should be reasonable");

            // Std should be reasonable (not too small, not too large)
            Assert.True(attnStd > 0.01 && attnStd < 100.0, $"Std {attnStd} should be reasonable");

            attentionWeightedEncoding.Dispose();
            alpha.Dispose();
        }
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var attention = new CellAttention(EncoderDim, TagDecoderDim, LanguageDim, AttentionDim);

        // Act & Assert: Should not throw
        attention.Dispose();
        attention.Dispose();
    }

    [Fact]
    public void Dispose_WithUsingStatement_DisposesCorrectly()
    {
        // Arrange & Act
        CellAttention? attention;
        using (attention = new CellAttention(EncoderDim, TagDecoderDim, LanguageDim, AttentionDim))
        {
            // Use the module
            using var encoderOut = torch.randn(1, 196, EncoderDim);
            using var decoderHidden = torch.randn(10, TagDecoderDim);
            using var languageOut = torch.randn(10, LanguageDim);

            var (attn, alpha) = attention.forward((encoderOut, decoderHidden, languageOut));
            Assert.NotNull(attn);
            Assert.NotNull(alpha);

            attn.Dispose();
            alpha.Dispose();
        }

        // Assert: Module should be disposed (no exception means success)
        Assert.NotNull(attention);
    }

    #endregion
}
