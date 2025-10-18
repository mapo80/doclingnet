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
/// Comprehensive unit tests for BBoxDecoder module.
/// Target: >=90% code coverage.
/// </summary>
public sealed class BBoxDecoderTests : IDisposable
{
    private readonly BBoxDecoder _decoder;
    private const long AttentionDim = 128;
    private const long EmbedDim = 256;
    private const long TagDecoderDim = 256;
    private const long DecoderDim = 512;
    private const long EncoderRawDim = 256;
    private const long NumClasses = 3;  // cell, row header, column header
    private const long EncoderDim = 512;

    public BBoxDecoderTests()
    {
        _decoder = new BBoxDecoder(
            attentionDim: AttentionDim,
            embedDim: EmbedDim,
            tagDecoderDim: TagDecoderDim,
            decoderDim: DecoderDim,
            numClasses: NumClasses,
            encoderRawDim: EncoderRawDim,
            encoderDim: EncoderDim,
            dropout: 0.1);
    }

    public void Dispose()
    {
        _decoder?.Dispose();
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultEncoderDim_CreatesModule()
    {
        // Arrange & Act
        using var decoder = new BBoxDecoder(
            attentionDim: 128,
            embedDim: 256,
            tagDecoderDim: 256,
            decoderDim: 512,
            numClasses: 3);

        // Assert
        Assert.NotNull(decoder);
    }

    [Fact]
    public void Constructor_WithCustomEncoderDim_CreatesModule()
    {
        // Arrange & Act
        using var decoder = new BBoxDecoder(
            attentionDim: 128,
            embedDim: 256,
            tagDecoderDim: 256,
            decoderDim: 512,
            numClasses: 3,
            encoderRawDim: 256,
            encoderDim: 256);

        // Assert
        Assert.NotNull(decoder);
    }

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.3)]
    [InlineData(0.5)]
    public void Constructor_WithVariousDropout_CreatesModule(double dropout)
    {
        // Arrange & Act
        using var decoder = new BBoxDecoder(
            attentionDim: 128,
            embedDim: 256,
            tagDecoderDim: 256,
            decoderDim: 512,
            numClasses: 3,
            dropout: dropout);

        // Assert
        Assert.NotNull(decoder);
    }

    #endregion

    #region Forward Pass Shape Tests

    [Fact]
    public void Forward_WithSingleCell_ReturnsCorrectShapes()
    {
        // Arrange
        _decoder.eval();
        const long numCells = 1;
        const long height = 14;
        const long width = 14;
        const long batchSize = 1;

        using var encoderOut = torch.randn(batchSize, height, width, EncoderRawDim);
        using var tagH = torch.randn(numCells, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.Equal(2, classes.ndim);
            Assert.Equal(numCells, classes.size(0));
            Assert.Equal(NumClasses + 1, classes.size(1));  // +1 for no-object class

            Assert.Equal(2, boxes.ndim);
            Assert.Equal(numCells, boxes.size(0));
            Assert.Equal(4, boxes.size(1));  // x, y, w, h

            classes.Dispose();
            boxes.Dispose();
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(50)]
    public void Forward_WithVariousCellCounts_ReturnsCorrectShapes(long numCells)
    {
        // Arrange
        _decoder.eval();
        const long height = 14;
        const long width = 14;

        using var encoderOut = torch.randn(1, height, width, EncoderRawDim);
        using var tagH = torch.randn(numCells, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.Equal(numCells, classes.size(0));
            Assert.Equal(NumClasses + 1, classes.size(1));
            Assert.Equal(numCells, boxes.size(0));
            Assert.Equal(4, boxes.size(1));

            classes.Dispose();
            boxes.Dispose();
        }
    }

    [Fact]
    public void Forward_WithNoCells_ReturnsEmptyTensors()
    {
        // Arrange
        _decoder.eval();
        const long height = 14;
        const long width = 14;

        using var encoderOut = torch.randn(1, height, width, EncoderRawDim);
        using var tagH = torch.empty(0, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.Equal(0, classes.size(0));
            Assert.Equal(0, boxes.size(0));

            classes.Dispose();
            boxes.Dispose();
        }
    }

    [Theory]
    [InlineData(7, 7)]
    [InlineData(14, 14)]
    [InlineData(28, 28)]
    public void Forward_WithVariousEncoderSizes_ProcessesCorrectly(long height, long width)
    {
        // Arrange
        _decoder.eval();
        const long numCells = 3;

        using var encoderOut = torch.randn(1, height, width, EncoderRawDim);
        using var tagH = torch.randn(numCells, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.Equal(numCells, classes.size(0));
            Assert.Equal(numCells, boxes.size(0));

            classes.Dispose();
            boxes.Dispose();
        }
    }

    #endregion

    #region Output Value Tests

    [Fact]
    public void Forward_ProducesFiniteValues()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.randn(1, 14, 14, EncoderRawDim);
        using var tagH = torch.randn(5, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.True(classes.isfinite().all().item<bool>(),
                "Classes should contain only finite values");
            Assert.True(boxes.isfinite().all().item<bool>(),
                "Boxes should contain only finite values");

            classes.Dispose();
            boxes.Dispose();
        }
    }

    [Fact]
    public void Forward_BBoxCoordinates_AreBetweenZeroAndOne()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.randn(1, 14, 14, EncoderRawDim);
        using var tagH = torch.randn(5, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert: Sigmoid should ensure values are in [0, 1]
            var minVal = boxes.min().item<float>();
            var maxVal = boxes.max().item<float>();

            Assert.True(minVal >= 0.0f, $"Min bbox value should be >= 0, got {minVal}");
            Assert.True(maxVal <= 1.0f, $"Max bbox value should be <= 1, got {maxVal}");

            classes.Dispose();
            boxes.Dispose();
        }
    }

    [Fact]
    public void Forward_ClassLogits_HaveReasonableScale()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.randn(1, 14, 14, EncoderRawDim);
        using var tagH = torch.randn(5, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            var mean = classes.mean().item<float>();
            var std = classes.std().item<float>();

            Assert.True(Math.Abs(mean) < 20.0,
                $"Class logits mean should be reasonable, got {mean}");
            Assert.True(std > 0.01 && std < 50.0,
                $"Class logits std should be reasonable, got {std}");

            classes.Dispose();
            boxes.Dispose();
        }
    }

    #endregion

    #region Train/Eval Mode Tests

    [Fact]
    public void TrainMode_CanBeEnabled()
    {
        // Arrange & Act
        _decoder.train();

        // Assert: No exception means success
        Assert.NotNull(_decoder);
    }

    [Fact]
    public void EvalMode_CanBeEnabled()
    {
        // Arrange & Act
        _decoder.eval();

        // Assert: No exception means success
        Assert.NotNull(_decoder);
    }

    [Fact]
    public void EvalMode_ProducesDeterministicResults()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.randn(1, 14, 14, EncoderRawDim);
        using var tagH = torch.randn(3, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes1, boxes1) = _decoder.forward((encoderOut, tagH));
            var (classes2, boxes2) = _decoder.forward((encoderOut, tagH));

            // Assert
            using var classesDiff = (classes1 - classes2).abs().max();
            using var boxesDiff = (boxes1 - boxes2).abs().max();

            var classesMaxDiff = classesDiff.item<float>();
            var boxesMaxDiff = boxesDiff.item<float>();

            Assert.True(classesMaxDiff < 1e-6,
                $"Classes should be identical in eval mode (diff: {classesMaxDiff})");
            Assert.True(boxesMaxDiff < 1e-6,
                $"Boxes should be identical in eval mode (diff: {boxesMaxDiff})");

            classes1.Dispose();
            boxes1.Dispose();
            classes2.Dispose();
            boxes2.Dispose();
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void Forward_WithZeroEncoderOut_ProducesValidOutput()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.zeros(1, 14, 14, EncoderRawDim);
        using var tagH = torch.randn(3, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.True(classes.isfinite().all().item<bool>(),
                "Classes should be finite even with zero encoder output");
            Assert.True(boxes.isfinite().all().item<bool>(),
                "Boxes should be finite even with zero encoder output");

            classes.Dispose();
            boxes.Dispose();
        }
    }

    [Fact]
    public void Forward_WithZeroTagH_ProducesValidOutput()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.randn(1, 14, 14, EncoderRawDim);
        using var tagH = torch.zeros(3, TagDecoderDim);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.True(classes.isfinite().all().item<bool>(),
                "Classes should be finite even with zero tag hidden states");
            Assert.True(boxes.isfinite().all().item<bool>(),
                "Boxes should be finite even with zero tag hidden states");

            classes.Dispose();
            boxes.Dispose();
        }
    }

    [Fact]
    public void Forward_WithLargeValues_ProducesFiniteOutput()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.full(new long[] { 1, 14, 14, EncoderRawDim }, 10.0f);
        using var tagH = torch.full(new long[] { 3, TagDecoderDim }, 10.0f);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.True(classes.isfinite().all().item<bool>(),
                "Output should be finite even with large input values");
            Assert.True(boxes.isfinite().all().item<bool>(),
                "Boxes should be finite even with large input values");

            classes.Dispose();
            boxes.Dispose();
        }
    }

    [Fact]
    public void Forward_WithNegativeValues_ProducesValidOutput()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.full(new long[] { 1, 14, 14, EncoderRawDim }, -5.0f);
        using var tagH = torch.full(new long[] { 3, TagDecoderDim }, -5.0f);

        // Act
        using (torch.no_grad())
        {
            var (classes, boxes) = _decoder.forward((encoderOut, tagH));

            // Assert
            Assert.True(classes.isfinite().all().item<bool>(),
                "Output should be finite even with negative input values");
            Assert.True(boxes.isfinite().all().item<bool>(),
                "Boxes should be finite even with negative input values");

            classes.Dispose();
            boxes.Dispose();
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Forward_RepeatedCalls_ProducesConsistentShapes()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.randn(1, 14, 14, EncoderRawDim);
        using var tagH = torch.randn(5, TagDecoderDim);

        // Act & Assert
        using (torch.no_grad())
        {
            for (int i = 0; i < 5; i++)
            {
                var (classes, boxes) = _decoder.forward((encoderOut, tagH));

                Assert.Equal(5, classes.size(0));
                Assert.Equal(NumClasses + 1, classes.size(1));
                Assert.Equal(5, boxes.size(0));
                Assert.Equal(4, boxes.size(1));

                classes.Dispose();
                boxes.Dispose();
            }
        }
    }

    [Fact]
    public void Forward_WithDifferentCellsSequentially_ProcessesCorrectly()
    {
        // Arrange
        _decoder.eval();
        using var encoderOut = torch.randn(1, 14, 14, EncoderRawDim);

        // Act & Assert
        using (torch.no_grad())
        {
            for (long numCells = 1; numCells <= 10; numCells++)
            {
                using var tagH = torch.randn(numCells, TagDecoderDim);
                var (classes, boxes) = _decoder.forward((encoderOut, tagH));

                Assert.Equal(numCells, classes.size(0));
                Assert.Equal(numCells, boxes.size(0));

                classes.Dispose();
                boxes.Dispose();
            }
        }
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var decoder = new BBoxDecoder(
            attentionDim: 128,
            embedDim: 256,
            tagDecoderDim: 256,
            decoderDim: 512,
            numClasses: 3);

        // Act & Assert: Should not throw
        decoder.Dispose();
        decoder.Dispose();
    }

    [Fact]
    public void Dispose_WithUsingStatement_DisposesCorrectly()
    {
        // Arrange & Act
        BBoxDecoder? decoder;
        using (decoder = new BBoxDecoder(
            attentionDim: 128,
            embedDim: 256,
            tagDecoderDim: 256,
            decoderDim: 512,
            numClasses: 3))
        {
            // Use the module
            using var encoderOut = torch.randn(1, 14, 14, EncoderRawDim);
            using var tagH = torch.randn(3, 256);
            var (classes, boxes) = decoder.forward((encoderOut, tagH));

            classes.Dispose();
            boxes.Dispose();

            Assert.NotNull(decoder);
        }

        // Assert: Module should be disposed (no exception means success)
        Assert.NotNull(decoder);
    }

    #endregion
}
