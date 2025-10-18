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
/// Comprehensive unit tests for Encoder04 module.
/// Target: 90%+ code coverage.
/// </summary>
public sealed class Encoder04Tests : IDisposable
{
    private readonly Encoder04 _encoder;
    private const long EncImageSize = 14;  // Typical encoded image size (14x14)
    private const long EncDim = 512;

    public Encoder04Tests()
    {
        // Initialize with typical TableFormer parameters
        _encoder = new Encoder04(EncImageSize, EncDim);
    }

    public void Dispose()
    {
        _encoder?.Dispose();
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultParameters_CreatesModule()
    {
        // Arrange & Act
        using var encoder = new Encoder04(EncImageSize);

        // Assert
        Assert.NotNull(encoder);
    }

    [Fact]
    public void Constructor_WithCustomParameters_CreatesModule()
    {
        // Arrange & Act
        using var encoder = new Encoder04(
            encImageSize: 16,
            encDim: 1024,
            name: "CustomEncoder");

        // Assert
        Assert.NotNull(encoder);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(14)]
    [InlineData(16)]
    [InlineData(28)]
    public void Constructor_WithVariousImageSizes_CreatesModule(long encImageSize)
    {
        // Arrange & Act
        using var encoder = new Encoder04(encImageSize);

        // Assert
        Assert.NotNull(encoder);
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void Constructor_WithVariousEncoderDims_CreatesModule(long encDim)
    {
        // Arrange & Act
        using var encoder = new Encoder04(EncImageSize, encDim);

        // Assert
        Assert.NotNull(encoder);
    }

    #endregion

    #region GetEncoderDim Tests

    [Fact]
    public void GetEncoderDim_ReturnsCorrectValue()
    {
        // Arrange & Act
        var encoderDim = _encoder.GetEncoderDim();

        // Assert
        Assert.Equal(EncDim, encoderDim);
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void GetEncoderDim_WithVariousDims_ReturnsCorrectValue(long expectedDim)
    {
        // Arrange
        using var encoder = new Encoder04(EncImageSize, expectedDim);

        // Act
        var encoderDim = encoder.GetEncoderDim();

        // Assert
        Assert.Equal(expectedDim, encoderDim);
    }

    #endregion

    #region Forward Pass Shape Tests

    [Fact]
    public void Forward_WithStandardInput_ReturnsCorrectShape()
    {
        // Arrange
        const long batchSize = 1;
        const long channels = 3;
        const long imageSize = 448;  // Standard TableFormer input

        using var images = torch.randn(batchSize, channels, imageSize, imageSize);

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.Equal(4, output.ndim);
        Assert.Equal(batchSize, output.size(0));
        Assert.Equal(EncImageSize, output.size(1));
        Assert.Equal(EncImageSize, output.size(2));
        Assert.Equal(256, output.size(3));  // ResNet output channels
    }

    [Theory]
    [InlineData(1, 224)]   // Smaller input
    [InlineData(1, 448)]   // Standard input
    [InlineData(2, 448)]   // Batch of 2
    [InlineData(4, 448)]   // Batch of 4
    public void Forward_WithVariousBatchSizesAndImageSizes_ReturnsCorrectShape(
        long batchSize,
        long imageSize)
    {
        // Arrange
        const long channels = 3;
        using var images = torch.randn(batchSize, channels, imageSize, imageSize);

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.Equal(batchSize, output.size(0));
        Assert.Equal(EncImageSize, output.size(1));
        Assert.Equal(EncImageSize, output.size(2));
        Assert.Equal(256, output.size(3));
    }

    [Fact]
    public void Forward_OutputSize_MatchesEncImageSize()
    {
        // Arrange
        const long customEncSize = 16;
        using var encoder = new Encoder04(customEncSize);
        using var images = torch.randn(1, 3, 448, 448);

        // Act
        using var output = encoder.forward(images);

        // Assert
        Assert.Equal(customEncSize, output.size(1));
        Assert.Equal(customEncSize, output.size(2));
    }

    #endregion

    #region Output Format Tests

    [Fact]
    public void Forward_OutputFormat_IsCorrect()
    {
        // Arrange
        _encoder.eval();
        using var images = torch.randn(1, 3, 448, 448);

        // Act
        using (torch.no_grad())
        {
            using var output = _encoder.forward(images);

            // Assert
            // Output should be (batch_size, enc_image_size, enc_image_size, 256)
            // This is NHWC format (batch, height, width, channels)
            Assert.Equal(1, output.size(0));      // Batch
            Assert.Equal(EncImageSize, output.size(1));  // Height
            Assert.Equal(EncImageSize, output.size(2));  // Width
            Assert.Equal(256, output.size(3));    // Channels
        }
    }

    [Fact]
    public void Forward_ProducesFiniteValues()
    {
        // Arrange
        _encoder.eval();
        using var images = torch.randn(1, 3, 448, 448);

        // Act
        using (torch.no_grad())
        {
            using var output = _encoder.forward(images);

            // Assert
            Assert.True(output.isfinite().all().item<bool>(),
                "Output should contain only finite values");
        }
    }

    #endregion

    #region Train/Eval Mode Tests

    [Fact]
    public void TrainMode_CanBeEnabled()
    {
        // Arrange & Act
        _encoder.train();

        // Assert: No exception means success
        Assert.NotNull(_encoder);
    }

    [Fact]
    public void EvalMode_CanBeEnabled()
    {
        // Arrange & Act
        _encoder.eval();

        // Assert: No exception means success
        Assert.NotNull(_encoder);
    }

    [Fact]
    public void EvalMode_ProducesDeterministicResults()
    {
        // Arrange
        _encoder.eval();
        using var images = torch.randn(1, 3, 448, 448);

        // Act
        using (torch.no_grad())
        {
            using var output1 = _encoder.forward(images);
            using var output2 = _encoder.forward(images);

            // Assert: Results should be identical in eval mode
            using var diff = (output1 - output2).abs().max();
            var maxDiff = diff.item<float>();

            Assert.True(maxDiff < 1e-6,
                $"Outputs should be identical in eval mode (diff: {maxDiff})");
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void Forward_WithZeroInput_ProducesValidOutput()
    {
        // Arrange
        using var images = torch.zeros(1, 3, 448, 448);

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Output should contain only finite values even with zero input");
    }

    [Fact]
    public void Forward_WithOnesInput_ProducesValidOutput()
    {
        // Arrange
        using var images = torch.ones(1, 3, 448, 448);

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Output should contain only finite values even with ones input");
    }

    [Fact]
    public void Forward_WithLargeValues_ProducesFiniteOutput()
    {
        // Arrange
        using var images = torch.full(new long[] { 1, 3, 448, 448 }, 100.0f);

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Output should contain only finite values even with large inputs");
    }

    [Fact]
    public void Forward_WithNegativeValues_ProducesValidOutput()
    {
        // Arrange
        using var images = torch.full(new long[] { 1, 3, 448, 448 }, -5.0f);

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Output should contain only finite values even with negative inputs");
    }

    [Fact]
    public void Forward_WithMixedValues_ProducesValidOutput()
    {
        // Arrange
        using var images = torch.randn(1, 3, 448, 448) * 10.0f;  // Wider range

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.True(output.isfinite().all().item<bool>(),
            "Output should contain only finite values with mixed value ranges");
    }

    #endregion

    #region Adaptive Pooling Tests

    [Theory]
    [InlineData(7, 224)]   // Small output, small input
    [InlineData(14, 448)]  // Standard
    [InlineData(16, 512)]  // Larger output
    [InlineData(28, 896)]  // Large output, large input
    public void Forward_AdaptivePooling_ProducesCorrectOutputSize(
        long encImageSize,
        long inputImageSize)
    {
        // Arrange
        using var encoder = new Encoder04(encImageSize);
        using var images = torch.randn(1, 3, inputImageSize, inputImageSize);

        // Act
        using var output = encoder.forward(images);

        // Assert
        Assert.Equal(encImageSize, output.size(1));
        Assert.Equal(encImageSize, output.size(2));
    }

    #endregion

    #region Batch Processing Tests

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void Forward_WithVariousBatchSizes_ProcessesCorrectly(long batchSize)
    {
        // Arrange
        _encoder.eval();
        using var images = torch.randn(batchSize, 3, 448, 448);

        // Act
        using (torch.no_grad())
        {
            using var output = _encoder.forward(images);

            // Assert
            Assert.Equal(batchSize, output.size(0));
            Assert.True(output.isfinite().all().item<bool>(),
                $"Output should be finite for batch size {batchSize}");
        }
    }

    [Fact]
    public void Forward_BatchConsistency_EachItemProcessedIndependently()
    {
        // Arrange
        _encoder.eval();
        const long batchSize = 2;
        using var images = torch.randn(batchSize, 3, 448, 448);

        // Act
        using (torch.no_grad())
        {
            using var batchOutput = _encoder.forward(images);

            // Process first image separately
            using var image1 = images[0].unsqueeze(0);
            using var output1 = _encoder.forward(image1);

            // Assert: First item in batch should match single processing
            using var batchItem1 = batchOutput[0];
            using var diff = (batchItem1 - output1[0]).abs().max();
            var maxDiff = diff.item<float>();

            Assert.True(maxDiff < 1e-5,
                $"Batch processing should be consistent with single processing (diff: {maxDiff})");
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Forward_RepeatedCalls_ProducesConsistentShapes()
    {
        // Arrange
        _encoder.eval();
        using var images = torch.randn(1, 3, 448, 448);

        // Act & Assert
        using (torch.no_grad())
        {
            for (int i = 0; i < 5; i++)
            {
                using var output = _encoder.forward(images);

                Assert.Equal(1, output.size(0));
                Assert.Equal(EncImageSize, output.size(1));
                Assert.Equal(EncImageSize, output.size(2));
                Assert.Equal(256, output.size(3));
            }
        }
    }

    [Fact]
    public void Forward_WithNormalizedImageInput_ProducesReasonableOutput()
    {
        // Arrange
        _encoder.eval();

        // Simulate normalized image input (mean~0, std~1)
        using var images = torch.randn(1, 3, 448, 448);

        // Act
        using (torch.no_grad())
        {
            using var output = _encoder.forward(images);

            // Assert: Check that output has reasonable statistics
            var mean = output.mean().item<float>();
            var std = output.std().item<float>();

            // Mean and std should be reasonable (not extreme values)
            Assert.True(Math.Abs(mean) < 100.0,
                $"Output mean should be reasonable, got {mean}");
            Assert.True(std > 0.01 && std < 1000.0,
                $"Output std should be reasonable, got {std}");
        }
    }

    [Fact]
    public void Forward_OutputChannels_Are256()
    {
        // Arrange
        using var images = torch.randn(1, 3, 448, 448);

        // Act
        using var output = _encoder.forward(images);

        // Assert
        // ResNet-18 layer3 outputs 256 channels
        Assert.Equal(256, output.size(3));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Forward_WithSmallInput_ProcessesCorrectly()
    {
        // Arrange
        using var images = torch.randn(1, 3, 32, 32);  // Very small input

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.Equal(EncImageSize, output.size(1));
        Assert.Equal(EncImageSize, output.size(2));
        Assert.True(output.isfinite().all().item<bool>(),
            "Should handle small inputs gracefully");
    }

    [Fact]
    public void Forward_WithLargeInput_ProcessesCorrectly()
    {
        // Arrange
        using var images = torch.randn(1, 3, 1024, 1024);  // Large input

        // Act
        using var output = _encoder.forward(images);

        // Assert
        Assert.Equal(EncImageSize, output.size(1));
        Assert.Equal(EncImageSize, output.size(2));
        Assert.True(output.isfinite().all().item<bool>(),
            "Should handle large inputs gracefully");
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var encoder = new Encoder04(EncImageSize);

        // Act & Assert: Should not throw
        encoder.Dispose();
        encoder.Dispose();
    }

    [Fact]
    public void Dispose_WithUsingStatement_DisposesCorrectly()
    {
        // Arrange & Act
        Encoder04? encoder;
        using (encoder = new Encoder04(EncImageSize))
        {
            // Use the module
            using var images = torch.randn(1, 3, 448, 448);
            using var output = encoder.forward(images);
            Assert.NotNull(output);
        }

        // Assert: Module should be disposed (no exception means success)
        Assert.NotNull(encoder);
    }

    #endregion
}
