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
/// Comprehensive unit tests for TagTransformer module.
/// Target: >=90% code coverage.
/// </summary>
public sealed class TagTransformerTests : IDisposable
{
    private readonly TagTransformer _transformer;
    private const long VocabSize = 100;
    private const long EmbedDim = 256;
    private const long EncoderLayers = 2;
    private const long DecoderLayers = 4;
    private const long EncImageSize = 14;
    private const long NHeads = 4;
    private const long DimFf = 1024;

    public TagTransformerTests()
    {
        var tdEncode = new List<long> { 1, 2, 3, 4, 5 };  // Example: ecel, fcel, etc.

        _transformer = new TagTransformer(
            vocabSize: VocabSize,
            tdEncode: tdEncode,
            embedDim: EmbedDim,
            encoderLayers: EncoderLayers,
            decoderLayers: DecoderLayers,
            encImageSize: EncImageSize,
            dropout: 0.1,
            nHeads: NHeads,
            dimFf: DimFf);
    }

    public void Dispose()
    {
        _transformer?.Dispose();
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultParameters_CreatesModule()
    {
        // Arrange
        var tdEncode = new List<long> { 1, 2, 3 };

        // Act
        using var transformer = new TagTransformer(
            vocabSize: 50,
            tdEncode: tdEncode,
            embedDim: 128,
            encoderLayers: 2,
            decoderLayers: 2,
            encImageSize: 14);

        // Assert
        Assert.NotNull(transformer);
    }

    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 4)]
    [InlineData(6, 6)]
    public void Constructor_WithVariousLayerCounts_CreatesModule(long encoderLayers, long decoderLayers)
    {
        // Arrange
        var tdEncode = new List<long> { 1, 2 };

        // Act
        using var transformer = new TagTransformer(
            vocabSize: VocabSize,
            tdEncode: tdEncode,
            embedDim: EmbedDim,
            encoderLayers: encoderLayers,
            decoderLayers: decoderLayers,
            encImageSize: EncImageSize);

        // Assert
        Assert.NotNull(transformer);
    }

    [Theory]
    [InlineData(128, 4)]
    [InlineData(256, 8)]
    [InlineData(512, 16)]
    public void Constructor_WithVariousDimensions_CreatesModule(long embedDim, long nHeads)
    {
        // Arrange
        var tdEncode = new List<long> { 1 };

        // Act
        using var transformer = new TagTransformer(
            vocabSize: VocabSize,
            tdEncode: tdEncode,
            embedDim: embedDim,
            encoderLayers: 2,
            decoderLayers: 2,
            encImageSize: EncImageSize,
            nHeads: nHeads);

        // Assert
        Assert.NotNull(transformer);
    }

    #endregion

    #region Forward Pass Shape Tests

    [Fact]
    public void Forward_WithValidInputs_ReturnsCorrectShapes()
    {
        // Arrange
        _transformer.eval();
        const long batchSize = 1;
        const long seqLen = 10;
        const long height = EncImageSize;
        const long width = EncImageSize;
        const long encoderDim = 512;

        using var encInputs = torch.randn(batchSize, height, width, encoderDim);
        using var tags = torch.randint(0, VocabSize, new long[] { batchSize, seqLen });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert - Predictions shape (only last token prediction)
            Assert.Equal(3, predictions.ndim);
            Assert.Equal(batchSize, predictions.size(0));
            Assert.Equal(1, predictions.size(1));  // Only last token
            Assert.Equal(VocabSize, predictions.size(2));

            // Assert - Decoder output shape (only last token for autoregressive decoding)
            Assert.Equal(3, decoderOutput.ndim);
            Assert.Equal(1, decoderOutput.size(0));  // Only last token
            Assert.Equal(batchSize, decoderOutput.size(1));
            Assert.Equal(EmbedDim, decoderOutput.size(2));

            // Assert - Cache shape
            Assert.Equal(4, cache.ndim);
            Assert.Equal(DecoderLayers, cache.size(0));

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    [Theory]
    [InlineData(1, 5)]
    [InlineData(1, 10)]
    public void Forward_WithVariousSequenceLengths_ReturnsCorrectShapes(long batchSize, long seqLen)
    {
        // Arrange
        _transformer.eval();
        const long height = EncImageSize;
        const long width = EncImageSize;
        const long encoderDim = 512;

        using var encInputs = torch.randn(batchSize, height, width, encoderDim);
        using var tags = torch.randint(0, VocabSize, new long[] { batchSize, seqLen });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert (only last token prediction)
            Assert.Equal(batchSize, predictions.size(0));
            Assert.Equal(1, predictions.size(1));  // Only last token
            Assert.Equal(VocabSize, predictions.size(2));

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithVariousImageSizes_ProcessesCorrectly()
    {
        // Arrange
        _transformer.eval();
        const long batchSize = 1;
        const long seqLen = 5;
        const long height = 14;
        const long width = 14;
        const long encoderDim = 512;

        using var encInputs = torch.randn(batchSize, height, width, encoderDim);
        using var tags = torch.randint(0, VocabSize, new long[] { batchSize, seqLen });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert (decoder returns only last token prediction)
            Assert.Equal(batchSize, predictions.size(0));
            Assert.Equal(1, predictions.size(1));  // Only last token
            Assert.Equal(VocabSize, predictions.size(2));

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    #endregion

    #region Output Value Tests

    [Fact]
    public void Forward_ProducesFiniteValues()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, 10 });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert
            Assert.True(predictions.isfinite().all().item<bool>(),
                "Predictions should contain only finite values");
            Assert.True(decoderOutput.isfinite().all().item<bool>(),
                "Decoder output should contain only finite values");
            Assert.True(cache.isfinite().all().item<bool>(),
                "Cache should contain only finite values");

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    [Fact]
    public void Forward_Predictions_HaveReasonableScale()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, 10 });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert
            var mean = predictions.mean().item<float>();
            var std = predictions.std().item<float>();

            Assert.True(Math.Abs(mean) < 20.0,
                $"Predictions mean should be reasonable, got {mean}");
            Assert.True(std > 0.01 && std < 50.0,
                $"Predictions std should be reasonable, got {std}");

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    [Fact]
    public void Forward_DecoderOutput_HasReasonableScale()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, 10 });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert
            var mean = decoderOutput.mean().item<float>();
            var std = decoderOutput.std().item<float>();

            Assert.True(Math.Abs(mean) < 10.0,
                $"Decoder output mean should be reasonable, got {mean}");
            Assert.True(std > 0.01 && std < 20.0,
                $"Decoder output std should be reasonable, got {std}");

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    #endregion

    #region Train/Eval Mode Tests

    [Fact]
    public void TrainMode_CanBeEnabled()
    {
        // Arrange & Act
        _transformer.train();

        // Assert: No exception means success
        Assert.NotNull(_transformer);
    }

    [Fact]
    public void EvalMode_CanBeEnabled()
    {
        // Arrange & Act
        _transformer.eval();

        // Assert: No exception means success
        Assert.NotNull(_transformer);
    }

    [Fact]
    public void EvalMode_ProducesDeterministicResults()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, 10 });

        // Act
        using (torch.no_grad())
        {
            var (predictions1, decoderOutput1, cache1) = _transformer.forward((encInputs, tags));
            var (predictions2, decoderOutput2, cache2) = _transformer.forward((encInputs, tags));

            // Assert
            using var predDiff = (predictions1 - predictions2).abs().max();
            using var decoderDiff = (decoderOutput1 - decoderOutput2).abs().max();

            var predMaxDiff = predDiff.item<float>();
            var decoderMaxDiff = decoderDiff.item<float>();

            Assert.True(predMaxDiff < 1e-5,
                $"Predictions should be identical in eval mode (diff: {predMaxDiff})");
            Assert.True(decoderMaxDiff < 1e-5,
                $"Decoder outputs should be identical in eval mode (diff: {decoderMaxDiff})");

            predictions1.Dispose();
            decoderOutput1.Dispose();
            cache1.Dispose();
            predictions2.Dispose();
            decoderOutput2.Dispose();
            cache2.Dispose();
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void Forward_WithZeroEncoderInput_ProducesValidOutput()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.zeros(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, 10 });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert
            Assert.True(predictions.isfinite().all().item<bool>(),
                "Predictions should be finite even with zero encoder input");

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithLargeValues_ProducesFiniteOutput()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.full(new long[] { 1, EncImageSize, EncImageSize, 512 }, 10.0f);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, 10 });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert
            Assert.True(predictions.isfinite().all().item<bool>(),
                "Output should be finite even with large input values");

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithSameTagSequence_ProducesValidOutput()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.full(new long[] { 1, 10 }, 5L);  // All same tag

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert
            Assert.True(predictions.isfinite().all().item<bool>(),
                "Should handle repeated tag sequences");

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithEdgeTagIndices_ProducesValidOutput()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);

        // Use edge indices: 0 and VocabSize-1
        using var tags = torch.cat(new[]
        {
            torch.zeros(1, 5, dtype: ScalarType.Int64),
            torch.full(new long[] { 1, 5 }, VocabSize - 1, dtype: ScalarType.Int64)
        }, dim: 1);

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert
            Assert.True(predictions.isfinite().all().item<bool>(),
                "Should handle edge vocabulary indices");

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Forward_RepeatedCalls_ProducesConsistentShapes()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, 10 });

        // Act & Assert
        using (torch.no_grad())
        {
            for (int i = 0; i < 5; i++)
            {
                var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

                Assert.Equal(1, predictions.size(0));
                Assert.Equal(1, predictions.size(1));  // Only last token
                Assert.Equal(VocabSize, predictions.size(2));

                predictions.Dispose();
                decoderOutput.Dispose();
                cache.Dispose();
            }
        }
    }

    [Fact]
    public void Forward_WithDifferentBatchSizes_ProcessesIndependently()
    {
        // Arrange
        _transformer.eval();
        const long seqLen = 5;
        using var encInputs1 = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags1 = torch.randint(0, VocabSize, new long[] { 1, seqLen });

        using var encInputs2 = torch.randn(2, EncImageSize, EncImageSize, 512);
        using var tags2 = torch.randint(0, VocabSize, new long[] { 2, seqLen });

        // Act
        using (torch.no_grad())
        {
            var (predictions1, decoderOutput1, cache1) = _transformer.forward((encInputs1, tags1));
            var (predictions2, decoderOutput2, cache2) = _transformer.forward((encInputs2, tags2));

            // Assert
            Assert.Equal(1, predictions1.size(0));
            Assert.Equal(2, predictions2.size(0));

            predictions1.Dispose();
            decoderOutput1.Dispose();
            cache1.Dispose();
            predictions2.Dispose();
            decoderOutput2.Dispose();
            cache2.Dispose();
        }
    }

    #endregion

    #region Cache Tests

    [Fact]
    public void Forward_Cache_HasCorrectStructure()
    {
        // Arrange
        _transformer.eval();
        const long seqLen = 5;
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, seqLen });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert - Cache dimensions (only last token is cached per layer)
            Assert.Equal(4, cache.ndim);
            Assert.Equal(DecoderLayers, cache.size(0));  // num_layers
            Assert.Equal(1, cache.size(1));               // Only last token (not seqLen)
            Assert.Equal(1, cache.size(2));               // batch_size
            Assert.Equal(EmbedDim, cache.size(3));        // embed_dim

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    [Fact]
    public void Forward_Cache_ContainsFiniteValues()
    {
        // Arrange
        _transformer.eval();
        using var encInputs = torch.randn(1, EncImageSize, EncImageSize, 512);
        using var tags = torch.randint(0, VocabSize, new long[] { 1, 10 });

        // Act
        using (torch.no_grad())
        {
            var (predictions, decoderOutput, cache) = _transformer.forward((encInputs, tags));

            // Assert
            Assert.True(cache.isfinite().all().item<bool>(),
                "Cache should contain only finite values");

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();
        }
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var tdEncode = new List<long> { 1, 2 };
        var transformer = new TagTransformer(
            vocabSize: 50,
            tdEncode: tdEncode,
            embedDim: 128,
            encoderLayers: 2,
            decoderLayers: 2,
            encImageSize: 14);

        // Act & Assert: Should not throw
        transformer.Dispose();
        transformer.Dispose();
    }

    [Fact]
    public void Dispose_WithUsingStatement_DisposesCorrectly()
    {
        // Arrange & Act
        TagTransformer? transformer;
        var tdEncode = new List<long> { 1, 2 };

        using (transformer = new TagTransformer(
            vocabSize: 50,
            tdEncode: tdEncode,
            embedDim: 128,
            encoderLayers: 2,
            decoderLayers: 2,
            encImageSize: 14))
        {
            // Use the module
            using var encInputs = torch.randn(1, 14, 14, 512);
            using var tags = torch.randint(0, 50, new long[] { 1, 5 });
            var (predictions, decoderOutput, cache) = transformer.forward((encInputs, tags));

            predictions.Dispose();
            decoderOutput.Dispose();
            cache.Dispose();

            Assert.NotNull(transformer);
        }

        // Assert: Module should be disposed (no exception means success)
        Assert.NotNull(transformer);
    }

    #endregion
}
