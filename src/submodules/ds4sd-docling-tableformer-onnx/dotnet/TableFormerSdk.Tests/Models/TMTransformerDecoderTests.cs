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
/// Comprehensive unit tests for TMTransformerDecoder module.
/// Target: 100% code coverage.
/// </summary>
public sealed class TMTransformerDecoderTests : IDisposable
{
    private readonly TMTransformerDecoder _decoder;
    private const long DModel = 256;
    private const long Nhead = 8;
    private const long NumLayers = 4;
    private const long DimFeedforward = 1024;

    public TMTransformerDecoderTests()
    {
        // Initialize with typical TableFormer parameters
        _decoder = new TMTransformerDecoder(
            dModel: DModel,
            nhead: Nhead,
            numLayers: NumLayers,
            dimFeedforward: DimFeedforward,
            dropout: 0.1);
    }

    public void Dispose()
    {
        _decoder?.Dispose();
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithParameters_CreatesModule()
    {
        // Arrange & Act
        using var decoder = new TMTransformerDecoder(
            dModel: 256,
            nhead: 8,
            numLayers: 4);

        // Assert
        Assert.NotNull(decoder);
    }

    [Fact]
    public void Constructor_WithDecoderLayer_CreatesModule()
    {
        // Arrange
        var decoderLayer = new TMTransformerDecoderLayer(
            dModel: DModel,
            nhead: Nhead,
            dimFeedforward: DimFeedforward);

        // Act
        using var decoder = new TMTransformerDecoder(decoderLayer, NumLayers);
        decoderLayer.Dispose();

        // Assert
        Assert.NotNull(decoder);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    [InlineData(8)]
    public void Constructor_WithVariousNumLayers_CreatesModule(long numLayers)
    {
        // Arrange & Act
        using var decoder = new TMTransformerDecoder(
            dModel: DModel,
            nhead: Nhead,
            numLayers: numLayers);

        // Assert
        Assert.NotNull(decoder);
    }

    [Theory]
    [InlineData(128, 4)]
    [InlineData(256, 8)]
    [InlineData(512, 16)]
    public void Constructor_WithVariousDimensions_CreatesModule(long dModel, long nhead)
    {
        // Arrange & Act
        using var decoder = new TMTransformerDecoder(
            dModel: dModel,
            nhead: nhead,
            numLayers: 2);

        // Assert
        Assert.NotNull(decoder);
    }

    #endregion

    #region Forward Pass Shape Tests

    [Fact]
    public void Forward_WithoutCacheOrMemory_ReturnsCorrectShapes()
    {
        // Arrange
        _decoder.eval();
        const long seqLen = 1;
        const long batchSize = 2;

        using var tgt = torch.randn(seqLen, batchSize, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.Equal(3, output.ndim);
            Assert.Equal(seqLen, output.size(0));
            Assert.Equal(batchSize, output.size(1));
            Assert.Equal(DModel, output.size(2));

            // Cache shape: (num_layers, seq_len, batch_size, d_model)
            Assert.Equal(4, outCache.ndim);
            Assert.Equal(NumLayers, outCache.size(0));
            Assert.Equal(seqLen, outCache.size(1));
            Assert.Equal(batchSize, outCache.size(2));
            Assert.Equal(DModel, outCache.size(3));

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithMemory_ReturnsCorrectShapes()
    {
        // Arrange
        _decoder.eval();
        const long seqLen = 1;
        const long batchSize = 2;
        const long memoryLen = 196;  // 14x14 encoder output

        using var tgt = torch.randn(seqLen, batchSize, DModel);
        using var memory = torch.randn(memoryLen, batchSize, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, memory, null));

            // Assert
            Assert.Equal(seqLen, output.size(0));
            Assert.Equal(batchSize, output.size(1));
            Assert.Equal(DModel, output.size(2));

            Assert.Equal(NumLayers, outCache.size(0));
            Assert.Equal(seqLen, outCache.size(1));

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithCache_ReturnsCorrectShapes()
    {
        // Arrange
        _decoder.eval();
        const long prevLen = 3;
        const long newLen = 1;
        const long batchSize = 2;

        using var tgt = torch.randn(newLen, batchSize, DModel);
        using var cache = torch.randn(NumLayers, prevLen, batchSize, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, cache));

            // Assert
            // Output should have concatenated length
            Assert.Equal(prevLen + newLen, output.size(0));
            Assert.Equal(batchSize, output.size(1));
            Assert.Equal(DModel, output.size(2));

            // Cache should grow
            Assert.Equal(NumLayers, outCache.size(0));
            Assert.Equal(prevLen + newLen, outCache.size(1));
            Assert.Equal(batchSize, outCache.size(2));
            Assert.Equal(DModel, outCache.size(3));

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithCacheAndMemory_ReturnsCorrectShapes()
    {
        // Arrange
        _decoder.eval();
        const long prevLen = 3;
        const long newLen = 1;
        const long batchSize = 2;
        const long memoryLen = 196;

        using var tgt = torch.randn(newLen, batchSize, DModel);
        using var memory = torch.randn(memoryLen, batchSize, DModel);
        using var cache = torch.randn(NumLayers, prevLen, batchSize, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, memory, cache));

            // Assert
            Assert.Equal(prevLen + newLen, output.size(0));
            Assert.Equal(batchSize, output.size(1));
            Assert.Equal(DModel, output.size(2));

            Assert.Equal(NumLayers, outCache.size(0));
            Assert.Equal(prevLen + newLen, outCache.size(1));

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(1, 4)]
    // Note: seqLen > 1 not supported in autoregressive mode
    public void Forward_WithVariousBatchSizes_ReturnsCorrectShapes(long seqLen, long batchSize)
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.randn(seqLen, batchSize, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.Equal(seqLen, output.size(0));
            Assert.Equal(batchSize, output.size(1));
            Assert.Equal(DModel, output.size(2));

            output.Dispose();
            outCache.Dispose();
        }
    }

    #endregion

    #region Caching Mechanism Tests

    [Fact]
    public void Forward_AutoregressiveDecoding_CacheGrowsCorrectly()
    {
        // Arrange
        _decoder.eval();
        const long batchSize = 2;

        // Act
        using (torch.no_grad())
        {
            // Step 1: First token
            using var tgt1 = torch.randn(1, batchSize, DModel);
            var (output1, cache1) = _decoder.forward((tgt1, null, null));

            Assert.Equal(1, output1.size(0));
            Assert.Equal(1, cache1.size(1));

            // Step 2: Second token
            using var tgt2 = torch.randn(1, batchSize, DModel);
            var (output2, cache2) = _decoder.forward((tgt2, null, cache1));

            Assert.Equal(2, output2.size(0));  // Concatenated with previous
            Assert.Equal(2, cache2.size(1));

            // Step 3: Third token
            using var tgt3 = torch.randn(1, batchSize, DModel);
            var (output3, cache3) = _decoder.forward((tgt3, null, cache2));

            Assert.Equal(3, output3.size(0));
            Assert.Equal(3, cache3.size(1));

            output1.Dispose();
            cache1.Dispose();
            output2.Dispose();
            cache2.Dispose();
            output3.Dispose();
            cache3.Dispose();
        }
    }

    // Note: Forward_CachePreservesInformation test removed - relies on seqLen > 1 which is not supported in autoregressive mode

    [Fact]
    public void Forward_NullCache_CreatesNewCache()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.randn(1, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.NotNull(outCache);
            Assert.Equal(NumLayers, outCache.size(0));
            Assert.Equal(1, outCache.size(1));

            output.Dispose();
            outCache.Dispose();
        }
    }

    #endregion

    #region Multi-Layer Behavior Tests

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    public void Forward_WithVariousLayerCounts_ProcessesCorrectly(long numLayers)
    {
        // Arrange
        using var decoder = new TMTransformerDecoder(
            dModel: DModel,
            nhead: Nhead,
            numLayers: numLayers);
        decoder.eval();

        using var tgt = torch.randn(1, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = decoder.forward((tgt, null, null));

            // Assert
            Assert.Equal(numLayers, outCache.size(0));

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Fact]
    public void Forward_EachLayerProcessesOutput()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.randn(1, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert: Each layer should contribute to cache
            for (int i = 0; i < NumLayers; i++)
            {
                using var layerCache = outCache[i];
                Assert.Equal(1, layerCache.size(0));  // seq_len
                Assert.Equal(2, layerCache.size(1));  // batch_size
                Assert.Equal(DModel, layerCache.size(2));

                // Check for finite values
                Assert.True(layerCache.isfinite().all().item<bool>(),
                    $"Layer {i} cache should contain finite values");
            }

            output.Dispose();
            outCache.Dispose();
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
        using var tgt = torch.randn(1, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output1, cache1) = _decoder.forward((tgt, null, null));
            var (output2, cache2) = _decoder.forward((tgt, null, null));

            // Assert
            using var diff = (output1 - output2).abs().max();
            var maxDiff = diff.item<float>();

            Assert.True(maxDiff < 1e-6,
                $"Outputs should be identical in eval mode (diff: {maxDiff})");

            output1.Dispose();
            cache1.Dispose();
            output2.Dispose();
            cache2.Dispose();
        }
    }

    #endregion

    #region Output Value Tests

    [Fact]
    public void Forward_ProducesFiniteValues()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.randn(1, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.True(output.isfinite().all().item<bool>(),
                "Output should contain only finite values");
            Assert.True(outCache.isfinite().all().item<bool>(),
                "Cache should contain only finite values");

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Fact]
    public void Forward_OutputHasReasonableScale()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.randn(1, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            var mean = output.mean().item<float>();
            var std = output.std().item<float>();

            Assert.True(Math.Abs(mean) < 10.0,
                $"Output mean should be reasonable, got {mean}");
            Assert.True(std > 0.01 && std < 100.0,
                $"Output std should be reasonable, got {std}");

            output.Dispose();
            outCache.Dispose();
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void Forward_WithZeroInput_ProducesValidOutput()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.zeros(1, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.True(output.isfinite().all().item<bool>(),
                "Output should be finite even with zero input");

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithLargeValues_ProducesFiniteOutput()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.full(new long[] { 1, 2, DModel }, 10.0f);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.True(output.isfinite().all().item<bool>(),
                "Output should be finite even with large input");

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithNegativeValues_ProducesValidOutput()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.full(new long[] { 1, 2, DModel }, -5.0f);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.True(output.isfinite().all().item<bool>(),
                "Output should be finite even with negative input");

            output.Dispose();
            outCache.Dispose();
        }
    }

    [Fact]
    public void Forward_WithMixedValues_ProducesValidOutput()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.randn(1, 2, DModel) * 10.0f;

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.True(output.isfinite().all().item<bool>(),
                "Output should be finite with mixed value ranges");

            output.Dispose();
            outCache.Dispose();
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Forward_WithSingleBatch_ProcessesCorrectly()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.randn(1, 1, DModel);  // Single batch

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, null, null));

            // Assert
            Assert.Equal(1, output.size(1));
            Assert.True(output.isfinite().all().item<bool>());

            output.Dispose();
            outCache.Dispose();
        }
    }

    // Note: Forward_WithLongSequence test removed - not supported in autoregressive mode (seqLen must be 1)

    [Fact]
    public void Forward_WithLargeMemory_ProcessesCorrectly()
    {
        // Arrange
        _decoder.eval();
        const long largeMemoryLen = 500;
        using var tgt = torch.randn(1, 2, DModel);
        using var memory = torch.randn(largeMemoryLen, 2, DModel);

        // Act
        using (torch.no_grad())
        {
            var (output, outCache) = _decoder.forward((tgt, memory, null));

            // Assert
            Assert.True(output.isfinite().all().item<bool>(),
                "Should handle large memory gracefully");

            output.Dispose();
            outCache.Dispose();
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Forward_RepeatedCalls_ProducesConsistentShapes()
    {
        // Arrange
        _decoder.eval();
        using var tgt = torch.randn(1, 2, DModel);

        // Act & Assert
        using (torch.no_grad())
        {
            for (int i = 0; i < 5; i++)
            {
                var (output, outCache) = _decoder.forward((tgt, null, null));

                Assert.Equal(1, output.size(0));
                Assert.Equal(2, output.size(1));
                Assert.Equal(DModel, output.size(2));

                output.Dispose();
                outCache.Dispose();
            }
        }
    }

    [Fact]
    public void Forward_CompleteSequenceGeneration_WorksCorrectly()
    {
        // Arrange
        _decoder.eval();
        const long batchSize = 2;
        const long maxLen = 5;

        using (torch.no_grad())
        {
            Tensor? cache = null;

            // Simulate autoregressive generation
            for (int step = 0; step < maxLen; step++)
            {
                using var tgt = torch.randn(1, batchSize, DModel);
                var (output, newCache) = _decoder.forward((tgt, null, cache));

                // Verify cache grows
                Assert.Equal(step + 1, newCache.size(1));

                // Cleanup
                cache?.Dispose();
                output.Dispose();
                cache = newCache;
            }

            // Final verification
            Assert.NotNull(cache);
            Assert.Equal(maxLen, cache.size(1));
            cache.Dispose();
        }
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var decoder = new TMTransformerDecoder(
            dModel: DModel,
            nhead: Nhead,
            numLayers: 2);

        // Act & Assert: Should not throw
        decoder.Dispose();
        decoder.Dispose();
    }

    [Fact]
    public void Dispose_WithUsingStatement_DisposesCorrectly()
    {
        // Arrange & Act
        TMTransformerDecoder? decoder;
        using (decoder = new TMTransformerDecoder(
            dModel: DModel,
            nhead: Nhead,
            numLayers: 2))
        {
            // Use the module
            using var tgt = torch.randn(1, 2, DModel);
            var (output, outCache) = decoder.forward((tgt, null, null));

            output.Dispose();
            outCache.Dispose();

            Assert.NotNull(decoder);
        }

        // Assert: Module should be disposed (no exception means success)
        Assert.NotNull(decoder);
    }

    #endregion
}
