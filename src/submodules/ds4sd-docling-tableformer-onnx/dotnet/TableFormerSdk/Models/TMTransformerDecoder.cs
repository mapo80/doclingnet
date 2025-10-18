//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace TableFormerSdk.Models;

/// <summary>
/// Custom Transformer Decoder for TableFormer.
/// Manages multiple decoder layers with optional caching for inference.
/// Ported from Python implementation in docling-ibm-models.
/// </summary>
public sealed class TMTransformerDecoder : Module<(Tensor tgt, Tensor? memory, Tensor? cache), (Tensor output, Tensor outCache)>
{
    private readonly ModuleList<TMTransformerDecoderLayer> _layers;
    private readonly long _numLayers;

    /// <summary>
    /// Initializes a new instance of the <see cref="TMTransformerDecoder"/> class.
    /// </summary>
    /// <param name="decoderLayer">The decoder layer to replicate.</param>
    /// <param name="numLayers">The number of decoder layers.</param>
    /// <param name="name">The name of the module.</param>
    public TMTransformerDecoder(
        TMTransformerDecoderLayer decoderLayer,
        long numLayers,
        string name = "TMTransformerDecoder")
        : base(name)
    {
        _numLayers = numLayers;

        // Create list of decoder layers
        _layers = new ModuleList<TMTransformerDecoderLayer>();

        for (long i = 0; i < numLayers; i++)
        {
            // Create a new instance for each layer
            // Note: In production, layers should be independent instances with separate parameters
            var layer = new TMTransformerDecoderLayer(
                decoderLayer.GetType().GetProperty("DModel")?.GetValue(decoderLayer) as long? ?? 256,
                decoderLayer.GetType().GetProperty("Nhead")?.GetValue(decoderLayer) as long? ?? 8);

            _layers.Add(layer);
            register_module($"layers_{i}", layer);
        }
    }

    /// <summary>
    /// Alternative constructor that creates layers from parameters.
    /// </summary>
    /// <param name="dModel">The number of expected features in the input.</param>
    /// <param name="nhead">The number of heads in the multiheadattention models.</param>
    /// <param name="numLayers">The number of decoder layers.</param>
    /// <param name="dimFeedforward">The dimension of the feedforward network model.</param>
    /// <param name="dropout">The dropout value.</param>
    /// <param name="name">The name of the module.</param>
    public TMTransformerDecoder(
        long dModel,
        long nhead,
        long numLayers,
        long dimFeedforward = 2048,
        double dropout = 0.1,
        string name = "TMTransformerDecoder")
        : base(name)
    {
        _numLayers = numLayers;

        // Create list of decoder layers
        _layers = new ModuleList<TMTransformerDecoderLayer>();

        for (long i = 0; i < numLayers; i++)
        {
            var layer = new TMTransformerDecoderLayer(dModel, nhead, dimFeedforward, dropout);
            _layers.Add(layer);
            register_module($"layers_{i}", layer);
        }
    }

    /// <summary>
    /// Forward pass through the transformer decoder.
    /// </summary>
    /// <param name="input">Tuple containing:
    ///   - tgt: Target sequence (seq_len, batch_size, d_model)
    ///   - memory: Memory sequence from encoder (memory_len, batch_size, d_model), can be null
    ///   - cache: Optional cache from previous steps (num_layers, prev_len, batch_size, d_model)
    /// </param>
    /// <returns>Tuple containing:
    ///   - output: Output tensor (seq_len, batch_size, d_model)
    ///   - outCache: Cache for next step (num_layers, seq_len, batch_size, d_model)
    /// </returns>
    public override (Tensor output, Tensor outCache) forward((Tensor tgt, Tensor? memory, Tensor? cache) input)
    {
        var (tgt, memory, cache) = input;

        var output = tgt;
        var tagCacheList = new List<Tensor>();

        // Process through each decoder layer
        for (int i = 0; i < _layers.Count; i++)
        {
            var layer = _layers[i];

            // Pass through current layer (returns only last token)
            var layerOutput = layer.forward((output, memory));

            // Store layer output for caching (only the last token)
            tagCacheList.Add(layerOutput);

            // If we have cache from previous steps, concatenate for next layer
            if (cache is not null)
            {
                // cache[i] has shape: (prev_len, batch_size, d_model)
                // layerOutput has shape: (1, batch_size, d_model)
                using var cacheI = cache[i];
                output = torch.cat(new[] { cacheI, layerOutput }, dim: 0);
            }
            else
            {
                // No cache, use layer output directly
                output = layerOutput;
            }
        }

        // Build output cache
        Tensor outCache;
        if (cache is not null)
        {
            // Stack new outputs: (num_layers, 1, batch_size, d_model)
            using var newCache = torch.stack(tagCacheList.ToArray(), dim: 0);

            // Concatenate with existing cache along sequence dimension
            // cache: (num_layers, prev_len, batch_size, d_model)
            // newCache: (num_layers, 1, batch_size, d_model)
            // result: (num_layers, prev_len+1, batch_size, d_model)
            outCache = torch.cat(new[] { cache, newCache }, dim: 1);

            // Dispose intermediate tensors after stacking (they're copied into newCache)
            foreach (var tensor in tagCacheList)
            {
                tensor.Dispose();
            }
        }
        else
        {
            // No existing cache, just stack current outputs
            // Shape: (num_layers, 1, batch_size, d_model)
            outCache = torch.stack(tagCacheList.ToArray(), dim: 0);

            // Don't dispose tagCacheList tensors when no cache exists
            // because output might reference the last one
        }

        return (output, outCache);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            if (_layers is not null)
            {
                foreach (var layer in _layers)
                {
                    layer?.Dispose();
                }
                _layers.Dispose();
            }
        }
        base.Dispose(disposing);
    }
}
