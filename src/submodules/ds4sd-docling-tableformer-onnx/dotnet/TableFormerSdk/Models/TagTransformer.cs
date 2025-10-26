//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace TableFormerSdk.Models;

// NOTE: This class is refactored. It no longer implements a single 'forward' pass.
// Instead, it has separate methods for the one-time image encoding and the per-step decoding.
public sealed class TagTransformer : Module
{
    private readonly long _nHeads;
    private readonly long _decoderDim;
    private readonly long _encImageSize;
    private readonly IReadOnlyList<long> _tdEncode;

    private readonly Embedding _embedding;
    private readonly PositionalEncoding _positionalEncoding;
    private readonly ResNetBasicBlock _input_filter_0;  // First BasicBlock 256->512 with downsample
    private readonly ResNetBasicBlock _input_filter_1;  // Second BasicBlock 512->512
    private readonly TransformerEncoder _encoder;
    private readonly TMTransformerDecoder _decoder;
    public Linear _fc;

    public IEnumerable<(string name, Parameter parameter)> GetEncoderParameters() => _encoder.named_parameters();

    /// <summary>
    /// Initializes a new instance of the <see cref="TagTransformer"/> class.
    /// </summary>
    public TagTransformer(
        long vocabSize,
        IReadOnlyList<long> tdEncode,
        long embedDim,
        long encoderLayers,
        long decoderLayers,
        long encImageSize,
        double dropout = 0.1,
        long nHeads = 4,
        long dimFf = 1024,
        long encoderDim = 512,
        string name = "TagTransformer")
        : base(name)
    {
        _nHeads = nHeads;
        _decoderDim = embedDim;
        _encImageSize = encImageSize;
        _tdEncode = tdEncode;

        // Token embedding layer
        _embedding = Embedding(vocabSize, embedDim);
        register_module(nameof(_embedding), _embedding);

        // Positional encoding
        _positionalEncoding = new PositionalEncoding(embedDim, dropout, maxLen: 1024);
        register_module(nameof(_positionalEncoding), _positionalEncoding);

        // Input filter: 2 ResNet BasicBlocks to project encoder features from 256 -> 512
        bool needsDownsample = encoderDim != embedDim;
        _input_filter_0 = new ResNetBasicBlock(encoderDim, embedDim, stride: 1, useDownsample: needsDownsample);
        register_module("_input_filter_0", _input_filter_0);
        
        _input_filter_1 = new ResNetBasicBlock(embedDim, embedDim, stride: 1, useDownsample: false);
        register_module("_input_filter_1", _input_filter_1);

        // Transformer Encoder (for image features)
        var encoderLayer = TransformerEncoderLayer(
            d_model: embedDim,
            nhead: nHeads,
            dim_feedforward: dimFf,
            dropout: dropout);

        _encoder = TransformerEncoder(encoderLayer, encoderLayers);
        register_module(nameof(_encoder), _encoder);

        // Transformer Decoder (for tag sequence)
        _decoder = new TMTransformerDecoder(
            dModel: embedDim,
            nhead: nHeads,
            numLayers: decoderLayers,
            dimFeedforward: dimFf,
            dropout: dropout);
        register_module(nameof(_decoder), _decoder);

        // Final classification layer
        _fc = Linear(embedDim, vocabSize);
        register_module(nameof(_fc), _fc);
    }

    public PositionalEncoding GetPositionalEncoding() => _positionalEncoding;

    /// <summary>
    /// Encodes the image features once. This should be called before the autoregressive loop.
    /// Corresponds to the initial part of the original Python forward pass.
    /// </summary>
    /// <param name="encInputs">Raw encoded image features from Encoder04, shape (batch, H, W, 256).</param>
    /// <returns>The processed memory bank for the decoder, shape (H*W, batch, 512).</returns>
    public Tensor EncodeImageFeatures(Tensor encInputs)
    {
        var batchSize = encInputs.size(0);

        // Permute from (batch, height, width, channels) to (batch, channels, height, width) for Conv2d
        using var encInputsNCHW = encInputs.permute(0, 3, 1, 2);

        // Apply input_filter: ResNet BasicBlocks (256 -> 512)
        using var filtered_0 = _input_filter_0.forward(encInputsNCHW);
        using var filtered = _input_filter_1.forward(filtered_0);

        // Permute back to (batch, height, width, channels=512)
        using var filteredNHWC = filtered.permute(0, 2, 3, 1);

        // Flatten spatial dimensions: (batch_size, height * width, 512)
        using var encInputsFlat = filteredNHWC.view(batchSize, -1, _decoderDim);

        // Permute to (seq_len, batch_size, 512) for transformer
        using var encInputsPermuted = encInputsFlat.permute(1, 0, 2);

        // Encode image features. This is the 'memory' for the decoder.
        var encoderOut = _encoder.forward(encInputsPermuted, src_mask: null, src_key_padding_mask: null);

        return encoderOut;
    }

    /// <summary>
    /// Performs one step of autoregressive decoding.
    /// </summary>
    /// <param name="tags">Current sequence of decoded tags, shape (batch, seq_len).</param>
    /// <param name="memory">The pre-encoded image features (from EncodeImageFeatures).</param>
    /// <param name="cache">The decoder cache from the previous step.</param>
    /// <returns>A tuple containing predictions, decoder hidden state, and the new cache.</returns>
    public (Tensor predictions, Tensor decoderOutput, Tensor newCache) DecodeStep(Tensor tags, Tensor memory, Tensor? cache)
    {
        // Embed and encode tag sequence
        // tags: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        using var tagsEmbedded = _embedding.forward(tags);

        // Permute to (seq_len, batch_size, embed_dim)
        using var tagsPermuted = tagsEmbedded.permute(1, 0, 2);

        // Add positional encoding
        using var tgt = _positionalEncoding.forward(tagsPermuted);

        // Decode with transformer decoder
        var (decoded, newCache) = _decoder.forward((tgt, memory, cache));

        // Permute back to (batch_size, seq_len, embed_dim)
        using var decodedPermuted = decoded.permute(1, 0, 2);

        // Apply final classification layer
        var predictions = _fc.forward(decodedPermuted);

        // Keep 'decoded' for returning (don't dispose)
        return (predictions, decoded, newCache);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _embedding?.Dispose();
            _positionalEncoding?.Dispose();
            _input_filter_0?.Dispose();
            _input_filter_1?.Dispose();
            _encoder?.Dispose();
            _decoder?.Dispose();
            _fc?.Dispose();
        }
        base.Dispose(disposing);
    }
}