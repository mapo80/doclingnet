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
/// Tag_Transformer: Transformer-based sequence decoder for table structure tags.
/// Based on "Attention Is All You Need" (Vaswani et al., 2017).
/// Ported from Python implementation in docling-ibm-models.
/// </summary>
public sealed class TagTransformer : Module<(Tensor encInputs, Tensor tags), (Tensor predictions, Tensor decoderOutput, Tensor cache)>
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
    private readonly Linear _fc;

    /// <summary>
    /// Initializes a new instance of the <see cref="TagTransformer"/> class.
    /// </summary>
    /// <param name="vocabSize">Vocabulary size for tag embeddings.</param>
    /// <param name="tdEncode">List of tag indices that represent table data cells (e.g., ecel, fcel).</param>
    /// <param name="embedDim">Embedding dimension (d_model).</param>
    /// <param name="encoderLayers">Number of transformer encoder layers.</param>
    /// <param name="decoderLayers">Number of transformer decoder layers.</param>
    /// <param name="encImageSize">Encoded image size (height/width).</param>
    /// <param name="dropout">Dropout probability (default: 0.1).</param>
    /// <param name="nHeads">Number of attention heads (default: 4).</param>
    /// <param name="dimFf">Feedforward network dimension (default: 1024).</param>
    /// <param name="encoderDim">Encoder feature dimension (default: 512 for ResNet).</param>
    /// <param name="name">Module name.</param>
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
        // This matches Python's _tag_transformer._input_filter
        // Input filter: 2 ResNet BasicBlocks to project encoder features from 256 -> 512
        // This matches Python's _tag_transformer._input_filter
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

    /// <summary>
    /// Creates the input filter: 2 ResNet BasicBlocks for dimension projection.
    /// Matches Python's _tag_transformer._input_filter structure.
    /// </summary>
    /// <summary>
    /// Forward pass for inference.
    /// </summary>
    /// <param name="input">Tuple of (encInputs, tags) where:
    ///   - encInputs: Encoded image features of shape (batch_size, height, width, encoder_dim)
    ///   - tags: Tag sequence tensor of shape (batch_size, seq_len)
    /// </param>
    /// <returns>Tuple of (predictions, decoderOutput, cache) where:
    ///   - predictions: Class logits of shape (batch_size, seq_len, vocab_size)
    ///   - decoderOutput: Decoder hidden states of shape (seq_len, batch_size, embed_dim)
    ///   - cache: Decoder cache for autoregressive generation
    /// </returns>
    public override (Tensor predictions, Tensor decoderOutput, Tensor cache) forward((Tensor encInputs, Tensor tags) input)
    {
        var (encInputs, tags) = input;

        // encInputs: (batch_size, height, width, encoder_dim=256)
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

        // Encode image features (no masking needed - all positions can attend to all)
        // src_mask: null (no attention masking)
        // src_key_padding_mask: null (no padding mask needed for image features)
        using var encoderOut = _encoder.forward(encInputsPermuted, src_mask: null, src_key_padding_mask: null);

        // Embed and encode tag sequence
        // tags: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        using var tagsEmbedded = _embedding.forward(tags);

        // Permute to (seq_len, batch_size, embed_dim)
        using var tagsPermuted = tagsEmbedded.permute(1, 0, 2);

        // Add positional encoding
        using var tgt = _positionalEncoding.forward(tagsPermuted);

        // Decode with transformer decoder
        var (decoded, cache) = _decoder.forward((tgt, encoderOut, null));

        // Permute back to (batch_size, seq_len, embed_dim)
        using var decodedPermuted = decoded.permute(1, 0, 2);

        // Apply final classification layer
        var predictions = _fc.forward(decodedPermuted);

        // Keep decoded for returning (don't dispose)
        return (predictions, decoded, cache);
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
