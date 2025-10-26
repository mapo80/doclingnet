//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using TorchSharp.Modules;

namespace TableFormerSdk.Models;

/// <summary>
/// Custom Transformer Decoder Layer for TableFormer.
/// Modified to only use the last token for autoregressive decoding.
/// Ported from Python implementation in docling-ibm-models.
/// </summary>
public sealed class TMTransformerDecoderLayer : Module<(Tensor tgt, Tensor? memory), Tensor>
{
    private readonly MultiheadAttention _self_attn;
    private readonly MultiheadAttention _multihead_attn;
    private readonly Linear _linear1;
    private readonly Linear _linear2;
    private readonly Dropout _dropout;
    private readonly Dropout _dropout1;
    private readonly Dropout _dropout2;
    private readonly Dropout _dropout3;
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;
    private readonly LayerNorm _norm3;
    private readonly Func<Tensor, Tensor> _activation;

    /// <summary>
    /// Initializes a new instance of the <see cref="TMTransformerDecoderLayer"/> class.
    /// </summary>
    /// <param name="dModel">The number of expected features in the input.</param>
    /// <param name="nhead">The number of heads in the multiheadattention models.</param>
    /// <param name="dimFeedforward">The dimension of the feedforward network model. Default: 2048.</param>
    /// <param name="dropout">The dropout value. Default: 0.1.</param>
    /// <param name="name">The name of the module.</param>
    public TMTransformerDecoderLayer(
        long dModel,
        long nhead,
        long dimFeedforward = 2048,
        double dropout = 0.1,
        string name = "TMTransformerDecoderLayer")
        : base(name)
    {
        // Self-attention
        _self_attn = MultiheadAttention(dModel, nhead, dropout: dropout);
        register_module("self_attn", _self_attn);

        // Multihead attention (cross-attention)
        _multihead_attn = MultiheadAttention(dModel, nhead, dropout: dropout);
        register_module("multihead_attn", _multihead_attn);

        // Feedforward network
        _linear1 = Linear(dModel, dimFeedforward);
        register_module("linear1", _linear1);

        _linear2 = Linear(dimFeedforward, dModel);
        register_module("linear2", _linear2);

        // Layer norms
        _norm1 = LayerNorm(dModel);
        register_module("norm1", _norm1);

        _norm2 = LayerNorm(dModel);
        register_module("norm2", _norm2);

        _norm3 = LayerNorm(dModel);
        register_module("norm3", _norm3);

        // Dropouts
        _dropout = Dropout(dropout);
        register_module("dropout", _dropout);

        _dropout1 = Dropout(dropout);
        register_module("dropout1", _dropout1);

        _dropout2 = Dropout(dropout);
        register_module("dropout2", _dropout2);

        _dropout3 = Dropout(dropout);
        register_module("dropout3", _dropout3);

        // Activation function
        _activation = (x) => relu(x);
    }

    /// <summary>
    /// Forward pass through the transformer decoder layer.
    /// Modified to only use the last token for autoregressive decoding.
    /// </summary>
    /// <param name="input">Tuple containing:
    ///   - tgt: Target sequence (seq_len, batch_size, d_model)
    ///   - memory: Memory sequence from encoder (memory_len, batch_size, d_model), can be null
    /// </param>
    /// <returns>Output tensor (1, batch_size, d_model) - only last token embedding</returns>
    public override Tensor forward((Tensor tgt, Tensor? memory) input)
    {
        var (tgt, memory) = input;

        // From PyTorch but modified to only use the last tag
        // tgt_last_tok shape: (1, batch_size, d_model)
        using var tgtLastTok = tgt.index(TensorIndex.Slice(-1, null), TensorIndex.Colon, TensorIndex.Colon);

        // IMPORTANT: Create causal mask for autoregressive decoding
        // The causal mask ensures that position i can only attend to positions <= i
        // This prevents the model from "cheating" by looking at future tokens
        // For MultiheadAttention with query=(1, batch, d) and key=(seq_len, batch, d),
        // the mask shape should be (1, seq_len) where mask[i,j] = -inf means query i CANNOT attend to key j
        var tgtLen = tgt.size(0);
        Tensor? attnMask = null;

        // Only create mask if we have more than one token (no mask needed for first token)
        if (tgtLen > 1)
        {
            // Since we're only querying the last token (query shape: 1, batch, d_model)
            // and attending to all tokens (key shape: seq_len, batch, d_model),
            // we don't need a causal mask because the query is always the last token
            // and can naturally only attend to past tokens (positions 0 to seq_len-1)
            // The causal constraint is inherently satisfied by using only the last token as query
            attnMask = null;
        }

        // Self-attention: attend to all previous tokens including current (with causal mask)
        // Query: last token, Key/Value: all tokens
        var (selfAttnOut, _) = _self_attn.forward(
            tgtLastTok,  // query: last token (1, batch, d_model)
            tgt,         // key: all tokens (seq_len, batch, d_model)
            tgt,         // value: all tokens (seq_len, batch, d_model)
            attn_mask: attnMask,  // Causal mask (seq_len, seq_len) - CRITICAL for autoregressive generation!
            need_weights: false,
            key_padding_mask: null);  // No padding mask

        attnMask?.Dispose();

        // Residual connection + dropout + norm
        using var tmpTgt1 = _dropout1.forward(selfAttnOut);
        using var tgtAfterSelfAttn = tgtLastTok + tmpTgt1;
        var tgtNorm1 = _norm1.forward(tgtAfterSelfAttn);

        // Cross-attention with memory (if provided)
        Tensor tgtNorm2;
        if (memory is not null)
        {
            var (crossAttnOut, _) = _multihead_attn.forward(
                tgtNorm1,  // query: last token after self-attention
                memory,    // key: encoder output
                memory,    // value: encoder output
                attn_mask: null,  // No attention mask
                need_weights: false,
                key_padding_mask: null);  // No padding mask

            // Residual connection + dropout + norm
            using var tmpTgt2 = _dropout2.forward(crossAttnOut);
            using var tgtAfterCrossAttn = tgtNorm1 + tmpTgt2;
            tgtNorm2 = _norm2.forward(tgtAfterCrossAttn);
            tgtNorm1.Dispose();
        }
        else
        {
            tgtNorm2 = tgtNorm1;
        }

        // Feedforward network
        using var ff1 = _linear1.forward(tgtNorm2);
        using var ff1Activated = _activation(ff1);
        using var ff1Dropped = _dropout.forward(ff1Activated);
        using var ff2 = _linear2.forward(ff1Dropped);

        // Residual connection + dropout + norm
        using var tmpTgt3 = _dropout3.forward(ff2);
        using var tgtAfterFF = tgtNorm2 + tmpTgt3;
        var output = _norm3.forward(tgtAfterFF);

        selfAttnOut.Dispose();

        return output;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _self_attn?.Dispose();
            _multihead_attn?.Dispose();
            _linear1?.Dispose();
            _linear2?.Dispose();
            _dropout?.Dispose();
            _dropout1?.Dispose();
            _dropout2?.Dispose();
            _dropout3?.Dispose();
            _norm1?.Dispose();
            _norm2?.Dispose();
            _norm3?.Dispose();
        }
        base.Dispose(disposing);
    }
}
