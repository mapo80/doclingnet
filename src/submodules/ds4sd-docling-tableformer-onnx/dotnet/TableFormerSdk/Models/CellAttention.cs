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
/// Attention network for cell-level attention mechanism.
/// Ported from Python implementation in docling-ibm-models.
/// </summary>
public sealed class CellAttention : Module<(Tensor encoderOut, Tensor decoderHidden, Tensor languageOut), (Tensor attentionWeightedEncoding, Tensor alpha)>
{
    private readonly Linear _encoder_att;
    private readonly Linear _tag_decoder_att;
    private readonly Linear _language_att;
    private readonly Linear _full_att;
    private readonly ReLU _relu;
    private readonly Softmax _softmax;

    /// <summary>
    /// Initializes a new instance of the <see cref="CellAttention"/> class.
    /// </summary>
    /// <param name="encoderDim">Feature size of encoded images.</param>
    /// <param name="tagDecoderDim">Size of tag decoder's RNN.</param>
    /// <param name="languageDim">Size of language model's RNN.</param>
    /// <param name="attentionDim">Size of the attention network.</param>
    /// <param name="name">The name of the module.</param>
    public CellAttention(
        long encoderDim,
        long tagDecoderDim,
        long languageDim,
        long attentionDim,
        string name = "CellAttention")
        : base(name)
    {
        // Linear layer to transform encoded image
        _encoder_att = Linear(encoderDim, attentionDim);
        register_module("encoder_att", _encoder_att);

        // Linear layer to transform tag decoder output
        _tag_decoder_att = Linear(tagDecoderDim, attentionDim);
        register_module("tag_decoder_att", _tag_decoder_att);

        // Linear layer to transform language model output
        _language_att = Linear(languageDim, attentionDim);
        register_module("language_att", _language_att);

        // Linear layer to calculate values to be softmax-ed
        _full_att = Linear(attentionDim, 1);
        register_module("full_att", _full_att);

        // Activation and normalization
        _relu = ReLU();
        register_module("relu", _relu);

        _softmax = Softmax(dim: 1);  // Softmax layer to calculate weights
        register_module("softmax", _softmax);
    }

    /// <summary>
    /// Forward propagation through the attention network.
    /// </summary>
    /// <param name="input">Tuple containing:
    ///   - encoderOut: Encoded images, tensor of dimension (1, num_pixels, encoder_dim)
    ///   - decoderHidden: Tag decoder output, tensor of dimension (num_cells, tag_decoder_dim)
    ///   - languageOut: Language model output, tensor of dimension (num_cells, language_dim)
    /// </param>
    /// <returns>Tuple containing:
    ///   - attentionWeightedEncoding: Attention-weighted encoding (num_cells, encoder_dim)
    ///   - alpha: Attention weights (num_cells, num_pixels)
    /// </returns>
    public override (Tensor attentionWeightedEncoding, Tensor alpha) forward((Tensor encoderOut, Tensor decoderHidden, Tensor languageOut) input)
    {
        var (encoderOut, decoderHidden, languageOut) = input;

        // Transform encoder output: (1, num_pixels, attention_dim)
        var att1 = _encoder_att.forward(encoderOut);

        // Transform decoder hidden state: (num_cells, attention_dim)
        var att2 = _tag_decoder_att.forward(decoderHidden);

        // Transform language output: (num_cells, attention_dim)
        var att3 = _language_att.forward(languageOut);

        // Compute attention scores
        // att1: (1, num_pixels, attention_dim)
        // att2.unsqueeze(1): (num_cells, 1, attention_dim)
        // att3.unsqueeze(1): (num_cells, 1, attention_dim)
        // Broadcasting: (num_cells, num_pixels, attention_dim)
        var combined = att1 + att2.unsqueeze(1) + att3.unsqueeze(1);

        // Apply ReLU and project to scalar
        var reluOut = _relu.forward(combined);
        var fullAtt = _full_att.forward(reluOut);  // (num_cells, num_pixels, 1)

        // Remove last dimension: (num_cells, num_pixels)
        var att = fullAtt.squeeze(2);

        // Apply softmax to get attention weights: (num_cells, num_pixels)
        var alpha = _softmax.forward(att);

        // Compute attention-weighted encoding
        // encoderOut: (1, num_pixels, encoder_dim)
        // alpha.unsqueeze(2): (num_cells, num_pixels, 1)
        // Broadcasting and sum over pixels: (num_cells, encoder_dim)
        var attentionWeightedEncoding = (encoderOut * alpha.unsqueeze(2)).sum(dim: 1);

        return (attentionWeightedEncoding, alpha);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _encoder_att?.Dispose();
            _tag_decoder_att?.Dispose();
            _language_att?.Dispose();
            _full_att?.Dispose();
            _relu?.Dispose();
            _softmax?.Dispose();
        }
        base.Dispose(disposing);
    }
}
