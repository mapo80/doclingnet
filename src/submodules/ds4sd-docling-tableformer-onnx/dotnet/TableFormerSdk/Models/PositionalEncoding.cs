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
/// Positional encoding for transformer models.
/// Ported from Python implementation in docling-ibm-models.
/// </summary>
public sealed class PositionalEncoding : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Dropout _dropout;
    private readonly Tensor _pe;

    /// <summary>
    /// Initializes a new instance of the <see cref="PositionalEncoding"/> class.
    /// </summary>
    /// <param name="dModel">The dimensionality of the model embeddings.</param>
    /// <param name="dropout">The dropout probability. Default is 0.1.</param>
    /// <param name="maxLen">The maximum sequence length. Default is 1024.</param>
    /// <param name="name">The name of the module.</param>
    public PositionalEncoding(long dModel, double dropout = 0.1, long maxLen = 1024, string name = "PositionalEncoding")
        : base(name)
    {
        _dropout = Dropout(dropout);
        register_module("dropout", _dropout);

        // Create positional encoding matrix
        // pe shape: (max_len, d_model)
        var pe = torch.zeros(maxLen, dModel);

        // position shape: (max_len, 1)
        var position = torch.arange(0, maxLen, dtype: ScalarType.Float32).unsqueeze(1);

        // div_term calculation: exp(arange(0, d_model, 2) * (-log(10000.0) / d_model))
        var divTerm = torch.exp(
            torch.arange(0, dModel, 2, dtype: ScalarType.Float32) * (-Math.Log(10000.0) / dModel)
        );

        // Apply sin to even indices (0, 2, 4, ...)
        // pe[:, 0::2] = torch.sin(position * div_term)
        pe.index_put_(torch.sin(position * divTerm), TensorIndex.Colon, TensorIndex.Slice(0, null, 2));

        // Apply cos to odd indices (1, 3, 5, ...)
        // pe[:, 1::2] = torch.cos(position * div_term)
        pe.index_put_(torch.cos(position * divTerm), TensorIndex.Colon, TensorIndex.Slice(1, null, 2));

        // Reshape pe from (max_len, d_model) to (1, max_len, d_model)
        // In Python: pe = pe.unsqueeze(0)  (NO transpose!)
        pe = pe.unsqueeze(0);

        // Register as buffer (not a trainable parameter)
        register_buffer("pe", pe);
        _pe = pe;
    }

    /// <summary>
    /// Forward pass of the positional encoding.
    /// </summary>
    /// <param name="x">Input tensor of shape (seq_len, batch_size, d_model).</param>
    /// <returns>Output tensor with positional encoding added, shape (seq_len, batch_size, d_model).</returns>
    public override Tensor forward(Tensor x)
    {
        // Add positional encoding to input
        // x shape: (seq_len, batch_size, d_model)
        // pe shape: (1, max_len, d_model)
        // We need to permute pe to match: (max_len, 1, d_model) then slice to (seq_len, 1, d_model)
        var seqLen = x.size(0);

        // Permute pe from (1, max_len, d_model) to (max_len, 1, d_model)
        using var pePermuted = _pe.permute(1, 0, 2);

        // Slice to match sequence length: (seq_len, 1, d_model)
        var peSlice = pePermuted.index(TensorIndex.Slice(null, seqLen), TensorIndex.Colon, TensorIndex.Colon);

        x = x + peSlice;

        // Apply dropout
        return _dropout.forward(x);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _dropout?.Dispose();
            _pe?.Dispose();
        }
        base.Dispose(disposing);
    }
}
