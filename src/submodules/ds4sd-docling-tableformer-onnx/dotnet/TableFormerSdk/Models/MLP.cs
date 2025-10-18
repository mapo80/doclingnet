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
/// Very simple multi-layer perceptron (also called FFN - Feed Forward Network).
/// </summary>
public sealed class MLP : Module<Tensor, Tensor>
{
    private readonly List<Linear> _layers;
    private readonly long _numLayers;

    /// <summary>
    /// Initializes a new instance of the <see cref="MLP"/> class.
    /// </summary>
    /// <param name="inputDim">Input dimension.</param>
    /// <param name="hiddenDim">Hidden layer dimension.</param>
    /// <param name="outputDim">Output dimension.</param>
    /// <param name="numLayers">Number of layers.</param>
    public MLP(long inputDim, long hiddenDim, long outputDim, long numLayers)
        : base(nameof(MLP))
    {
        _numLayers = numLayers;
        _layers = new List<Linear>();

        // Build layer dimensions: [inputDim, hiddenDim, ..., hiddenDim, outputDim]
        var dims = new List<long>();
        dims.Add(inputDim);
        for (int i = 0; i < numLayers - 1; i++)
        {
            dims.Add(hiddenDim);
        }
        dims.Add(outputDim);

        // Create linear layers
        for (int i = 0; i < numLayers; i++)
        {
            var layer = Linear(dims[i], dims[i + 1]);
            _layers.Add(layer);
            register_module($"layers_{i}", layer);
        }
    }

    /// <summary>
    /// Forward pass through the MLP.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor after passing through all layers with ReLU activation (except the last layer).</returns>
    public override Tensor forward(Tensor input)
    {
        var x = input;

        for (int i = 0; i < _numLayers; i++)
        {
            var layer = _layers[i];
            x = layer.forward(x);

            // Apply ReLU to all layers except the last one
            if (i < _numLayers - 1)
            {
                x = functional.relu(x);
            }
        }

        return x;
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
            }
        }
        base.Dispose(disposing);
    }
}
