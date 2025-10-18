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
/// Basic residual block for ResNet-18/34.
/// Consists of two 3x3 convolutions with batch normalization and ReLU activation.
/// </summary>
public sealed class ResNetBasicBlock : Module<Tensor, Tensor>
{
    private readonly Conv2d _conv1;
    private readonly BatchNorm2d _bn1;
    private readonly Conv2d _conv2;
    private readonly BatchNorm2d _bn2;
    private readonly Conv2d? _downsample_conv;
    private readonly BatchNorm2d? _downsample_bn;
    private readonly long _stride;
    private readonly bool _has_downsample;

    /// <summary>
    /// Initializes a new instance of the <see cref="ResNetBasicBlock"/> class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="stride">Stride for the first convolution. Default is 1.</param>
    /// <param name="useDownsample">Whether to use downsample for dimension matching.</param>
    /// <param name="name">Module name.</param>
    public ResNetBasicBlock(
        long inChannels,
        long outChannels,
        long stride = 1,
        bool useDownsample = false,
        string name = "BasicBlock")
        : base(name)
    {
        _stride = stride;
        _has_downsample = useDownsample;

        // First convolutional layer
        _conv1 = Conv2d(inChannels, outChannels, kernel_size: 3, stride: stride, padding: 1, bias: false);
        register_module("conv1", _conv1);

        _bn1 = BatchNorm2d(outChannels);
        register_module("bn1", _bn1);

        // Second convolutional layer
        _conv2 = Conv2d(outChannels, outChannels, kernel_size: 3, stride: 1, padding: 1, bias: false);
        register_module("conv2", _conv2);

        _bn2 = BatchNorm2d(outChannels);
        register_module("bn2", _bn2);

        // Create downsample layers if needed (for dimension matching)
        if (_has_downsample)
        {
            _downsample_conv = Conv2d(inChannels, outChannels, kernel_size: 1, stride: stride, bias: false);
            register_module("downsample_0", _downsample_conv);

            _downsample_bn = BatchNorm2d(outChannels);
            register_module("downsample_1", _downsample_bn);
        }
    }

    /// <summary>
    /// Forward pass through the basic block.
    /// </summary>
    /// <param name="x">Input tensor.</param>
    /// <returns>Output tensor after residual connection.</returns>
    public override Tensor forward(Tensor x)
    {
        var identity = x;

        // First conv block
        var @out = _conv1.forward(x);
        @out = _bn1.forward(@out);
        @out = functional.relu(@out, inplace: true);

        // Second conv block
        @out = _conv2.forward(@out);
        @out = _bn2.forward(@out);

        // Apply downsample to identity if needed (for dimension matching)
        if (_has_downsample)
        {
            identity = _downsample_conv!.forward(x);
            identity = _downsample_bn!.forward(identity);
        }

        // Residual connection
        @out = @out.add_(identity);
        @out = functional.relu(@out, inplace: true);

        return @out;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _conv1?.Dispose();
            _bn1?.Dispose();
            _conv2?.Dispose();
            _bn2?.Dispose();
            _downsample_conv?.Dispose();
            _downsample_bn?.Dispose();
        }
        base.Dispose(disposing);
    }
}
