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
/// Encoder based on ResNet-18 with BasicBlocks.
/// This implementation matches the PyTorch ResNet18 structure to enable proper weight loading.
/// </summary>
public sealed class Encoder04 : Module<Tensor, Tensor>
{
    private readonly ModuleList<Module<Tensor, Tensor>> _resnet;
    private readonly AdaptiveAvgPool2d _adaptive_pool;
    private readonly long _enc_image_size;
    private readonly long _encoder_dim;  // Note: encoder actually outputs 256, not 512

    /// <summary>
    /// Initializes a new instance of the <see cref="Encoder04"/> class.
    /// </summary>
    /// <param name="encImageSize">Encoded image size (assuming square output).</param>
    /// <param name="encDim">Encoder output dimensionality. Default is 512.</param>
    /// <param name="name">The name of the module.</param>
    public Encoder04(long encImageSize, long encDim = 512, string name = "Encoder04")
        : base(name)
    {
        _enc_image_size = encImageSize;
        _encoder_dim = encDim;

        // Build ResNet-18 architecture matching PyTorch structure
        // ResNet-18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4

        _resnet = new ModuleList<Module<Tensor, Tensor>>();

        // Layer 0: Initial 7x7 convolution
        _resnet.Add(Conv2d(3, 64, kernel_size: 7, stride: 2, padding: 3, bias: false));

        // Layer 1: Batch normalization
        _resnet.Add(BatchNorm2d(64));

        // Layer 2: ReLU (stored as identity since we'll apply it inline)
        _resnet.Add(Identity());

        // Layer 3: MaxPool2d
        _resnet.Add(MaxPool2d(kernel_size: 3, stride: 2, padding: 1));

        // Layer 4: layer1 - 2 BasicBlocks (64 -> 64)
        _resnet.Add(_make_layer(64, 64, blocks: 2, stride: 1, layer_idx: 0));

        // Layer 5: layer2 - 2 BasicBlocks (64 -> 128)
        _resnet.Add(_make_layer(64, 128, blocks: 2, stride: 2, layer_idx: 1));

        // Layer 6: layer3 - 2 BasicBlocks (128 -> 256)
        _resnet.Add(_make_layer(128, 256, blocks: 2, stride: 2, layer_idx: 2));

        // Register resnet as sequential for easier named_parameters access
        register_module("resnet", _resnet);

        // Adaptive pooling to ensure output size
        // Output will be (batch, 256, enc_image_size, enc_image_size)
        _adaptive_pool = AdaptiveAvgPool2d(new long[] { _enc_image_size, _enc_image_size });
        register_module("adaptive_pool", _adaptive_pool);
    }

    /// <summary>
    /// Creates a ResNet layer consisting of multiple BasicBlocks.
    /// </summary>
    private Sequential _make_layer(long inChannels, long outChannels, int blocks, long stride, int layer_idx)
    {
        var layers = new List<Module<Tensor, Tensor>>();

        // First block (may have downsample if dimensions change)
        bool needsDownsample = (stride != 1) || (inChannels != outChannels);
        layers.Add(new ResNetBasicBlock(inChannels, outChannels, stride, useDownsample: needsDownsample));

        // Remaining blocks (no downsample needed)
        for (int i = 1; i < blocks; i++)
        {
            layers.Add(new ResNetBasicBlock(outChannels, outChannels, stride: 1, useDownsample: false));
        }

        return Sequential(layers.ToArray());
    }

    /// <summary>
    /// Gets the encoder output dimensionality.
    /// </summary>
    public long GetEncoderDim() => _encoder_dim;

    /// <summary>
    /// Forward propagation through the encoder.
    /// </summary>
    /// <param name="images">Input images of shape (batch_size, image_channels, resized_image, resized_image).</param>
    /// <returns>Encoded images of shape (batch_size, enc_image_size, enc_image_size, encoder_dim).</returns>
    public override Tensor forward(Tensor images)
    {
        var x = images;

        // Pass through ResNet layers
        // Layer 0: conv1
        x = ((Conv2d)_resnet[0]).forward(x);

        // Layer 1: bn1
        x = ((BatchNorm2d)_resnet[1]).forward(x);

        // Layer 2: relu (inline)
        x = functional.relu(x, inplace: true);

        // Layer 3: maxpool
        x = ((MaxPool2d)_resnet[3]).forward(x);

        // Layers 4-6: ResNet blocks
        x = _resnet[4].forward(x);  // layer1
        x = _resnet[5].forward(x);  // layer2
        x = _resnet[6].forward(x);  // layer3 (outputs 256 channels)

        // DEBUG: Check shape before pooling
        if (System.Environment.GetEnvironmentVariable("DEBUG_ENCODER") == "1")
        {
            Console.WriteLine($"[ENC DEBUG] Before adaptive_pool: shape=[{x.size(0)}, {x.size(1)}, {x.size(2)}, {x.size(3)}]");
        }

        // Adaptive pooling to fixed size (256, enc_image_size, enc_image_size)
        var pooled = _adaptive_pool.forward(x);

        // DEBUG: Check shape after pooling
        if (System.Environment.GetEnvironmentVariable("DEBUG_ENCODER") == "1")
        {
            Console.WriteLine($"[ENC DEBUG] After adaptive_pool: shape=[{pooled.size(0)}, {pooled.size(1)}, {pooled.size(2)}, {pooled.size(3)}]");
            Console.WriteLine($"[ENC DEBUG] Target enc_image_size: {_enc_image_size}");
        }

        // Permute to match expected output format
        // From: (batch_size, 256, enc_image_size, enc_image_size)
        // To:   (batch_size, enc_image_size, enc_image_size, 256)
        var output = pooled.permute(0, 2, 3, 1);

        return output;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _resnet?.Dispose();
            _adaptive_pool?.Dispose();
        }
        base.Dispose(disposing);
    }
}
