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
/// BBoxDecoder generates bounding box predictions for detected table cells.
/// Ported from Python implementation in docling-ibm-models.
/// </summary>
public sealed class BBoxDecoder : Module<(Tensor encoderOut, Tensor tagH), (Tensor classes, Tensor boxes)>
{
    private readonly long _encoderDim;
    private readonly long _encoderRawDim;  // Raw encoder output dimension (256)
    private readonly long _attentionDim;
    private readonly long _embedDim;
    private readonly long _decoderDim;
    private readonly long _numClasses;
    private readonly double _dropout;

    private readonly ResNetBasicBlock _input_filter_0;  // First BasicBlock 256->512 with downsample
    private readonly ResNetBasicBlock _input_filter_1;  // Second BasicBlock 512->512
    private readonly CellAttention _attention;
    private readonly Linear _init_h;
    private readonly Linear _f_beta;
    private readonly Sigmoid _sigmoid;
    private readonly Dropout _dropoutLayer;
    private readonly Linear _class_embed;
    private readonly MLP _bbox_embed;

    /// <summary>
    /// Initializes a new instance of the <see cref="BBoxDecoder"/> class.
    /// </summary>
    /// <param name="attentionDim">Size of attention network.</param>
    /// <param name="embedDim">Embedding size.</param>
    /// <param name="tagDecoderDim">Size of tag decoder's RNN.</param>
    /// <param name="decoderDim">Size of decoder's RNN.</param>
    /// <param name="numClasses">Number of bbox classes (usually 3: cell, row header, column header).</param>
    /// <param name="encoderRawDim">Raw encoder output dimension (default: 256).</param>
    /// <param name="encoderDim">Feature size after input_filter projection (default: 512).</param>
    /// <param name="dropout">Dropout probability (default: 0.5).</param>
    /// <param name="name">Module name.</param>
    public BBoxDecoder(
        long attentionDim,
        long embedDim,
        long tagDecoderDim,
        long decoderDim,
        long numClasses,
        long encoderRawDim = 256,
        long encoderDim = 512,
        double dropout = 0.5,
        string name = "BBoxDecoder")
        : base(name)
    {
        _encoderRawDim = encoderRawDim;
        _encoderDim = encoderDim;
        _attentionDim = attentionDim;
        _embedDim = embedDim;
        _decoderDim = decoderDim;
        _numClasses = numClasses;
        _dropout = dropout;

        // Input filter: 2 ResNet BasicBlocks to project encoder features from 256 -> 512
        bool needsDownsample = encoderRawDim != encoderDim;
        _input_filter_0 = new ResNetBasicBlock(encoderRawDim, encoderDim, stride: 1, useDownsample: needsDownsample);
        register_module("_input_filter_0", _input_filter_0);

        _input_filter_1 = new ResNetBasicBlock(encoderDim, encoderDim, stride: 1, useDownsample: false);
        register_module("_input_filter_1", _input_filter_1);

        // Attention network
        _attention = new CellAttention(encoderDim, tagDecoderDim, decoderDim, attentionDim);
        register_module(nameof(_attention), _attention);

        // Initialize hidden state from encoder output
        _init_h = Linear(encoderDim, decoderDim);
        register_module(nameof(_init_h), _init_h);

        // Gating mechanism for attention
        _f_beta = Linear(decoderDim, encoderDim);
        register_module(nameof(_f_beta), _f_beta);

        _sigmoid = Sigmoid();
        register_module(nameof(_sigmoid), _sigmoid);

        _dropoutLayer = Dropout(_dropout);
        register_module(nameof(_dropoutLayer), _dropoutLayer);

        // Classification head (numClasses + 1 for no-object class)
        // Input dimension is decoderDim (the hidden state dimension)
        _class_embed = Linear(encoderDim, _numClasses + 1);
        register_module(nameof(_class_embed), _class_embed);

        // Bounding box regression head (MLP with 3 layers)
        // Input dimension is decoderDim (the hidden state dimension)
        _bbox_embed = new MLP(encoderDim, 256, 4, 3);
        register_module(nameof(_bbox_embed), _bbox_embed);
    }

    /// <summary>
    /// Initialize hidden state from encoder output.
    /// </summary>
    /// <param name="encoderOut">Encoder output tensor of shape (batch_size, num_pixels, encoder_dim).</param>
    /// <param name="batchSize">Batch size.</param>
    /// <returns>Initial hidden state of shape (batch_size, decoder_dim).</returns>
    private Tensor InitHiddenState(Tensor encoderOut, long batchSize)
    {
        // Take mean over spatial dimensions (dimension 1)
        using var meanEncoderOut = encoderOut.mean(new long[] { 1 });

        // Project to decoder dimension and expand to batch size
        var h = _init_h.forward(meanEncoderOut).expand(batchSize, -1);

        return h;
    }

    /// <summary>
    /// Forward pass for inference. Processes each cell independently.
    /// </summary>
    /// <param name="input">Tuple of (encoderOut, tagH) where:
    ///   - encoderOut: Encoded image features of shape (batch_size, height, width, encoder_dim)
    ///   - tagH: Tag decoder hidden states, list/tensor for each cell
    /// </param>
    /// <returns>Tuple of (classes, boxes) where:
    ///   - classes: Predicted classes of shape (num_cells, num_classes+1)
    ///   - boxes: Predicted bounding boxes of shape (num_cells, 4) with sigmoid applied
    /// </returns>
    public override (Tensor classes, Tensor boxes) forward((Tensor encoderOut, Tensor tagH) input)
    {
        var (encoderOut, tagH) = input;

        // Apply input_filter: (batch_size, height, width, 256) -> (batch_size, height, width, 512)
        // Permute to NCHW for Conv2d: (batch, height, width, channels) -> (batch, channels, height, width)
        using var encoderNCHW = encoderOut.permute(0, 3, 1, 2);

        // Apply input_filter: ResNet BasicBlocks (256 -> 512)
        using var filtered_0 = _input_filter_0.forward(encoderNCHW);
        using var filtered = _input_filter_1.forward(filtered_0);

        // Permute back to NHWC: (batch, channels, height, width) -> (batch, height, width, channels)
        using var filteredNHWC = filtered.permute(0, 2, 3, 1);

        // Flatten encoder output: (batch_size, height, width, 512) -> (1, num_pixels, 512)
        using var encoderOutFlat = filteredNHWC.view(1, -1, _encoderDim);

        // Get number of cells from tagH
        // tagH should have shape (num_cells, tag_decoder_dim)
        var numCells = tagH.size(0);

        var predictionsBboxes = new List<Tensor>();
        var predictionsClasses = new List<Tensor>();

        // Process each cell
        for (long cId = 0; cId < numCells; cId++)
        {
            // Initialize hidden state
            using var h = InitHiddenState(encoderOutFlat, 1);

            // Get current cell's tag hidden state
            using var cellTagH = tagH[cId].unsqueeze(0);  // (1, tag_decoder_dim)

            // Apply attention
            var (awe, alpha) = _attention.forward((encoderOutFlat, cellTagH, h));
            alpha.Dispose();

            // Apply gating and dropout to the attention context
            using var gate = _sigmoid.forward(_f_beta.forward(h));
            using var gatedContext = gate * awe;
            awe.Dispose();

            using var context = _dropoutLayer.forward(gatedContext);

            // Predict bounding box (apply sigmoid for normalized coordinates)
            using var bboxRaw = _bbox_embed.forward(context);
            var bbox = bboxRaw.sigmoid();
            predictionsBboxes.Add(bbox);

            // Predict class logits
            var classLogits = _class_embed.forward(context);
            predictionsClasses.Add(classLogits);
        }

        // Stack predictions
        Tensor classes;
        Tensor boxes;

        if (predictionsBboxes.Count > 0)
        {
            // Stack and squeeze: (num_cells, 1, 4) -> (num_cells, 4)
            using var boxesStacked = torch.stack(predictionsBboxes.ToArray(), dim: 0);
            boxes = boxesStacked.squeeze(1);

            using var classesStacked = torch.stack(predictionsClasses.ToArray(), dim: 0);
            classes = classesStacked.squeeze(1);
        }
        else
        {
            // No cells, return empty tensors
            boxes = torch.empty(0, 4);
            classes = torch.empty(0, _numClasses + 1);
        }

        // Dispose intermediate tensors
        foreach (var bbox in predictionsBboxes)
        {
            bbox.Dispose();
        }
        foreach (var cls in predictionsClasses)
        {
            cls.Dispose();
        }

        return (classes, boxes);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _input_filter_0?.Dispose();
            _input_filter_1?.Dispose();
            _attention?.Dispose();
            _init_h?.Dispose();
            _f_beta?.Dispose();
            _sigmoid?.Dispose();
            _dropoutLayer?.Dispose();
            _class_embed?.Dispose();
            _bbox_embed?.Dispose();
        }
        base.Dispose(disposing);
    }
}
