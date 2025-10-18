//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using System.Text.Json;

namespace TableFormerSdk.Models;

/// <summary>
/// Configuration for TableModel04.
/// </summary>
public sealed class TableModel04Config
{
    public required Dictionary<string, long> WordMapTag { get; init; }
    public required long EncImageSize { get; init; }
    public required long EncoderDim { get; init; }
    public required long TagAttentionDim { get; init; }
    public required long TagEmbedDim { get; init; }
    public required long TagDecoderDim { get; init; }
    public required long BBoxAttentionDim { get; init; }
    public required long BBoxEmbedDim { get; init; }
    public required long BBoxDecoderDim { get; init; }
    public required long EncLayers { get; init; }
    public required long DecLayers { get; init; }
    public required long NumHeads { get; init; }
    public required double Dropout { get; init; }
    public required long NumClasses { get; init; }
    public required long MaxSteps { get; init; }

    /// <summary>
    /// Load configuration from JSON file (tm_config.json).
    /// </summary>
    public static TableModel04Config FromJsonFile(string path)
    {
        var json = File.ReadAllText(path);
        var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var model = root.GetProperty("model");
        var predict = root.GetProperty("predict");
        var wordMapTag = new Dictionary<string, long>();

        foreach (var prop in root.GetProperty("dataset_wordmap").GetProperty("word_map_tag").EnumerateObject())
        {
            wordMapTag[prop.Name] = prop.Value.GetInt64();
        }

        return new TableModel04Config
        {
            WordMapTag = wordMapTag,
            EncImageSize = model.GetProperty("enc_image_size").GetInt64(),
            EncoderDim = model.GetProperty("hidden_dim").GetInt64(),
            TagAttentionDim = model.GetProperty("tag_attention_dim").GetInt64(),
            TagEmbedDim = model.GetProperty("tag_embed_dim").GetInt64(),
            TagDecoderDim = model.GetProperty("tag_decoder_dim").GetInt64(),
            BBoxAttentionDim = model.GetProperty("bbox_attention_dim").GetInt64(),
            BBoxEmbedDim = model.GetProperty("bbox_embed_dim").GetInt64(),
            BBoxDecoderDim = model.GetProperty("hidden_dim").GetInt64(),
            EncLayers = model.GetProperty("enc_layers").GetInt64(),
            DecLayers = model.GetProperty("dec_layers").GetInt64(),
            NumHeads = model.GetProperty("nheads").GetInt64(),
            Dropout = model.GetProperty("dropout").GetDouble(),
            NumClasses = model.GetProperty("bbox_classes").GetInt64(),
            MaxSteps = predict.GetProperty("max_steps").GetInt64()
        };
    }
}

/// <summary>
/// Prediction result from TableModel04.
/// </summary>
public sealed class TableModel04Result
{
    /// <summary>
    /// Tag sequence as indices into vocabulary.
    /// </summary>
    public required IReadOnlyList<long> Sequence { get; init; }

    /// <summary>
    /// Predicted bounding box classes of shape (num_cells, num_classes+1).
    /// </summary>
    public required Tensor BBoxClasses { get; init; }

    /// <summary>
    /// Predicted bounding box coordinates of shape (num_cells, 4) in [cx, cy, w, h] format.
    /// Values are normalized to [0, 1].
    /// </summary>
    public required Tensor BBoxCoords { get; init; }
}

/// <summary>
/// TableModel04: Transformer-based encoder-decoder for table structure recognition.
/// Orchestrates Encoder04, TagTransformer, and BBoxDecoder.
/// Ported from Python implementation in docling-ibm-models.
/// </summary>
public sealed class TableModel04 : Module<Tensor, TableModel04Result>
{
    private readonly TableModel04Config _config;
    private readonly IReadOnlyList<long> _tdEncode;
    private readonly long _startToken;
    private readonly long _endToken;
    private readonly long _fcelToken;
    private readonly long _ucelToken;
    private readonly long _lcelToken;
    private readonly long _xcelToken;

    private readonly Encoder04 _encoder;
    private readonly TagTransformer _tagTransformer;
    private readonly BBoxDecoder _bboxDecoder;

    /// <summary>
    /// Initializes a new instance of the <see cref="TableModel04"/> class.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="name">Module name.</param>
    public TableModel04(TableModel04Config config, string name = "TableModel04")
        : base(name)
    {
        _config = config;

        // Build list of table data cell tags (ecel, fcel, ched, rhed, srow)
        var tdEncode = new List<long>();
        foreach (var tag in new[] { "ecel", "fcel", "ched", "rhed", "srow" })
        {
            if (_config.WordMapTag.TryGetValue(tag, out var idx))
            {
                tdEncode.Add(idx);
            }
        }
        _tdEncode = tdEncode;

        // Cache frequently used tokens
        _startToken = _config.WordMapTag["<start>"];
        _endToken = _config.WordMapTag["<end>"];
        _fcelToken = _config.WordMapTag["fcel"];
        _ucelToken = _config.WordMapTag["ucel"];
        _lcelToken = _config.WordMapTag["lcel"];
        _xcelToken = _config.WordMapTag["xcel"];

        // Create submodules
        // NOTE: Python uses enc_image_size=14 instead of config value (28)
        // This is the spatial size after adaptive pooling
        const long ACTUAL_ENC_IMAGE_SIZE = 14;
        _encoder = new Encoder04(ACTUAL_ENC_IMAGE_SIZE, _config.EncoderDim);
        register_module(nameof(_encoder), _encoder);

        var vocabSize = _config.WordMapTag.Count;
        _tagTransformer = new TagTransformer(
            vocabSize: vocabSize,
            tdEncode: _tdEncode,
            embedDim: _config.TagDecoderDim,
            encoderLayers: _config.EncLayers,
            decoderLayers: _config.DecLayers,
            encImageSize: ACTUAL_ENC_IMAGE_SIZE,
            dropout: _config.Dropout,
            nHeads: _config.NumHeads,
            encoderDim: 256);
        register_module(nameof(_tagTransformer), _tagTransformer);

        _bboxDecoder = new BBoxDecoder(
            attentionDim: _config.BBoxAttentionDim,
            embedDim: _config.BBoxEmbedDim,
            tagDecoderDim: _config.TagDecoderDim,
            decoderDim: _config.BBoxDecoderDim,
            numClasses: _config.NumClasses,
            encoderRawDim: 256,
            encoderDim: 512,
            dropout: _config.Dropout);
        register_module(nameof(_bboxDecoder), _bboxDecoder);
    }

    /// <summary>
    /// Create TableModel04 and load weights from SafeTensors file.
    /// </summary>
    public static TableModel04 FromSafeTensors(TableModel04Config config, string safetensorsPath)
    {
        var model = new TableModel04(config);
        TableModel04Loader.LoadWeights(model, safetensorsPath);
        return model;
    }

    /// <summary>
    /// Merge two bounding boxes (used for horizontal spans like lcel).
    /// </summary>
    /// <param name="bbox1">First box in [cx, cy, w, h] format.</param>
    /// <param name="bbox2">Last box in [cx, cy, w, h] format.</param>
    /// <returns>Merged bounding box in [cx, cy, w, h] format.</returns>
    private Tensor MergeBBoxes(Tensor bbox1, Tensor bbox2)
    {
        // Extract values
        var cx1 = bbox1[0].item<float>();
        var cy1 = bbox1[1].item<float>();
        var w1 = bbox1[2].item<float>();
        var h1 = bbox1[3].item<float>();

        var cx2 = bbox2[0].item<float>();
        var cy2 = bbox2[1].item<float>();
        var w2 = bbox2[2].item<float>();
        var h2 = bbox2[3].item<float>();

        // Calculate merged dimensions
        var newW = (cx2 + w2 / 2) - (cx1 - w1 / 2);
        var newH = (cy2 + h2 / 2) - (cy1 - h1 / 2);

        var newLeft = cx1 - w1 / 2;
        var newTop = Math.Min(cy2 - h2 / 2, cy1 - h1 / 2);

        var newCx = newLeft + newW / 2;
        var newCy = newTop + newH / 2;

        return torch.tensor(new[] { newCx, newCy, newW, newH });
    }

    /// <summary>
    /// Forward pass for inference (autoregressive tag generation + bbox prediction).
    /// </summary>
    /// <param name="images">Input images of shape (batch_size, height, width, 3) or (batch_size, 3, height, width).</param>
    /// <returns>Prediction result containing tag sequence, bbox classes, and coordinates.</returns>
    public override TableModel04Result forward(Tensor images)
    {
        // Encode image
        using var encOut = _encoder.forward(images);

        // DEBUG: Check encoder output
        if (Environment.GetEnvironmentVariable("DEBUG_ENCODER") == "1")
        {
            Console.WriteLine($"[DEBUG] Encoder output shape: [{encOut.size(0)}, {encOut.size(1)}, {encOut.size(2)}, {encOut.size(3)}]");
            Console.WriteLine($"[DEBUG] Encoder output range: [{encOut.min().item<float>():F6}, {encOut.max().item<float>():F6}]");
            Console.WriteLine($"[DEBUG] Encoder output mean: {encOut.mean().item<float>():F6}");
        }

        // Tag sequence generation (autoregressive)
        var outputTags = new List<long>();
        var tagHiddenStates = new List<Tensor>();

        // Cache commonly used tokens
        var nlToken = _config.WordMapTag["nl"];
        var chedToken = _config.WordMapTag["ched"];
        var rhedToken = _config.WordMapTag["rhed"];
        var srowToken = _config.WordMapTag["srow"];
        var ecelToken = _config.WordMapTag["ecel"];

        // Start with <start> token
        using var decodedTags = torch.tensor(new long[] { _startToken }).unsqueeze(0);  // (1, 1)

        Tensor? cache = null;
        var prevTagUcel = false;
        var isFirstLine = true;

        // Track bounding boxes to merge (for horizontal spans)
        var firstLcel = true;
        var bboxesToMerge = new Dictionary<long, long>();
        var curBboxInd = -1L;
        var bboxInd = 0L;

        var currentDecodedTags = decodedTags;
        var isFirstIteration = true;

        // Autoregressive generation
        for (int step = 0; step < _config.MaxSteps; step++)
        {
            // Run tag transformer
            var (predictions, decoderOutput, newCache) = _tagTransformer.forward((encOut, currentDecodedTags));

            // Cleanup old cache
            cache?.Dispose();
            cache = newCache;

            // Get logits for the last predicted token
            using var lastLogits = predictions[0, predictions.size(1) - 1, TensorIndex.Colon];  // (vocab_size,)
            var newTag = lastLogits.argmax().item<long>();

            // DEBUG: Print logits for first 10 steps
            if (Environment.GetEnvironmentVariable("DEBUG_LOGITS") == "1" && step < 10)
            {
                var logitsArray = lastLogits.data<float>().ToArray();

                Console.WriteLine($"\n[DEBUG Step {step}] ========================================");
                Console.WriteLine($"  Current sequence: [{string.Join(", ", outputTags.Select(t => _config.WordMapTag.FirstOrDefault(kvp => kvp.Value == t).Key ?? $"idx{t}"))}]");
                Console.WriteLine($"  Sequence length: {outputTags.Count + 1}");

                // Show all 13 logits
                Console.WriteLine($"  All logits:");
                foreach (var kvp in _config.WordMapTag.OrderBy(x => x.Value))
                {
                    var idx = kvp.Value;
                    var tokenName = kvp.Key;
                    var logit = logitsArray[idx];
                    var marker = (idx == newTag) ? " ← PREDICTED" : "";
                    Console.WriteLine($"    [{idx:D2}] {tokenName,-10}: {logit,8:F4}{marker}");
                }

                // Show top-5 for quick reference
                var topK = logitsArray
                    .Select((val, idx) => (val, idx))
                    .OrderByDescending(x => x.val)
                    .Take(5)
                    .ToList();

                Console.WriteLine($"\n  Top-5 logits:");
                foreach (var (val, idx) in topK)
                {
                    var tokenName = _config.WordMapTag.FirstOrDefault(kvp => kvp.Value == idx).Key ?? $"idx{idx}";
                    Console.WriteLine($"    {tokenName}: {val:F4}");
                }

                // Highlight <end> token
                var endLogit = logitsArray[_endToken];
                Console.WriteLine($"\n  <end> token analysis:");
                Console.WriteLine($"    Logit: {endLogit:F4}");
                Console.WriteLine($"    Rank: {logitsArray.OrderByDescending(x => x).ToList().IndexOf(endLogit) + 1}/13");
                Console.WriteLine($"    Distance from max: {logitsArray.Max() - endLogit:F4}");
            }

            // Structure error correction (line 199-208 in Python)
            // Correction for first line xcel...
            if (isFirstLine && newTag == _xcelToken)
            {
                newTag = _lcelToken;
            }

            // Correction for ucel, lcel sequence...
            if (prevTagUcel && newTag == _lcelToken)
            {
                newTag = _fcelToken;
            }

            // Check for end token
            if (newTag == _endToken)
            {
                outputTags.Add(newTag);
                break;
            }

            outputTags.Add(newTag);

            // Extract hidden state for bbox prediction (only for data cell tags)
            // Skip next tag after nl, ucel, xcel (line 255-258)
            var skipNextTag = step > 0 && (
                outputTags[step - 1] == nlToken ||
                outputTags[step - 1] == _ucelToken ||
                outputTags[step - 1] == _xcelToken);

            if (!skipNextTag)
            {
                if (newTag == _fcelToken || newTag == ecelToken || newTag == chedToken ||
                    newTag == rhedToken || newTag == srowToken || newTag == nlToken || newTag == _ucelToken)
                {
                    // Save hidden state for bbox prediction
                    using var lastHidden = decoderOutput[decoderOutput.size(0) - 1, TensorIndex.Colon, TensorIndex.Colon];
                    tagHiddenStates.Add(lastHidden.clone());

                    if (!firstLcel)
                    {
                        // Mark end index for horizontal cell bbox merge
                        bboxesToMerge[curBboxInd] = bboxInd;
                    }
                    bboxInd++;
                }
            }

            // Handle horizontal span bboxes (lcel)
            if (newTag != _lcelToken)
            {
                firstLcel = true;
            }
            else
            {
                if (firstLcel)
                {
                    // Beginning of horizontal span
                    using var lastHidden = decoderOutput[decoderOutput.size(0) - 1, TensorIndex.Colon, TensorIndex.Colon];
                    tagHiddenStates.Add(lastHidden.clone());
                    firstLcel = false;

                    // Mark start index for cell bbox merge
                    curBboxInd = bboxInd;
                    bboxesToMerge[curBboxInd] = -1;
                    bboxInd++;
                }
            }

            // Update state for next iteration
            if (newTag == _ucelToken)
            {
                prevTagUcel = true;
            }
            else
            {
                prevTagUcel = false;
            }

            // Update first line flag
            if (newTag == nlToken)
            {
                isFirstLine = false;
            }

            // Append new tag to sequence
            using var newTagTensor = torch.tensor(new long[] { newTag }).unsqueeze(0);  // (1, 1)
            var nextDecodedTags = torch.cat(new[] { currentDecodedTags, newTagTensor }, dim: 1);

            if (!isFirstIteration)
            {
                currentDecodedTags.Dispose();
            }
            currentDecodedTags = nextDecodedTags;
            isFirstIteration = false;

            // Cleanup
            predictions.Dispose();
            decoderOutput.Dispose();
        }

        // Predict bounding boxes
        Tensor bboxClasses;
        Tensor bboxCoords;

        if (tagHiddenStates.Count > 0)
        {
            // Stack tag hidden states: (num_cells, decoder_dim)
            using var tagHStacked = torch.stack(tagHiddenStates.ToArray(), dim: 0).squeeze(1);
            (bboxClasses, bboxCoords) = _bboxDecoder.forward((encOut, tagHStacked));
        }
        else
        {
            bboxClasses = torch.empty(0, _config.NumClasses + 1);
            bboxCoords = torch.empty(0, 4);
        }

        // Merge bounding boxes for horizontal spans
        var finalBBoxClasses = new List<Tensor>();
        var finalBBoxCoords = new List<Tensor>();
        var boxesToSkip = new HashSet<long>();

        for (long boxInd = 0; boxInd < bboxCoords.size(0); boxInd++)
        {
            using var box1 = bboxCoords[boxInd];
            using var cls1 = bboxClasses[boxInd];

            if (bboxesToMerge.TryGetValue(boxInd, out var endIdx))
            {
                if (endIdx >= 0)
                {
                    boxesToSkip.Add(endIdx);
                    using var box2 = bboxCoords[endIdx];
                    var boxMerged = MergeBBoxes(box1, box2);
                    finalBBoxCoords.Add(boxMerged);
                    finalBBoxClasses.Add(cls1.clone());
                }
            }
            else
            {
                if (!boxesToSkip.Contains(boxInd))
                {
                    finalBBoxCoords.Add(box1.clone());
                    finalBBoxClasses.Add(cls1.clone());
                }
            }
        }

        // Stack final outputs
        Tensor finalClasses;
        Tensor finalCoords;

        if (finalBBoxClasses.Count > 0)
        {
            finalClasses = torch.stack(finalBBoxClasses.ToArray(), dim: 0);
            finalCoords = torch.stack(finalBBoxCoords.ToArray(), dim: 0);
        }
        else
        {
            finalClasses = torch.empty(0, _config.NumClasses + 1);
            finalCoords = torch.empty(0, 4);
        }

        // Cleanup
        cache?.Dispose();
        bboxClasses.Dispose();
        bboxCoords.Dispose();
        foreach (var h in tagHiddenStates)
        {
            h.Dispose();
        }
        foreach (var bbox in finalBBoxCoords)
        {
            bbox.Dispose();
        }
        foreach (var cls in finalBBoxClasses)
        {
            cls.Dispose();
        }
        if (!isFirstIteration)
        {
            currentDecodedTags.Dispose();
        }

        return new TableModel04Result
        {
            Sequence = outputTags,
            BBoxClasses = finalClasses,
            BBoxCoords = finalCoords
        };
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _encoder?.Dispose();
            _tagTransformer?.Dispose();
            _bboxDecoder?.Dispose();
        }
        base.Dispose(disposing);
    }
}
