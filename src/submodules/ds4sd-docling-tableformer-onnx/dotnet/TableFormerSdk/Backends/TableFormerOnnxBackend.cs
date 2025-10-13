using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TableFormerSdk.Configuration;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

/// <summary>
/// Complete TableFormer backend using the 4-component ONNX architecture.
/// Implements the full pipeline: preprocessing → encoder → tag transformer → autoregressive decoding → bbox prediction.
/// </summary>
internal sealed class TableFormerOnnxBackend : ITableFormerBackend, IDisposable
{
    private readonly TableFormerOnnxComponents _components;
    private readonly TableFormerAutoregressive _autoregressive;
    private readonly string _modelsDirectory;
    private readonly TableFormerConfig? _config;
    private readonly TableFormerWordMap _wordMap;
    private readonly NormalizationParameters _normalization;
    private readonly int _targetImageSize;
    private bool _disposed;

    public TableFormerOnnxBackend(TableFormerVariantModelPaths modelPaths)
    {
        // Extract the directory path from the encoder path to get the models directory
        _modelsDirectory = Path.GetDirectoryName(modelPaths.EncoderPath)
            ?? throw new ArgumentException("Could not determine models directory from encoder path", nameof(modelPaths));

        // Load configuration
        _config = TableFormerConfig.LoadFromFile(modelPaths.ConfigPath);
        _normalization = _config.NormalizationParameters;
        _targetImageSize = _config.TargetImageSize;

        // Load word map
        _wordMap = TableFormerWordMap.LoadFromFile(modelPaths.WordMapPath);

        _components = new TableFormerOnnxComponents(_modelsDirectory);
        _autoregressive = new TableFormerAutoregressive(_components, _wordMap);
    }

    // Costruttore temporaneo per compatibilità durante la transizione
    public TableFormerOnnxBackend(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path is empty", nameof(modelPath));
        }

        // Try to infer models directory from the single model path
        _modelsDirectory = Path.GetDirectoryName(modelPath)
            ?? throw new ArgumentException("Could not determine models directory from model path", nameof(modelPath));

        // Try to load config, fallback to defaults if not found
        var configPath = Path.Combine(_modelsDirectory, "tableformer_fast_config.json");
        var wordMapPath = Path.Combine(_modelsDirectory, "tableformer_fast_wordmap.json");

        if (File.Exists(configPath))
        {
            _config = TableFormerConfig.LoadFromFile(configPath);
            _normalization = _config.NormalizationParameters;
            _targetImageSize = _config.TargetImageSize;
        }
        else
        {
            // Use default PubTabNet normalization
            _config = null;
            _normalization = NormalizationParameters.Default;
            _targetImageSize = 448;
        }

        // Load word map or use default
        _wordMap = File.Exists(wordMapPath)
            ? TableFormerWordMap.LoadFromFile(wordMapPath)
            : TableFormerWordMap.CreateDefault();

        _components = new TableFormerOnnxComponents(_modelsDirectory);
        _autoregressive = new TableFormerAutoregressive(_components, _wordMap);
    }

    /// <summary>
    /// Process table image using the complete 4-component architecture.
    /// </summary>
    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        try
        {
            // Step 1: Preprocess image with PubTabNet normalization
            var preprocessedTensor = PreprocessImage(image, _targetImageSize, _normalization);

            // Step 2: Run encoder
            var encoderOutput = _components.RunEncoder(preprocessedTensor);

            // Step 3: Run tag transformer encoder
            var encoderMask = CreateEncoderMask(encoderOutput);
            var memory = _components.RunTagTransformerEncoder(encoderOutput);

            // Step 4: Generate OTSL tags autoregressively
            var autoregressiveResult = _autoregressive.GenerateTags(memory, encoderMask);

            // Return empty if no cells generated
            if (autoregressiveResult.TagHiddenStates.Count == 0)
            {
                return new List<TableRegion>();
            }

            // Step 5: Run bbox decoder to get bounding boxes
            var (bboxClasses, bboxCoords) = _components.RunBboxDecoder(
                encoderOutput,
                CreateTagHiddensTensor(autoregressiveResult.TagHiddenStates));

            // Step 6: Parse OTSL and convert to table regions
            var tableStructure = OtslParser.ParseOtsl(autoregressiveResult.GeneratedTokens);

            var regions = ConvertToTableRegions(
                tableStructure,
                bboxClasses,
                bboxCoords,
                new BoundingBox(0, 0, image.Width, image.Height),
                image.Width,
                image.Height);

            return regions;
        }
        catch (Exception ex)
        {
            // Log error and return empty results
            Console.WriteLine($"TableFormer inference error: {ex.Message}");
            return new List<TableRegion>();
        }
    }

    /// <summary>
    /// Preprocess image with PubTabNet normalization.
    /// Formula: (pixel / 255.0 - mean) / std
    /// </summary>
    private static DenseTensor<float> PreprocessImage(
        SKBitmap bitmap,
        int targetSize,
        NormalizationParameters normalization)
    {
        // Resize to target size (default 448x448)
        var samplingOptions = new SKSamplingOptions(SKCubicResampler.CatmullRom);
        var resized = bitmap.Resize(new SKImageInfo(targetSize, targetSize), samplingOptions);
        if (resized is null)
        {
            throw new InvalidOperationException("Failed to resize image for TableFormer");
        }

        using (resized)
        {
            // Convert to tensor format (1, 3, H, W) with PubTabNet normalization
            var tensor = new DenseTensor<float>(new[] { 1, 3, targetSize, targetSize });

            // Get normalization parameters
            var meanR = normalization.Mean.Length > 0 ? normalization.Mean[0] : 0.94247851f;
            var meanG = normalization.Mean.Length > 1 ? normalization.Mean[1] : 0.94254675f;
            var meanB = normalization.Mean.Length > 2 ? normalization.Mean[2] : 0.94292611f;

            var stdR = normalization.Std.Length > 0 ? normalization.Std[0] : 0.17910956f;
            var stdG = normalization.Std.Length > 1 ? normalization.Std[1] : 0.17940403f;
            var stdB = normalization.Std.Length > 2 ? normalization.Std[2] : 0.17931663f;

            for (int y = 0; y < targetSize; y++)
            {
                for (int x = 0; x < targetSize; x++)
                {
                    var color = resized.GetPixel(x, y);

                    // Apply PubTabNet normalization: (pixel / 255.0 - mean) / std
                    if (normalization.Enabled)
                    {
                        tensor[0, 0, y, x] = ((color.Red / 255f) - meanR) / stdR;
                        tensor[0, 1, y, x] = ((color.Green / 255f) - meanG) / stdG;
                        tensor[0, 2, y, x] = ((color.Blue / 255f) - meanB) / stdB;
                    }
                    else
                    {
                        // Fallback to simple division by 255 if normalization disabled
                        tensor[0, 0, y, x] = color.Red / 255f;
                        tensor[0, 1, y, x] = color.Green / 255f;
                        tensor[0, 2, y, x] = color.Blue / 255f;
                    }
                }
            }

            return tensor;
        }
    }

    private static DenseTensor<bool> CreateEncoderMask(DenseTensor<float> encoderOutput)
    {
        // Create attention mask for transformer encoder
        var batchSize = encoderOutput.Dimensions[0];
        var seqLength = encoderOutput.Dimensions[1] * encoderOutput.Dimensions[2]; // 28 * 28 = 784
        var mask = new DenseTensor<bool>(new[] { batchSize, 1, 1, seqLength, seqLength });

        // Simple full attention mask (all positions attend to all positions)
        for (int i = 0; i < seqLength; i++)
        {
            for (int j = 0; j < seqLength; j++)
            {
                mask[0, 0, 0, i, j] = true;
            }
        }

        return mask;
    }

    private static DenseTensor<float> CreateTagHiddensTensor(List<DenseTensor<float>> tagHiddenStates)
    {
        if (tagHiddenStates.Count == 0)
        {
            throw new ArgumentException("No tag hidden states provided");
        }

        var firstTensor = tagHiddenStates[0];
        var batchSize = firstTensor.Dimensions[0];
        var hiddenSize = firstTensor.Dimensions[1];
        var tensor = new DenseTensor<float>(new[] { tagHiddenStates.Count, batchSize, hiddenSize });

        for (int i = 0; i < tagHiddenStates.Count; i++)
        {
            var hiddenState = tagHiddenStates[i];
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    tensor[i, b, h] = hiddenState[b, h];
                }
            }
        }

        return tensor;
    }

    /// <summary>
    /// Convert table structure and bbox predictions to TableRegion list.
    /// Applies softmax to bbox_classes and filters cells based on confidence threshold.
    /// </summary>
    private static IReadOnlyList<TableRegion> ConvertToTableRegions(
        OtslParser.TableStructure tableStructure,
        DenseTensor<float> bboxClasses,
        DenseTensor<float> bboxCoords,
        BoundingBox tableBounds,
        int imageWidth,
        int imageHeight)
    {
        var regions = new List<TableRegion>();

        // Convert tensors to arrays for easier access
        var classesArray = bboxClasses.ToArray();
        var coordsArray = bboxCoords.ToArray();

        // Get number of classes from tensor shape: (num_cells, num_classes)
        var numCells = bboxClasses.Dimensions[0];
        var numClasses = bboxClasses.Dimensions.Length > 1 ? bboxClasses.Dimensions[1] : 1;

        var cellIndex = 0;

        foreach (var row in tableStructure.Rows)
        {
            foreach (var cell in row)
            {
                // Skip linked and spanned cells (they don't have their own bbox)
                if (cell.CellType == "linked" || cell.CellType == "spanned")
                {
                    continue;
                }

                // Check if we have bbox data for this cell
                if (cellIndex >= numCells)
                {
                    break;
                }

                // Apply softmax to get class probabilities
                var classProbabilities = ApplySoftmax(classesArray, cellIndex * numClasses, numClasses);

                // Class 0: background, Class 1+: cells (typically class 1 = cell, class 2 = header)
                // Filter out cells with low confidence (background probability > 0.5)
                const float confidenceThreshold = 0.5f;
                if (classProbabilities[0] > confidenceThreshold)
                {
                    cellIndex++;
                    continue; // Skip this cell, it's likely background
                }

                // Get bbox coordinates (cx, cy, w, h) - normalized [0,1]
                var coordOffset = cellIndex * 4;
                if (coordOffset + 3 >= coordsArray.Length)
                {
                    break;
                }

                var cx = coordsArray[coordOffset];
                var cy = coordsArray[coordOffset + 1];
                var w = coordsArray[coordOffset + 2];
                var h = coordsArray[coordOffset + 3];

                // Convert from normalized [0,1] center coordinates to absolute pixel coordinates
                var absLeft = tableBounds.Left + (cx - w / 2) * tableBounds.Width;
                var absTop = tableBounds.Top + (cy - h / 2) * tableBounds.Height;
                var absRight = absLeft + w * tableBounds.Width;
                var absBottom = absTop + h * tableBounds.Height;

                // Clamp to table boundaries
                absLeft = Math.Max(tableBounds.Left, Math.Min(tableBounds.Right, absLeft));
                absTop = Math.Max(tableBounds.Top, Math.Min(tableBounds.Bottom, absTop));
                absRight = Math.Max(tableBounds.Left, Math.Min(tableBounds.Right, absRight));
                absBottom = Math.Max(tableBounds.Top, Math.Min(tableBounds.Bottom, absBottom));

                // Convert to normalized coordinates relative to table bounds
                var normX = (absLeft - tableBounds.Left) / tableBounds.Width;
                var normY = (absTop - tableBounds.Top) / tableBounds.Height;
                var normWidth = (absRight - absLeft) / tableBounds.Width;
                var normHeight = (absBottom - absTop) / tableBounds.Height;

                // Clamp normalized coordinates to [0, 1]
                normX = Math.Max(0, Math.Min(1, normX));
                normY = Math.Max(0, Math.Min(1, normY));
                normWidth = Math.Max(0, Math.Min(1 - normX, normWidth));
                normHeight = Math.Max(0, Math.Min(1 - normY, normHeight));

                regions.Add(new TableRegion(
                    (float)normX,
                    (float)normY,
                    (float)normWidth,
                    (float)normHeight,
                    $"cell_{cellIndex}"));

                cellIndex++;
            }
        }

        return regions;
    }

    /// <summary>
    /// Apply softmax to a slice of the array to get probabilities.
    /// </summary>
    private static float[] ApplySoftmax(float[] array, int offset, int length)
    {
        var probabilities = new float[length];
        var maxLogit = float.NegativeInfinity;

        // Find max for numerical stability
        for (int i = 0; i < length; i++)
        {
            maxLogit = Math.Max(maxLogit, array[offset + i]);
        }

        // Compute exp and sum
        var sumExp = 0.0f;
        for (int i = 0; i < length; i++)
        {
            var exp = (float)Math.Exp(array[offset + i] - maxLogit);
            probabilities[i] = exp;
            sumExp += exp;
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            probabilities[i] /= sumExp;
        }

        return probabilities;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _components?.Dispose();
            _disposed = true;
        }
    }
}
