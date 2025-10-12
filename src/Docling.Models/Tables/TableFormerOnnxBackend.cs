using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using Docling.Core.Geometry;

namespace Docling.Core.Models.Tables;

/// <summary>
/// New TableFormer backend using the 4-component ONNX architecture.
/// Replaces the old single-model approach with component-wise processing.
/// </summary>
internal sealed class TableFormerOnnxBackend : IDisposable
{
    private readonly TableFormerOnnxComponents _components;
    private readonly TableFormerAutoregressive _autoregressive;
    private readonly OtslParser _otslParser;
    private readonly string _modelsDirectory;
    private bool _disposed;

    public TableFormerOnnxBackend(string modelsDirectory)
    {
        _modelsDirectory = modelsDirectory ?? throw new ArgumentNullException(nameof(modelsDirectory));

        _components = new TableFormerOnnxComponents(modelsDirectory);
        _autoregressive = new TableFormerAutoregressive(_components);
        _otslParser = new OtslParser();
    }

    /// <summary>
    /// Process table image using the new 4-component architecture.
    /// </summary>
    public IReadOnlyList<TableFormerRegion> Infer(SKBitmap image, BoundingBox tableBounds)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (tableBounds.IsEmpty)
        {
            throw new ArgumentException("Table bounds cannot be empty", nameof(tableBounds));
        }

        // Step 1: Preprocess image to 448x448 as required by the model
        var preprocessedTensor = PreprocessImage(image);

        // Step 2: Run encoder
        var encoderOutput = _components.RunEncoder(preprocessedTensor);

        // Step 3: Run tag transformer encoder
        var encoderMask = CreateEncoderMask(encoderOutput);
        var memory = _components.RunTagTransformerEncoder(encoderOutput);

        // Step 4: Generate OTSL tags autoregressively
        var tagHiddenStates = _autoregressive.GenerateTags(memory, encoderMask);

        // Step 5: Run bbox decoder to get bounding boxes
        var (bboxClasses, bboxCoords) = _components.RunBboxDecoder(encoderOutput, CreateTagHiddensTensor(tagHiddenStates));

        // Step 6: Parse OTSL and convert to table regions
        var otslTokens = GenerateOtslSequence(tagHiddenStates.Count);
        var tableStructure = OtslParser.ParseOtsl(otslTokens);

        var regions = ConvertToTableRegions(tableStructure, bboxCoords, tableBounds, image.Width, image.Height);

        // Note: DenseTensor doesn't implement IDisposable, so no cleanup needed

        return regions;
    }

    private static DenseTensor<float> PreprocessImage(SKBitmap bitmap)
    {
        // Resize to 448x448 as required by the model
        const int targetSize = 448;
        var samplingOptions = new SKSamplingOptions(SKCubicResampler.CatmullRom);
        var resized = bitmap.Resize(new SKImageInfo(targetSize, targetSize), samplingOptions);
        if (resized is null)
        {
            throw new InvalidOperationException("Failed to resize image for TableFormer");
        }

        using (resized)
        {
            // Convert to tensor format (1, 3, 448, 448)
            var tensor = new DenseTensor<float>(new[] { 1, 3, targetSize, targetSize });

            for (int y = 0; y < targetSize; y++)
            {
                for (int x = 0; x < targetSize; x++)
                {
                    var color = resized.GetPixel(x, y);
                    tensor[0, 0, y, x] = color.Red / 255f;
                    tensor[0, 1, y, x] = color.Green / 255f;
                    tensor[0, 2, y, x] = color.Blue / 255f;
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

    private static List<string> GenerateOtslSequence(int cellCount)
    {
        var tokens = new List<string> { "<start>" };

        // Generate a simple OTSL sequence - this is a simplified version
        // In practice, this would come from the autoregressive decoder
        for (int i = 0; i < cellCount; i++)
        {
            tokens.Add("fcel");
        }

        tokens.Add("<end>");
        return tokens;
    }

    private static IReadOnlyList<TableFormerRegion> ConvertToTableRegions(
        OtslParser.TableStructure tableStructure,
        DenseTensor<float> bboxCoords,
        BoundingBox tableBounds,
        int imageWidth,
        int imageHeight)
    {
        var regions = new List<TableFormerRegion>();

        var coordsArray = bboxCoords.ToArray();
        var coordIndex = 0;

        foreach (var row in tableStructure.Rows)
        {
            foreach (var cell in row)
            {
                if (cell.CellType != "linked" && cell.CellType != "spanned" && coordIndex + 3 < coordsArray.Length)
                {
                    // Get normalized coordinates (cx, cy, w, h)
                    var cx = coordsArray[coordIndex];
                    var cy = coordsArray[coordIndex + 1];
                    var w = coordsArray[coordIndex + 2];
                    var h = coordsArray[coordIndex + 3];

                    // Convert from normalized [0,1] to table coordinates
                    var left = tableBounds.Left + (cx - w / 2) * tableBounds.Width;
                    var top = tableBounds.Top + (cy - h / 2) * tableBounds.Height;
                    var right = left + w * tableBounds.Width;
                    var bottom = top + h * tableBounds.Height;

                    // Clamp to table boundaries
                    left = Math.Max(tableBounds.Left, Math.Min(tableBounds.Right, left));
                    top = Math.Max(tableBounds.Top, Math.Min(tableBounds.Bottom, top));
                    right = Math.Max(tableBounds.Left, Math.Min(tableBounds.Right, right));
                    bottom = Math.Max(tableBounds.Top, Math.Min(tableBounds.Bottom, bottom));

                    regions.Add(new TableFormerRegion
                    {
                        X = (float)((left - tableBounds.Left) / tableBounds.Width),
                        Y = (float)((top - tableBounds.Top) / tableBounds.Height),
                        Width = (float)((right - left) / tableBounds.Width),
                        Height = (float)((bottom - top) / tableBounds.Height),
                        Confidence = 0.9f // Placeholder confidence
                    });

                    coordIndex += 4;
                }
            }
        }

        return regions;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _components.Dispose();
            _disposed = true;
        }
    }

    /// <summary>
    /// Simple TableFormer region for compatibility with existing service.
    /// </summary>
    public sealed class TableFormerRegion
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
        public float Confidence { get; set; }
    }
}