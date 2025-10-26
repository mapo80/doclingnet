using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Collections.Generic;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TableFormerSdk.Backends;
using System.Text.Json;

namespace TableFormerBenchmarkOnnx;

sealed class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("TableFormer ONNX Benchmark - FinTabNet Dataset");
        Console.WriteLine("==============================================");
        Console.WriteLine();

        // Paths
        var modelDir = "/Users/politom/Documents/Workspace/personal/doclingnet/models/onnx-components";
        var datasetDir = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/benchmark";
        var outputDir = "/Users/politom/Documents/Workspace/personal/doclingnet/benchmark-results-onnx";
        var wordmapPath = Path.Combine(modelDir, "tableformer_fast_wordmap.json");

        // Load wordmap
        Console.WriteLine($"Loading wordmap from: {wordmapPath}");
        var wordMapTag = LoadWordMap(wordmapPath);
        Console.WriteLine($"✅ Loaded {wordMapTag.Count} tokens\n");

        // Create output directory
        Directory.CreateDirectory(outputDir);

        // Create inference engine
        using var inference = new TableFormerOnnxInference(modelDir, wordMapTag);
        Console.WriteLine();

        // Find images
        var imageFiles = Directory.GetFiles(datasetDir, "*.png");
        Console.WriteLine($"Found {imageFiles.Length} images to process\n");

        int processedCount = 0;
        var totalStopwatch = Stopwatch.StartNew();

        foreach (var imagePath in imageFiles)
        {
            var fileName = Path.GetFileNameWithoutExtension(imagePath);
            Console.WriteLine($"Processing: {Path.GetFileName(imagePath)}");

            try
            {
                // Load and preprocess image
                using var origImage = Image.Load<Rgb24>(imagePath);
                Console.WriteLine($"  Original size: {origImage.Width}x{origImage.Height}");

                // Resize to 448x448
                const int targetSize = 448;
                origImage.Mutate(x => x.Resize(targetSize, targetSize));
                Console.WriteLine($"  Resized to: {origImage.Width}x{origImage.Height}");

                // Convert to tensor
                var imageData = ImageToTensor(origImage);

                // Run inference
                var sw = Stopwatch.StartNew();
                var (tags, bboxClasses, bboxCoords) = inference.Predict(imageData, maxSteps: 1024);
                sw.Stop();

                Console.WriteLine($"  Inference time: {sw.ElapsedMilliseconds}ms");
                Console.WriteLine($"  Tag sequence length: {tags.Count}");
                Console.WriteLine($"  Number of bounding boxes: {bboxCoords.GetLength(0)}");

                // Decode tags
                var tagNames = DecodeTags(tags, wordMapTag);
                Console.WriteLine($"  Tag sequence: {string.Join(" ", tagNames.Take(20))}...");

                // Save results
                var outputPath = Path.Combine(outputDir, $"{fileName}_onnx_results.txt");
                SaveResults(outputPath, fileName, tags, tagNames, bboxClasses, bboxCoords, sw.ElapsedMilliseconds);

                Console.WriteLine($"  ✓ Success\n");
                processedCount++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ✗ Error: {ex.Message}\n");
            }
        }

        totalStopwatch.Stop();

        // Summary
        Console.WriteLine("==============================================");
        Console.WriteLine("Summary:");
        Console.WriteLine($"  Processed: {processedCount}/{imageFiles.Length} images");
        Console.WriteLine($"  Total time: {totalStopwatch.Elapsed.TotalSeconds:F2}s");
        if (processedCount > 0)
        {
            Console.WriteLine($"  Average time: {totalStopwatch.ElapsedMilliseconds / processedCount}ms per image");
        }
        Console.WriteLine($"  Results saved to: {outputDir}");
        Console.WriteLine();
    }

    static Dictionary<string, long> LoadWordMap(string path)
    {
        var json = File.ReadAllText(path);
        var doc = JsonDocument.Parse(json);
        var wordMap = new Dictionary<string, long>();

        // Extract word_map_tag from nested JSON
        var wordMapTag = doc.RootElement.GetProperty("word_map_tag");
        foreach (var prop in wordMapTag.EnumerateObject())
        {
            wordMap[prop.Name] = prop.Value.GetInt64();
        }

        return wordMap;
    }

    static float[,,,] ImageToTensor(Image<Rgb24> image)
    {
        // ImageNet normalization
        var mean = new[] { 0.94247851f, 0.94254675f, 0.94292611f };
        var std = new[] { 0.17910956f, 0.17940403f, 0.17931663f };

        var height = image.Height;
        var width = image.Width;
        var data = new float[1, 3, height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];

                // Channel R
                data[0, 0, y, x] = (pixel.R / 255.0f - mean[0]) / std[0];
                // Channel G
                data[0, 1, y, x] = (pixel.G / 255.0f - mean[1]) / std[1];
                // Channel B
                data[0, 2, y, x] = (pixel.B / 255.0f - mean[2]) / std[2];
            }
        }

        return data;
    }

    static List<string> DecodeTags(List<long> tags, Dictionary<string, long> wordMap)
    {
        var reverseMap = wordMap.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        return tags.Select(t => reverseMap.GetValueOrDefault(t, $"<unk_{t}>")).ToList();
    }

    static void SaveResults(
        string outputPath,
        string fileName,
        List<long> tags,
        List<string> tagNames,
        float[,] bboxClasses,
        float[,] bboxCoords,
        long inferenceTimeMs)
    {
        using var writer = new StreamWriter(outputPath);

        writer.WriteLine($"TableFormer ONNX Results: {fileName}.png");
        writer.WriteLine($"Inference Time: {inferenceTimeMs}ms");
        writer.WriteLine($"Timestamp: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        writer.WriteLine();

        writer.WriteLine("Tag Sequence:");
        writer.WriteLine(string.Join(" ", tagNames));
        writer.WriteLine();

        writer.WriteLine($"Number of Tags: {tags.Count}");
        writer.WriteLine();

        writer.WriteLine($"Number of Bounding Boxes: {bboxCoords.GetLength(0)}");
        writer.WriteLine();

        writer.WriteLine("Bounding Boxes (normalized [cx, cy, w, h]):");
        for (int i = 0; i < Math.Min(100, bboxCoords.GetLength(0)); i++)
        {
            writer.WriteLine($"  Box {i}: cx={bboxCoords[i, 0]:F4}, cy={bboxCoords[i, 1]:F4}, " +
                           $"w={bboxCoords[i, 2]:F4}, h={bboxCoords[i, 3]:F4}");
        }

        Console.WriteLine($"  Results saved to: {outputPath}");
    }
}
