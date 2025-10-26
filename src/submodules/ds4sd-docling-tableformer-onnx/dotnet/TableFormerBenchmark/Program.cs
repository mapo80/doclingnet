using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TableFormerSdk.Models;
using TorchSharp;

namespace TableFormerBenchmark;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("TableFormer Benchmark - FinTabNet Dataset");
        Console.WriteLine("==========================================");
        Console.WriteLine();

        // Paths
        var baseDir = "/Users/politom/Documents/Workspace/personal/doclingnet";
        var datasetDir = Path.Combine(baseDir, "src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/benchmark");
        var modelsDir = Path.Combine(baseDir, "models/model_artifacts/tableformer");
        var outputDir = Path.Combine(baseDir, "benchmark-results");

        // Create output directory
        Directory.CreateDirectory(outputDir);

        // Load model with weights from SafeTensors
        var configPath = Path.Combine(modelsDir, "fast/tm_config.json");
        var safetensorsPath = Path.Combine(modelsDir, "fast/tableformer_fast.safetensors");

        Console.WriteLine($"Config: {configPath}");
        Console.WriteLine($"Weights: {safetensorsPath}\n");

        var config = TableModel04Config.FromJsonFile(configPath);
        Console.WriteLine("Loading TableModel04 from SafeTensors...");
        using var model = TableModel04.FromSafeTensors(config, safetensorsPath);
        model.eval();

        Console.WriteLine($"\nModel ready!");
        Console.WriteLine($"- Vocab size: {config.WordMapTag.Count}");
        Console.WriteLine($"- Encoder dim: {config.EncoderDim}");
        Console.WriteLine($"- Max steps: {config.MaxSteps}");
        Console.WriteLine();

        // Process images
        var imageFiles = Directory.GetFiles(datasetDir, "*.png");
        Console.WriteLine($"Found {imageFiles.Length} images to process");
        Console.WriteLine();

        int successCount = 0;
        var totalTime = TimeSpan.Zero;

        foreach (var imagePath in imageFiles)
        {
            var fileName = Path.GetFileName(imagePath);
            Console.WriteLine($"Processing: {fileName}");

            try
            {
                // Load image
                using var origImage = Image.Load<Rgb24>(imagePath);
                Console.WriteLine($"  Original size: {origImage.Width}x{origImage.Height}");

                // Resize to 448x448 (expected input size)
                const int targetSize = 448;
                origImage.Mutate(x => x.Resize(targetSize, targetSize));
                Console.WriteLine($"  Resized to: {origImage.Width}x{origImage.Height}");

                // Convert to tensor (batch_size=1, 3 channels, height, width)
                var tensor = ImageToTensor(origImage);

                // Run inference
                var sw = Stopwatch.StartNew();
                TableModel04Result result;

                using (torch.no_grad())
                {
                    result = model.forward(tensor);
                }

                sw.Stop();
                totalTime += sw.Elapsed;

                // Print results
                Console.WriteLine($"  Inference time: {sw.ElapsedMilliseconds}ms");
                Console.WriteLine($"  Tag sequence length: {result.Sequence.Count}");
                Console.WriteLine($"  Number of bounding boxes: {result.BBoxCoords.size(0)}");

                // Decode tag sequence
                var tagNames = DecodeTagSequence(result.Sequence, config.WordMapTag);
                Console.WriteLine($"  Tag sequence: {string.Join(" ", tagNames.Take(20))}{(tagNames.Count > 20 ? "..." : "")}");

                // Save results
                SaveResults(result, tagNames, fileName, outputDir, sw.Elapsed);

                // Cleanup
                tensor.Dispose();
                result.BBoxClasses.Dispose();
                result.BBoxCoords.Dispose();

                successCount++;
                Console.WriteLine($"  ✓ Success");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ✗ Error: {ex.Message}");
            }

            Console.WriteLine();
        }

        // Summary
        Console.WriteLine("==========================================");
        Console.WriteLine("Summary:");
        Console.WriteLine($"  Processed: {successCount}/{imageFiles.Length} images");
        Console.WriteLine($"  Total time: {totalTime.TotalSeconds:F2}s");
        Console.WriteLine($"  Average time: {(successCount > 0 ? totalTime.TotalMilliseconds / successCount : 0):F0}ms per image");
        Console.WriteLine($"  Results saved to: {outputDir}");
        Console.WriteLine();
    }

    static torch.Tensor ImageToTensor(Image<Rgb24> image)
    {
        // ImageNet normalization from config
        // mean: [0.94247851, 0.94254675, 0.94292611]
        // std:  [0.17910956, 0.17940403, 0.17931663]
        var mean = new[] { 0.94247851f, 0.94254675f, 0.94292611f };
        var std = new[] { 0.17910956f, 0.17940403f, 0.17931663f };

        // Create tensor: (1, 3, height, width)
        var height = image.Height;
        var width = image.Width;
        var data = new float[3 * height * width];

        int idx = 0;
        // Channel R - normalized
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                data[idx++] = (pixel.R / 255.0f - mean[0]) / std[0];
            }
        }

        // Channel G - normalized
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                data[idx++] = (pixel.G / 255.0f - mean[1]) / std[1];
            }
        }

        // Channel B - normalized
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                data[idx++] = (pixel.B / 255.0f - mean[2]) / std[2];
            }
        }

        var tensor = torch.tensor(data).reshape(1, 3, height, width);
        return tensor;
    }

    static List<string> DecodeTagSequence(IReadOnlyList<long> sequence, Dictionary<string, long> wordMap)
    {
        // Reverse map
        var reverseMap = wordMap.ToDictionary(kv => kv.Value, kv => kv.Key);

        var result = new List<string>();
        foreach (var token in sequence)
        {
            if (reverseMap.TryGetValue(token, out var tag))
            {
                result.Add(tag);
            }
            else
            {
                result.Add($"UNK({token})");
            }
        }

        return result;
    }

    static void SaveResults(TableModel04Result result, List<string> tagNames, string fileName, string outputDir, TimeSpan inferenceTime)
    {
        var baseName = Path.GetFileNameWithoutExtension(fileName);
        var outputPath = Path.Combine(outputDir, $"{baseName}_results.txt");

        using var writer = new StreamWriter(outputPath);
        writer.WriteLine($"TableFormer Results: {fileName}");
        writer.WriteLine($"Inference Time: {inferenceTime.TotalMilliseconds:F2}ms");
        writer.WriteLine($"Timestamp: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        writer.WriteLine();

        writer.WriteLine("Tag Sequence:");
        writer.WriteLine(string.Join(" ", tagNames));
        writer.WriteLine();

        writer.WriteLine($"Number of Bounding Boxes: {result.BBoxCoords.size(0)}");
        writer.WriteLine();

        // Save bounding boxes
        if (result.BBoxCoords.size(0) > 0)
        {
            writer.WriteLine("Bounding Boxes (normalized [cx, cy, w, h]):");
            for (long i = 0; i < result.BBoxCoords.size(0); i++)
            {
                using var bbox = result.BBoxCoords[i];
                var cx = bbox[0].item<float>();
                var cy = bbox[1].item<float>();
                var w = bbox[2].item<float>();
                var h = bbox[3].item<float>();

                using var classes = result.BBoxClasses[i];
                var classLogits = new float[classes.size(0)];
                for (long j = 0; j < classes.size(0); j++)
                {
                    classLogits[j] = classes[j].item<float>();
                }
                var predictedClass = Array.IndexOf(classLogits, classLogits.Max());

                writer.WriteLine($"  Box {i}: cx={cx:F4}, cy={cy:F4}, w={w:F4}, h={h:F4}, class={predictedClass}");
            }
        }
    }
}
