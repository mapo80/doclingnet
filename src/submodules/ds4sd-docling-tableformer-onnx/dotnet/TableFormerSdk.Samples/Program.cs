using SkiaSharp;
using TableFormerSdk;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Samples;

/// <summary>
/// Sample program demonstrating TableFormerSdk usage
/// Matches the Python example.py functionality
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=".PadRight(70, '='));
        Console.WriteLine("TableFormer ONNX .NET SDK - Sample Application");
        Console.WriteLine("Based on HuggingFace asmud/ds4sd-docling-models-onnx");
        Console.WriteLine("=".PadRight(70, '='));
        Console.WriteLine();

        // Parse arguments
        var modelVariant = TableFormerModelVariant.Fast;
        string? imagePath = null;
        bool runBenchmark = false;
        int iterations = 100;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model" when i + 1 < args.Length:
                    modelVariant = args[++i].ToLower() == "accurate"
                        ? TableFormerModelVariant.Accurate
                        : TableFormerModelVariant.Fast;
                    break;
                case "--image" when i + 1 < args.Length:
                    imagePath = args[++i];
                    break;
                case "--benchmark":
                    runBenchmark = true;
                    break;
                case "--iterations" when i + 1 < args.Length:
                    iterations = int.Parse(args[++i]);
                    break;
                case "--help":
                    ShowHelp();
                    return;
            }
        }

        // Find models directory
        var modelsDir = FindModelsDirectory();
        if (modelsDir == null)
        {
            Console.WriteLine("❌ Error: Could not find models directory");
            Console.WriteLine("Expected: ../../models/ relative to executable");
            return;
        }

        Console.WriteLine($"📁 Models directory: {modelsDir}");
        Console.WriteLine();

        try
        {
            // Initialize SDK
            using var sdk = new TableFormer(modelsDir);

            // Run benchmark if requested
            if (runBenchmark)
            {
                Console.WriteLine($"🚀 Running benchmark ({modelVariant} model, {iterations} iterations)...");
                Console.WriteLine();

                var benchResult = sdk.Benchmark(modelVariant, iterations);

                Console.WriteLine("📊 Benchmark Results:");
                Console.WriteLine($"  Mean time: {benchResult.MeanTimeMs:F3}ms ± {benchResult.StdTimeMs:F3}ms");
                Console.WriteLine($"  Median time: {benchResult.MedianTimeMs:F3}ms");
                Console.WriteLine($"  Min time: {benchResult.MinTimeMs:F3}ms");
                Console.WriteLine($"  Max time: {benchResult.MaxTimeMs:F3}ms");
                Console.WriteLine($"  Throughput: {benchResult.ThroughputFps:F1} FPS");
                Console.WriteLine();
            }

            // Process image if provided
            if (!string.IsNullOrEmpty(imagePath))
            {
                if (!File.Exists(imagePath))
                {
                    Console.WriteLine($"❌ Error: Image not found: {imagePath}");
                    return;
                }

                Console.WriteLine($"🖼️  Processing image: {imagePath}");
                Console.WriteLine($"  Model: {modelVariant}");
                Console.WriteLine();

                var result = sdk.ExtractTableStructure(imagePath, modelVariant);

                Console.WriteLine("✓ Table structure extracted:");
                Console.WriteLine($"  Regions found: {result.Regions.Count}");
                Console.WriteLine($"  Inference time: {result.InferenceTime.TotalMilliseconds:F2}ms");
                Console.WriteLine($"  Model variant: {result.ModelVariant}");
                Console.WriteLine($"  Raw output shapes: {string.Join(", ", result.RawOutputShapes.Select(kvp => $"{kvp.Key}={string.Join("x", kvp.Value)}"))}");
                Console.WriteLine();

                if (result.Regions.Count > 0)
                {
                    Console.WriteLine("  Sample regions (first 5):");
                    foreach (var region in result.Regions.Take(5))
                    {
                        Console.WriteLine($"    {region}");
                    }
                    if (result.Regions.Count > 5)
                    {
                        Console.WriteLine($"    ... and {result.Regions.Count - 5} more");
                    }
                }
            }

            // Run demo with dummy data if no image provided
            if (string.IsNullOrEmpty(imagePath) && !runBenchmark)
            {
                Console.WriteLine($"🔬 Running demo with dummy data ({modelVariant} model)...");
                Console.WriteLine();

                // Create dummy image
                using var dummyImage = new SKBitmap(400, 300, SKColorType.Rgba8888, SKAlphaType.Premul);
                using var canvas = new SKCanvas(dummyImage);

                // Fill with random pixels
                var random = new Random(42);
                var pixels = dummyImage.GetPixelSpan();
                random.NextBytes(pixels);

                var result = sdk.ExtractTableStructure(dummyImage, modelVariant);

                Console.WriteLine("✓ Demo completed:");
                Console.WriteLine($"  Model: {result.ModelVariant}");
                Console.WriteLine($"  Regions: {result.Regions.Count}");
                Console.WriteLine($"  Inference time: {result.InferenceTime.TotalMilliseconds:F2}ms");
                Console.WriteLine($"  Raw outputs: {string.Join(", ", result.RawOutputShapes.Select(kvp => $"{kvp.Key}={string.Join("x", kvp.Value)}"))}");
                Console.WriteLine();
            }

            Console.WriteLine("✅ Example completed successfully!");
            Console.WriteLine();
            Console.WriteLine("To process a real image, use:");
            Console.WriteLine($"  dotnet run -- --model {modelVariant.ToString().ToLower()} --image path/to/table.png");
            Console.WriteLine();
            Console.WriteLine("To run a benchmark, use:");
            Console.WriteLine($"  dotnet run -- --model {modelVariant.ToString().ToLower()} --benchmark --iterations 100");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"   {ex.InnerException.Message}");
            }
        }
    }

    static string? FindModelsDirectory()
    {
        var baseDir = AppContext.BaseDirectory;

        // Try relative paths
        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "models")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "models")),
            Path.GetFullPath(Path.Combine(baseDir, "models")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "models")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "models"))
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate))
            {
                // Check if models exist
                if (File.Exists(Path.Combine(candidate, "tableformer_fast.onnx")) ||
                    File.Exists(Path.Combine(candidate, "tableformer_accurate.onnx")))
                {
                    return candidate;
                }
            }
        }

        return null;
    }

    static void ShowHelp()
    {
        Console.WriteLine("Usage: TableFormerSdk.Samples [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --model <fast|accurate>   Model variant to use (default: fast)");
        Console.WriteLine("  --image <path>            Path to table image to process");
        Console.WriteLine("  --benchmark               Run performance benchmark");
        Console.WriteLine("  --iterations <n>          Number of benchmark iterations (default: 100)");
        Console.WriteLine("  --help                    Show this help message");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  dotnet run -- --model fast --image table.png");
        Console.WriteLine("  dotnet run -- --model accurate --benchmark --iterations 100");
        Console.WriteLine("  dotnet run -- --help");
    }
}
