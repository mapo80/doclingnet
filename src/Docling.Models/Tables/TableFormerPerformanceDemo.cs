using SkiaSharp;
using System;
using System.IO;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Demonstration of TableFormer performance optimizations and benchmarking.
/// Shows how to use the optimized components and run performance comparisons.
/// </summary>
internal static class TableFormerPerformanceDemo
{
    public static void RunDemo()
    {
        Console.WriteLine("🚀 TableFormer Performance Optimization Demo");
        Console.WriteLine("=============================================");
        Console.WriteLine();

        // Check if models exist
        var modelsDir = "src/submodules/ds4sd-docling-tableformer-onnx/models";
        if (!Directory.Exists(modelsDir))
        {
            Console.WriteLine($"❌ Models directory not found: {modelsDir}");
            Console.WriteLine("Please ensure models are copied to the correct location.");
            return;
        }

        Console.WriteLine($"✅ Models found in: {modelsDir}");
        Console.WriteLine();

        // Create a sample image for testing
        var sampleImage = CreateSampleImage(800, 600);
        if (sampleImage == null)
        {
            Console.WriteLine("❌ Failed to create sample image");
            return;
        }

        try
        {
            // Demonstrate optimized components
            DemonstrateOptimizedComponents(modelsDir, sampleImage);

            // Run benchmark comparison
            RunBenchmarkComparison(modelsDir, sampleImage);
        }
        finally
        {
            sampleImage.Dispose();
        }

        Console.WriteLine("✅ Demo completed successfully!");
    }

    private static void DemonstrateOptimizedComponents(string modelsDir, SKBitmap sampleImage)
    {
        Console.WriteLine("🔧 Testing Optimized Components");
        Console.WriteLine("-----------------------------");

        try
        {
            // Test optimized components
            var optimizedComponents = new OptimizedTableFormerOnnxComponents(modelsDir, useCUDA: false);

            var preprocessor = new ImagePreprocessor();
            var input = preprocessor.PreprocessImage(sampleImage);

            // Run optimized inference
            var output = optimizedComponents.RunEncoderOptimized(input);

            // Get performance metrics
            var metrics = optimizedComponents.GetPerformanceMetrics();

            Console.WriteLine("✅ Optimized inference successful!");
            Console.WriteLine($"   Provider: {metrics.Provider}");
            Console.WriteLine($"   Optimization: {metrics.OptimizationLevel}");
            Console.WriteLine($"   Memory Usage: {metrics.EncoderMemoryUsage / 1024:F2} KB");
            Console.WriteLine($"   Sessions: {metrics.TotalSessions}");

            // DenseTensor doesn't implement IDisposable
            optimizedComponents.Dispose();
            preprocessor.Dispose();

            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Optimized components test failed: {ex.Message}");
            Console.WriteLine($"   This is expected if CUDA is not available or models have issues.");
            Console.WriteLine();
        }
    }

    private static void RunBenchmarkComparison(string modelsDir, SKBitmap sampleImage)
    {
        Console.WriteLine("📊 Running Performance Benchmark");
        Console.WriteLine("-------------------------------");

        try
        {
            var benchmark = new TableFormerBenchmark(modelsDir);
            var results = benchmark.RunComprehensiveBenchmark(sampleImage);

            Console.WriteLine("✅ Benchmark completed successfully!");
            Console.WriteLine();

            // Show key insights
            if (results.FastResults != null && results.AccurateResults != null)
            {
                Console.WriteLine("💡 Key Performance Insights:");
                Console.WriteLine($"   • Fast variant: {results.FastResults.Throughput:F1} images/sec");
                Console.WriteLine($"   • Accurate variant: {results.AccurateResults.Throughput:F1} images/sec");
                Console.WriteLine($"   • Speed ratio: {results.SpeedRatio:F2}x faster with Fast");
                Console.WriteLine($"   • Memory ratio: {results.MemoryRatio:F2}x more memory with Accurate");

                if (results.FastIsFaster)
                {
                    Console.WriteLine("   • Recommendation: Use Fast variant for most applications");
                }
                else
                {
                    Console.WriteLine("   • Recommendation: Use Accurate variant for highest quality");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Benchmark failed: {ex.Message}");
            Console.WriteLine("This may be due to missing models or incompatible formats.");
        }

        Console.WriteLine();
    }

    private static SKBitmap? CreateSampleImage(int width, int height)
    {
        try
        {
            var bitmap = new SKBitmap(width, height);
            using var canvas = new SKCanvas(bitmap);

            // Create a simple test image with a table-like pattern
            canvas.Clear(SKColors.White);

            // Draw some lines to simulate table structure
            var paint = new SKPaint
            {
                Color = SKColors.Black,
                StrokeWidth = 2,
                IsStroke = true
            };

            // Draw horizontal lines
            for (int y = 100; y < height; y += 100)
            {
                canvas.DrawLine(0, y, width, y, paint);
            }

            // Draw vertical lines
            for (int x = 100; x < width; x += 100)
            {
                canvas.DrawLine(x, 0, x, height, paint);
            }

            return bitmap;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to create sample image: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Example of how to use the optimized TableFormer in production code.
    /// </summary>
    public static async Task DemonstrateProductionUsage()
    {
        Console.WriteLine("🏭 Production Usage Example");
        Console.WriteLine("--------------------------");

        var modelsDir = "src/submodules/ds4sd-docling-tableformer-onnx/models";
        var useCUDA = false; // Set to true if CUDA is available
        var enableQuantization = false; // Set to true for better performance

        try
        {
            using var optimizedComponents = new OptimizedTableFormerOnnxComponents(modelsDir, useCUDA, enableQuantization);
            using var preprocessor = new ImagePreprocessor();

            // Load your image here
            // var image = SKBitmap.Decode("path/to/your/table/image.png");

            Console.WriteLine("✅ Production setup ready!");
            Console.WriteLine($"   CUDA Enabled: {useCUDA}");
            Console.WriteLine($"   Quantization: {enableQuantization}");
            Console.WriteLine($"   Provider: {(useCUDA ? "GPU" : "CPU")}");

            // Example usage:
            // var input = preprocessor.PreprocessImage(image);
            // var features = optimizedComponents.RunEncoderOptimized(input);
            // Process features...
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Production setup failed: {ex.Message}");
        }
    }
}