#if false
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Models.Tables;

/// <summary>
/// Example demonstrating the complete TableFormer integration with new features.
/// Shows configuration flexibility, hot-reload, metrics, and telemetry.
/// </summary>
public static class TableFormerUsageExample
{
    public static async Task DemonstrateTableFormerFeatures()
    {
        // Demonstrate comprehensive benchmarking
        await DemonstrateBenchmarking();

        await DemonstrateBasicFeatures();
    }

    private static async Task DemonstrateBenchmarking()
    {
        Console.WriteLine("🚀 TableFormer Benchmark - Confronto Performance");
        Console.WriteLine("===============================================");

        using var benchmark = new TableFormerBenchmark();
        benchmark.GenerateTestImages(count: 5);

        Console.WriteLine("Running performance benchmark...");
        var results = await benchmark.RunBenchmarkAsync(iterationsPerImage: 2);

        Console.WriteLine($"\n📊 Benchmark Results ({results.Timestamp})");
        Console.WriteLine($"Test Images: {results.TestImageCount}");
        Console.WriteLine($"Iterations per Image: {results.IterationsPerImage}");
        Console.WriteLine();

        // Display Fast model results
        var fast = results.FastModelResults;
        Console.WriteLine("🚀 Fast Model:");
        Console.WriteLine($"  Success Rate: {fast.SuccessRate:P1}");
        Console.WriteLine($"  Avg Inference Time: {fast.AverageInferenceTime.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Min/Max Time: {fast.MinInferenceTime.TotalMilliseconds:F1}ms / {fast.MaxInferenceTime.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Total Cells Detected: {fast.TotalCellsDetected}");
        Console.WriteLine($"  Avg Cells per Image: {fast.AverageCellsDetected:F1}");

        // Display Accurate model results
        var accurate = results.AccurateModelResults;
        Console.WriteLine("\n🎯 Accurate Model:");
        Console.WriteLine($"  Success Rate: {accurate.SuccessRate:P1}");
        Console.WriteLine($"  Avg Inference Time: {accurate.AverageInferenceTime.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Min/Max Time: {accurate.MinInferenceTime.TotalMilliseconds:F1}ms / {accurate.MaxInferenceTime.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Total Cells Detected: {accurate.TotalCellsDetected}");
        Console.WriteLine($"  Avg Cells per Image: {accurate.AverageCellsDetected:F1}");

        // Display comparison
        if (results.PerformanceComparison != null)
        {
            var comp = results.PerformanceComparison;
            Console.WriteLine("\n⚖️ Comparison:");
            Console.WriteLine($"  Fast is faster: {comp.FastIsFaster}");
            Console.WriteLine($"  Speed ratio (Accurate/Fast): {comp.SpeedRatio:F2}x");
            Console.WriteLine($"  Accuracy ratio (Accurate/Fast): {comp.AccuracyComparison:F2}x");

            if (comp.FastIsFaster)
            {
                Console.WriteLine($"  💡 Recommendation: Use Fast model for {comp.SpeedRatio:F1}x faster processing");
            }
            else
            {
                Console.WriteLine($"  💡 Recommendation: Use Accurate model for {comp.AccuracyComparison:F1}x more accurate results");
            }
        }
    }

    private static async Task DemonstrateBasicFeatures()
    {
        Console.WriteLine("🚀 TableFormer FASE 6 Demo - Ottimizzazioni e Performance");
        Console.WriteLine("========================================================");

        // 1. Initialize service with flexible configuration
        var service = new TableFormerTableStructureService(
            options: new TableFormerStructureServiceOptions
            {
                Variant = TableFormerModelVariant.Fast,
                Runtime = TableFormerRuntime.Onnx,
                GenerateOverlay = true,
                WorkingDirectory = Path.GetTempPath()
            },
            logger: CreateLogger()
        );

        try
        {
            // 2. Demonstrate configuration inspection
            Console.WriteLine("\n📋 Configurazione attuale:");
            var (fastEncoder, fastTagEncoder, accurateEncoder) = service.GetCurrentModelPaths();
            Console.WriteLine($"Fast Encoder: {Path.GetFileName(fastEncoder)}");
            Console.WriteLine($"Backend attivo: {(service.IsUsingOnnxBackend() ? "ONNX ✅" : "Stub ⚠️")}");

            // 3. Demonstrate metrics collection
            Console.WriteLine("\n📊 Metrics iniziali:");
            var initialMetrics = service.GetMetrics();
            Console.WriteLine($"Inferences totali: {initialMetrics.TotalInferences}");
            Console.WriteLine($"Success rate: {initialMetrics.SuccessRate:P1}");

            // 4. Demonstrate hot-reload capability
            Console.WriteLine("\n🔄 Test hot-reload modelli:");
            service.ReloadModels();
            Console.WriteLine("Modelli ricaricati con successo!");

            // 5. Process sample images and collect metrics
            Console.WriteLine("\n🖼️ Processamento immagini di test:");

            for (int i = 0; i < 3; i++)
            {
                var testImage = CreateSampleTableImage(400, 300);
                var request = CreateSampleRequest(testImage, i);

                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var result = await service.InferStructureAsync(request);
                stopwatch.Stop();

                Console.WriteLine($"Immagine {i + 1}: {result.Cells.Count} celle, {result.RowCount} righe, {result.ColumnCount} colonne ({stopwatch.ElapsedMilliseconds}ms)");
            }

            // 6. Display final metrics
            Console.WriteLine("\n📈 Metrics finali:");
            var finalMetrics = service.GetMetrics();
            Console.WriteLine($"Inferences totali: {finalMetrics.TotalInferences}");
            Console.WriteLine($"Success rate: {finalMetrics.SuccessRate:P1}");
            Console.WriteLine($"Tempo medio: {finalMetrics.AverageInferenceTime.TotalMilliseconds:F1}ms");
            Console.WriteLine($"Celle totali rilevate: {finalMetrics.TotalCellsDetected}");
            Console.WriteLine($"Backend utilizzato: {string.Join(", ", finalMetrics.BackendUsage)}");

            // 7. Demonstrate environment variable configuration
            Console.WriteLine("\n🔧 Configurazione da environment variables:");
            Console.WriteLine("Impostare le seguenti variabili per personalizzare:");
            Console.WriteLine("- TABLEFORMER_MODELS_ROOT: Path root modelli");
            Console.WriteLine("- TABLEFORMER_FAST_MODELS_PATH: Path modelli Fast specifici");
            Console.WriteLine("- TABLEFORMER_ACCURATE_MODELS_PATH: Path modelli Accurate specifici");

            // 8. Demonstrate batch processing for better performance
            Console.WriteLine("\n⚡ Test batch processing:");
            var batchRequests = new List<TableStructureRequest>();
            for (int i = 0; i < 5; i++)
            {
                var batchImage = CreateSampleTableImage(300, 200);
                batchRequests.Add(CreateSampleRequest(batchImage, i + 10));
            }

            var batchStopwatch = Stopwatch.StartNew();
            var batchResults = await service.InferStructureBatchAsync(batchRequests);
            batchStopwatch.Stop();

            Console.WriteLine($"Batch processing: {batchResults.Count} immagini in {batchStopwatch.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"Throughput: {batchResults.Count / batchStopwatch.Elapsed.TotalSeconds:F1} immagini/sec");

            // 9. Demonstrate performance recommendations
            Console.WriteLine("\n🎯 Performance recommendations:");
            var recommendations = service.GetPerformanceRecommendations();
            Console.WriteLine(recommendations.ToString());

        }
        finally
        {
            service.Dispose();
        }
    }

    private static ILogger<TableFormerTableStructureService> CreateLogger()
    {
        // Simple console logger for demo
        return LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Information);
        }).CreateLogger<TableFormerTableStructureService>();
    }

    private static SKBitmap CreateSampleTableImage(int width, int height)
    {
        var bitmap = new SKBitmap(width, height);

        // Draw a simple table structure
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);

        var paint = new SKPaint
        {
            Color = SKColors.Black,
            StrokeWidth = 2,
            IsStroke = true
        };

        // Draw table borders
        canvas.DrawRect(50, 50, width - 100, height - 100, paint);

        // Draw horizontal lines (simulate table rows)
        for (int i = 1; i < 4; i++)
        {
            var y = 50 + (i * (height - 100) / 4);
            canvas.DrawLine(50, y, width - 50, y, paint);
        }

        // Draw vertical lines (simulate table columns)
        for (int i = 1; i < 3; i++)
        {
            var x = 50 + (i * (width - 100) / 3);
            canvas.DrawLine(x, 50, x, height - 50, paint);
        }

        return bitmap;
    }

    private static TableStructureRequest CreateSampleRequest(SKBitmap image, int pageNumber)
    {
        using var imageData = image.Encode(SKEncodedImageFormat.Png, 90);
        using var stream = new MemoryStream();
        imageData.SaveTo(stream);

        return new TableStructureRequest(
            Page: new(pageNumber + 1),
            BoundingBox: new(0, 0, image.Width, image.Height),
            RasterizedImage: stream.ToArray()
        );
    }
}
#endif
