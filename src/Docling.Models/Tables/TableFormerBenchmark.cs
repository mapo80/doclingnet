using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Performance benchmarking suite for TableFormer models.
/// Measures latency, throughput, memory usage, and compares Fast vs Accurate variants.
/// </summary>
internal sealed class TableFormerBenchmark
{
    private readonly string _modelsDirectory;
    private readonly int _warmupIterations = 5;
    private readonly int _benchmarkIterations = 50;

    public TableFormerBenchmark(string modelsDirectory)
    {
        _modelsDirectory = modelsDirectory ?? throw new ArgumentNullException(nameof(modelsDirectory));
    }

    /// <summary>
    /// Run comprehensive benchmark comparing Fast vs Accurate variants.
    /// </summary>
    public BenchmarkResults RunComprehensiveBenchmark(SKBitmap sampleImage)
    {
        Console.WriteLine("🚀 Starting TableFormer Performance Benchmark");
        Console.WriteLine($"Models Directory: {_modelsDirectory}");
        Console.WriteLine($"Sample Image: {sampleImage.Width}x{sampleImage.Height}");
        Console.WriteLine($"Warmup Iterations: {_warmupIterations}");
        Console.WriteLine($"Benchmark Iterations: {_benchmarkIterations}");
        Console.WriteLine(new string('-', 80));

        var results = new BenchmarkResults();

        try
        {
            // Benchmark Fast variant
            Console.WriteLine("📊 Benchmarking Fast Variant...");
            results.FastResults = BenchmarkVariant("Fast", sampleImage);

            // Benchmark Accurate variant
            Console.WriteLine("📊 Benchmarking Accurate Variant...");
            results.AccurateResults = BenchmarkVariant("Accurate", sampleImage);

            // Calculate comparison metrics
            CalculateComparisonMetrics(results);

            // Print results
            PrintBenchmarkResults(results);

            return results;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Benchmark failed: {ex.Message}");
            throw;
        }
    }

    private VariantBenchmarkResult BenchmarkVariant(string variant, SKBitmap sampleImage)
    {
        var preprocessor = new ImagePreprocessor();
        var components = new TableFormerOnnxComponents(_modelsDirectory.Replace("fast", variant.ToLower(), StringComparison.Ordinal));

        var result = new VariantBenchmarkResult
        {
            Variant = variant,
            ModelSizes = GetModelSizes(variant)
        };

        try
        {
            // Warmup phase
            Console.WriteLine($"  Warming up {variant} variant...");
            for (int i = 0; i < _warmupIterations; i++)
            {
                var input = preprocessor.PreprocessImage(sampleImage);
                _ = components.RunEncoder(input);
                input.Dispose();
            }

            // Actual benchmarking
            Console.WriteLine($"  Running {variant} benchmark ({_benchmarkIterations} iterations)...");
            var stopwatch = new Stopwatch();
            var inferenceTimes = new List<long>();
            var memoryUsages = new List<long>();

            for (int i = 0; i < _benchmarkIterations; i++)
            {
                // Force garbage collection before measurement
                GC.Collect();
                GC.WaitForPendingFinalizers();

                var initialMemory = GC.GetTotalMemory(true);

                stopwatch.Restart();
                var input = preprocessor.PreprocessImage(sampleImage);
                var output = components.RunEncoder(input);
                stopwatch.Stop();

                var finalMemory = GC.GetTotalMemory(true);
                var memoryUsed = finalMemory - initialMemory;

                inferenceTimes.Add(stopwatch.ElapsedMilliseconds);
                memoryUsages.Add(memoryUsed);

                // DenseTensor doesn't implement IDisposable
            }

            // Calculate statistics
            result.AverageInferenceTime = inferenceTimes.Average();
            result.MinInferenceTime = inferenceTimes.Min();
            result.MaxInferenceTime = inferenceTimes.Max();
            result.StdDevInferenceTime = CalculateStdDev(inferenceTimes);
            result.AverageMemoryUsage = (long)memoryUsages.Average();
            result.PeakMemoryUsage = memoryUsages.Max();

            // Calculate throughput (images per second)
            result.Throughput = 1000.0 / result.AverageInferenceTime;

            Console.WriteLine($"  ✅ {variant} Results:");
            Console.WriteLine($"     Average Latency: {result.AverageInferenceTime:F2}ms");
            Console.WriteLine($"     Throughput: {result.Throughput:F2} images/sec");
            Console.WriteLine($"     Memory Usage: {result.AverageMemoryUsage / 1024:F2} KB avg, {result.PeakMemoryUsage / 1024:F2} KB peak");

            return result;
        }
        finally
        {
            components.Dispose();
            preprocessor.Dispose();
        }
    }

    private Dictionary<string, long> GetModelSizes(string variant)
    {
        var sizes = new Dictionary<string, long>();
        var variantLower = variant.ToUpperInvariant();

        var modelFiles = new[]
        {
            $"{variantLower}_encoder.onnx",
            $"{variantLower}_tag_transformer_encoder.onnx",
            $"{variantLower}_tag_transformer_decoder_step.onnx",
            $"{variantLower}_bbox_decoder.onnx"
        };

        foreach (var file in modelFiles)
        {
            var path = Path.Combine(_modelsDirectory.Replace("fast", variantLower, StringComparison.Ordinal), file);
            if (File.Exists(path))
            {
                sizes[file] = new FileInfo(path).Length;
            }
        }

        return sizes;
    }

    private void CalculateComparisonMetrics(BenchmarkResults results)
    {
        if (results.FastResults == null || results.AccurateResults == null)
            return;

        var fast = results.FastResults;
        var accurate = results.AccurateResults;

        // Speed comparison
        results.SpeedRatio = accurate.AverageInferenceTime / fast.AverageInferenceTime;
        results.FastIsFaster = fast.AverageInferenceTime < accurate.AverageInferenceTime;

        // Memory comparison
        results.MemoryRatio = accurate.AverageMemoryUsage / fast.AverageMemoryUsage;

        // Model size comparison
        var fastTotalSize = fast.ModelSizes.Values.Sum();
        var accurateTotalSize = accurate.ModelSizes.Values.Sum();
        results.SizeRatio = (double)accurateTotalSize / fastTotalSize;
    }

    private void PrintBenchmarkResults(BenchmarkResults results)
    {
        Console.WriteLine("\n" + new string('=', 80));
        Console.WriteLine("📈 BENCHMARK RESULTS SUMMARY");
        Console.WriteLine(new string('=', 80));

        if (results.FastResults != null && results.AccurateResults != null)
        {
            var fast = results.FastResults;
            var accurate = results.AccurateResults;

            Console.WriteLine("⚡ PERFORMANCE COMPARISON:");
            Console.WriteLine($"   Fast → Accurate Speed Ratio: {results.SpeedRatio:F2}x");
            Console.WriteLine($"   Fast → Accurate Memory Ratio: {results.MemoryRatio:F2}x");
            Console.WriteLine($"   Fast → Accurate Size Ratio: {results.SizeRatio:F2}x");
            Console.WriteLine($"   Winner: {(results.FastIsFaster ? "⚡ FAST (Speed)" : "🎯 ACCURATE (Quality)")}");

            Console.WriteLine("\n📊 DETAILED RESULTS:");
            Console.WriteLine($"   {"Variant",-10} {"Latency(ms)",-12} {"Throughput",-12} {"Memory(KB)",-12} {"Model Size",-12}");
            Console.WriteLine($"   {"",-10} {"",-12} {"(img/sec)",-12} {"",-12} {"(MB)",-12}");

            PrintVariantResult("Fast", fast);
            PrintVariantResult("Accurate", accurate);

            Console.WriteLine("\n💡 RECOMMENDATIONS:");
            if (results.SpeedRatio > 2.0)
            {
                Console.WriteLine("   • Fast variant is significantly faster - use for real-time applications");
            }
            if (results.SizeRatio > 1.5)
            {
                Console.WriteLine("   • Accurate variant uses significantly more memory - consider resource constraints");
            }
            if (results.FastIsFaster)
            {
                Console.WriteLine("   • Fast variant recommended for most use cases");
            }
            else
            {
                Console.WriteLine("   • Accurate variant recommended for highest quality results");
            }
        }

        Console.WriteLine(new string('=', 80));
    }

    private static void PrintVariantResult(string name, VariantBenchmarkResult result)
    {
        var totalSizeMB = result.ModelSizes.Values.Sum() / 1024.0 / 1024.0;
        Console.WriteLine($"   {name,-10} {result.AverageInferenceTime,10:F2} {result.Throughput,10:F2} {result.AverageMemoryUsage / 1024,10:F2} {totalSizeMB,10:F2}");
    }

    private static double CalculateStdDev(IEnumerable<long> values)
    {
        var average = values.Average();
        var sumOfSquares = values.Sum(v => Math.Pow(v - average, 2));
        return Math.Sqrt(sumOfSquares / values.Count());
    }

    /// <summary>
    /// Results of comprehensive benchmark run.
    /// </summary>
    public sealed class BenchmarkResults
    {
        public VariantBenchmarkResult? FastResults { get; set; }
        public VariantBenchmarkResult? AccurateResults { get; set; }
        public double SpeedRatio { get; set; } = 1.0;
        public double MemoryRatio { get; set; } = 1.0;
        public double SizeRatio { get; set; } = 1.0;
        public bool FastIsFaster { get; set; }
    }

    /// <summary>
    /// Results for a single variant benchmark.
    /// </summary>
    public sealed class VariantBenchmarkResult
    {
        public string Variant { get; set; } = "";
        public double AverageInferenceTime { get; set; }
        public double MinInferenceTime { get; set; }
        public double MaxInferenceTime { get; set; }
        public double StdDevInferenceTime { get; set; }
        public double Throughput { get; set; }
        public long AverageMemoryUsage { get; set; }
        public long PeakMemoryUsage { get; set; }
        public Dictionary<string, long> ModelSizes { get; set; } = new();
    }
}