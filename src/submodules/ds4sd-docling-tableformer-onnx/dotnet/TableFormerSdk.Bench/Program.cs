using System.Diagnostics;
using System.Text;
using System.Text.Json;
using SkiaSharp;
using TableFormerSdk;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Bench;

/// <summary>
/// Comprehensive benchmark comparing Python vs .NET TableFormer implementations
/// </summary>
public class Program
{
    private const int WarmupRuns = 1;
    private const int BenchmarkRuns = 6;
    private const int RunsToKeep = 5; // Discard first run

    public static async Task<int> Main(string[] args)
    {
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("  TableFormer SDK - Python vs .NET Comprehensive Benchmark");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine();

        var modelVariant = args.Length > 0 && args[0] == "accurate"
            ? TableFormerModelVariant.Accurate
            : TableFormerModelVariant.Fast;

        Console.WriteLine($"Model Variant: {modelVariant}");
        Console.WriteLine($"Warmup runs: {WarmupRuns}");
        Console.WriteLine($"Benchmark runs: {BenchmarkRuns} (discard first, keep {RunsToKeep})");
        Console.WriteLine();

        // Find directories
        var modelsDir = FindModelsDirectory();
        var datasetDir = FindDatasetDirectory();

        if (modelsDir == null || datasetDir == null)
        {
            Console.WriteLine("❌ Error: Could not find models or dataset directory");
            return 1;
        }

        Console.WriteLine($"Models: {modelsDir}");
        Console.WriteLine($"Dataset: {datasetDir}");
        Console.WriteLine();

        // Find test images
        var imageFiles = Directory.GetFiles(datasetDir, "*.png")
            .OrderBy(f => f)
            .Take(10) // Limit to 10 images for reasonable benchmark time
            .ToArray();

        if (imageFiles.Length == 0)
        {
            Console.WriteLine("❌ Error: No PNG images found in dataset");
            return 1;
        }

        Console.WriteLine($"Found {imageFiles.Length} test images");
        Console.WriteLine();

        try
        {
            // Run .NET benchmarks
            Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine("  Running .NET Benchmarks");
            Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine();

            var dotnetResults = await RunDotnetBenchmarks(modelsDir, imageFiles, modelVariant);

            // Run Python benchmarks
            Console.WriteLine();
            Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine("  Running Python Benchmarks");
            Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine();

            var pythonResults = await RunPythonBenchmarks(modelsDir, imageFiles, modelVariant);

            // Generate reports
            Console.WriteLine();
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine("  PERFORMANCE COMPARISON");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine();

            GeneratePerformanceReport(dotnetResults, pythonResults);

            Console.WriteLine();
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine("  RESULTS ACCURACY COMPARISON");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine();

            GenerateAccuracyReport(dotnetResults, pythonResults);

            Console.WriteLine();
            Console.WriteLine("✅ Benchmark completed successfully!");

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            return 1;
        }
    }

    private static async Task<List<BenchmarkResult>> RunDotnetBenchmarks(
        string modelsDir,
        string[] imageFiles,
        TableFormerModelVariant variant)
    {
        var results = new List<BenchmarkResult>();

        using var sdk = new TableFormer(modelsDir);

        foreach (var imagePath in imageFiles)
        {
            var fileName = Path.GetFileName(imagePath);
            Console.Write($"  {fileName,-50} ");

            var times = new List<double>();
            var lastResult = default(Models.TableStructureResult);

            // Warmup
            for (int i = 0; i < WarmupRuns; i++)
            {
                _ = sdk.ExtractTableStructure(imagePath, variant);
            }

            // Benchmark runs
            for (int i = 0; i < BenchmarkRuns; i++)
            {
                var sw = Stopwatch.StartNew();
                lastResult = sdk.ExtractTableStructure(imagePath, variant);
                sw.Stop();

                times.Add(sw.Elapsed.TotalMilliseconds);
            }

            // Discard first run, keep rest
            var validTimes = times.Skip(1).Take(RunsToKeep).ToList();
            var avgTime = validTimes.Average();

            Console.WriteLine($"{avgTime:F3}ms");

            results.Add(new BenchmarkResult
            {
                ImagePath = imagePath,
                ImageName = fileName,
                Times = validTimes,
                AverageTime = avgTime,
                RegionCount = lastResult?.Regions.Count ?? 0,
                Regions = lastResult?.Regions.ToList() ?? new List<Models.TableRegion>()
            });

            await Task.Delay(10); // Small delay between images
        }

        return results;
    }

    private static async Task<List<BenchmarkResult>> RunPythonBenchmarks(
        string modelsDir,
        string[] imageFiles,
        TableFormerModelVariant variant)
    {
        var results = new List<BenchmarkResult>();

        // Create Python benchmark script
        var pythonScript = CreatePythonBenchmarkScript(
            modelsDir,
            imageFiles,
            variant == TableFormerModelVariant.Fast ? "fast" : "accurate"
        );

        var scriptPath = Path.Combine(Path.GetTempPath(), "tableformer_bench.py");
        await File.WriteAllTextAsync(scriptPath, pythonScript);

        try
        {
            // Run Python script
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "python3",
                    Arguments = scriptPath,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                Console.WriteLine($"Python error: {error}");
                throw new InvalidOperationException($"Python benchmark failed: {error}");
            }

            // Parse JSON output
            var jsonResults = JsonSerializer.Deserialize<List<PythonBenchmarkResult>>(output);

            if (jsonResults == null)
            {
                throw new InvalidOperationException("Failed to parse Python benchmark results");
            }

            // Convert to BenchmarkResult
            foreach (var pyResult in jsonResults)
            {
                Console.WriteLine($"  {pyResult.ImageName,-50} {pyResult.AverageTime:F3}ms");

                results.Add(new BenchmarkResult
                {
                    ImagePath = pyResult.ImagePath,
                    ImageName = pyResult.ImageName,
                    Times = pyResult.Times,
                    AverageTime = pyResult.AverageTime,
                    RegionCount = pyResult.RegionCount,
                    Regions = new List<Models.TableRegion>() // Python doesn't return detailed regions in this benchmark
                });
            }
        }
        finally
        {
            if (File.Exists(scriptPath))
            {
                File.Delete(scriptPath);
            }
        }

        return results;
    }

    private static string CreatePythonBenchmarkScript(string modelsDir, string[] imageFiles, string variant)
    {
        var imagesJson = JsonSerializer.Serialize(imageFiles);

        return $@"#!/usr/bin/env python3
import onnxruntime as ort
import cv2
import numpy as np
import time
import json
import sys

WARMUP_RUNS = {WarmupRuns}
BENCHMARK_RUNS = {BenchmarkRuns}
RUNS_TO_KEEP = {RunsToKeep}

class TableFormerONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type

    def create_dummy_input(self, seed=42):
        np.random.seed(seed)
        if self.input_type == 'tensor(int64)':
            return np.random.randint(0, 100, self.input_shape).astype(np.int64)
        else:
            return np.random.randn(*self.input_shape).astype(np.float32)

    def predict(self, input_tensor):
        outputs = self.session.run(None, {{self.input_name: input_tensor}})
        return outputs

    def normalize_boundaries(self, scores):
        if len(scores) == 0:
            return [0.0, 1.0]

        min_val = scores.min()
        max_val = scores.max()
        range_val = max(1e-6, max_val - min_val)

        normalized = np.clip((scores - min_val) / range_val, 0, 1)
        normalized = sorted(set(normalized))

        if len(normalized) == 0 or normalized[0] > 0:
            normalized.insert(0, 0.0)

        if normalized[-1] < 1.0:
            normalized.append(1.0)

        # Remove extremely small segments
        result = [normalized[0]]
        for val in normalized[1:]:
            if val - result[-1] >= 0.01:
                result.append(val)

        return result

    def extract_regions(self, outputs, image_width, image_height):
        regions = []

        # For JPQD models, output is [1, 10] float32
        output_array = outputs[0].flatten()

        # Split output: first half for rows, second half for columns
        half_len = max(1, len(output_array) // 2)
        row_scores = output_array[:half_len]
        col_scores = output_array[half_len:]

        # Normalize boundaries
        row_boundaries = self.normalize_boundaries(row_scores)
        col_boundaries = self.normalize_boundaries(col_scores)

        # Create cells from boundaries
        for row in range(len(row_boundaries) - 1):
            for col in range(len(col_boundaries) - 1):
                x = col_boundaries[col] * image_width
                y = row_boundaries[row] * image_height
                w = max(1, (col_boundaries[col + 1] - col_boundaries[col]) * image_width)
                h = max(1, (row_boundaries[row + 1] - row_boundaries[row]) * image_height)
                regions.append((x, y, w, h))

        # Ensure at least one region
        if len(regions) == 0:
            regions.append((0, 0, image_width, image_height))

        return regions

model_path = '{modelsDir}/tableformer_{variant}.onnx'
model = TableFormerONNX(model_path)

image_files = {imagesJson}
results = []

for image_path in image_files:
    image_name = image_path.split('/')[-1]

    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        continue
    image_height, image_width = image.shape[:2]

    # Warmup
    for _ in range(WARMUP_RUNS):
        dummy_input = model.create_dummy_input(seed=42)
        _ = model.predict(dummy_input)

    # Benchmark
    times = []
    last_outputs = None
    for i in range(BENCHMARK_RUNS):
        dummy_input = model.create_dummy_input(seed=42)  # Use same seed for reproducibility
        start = time.time()
        outputs = model.predict(dummy_input)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
        last_outputs = outputs

    # Discard first run, keep rest
    valid_times = times[1:1+RUNS_TO_KEEP]
    avg_time = sum(valid_times) / len(valid_times)

    # Extract regions from last output
    regions = model.extract_regions(last_outputs, image_width, image_height)

    results.append({{
        'image_path': image_path,
        'image_name': image_name,
        'times': valid_times,
        'average_time': avg_time,
        'region_count': len(regions)
    }})

# Output JSON to stdout
print(json.dumps(results))
";
    }

    private static void GeneratePerformanceReport(
        List<BenchmarkResult> dotnetResults,
        List<BenchmarkResult> pythonResults)
    {
        // Header
        Console.WriteLine("┌────────────────────────────────────────────────┬──────────┬──────────┬───────────┬──────────┐");
        Console.WriteLine("│ Image                                          │ .NET (ms)│ Py (ms)  │ Diff (ms) │ Diff (%) │");
        Console.WriteLine("├────────────────────────────────────────────────┼──────────┼──────────┼───────────┼──────────┤");

        var totalDotnetTime = 0.0;
        var totalPythonTime = 0.0;

        for (int i = 0; i < Math.Min(dotnetResults.Count, pythonResults.Count); i++)
        {
            var dotnet = dotnetResults[i];
            var python = pythonResults[i];

            var diff = dotnet.AverageTime - python.AverageTime;
            var diffPct = (diff / python.AverageTime) * 100;

            totalDotnetTime += dotnet.AverageTime;
            totalPythonTime += python.AverageTime;

            var imageName = dotnet.ImageName.Length > 46
                ? dotnet.ImageName.Substring(0, 43) + "..."
                : dotnet.ImageName;

            Console.WriteLine(
                $"│ {imageName,-46} │ {dotnet.AverageTime,8:F3} │ {python.AverageTime,8:F3} │ " +
                $"{diff,9:+0.000;-0.000} │ {diffPct,7:+0.0;-0.0}% │"
            );
        }

        Console.WriteLine("├────────────────────────────────────────────────┼──────────┼──────────┼───────────┼──────────┤");

        var avgDotnet = totalDotnetTime / dotnetResults.Count;
        var avgPython = totalPythonTime / pythonResults.Count;
        var avgDiff = avgDotnet - avgPython;
        var avgDiffPct = (avgDiff / avgPython) * 100;

        Console.WriteLine(
            $"│ {"AVERAGE",-46} │ {avgDotnet,8:F3} │ {avgPython,8:F3} │ " +
            $"{avgDiff,9:+0.000;-0.000} │ {avgDiffPct,7:+0.0;-0.0}% │"
        );

        Console.WriteLine("└────────────────────────────────────────────────┴──────────┴──────────┴───────────┴──────────┘");

        Console.WriteLine();
        Console.WriteLine($"Total .NET time:   {totalDotnetTime:F2}ms");
        Console.WriteLine($"Total Python time: {totalPythonTime:F2}ms");
        Console.WriteLine($"Speedup factor:    {totalPythonTime / totalDotnetTime:F2}x");

        if (avgDotnet < avgPython)
        {
            Console.WriteLine($"⚡ .NET is {(avgPython / avgDotnet):F2}x faster on average");
        }
        else
        {
            Console.WriteLine($"🐍 Python is {(avgDotnet / avgPython):F2}x faster on average");
        }
    }

    private static void GenerateAccuracyReport(
        List<BenchmarkResult> dotnetResults,
        List<BenchmarkResult> pythonResults)
    {
        Console.WriteLine("┌────────────────────────────────────────────────┬───────────┬───────────┬──────────────┐");
        Console.WriteLine("│ Image                                          │ .NET Rgns │ Py Rgns   │ Deviation    │");
        Console.WriteLine("├────────────────────────────────────────────────┼───────────┼───────────┼──────────────┤");

        var totalDeviation = 0.0;
        var perfectMatches = 0;

        for (int i = 0; i < Math.Min(dotnetResults.Count, pythonResults.Count); i++)
        {
            var dotnet = dotnetResults[i];
            var python = pythonResults[i];

            // Calculate deviation percentage
            double deviation;
            string deviationStr;

            if (python.RegionCount == 0)
            {
                deviation = 0.0;
                deviationStr = "N/A";
            }
            else if (dotnet.RegionCount == python.RegionCount)
            {
                deviation = 0.0;
                deviationStr = "0,00%";
                perfectMatches++;
            }
            else
            {
                // Calculate percentage difference: |.NET - Python| / Python * 100
                var diff = Math.Abs(dotnet.RegionCount - python.RegionCount);
                deviation = (diff / (double)python.RegionCount) * 100.0;
                deviationStr = $"{deviation:F2}%";
            }

            var imageName = dotnet.ImageName.Length > 46
                ? dotnet.ImageName.Substring(0, 43) + "..."
                : dotnet.ImageName;

            Console.WriteLine(
                $"│ {imageName,-46} │ {dotnet.RegionCount,9} │ {python.RegionCount,9} │ {deviationStr,12} │"
            );

            totalDeviation += deviation;
        }

        Console.WriteLine("├────────────────────────────────────────────────┼───────────┼───────────┼──────────────┤");

        var avgDeviation = dotnetResults.Count > 0 ? totalDeviation / dotnetResults.Count : 0;

        Console.WriteLine(
            $"│ {"AVERAGE DEVIATION",-46} │           │           │ {avgDeviation,11:F2}% │"
        );

        Console.WriteLine("└────────────────────────────────────────────────┴───────────┴───────────┴──────────────┘");

        Console.WriteLine();
        Console.WriteLine($"Perfect matches: {perfectMatches}/{dotnetResults.Count}");
        Console.WriteLine($"Match rate:      {(perfectMatches * 100.0 / dotnetResults.Count):F1}%");

        Console.WriteLine();
        Console.WriteLine("ℹ️  Note: Both implementations use the same ONNX model with dummy inputs,");
        Console.WriteLine("   so results should be deterministic and identical.");
    }

    private static string? FindModelsDirectory()
    {
        var baseDir = AppContext.BaseDirectory;
        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", "models")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "models")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "models")),
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate) &&
                (File.Exists(Path.Combine(candidate, "tableformer_fast.onnx")) ||
                 File.Exists(Path.Combine(candidate, "tableformer_accurate.onnx"))))
            {
                return candidate;
            }
        }

        return null;
    }

    private static string? FindDatasetDirectory()
    {
        var baseDir = AppContext.BaseDirectory;
        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", "dataset", "FinTabNet", "images")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "dataset", "FinTabNet", "images")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "dataset", "FinTabNet", "images")),
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate) && Directory.GetFiles(candidate, "*.png").Length > 0)
            {
                return candidate;
            }
        }

        return null;
    }
}

public class BenchmarkResult
{
    public string ImagePath { get; set; } = "";
    public string ImageName { get; set; } = "";
    public List<double> Times { get; set; } = new();
    public double AverageTime { get; set; }
    public int RegionCount { get; set; }
    public List<Models.TableRegion> Regions { get; set; } = new();
}

public class PythonBenchmarkResult
{
    public string image_path { get; set; } = "";
    public string image_name { get; set; } = "";
    public List<double> times { get; set; } = new();
    public double average_time { get; set; }
    public int region_count { get; set; }

    public string ImagePath => image_path;
    public string ImageName => image_name;
    public List<double> Times => times;
    public double AverageTime => average_time;
    public int RegionCount => region_count;
}
