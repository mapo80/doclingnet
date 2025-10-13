#if false
using Microsoft.Extensions.Logging;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Docling.Models.Tables;

/// <summary>
/// Comprehensive benchmarking tool for TableFormer performance analysis.
/// Compares Fast vs Accurate models and provides detailed performance metrics.
/// </summary>
public sealed class TableFormerBenchmark : IDisposable
{
    private readonly ILogger<TableFormerBenchmark> _logger;
    private readonly TableFormerTableStructureService _fastService;
    private readonly TableFormerTableStructureService _accurateService;
    private readonly List<SKBitmap> _testImages = new();
    private bool _disposed;

    public TableFormerBenchmark(ILogger<TableFormerBenchmark>? logger = null)
    {
        _logger = logger ?? CreateDefaultLogger();

        _fastService = new TableFormerTableStructureService(
            new TableFormerStructureServiceOptions { Variant = TableFormerModelVariant.Fast },
            _logger);

        _accurateService = new TableFormerTableStructureService(
            new TableFormerStructureServiceOptions { Variant = TableFormerModelVariant.Accurate },
            _logger);
    }

    /// <summary>
    /// Generate test images with different table complexities.
    /// </summary>
    public void GenerateTestImages(int count = 10)
    {
        _logger.LogInformation("Generating {Count} test images for benchmarking", count);

        var random = new Random(42); // Deterministic seed for reproducible results

        for (int i = 0; i < count; i++)
        {
            var width = random.Next(200, 800);
            var height = random.Next(150, 600);
            var image = GenerateTestTableImage(width, height, random);
            _testImages.Add(image);
        }

        _logger.LogInformation("Generated {Count} test images", _testImages.Count);
    }

    /// <summary>
    /// Load test images from a directory.
    /// </summary>
    public void LoadTestImages(string directoryPath)
    {
        if (!Directory.Exists(directoryPath))
        {
            throw new DirectoryNotFoundException($"Test images directory not found: {directoryPath}");
        }

        var imageFiles = Directory.GetFiles(directoryPath, "*.*")
            .Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                       f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                       f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase))
            .ToArray();

        _logger.LogInformation("Loading {Count} test images from {Path}", imageFiles.Length, directoryPath);

        foreach (var file in imageFiles)
        {
            try
            {
                using var imageData = SKData.Create(file);
                var bitmap = SKBitmap.Decode(imageData);
                if (bitmap != null)
                {
                    _testImages.Add(bitmap);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load test image: {File}", file);
            }
        }

        _logger.LogInformation("Loaded {Count} test images", _testImages.Count);
    }

    /// <summary>
    /// Run comprehensive benchmark comparing Fast vs Accurate models.
    /// </summary>
    public async Task<BenchmarkResults> RunBenchmarkAsync(int iterationsPerImage = 3)
    {
        if (!_testImages.Any())
        {
            throw new InvalidOperationException("No test images available. Call GenerateTestImages() or LoadTestImages() first.");
        }

        _logger.LogInformation("Starting benchmark with {ImageCount} images, {IterationsPerImage} iterations each",
            _testImages.Count, iterationsPerImage);

        var results = new BenchmarkResults
        {
            Timestamp = DateTime.UtcNow,
            TestImageCount = _testImages.Count,
            IterationsPerImage = iterationsPerImage
        };

        // Benchmark Fast model
        _logger.LogInformation("Benchmarking Fast model...");
        var fastResults = await BenchmarkModelAsync(_fastService, "Fast", iterationsPerImage);
        results.FastModelResults = fastResults;

        // Benchmark Accurate model
        _logger.LogInformation("Benchmarking Accurate model...");
        var accurateResults = await BenchmarkModelAsync(_accurateService, "Accurate", iterationsPerImage);
        results.AccurateModelResults = accurateResults;

        // Calculate comparisons
        CalculateComparisons(results);

        _logger.LogInformation("Benchmark completed successfully");
        return results;
    }

    private async Task<ModelBenchmarkResults> BenchmarkModelAsync(
        TableFormerTableStructureService service,
        string modelName,
        int iterationsPerImage)
    {
        var results = new ModelBenchmarkResults { ModelName = modelName };
        var stopwatch = new Stopwatch();

        foreach (var image in _testImages)
        {
            var imageResults = new List<InferenceResult>();

            for (int i = 0; i < iterationsPerImage; i++)
            {
                var request = CreateBenchmarkRequest(image);

                stopwatch.Restart();

                try
                {
                    var tableResult = await service.InferStructureAsync(request);
                    stopwatch.Stop();

                    imageResults.Add(new InferenceResult
                    {
                        Success = true,
                        InferenceTime = stopwatch.Elapsed,
                        CellsDetected = tableResult.Cells.Count,
                        RowsDetected = tableResult.RowCount,
                        ColumnsDetected = tableResult.ColumnCount
                    });
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    _logger.LogWarning(ex, "Inference failed for {Model} on image {ImageIndex}, iteration {Iteration}",
                        modelName, _testImages.IndexOf(image), i);

                    imageResults.Add(new InferenceResult
                    {
                        Success = false,
                        InferenceTime = stopwatch.Elapsed,
                        Error = ex.Message
                    });
                }
            }

            results.ImageResults.Add(imageResults);
        }

        // Calculate aggregated metrics
        var allResults = results.ImageResults.SelectMany(r => r).ToList();
        var successfulResults = allResults.Where(r => r.Success).ToList();

        results.TotalInferences = allResults.Count;
        results.SuccessfulInferences = successfulResults.Count;
        results.FailedInferences = allResults.Count - successfulResults.Count;
        results.SuccessRate = allResults.Count > 0 ? (double)successfulResults.Count / allResults.Count : 0;

        if (successfulResults.Any())
        {
            results.AverageInferenceTime = TimeSpan.FromTicks((long)successfulResults.Average(r => r.InferenceTime.Ticks));
            results.MinInferenceTime = successfulResults.Min(r => r.InferenceTime);
            results.MaxInferenceTime = successfulResults.Max(r => r.InferenceTime);
            results.TotalInferenceTime = TimeSpan.FromTicks((long)successfulResults.Sum(r => r.InferenceTime.Ticks));

            results.AverageCellsDetected = successfulResults.Average(r => r.CellsDetected);
            results.TotalCellsDetected = successfulResults.Sum(r => r.CellsDetected);
        }

        return results;
    }

    private void CalculateComparisons(BenchmarkResults results)
    {
        if (results.FastModelResults.SuccessfulInferences > 0 && results.AccurateModelResults.SuccessfulInferences > 0)
        {
            results.PerformanceComparison = new PerformanceComparison
            {
                FastIsFaster = results.FastModelResults.AverageInferenceTime < results.AccurateModelResults.AverageInferenceTime,
                SpeedRatio = results.AccurateModelResults.AverageInferenceTime.TotalMilliseconds / results.FastModelResults.AverageInferenceTime.TotalMilliseconds,
                AccuracyComparison = results.AccurateModelResults.AverageCellsDetected / results.FastModelResults.AverageCellsDetected
            };
        }
    }

    private static TableStructureRequest CreateBenchmarkRequest(SKBitmap image)
    {
        using var imageData = image.Encode(SKEncodedImageFormat.Png, 90);
        using var stream = new MemoryStream();
        imageData.SaveTo(stream);

        return new TableStructureRequest(
            Page: new(1),
            BoundingBox: new(0, 0, image.Width, image.Height),
            RasterizedImage: stream.ToArray()
        );
    }

    private static SKBitmap GenerateTestTableImage(int width, int height, Random random)
    {
        var bitmap = new SKBitmap(width, height);

        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);

        var paint = new SKPaint
        {
            Color = SKColors.Black,
            StrokeWidth = 1,
            IsStroke = true
        };

        // Generate random table structure
        var rows = random.Next(2, 8);
        var cols = random.Next(2, 6);

        var cellWidth = (width - 100) / cols;
        var cellHeight = (height - 100) / rows;

        // Draw table grid
        for (int row = 0; row <= rows; row++)
        {
            var y = 50 + (row * cellHeight);
            canvas.DrawLine(50, y, width - 50, y, paint);
        }

        for (int col = 0; col <= cols; col++)
        {
            var x = 50 + (col * cellWidth);
            canvas.DrawLine(x, 50, x, height - 50, paint);
        }

        // Add some random content in cells
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                if (random.NextDouble() < 0.7) // 70% chance of content
                {
                    var textPaint = new SKPaint
                    {
                        Color = SKColors.DarkGray,
                        TextSize = 12,
                        IsAntialias = true
                    };

                    var cellX = 50 + (col * cellWidth) + 5;
                    var cellY = 50 + (row * cellHeight) + 15;
                    canvas.DrawText($"R{row}C{col}", cellX, cellY, textPaint);
                }
            }
        }

        return bitmap;
    }

    private static ILogger<TableFormerBenchmark> CreateDefaultLogger()
    {
        return LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Information);
        }).CreateLogger<TableFormerBenchmark>();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _fastService.Dispose();
            _accurateService.Dispose();

            foreach (var image in _testImages)
            {
                image.Dispose();
            }

            _testImages.Clear();
            _disposed = true;
        }
    }
}

/// <summary>
/// Results of a complete benchmark run.
/// </summary>
public sealed class BenchmarkResults
{
    public DateTime Timestamp { get; init; }
    public int TestImageCount { get; init; }
    public int IterationsPerImage { get; init; }
    public ModelBenchmarkResults FastModelResults { get; init; } = new();
    public ModelBenchmarkResults AccurateModelResults { get; init; } = new();
    public PerformanceComparison? PerformanceComparison { get; init; }
}

/// <summary>
/// Results for a single model benchmark.
/// </summary>
public sealed class ModelBenchmarkResults
{
    public string ModelName { get; init; } = string.Empty;
    public int TotalInferences { get; init; }
    public int SuccessfulInferences { get; init; }
    public int FailedInferences { get; init; }
    public double SuccessRate { get; init; }
    public TimeSpan TotalInferenceTime { get; init; }
    public TimeSpan AverageInferenceTime { get; init; }
    public TimeSpan MinInferenceTime { get; init; }
    public TimeSpan MaxInferenceTime { get; init; }
    public double AverageCellsDetected { get; init; }
    public int TotalCellsDetected { get; init; }
    public List<List<InferenceResult>> ImageResults { get; init; } = new();
}

/// <summary>
/// Result of a single inference operation.
/// </summary>
public sealed class InferenceResult
{
    public bool Success { get; init; }
    public TimeSpan InferenceTime { get; init; }
    public int CellsDetected { get; init; }
    public int RowsDetected { get; init; }
    public int ColumnsDetected { get; init; }
    public string? Error { get; init; }
}

/// <summary>
/// Performance comparison between Fast and Accurate models.
/// </summary>
public sealed class PerformanceComparison
{
    public bool FastIsFaster { get; init; }
    public double SpeedRatio { get; init; } // Accurate time / Fast time
    public double AccuracyComparison { get; init; } // Accurate cells / Fast cells
}
#endif
