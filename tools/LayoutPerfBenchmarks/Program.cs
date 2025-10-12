using System.Diagnostics;
using System.Text.Json;
using SkiaSharp;

namespace LayoutPerfBenchmarks;

internal sealed record BenchmarkResult(
    double BaselineMeanMs,
    double OptimizedMeanMs,
    double DeltaMs,
    double Ratio,
    int Iterations,
    int Warmup,
    string ImagePath,
    int Width,
    int Height);

internal static class Program
{
    private const int DefaultIterations = 50;
    private const int DefaultWarmup = 5;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
    };

    public static async Task Main(string[] args)
    {
        if (args.Length == 0)
        {
            await Console.Error.WriteLineAsync(
                "Usage: dotnet run --project tools/LayoutPerfBenchmarks -- <imagePath> [iterations]")
                .ConfigureAwait(false);
            Environment.Exit(1);
            return;
        }

        var imagePath = args[0];
        if (!File.Exists(imagePath))
        {
            await Console.Error.WriteLineAsync($"Image '{imagePath}' not found.")
                .ConfigureAwait(false);
            Environment.Exit(2);
            return;
        }

        var iterations = args.Length > 1 && int.TryParse(args[1], out var parsedIterations)
            ? Math.Max(parsedIterations, 1)
            : DefaultIterations;
        var warmup = DefaultWarmup;

        var payload = await File.ReadAllBytesAsync(imagePath).ConfigureAwait(false);
        using var stream = new SKMemoryStream(payload);
        using var codec = SKCodec.Create(stream);
        ArgumentNullException.ThrowIfNull(codec);

        var width = codec.Info.Width;
        var height = codec.Info.Height;

        RunWarmup(payload, warmup);

        var baseline = Measure(payload, width, height, iterations, runOptimized: false);
        var optimized = Measure(payload, width, height, iterations, runOptimized: true);

        var result = new BenchmarkResult(
            BaselineMeanMs: baseline,
            OptimizedMeanMs: optimized,
            DeltaMs: baseline - optimized,
            Ratio: optimized <= double.Epsilon ? double.PositiveInfinity : baseline / optimized,
            Iterations: iterations,
            Warmup: warmup,
            ImagePath: Path.GetFullPath(imagePath),
            Width: width,
            Height: height);

        var json = JsonSerializer.Serialize(result, JsonOptions);

        Console.WriteLine(json);
    }

    private static void RunWarmup(ReadOnlyMemory<byte> payload, int warmup)
    {
        for (var i = 0; i < warmup; i++)
        {
            using var data = SKData.CreateCopy(payload.Span);
            using var bitmap = SKBitmap.Decode(data);
            if (bitmap is null)
            {
                throw new InvalidOperationException("Warmup decode failed.");
            }
        }
    }

    private static double Measure(
        ReadOnlyMemory<byte> payload,
        int width,
        int height,
        int iterations,
        bool runOptimized)
    {
        var total = 0d;
        for (var i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            var path = Path.Combine(Path.GetTempPath(), $"layout-perf-{Guid.NewGuid():N}.png");
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 4096, FileOptions.DeleteOnClose | FileOptions.Asynchronous | FileOptions.SequentialScan))
            {
                stream.Write(payload.Span);
                stream.Flush();
            }

            if (!runOptimized)
            {
                using var data = SKData.CreateCopy(payload.Span);
                using var bitmap = SKBitmap.Decode(data) ??
                    throw new InvalidOperationException("Failed to decode image during baseline measurement.");

                _ = bitmap.Width;
                _ = bitmap.Height;
            }
            else
            {
                // Optimized path reuses metadata supplied by the pipeline.
                _ = width;
                _ = height;
            }

            sw.Stop();
            total += sw.Elapsed.TotalMilliseconds;

            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch
            {
                // ignored: the file is created in the temp directory with delete-on-close.
            }
        }

        return total / iterations;
    }
}
