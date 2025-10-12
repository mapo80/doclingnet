using System.Diagnostics;
using System.Text.Json;

namespace LayoutTensorCopyBenchmarks;

internal sealed record BenchmarkResult(
    double BaselineMeanMs,
    double OptimizedMeanMs,
    double DeltaMs,
    double Ratio,
    int Iterations,
    int Warmup,
    int Elements,
    int[] Shape);

internal static class Program
{
    private const int DefaultIterations = 100;
    private const int DefaultWarmup = 10;
    private const int DefaultChannels = 3;
    private const int DefaultWidth = 640;
    private const int DefaultHeight = 640;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
    };

    private static double _sink;

    public static void Main(string[] args)
    {
        var iterations = args.Length > 0 && int.TryParse(args[0], out var parsedIterations)
            ? Math.Max(parsedIterations, 1)
            : DefaultIterations;

        var warmup = args.Length > 1 && int.TryParse(args[1], out var parsedWarmup)
            ? Math.Max(parsedWarmup, 0)
            : DefaultWarmup;

        var channels = args.Length > 2 && int.TryParse(args[2], out var parsedChannels)
            ? Math.Max(parsedChannels, 1)
            : DefaultChannels;
        var height = args.Length > 3 && int.TryParse(args[3], out var parsedHeight)
            ? Math.Max(parsedHeight, 1)
            : DefaultHeight;
        var width = args.Length > 4 && int.TryParse(args[4], out var parsedWidth)
            ? Math.Max(parsedWidth, 1)
            : DefaultWidth;

        var elements = checked(channels * height * width);
        var shape = new[] { 1, channels, height, width };

        var buffer = CreateTensor(elements);

        RunWarmup(buffer, warmup);

        var baseline = MeasureBaseline(buffer, iterations);
        var optimized = MeasureOptimized(buffer, iterations);

        var result = new BenchmarkResult(
            BaselineMeanMs: baseline,
            OptimizedMeanMs: optimized,
            DeltaMs: baseline - optimized,
            Ratio: optimized <= double.Epsilon ? double.PositiveInfinity : baseline / optimized,
            Iterations: iterations,
            Warmup: warmup,
            Elements: elements,
            Shape: shape);

        var json = JsonSerializer.Serialize(result, JsonOptions);
        Console.WriteLine(json);
    }

    private static float[] CreateTensor(int elements)
    {
        var buffer = new float[elements];
        for (var i = 0; i < elements; i++)
        {
            buffer[i] = i * 0.001f;
        }

        return buffer;
    }

    private static void RunWarmup(float[] buffer, int warmup)
    {
        for (var i = 0; i < warmup; i++)
        {
            Consume(buffer.ToArray());
            ConsumeSpan(buffer);
        }
    }

    private static double MeasureBaseline(float[] buffer, int iterations)
    {
        var total = 0d;
        for (var i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            var copy = buffer.ToArray();
            sw.Stop();

            Consume(copy);
            total += sw.Elapsed.TotalMilliseconds;
        }

        return total / iterations;
    }

    private static double MeasureOptimized(float[] buffer, int iterations)
    {
        var total = 0d;
        for (var i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            var span = new ReadOnlySpan<float>(buffer);
            sw.Stop();

            ConsumeSpan(span);
            total += sw.Elapsed.TotalMilliseconds;
        }

        return total / iterations;
    }

    private static void Consume(float[] values)
    {
        if (values.Length > 0)
        {
            _sink += values[0];
        }
    }

    private static void ConsumeSpan(ReadOnlySpan<float> values)
    {
        if (!values.IsEmpty)
        {
            _sink += values[0];
        }
    }
}
