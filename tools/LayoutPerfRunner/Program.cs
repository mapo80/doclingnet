using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace LayoutPerfRunner;

internal sealed record LayoutPerfRunTelemetry(double TotalMilliseconds, LayoutSdkProfilingSnapshot? Profiling);

internal sealed record LayoutPerfSampleResult(
    string Path,
    int Width,
    int Height,
    double Dpi,
    int Runs,
    int Discard,
    bool AdvancedNonMaxSuppression,
    double TotalMeanMs,
    double PreprocessMeanMs,
    double InferenceMeanMs,
    double PostprocessMeanMs,
    IReadOnlyList<double> TotalsMs,
    IReadOnlyList<LayoutPerfRunTelemetry> RunTelemetry);

internal sealed record LayoutPerfOutput(
    string Runtime,
    int Runs,
    int Discard,
    bool AdvancedNonMaxSuppression,
    IReadOnlyList<LayoutPerfSampleResult> Samples);

internal sealed class RunnerOptions
{
    public List<string> Inputs { get; } = new();

    public int Runs { get; set; } = 6;

    public int Discard { get; set; } = 1;

    public bool? AdvancedNonMaxSuppression { get; set; }

    public string? OutputPath { get; set; }
}

internal static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
    };

    public static async Task<int> Main(string[] args)
    {
        if (!TryParseArgs(args, out var options, out var error))
        {
            await Console.Error.WriteLineAsync(error ?? "Invalid arguments supplied.").ConfigureAwait(false);
            PrintUsage();
            return 1;
        }

        if (options.Inputs.Count == 0)
        {
            PrintUsage();
            return 1;
        }

        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder
                .AddConsole()
                .SetMinimumLevel(LogLevel.Information);
        });

        var detectionOptions = new LayoutSdkDetectionOptions
        {
            ValidateModelFiles = true,
            MaxDegreeOfParallelism = 1,
            EnableAdvancedNonMaxSuppression = options.AdvancedNonMaxSuppression ?? true,
            EnableProfiling = true,
        };

        using var service = new LayoutSdkDetectionService(
            detectionOptions,
            loggerFactory.CreateLogger<LayoutSdkDetectionService>());

        var profilingSource = service as ILayoutProfilingTelemetrySource;
        var results = new List<LayoutPerfSampleResult>(options.Inputs.Count);
        var pageIndex = 0;

        foreach (var input in options.Inputs)
        {
            if (!File.Exists(input))
            {
                await Console.Error.WriteLineAsync($"Input file '{input}' was not found.").ConfigureAwait(false);
                continue;
            }

            pageIndex++;
            var payload = CreatePayload(input, pageIndex);
        var request = new LayoutRequest(
            documentId: Path.GetFileNameWithoutExtension(input) ?? $"document_{pageIndex:0000}",
            modelIdentifier: "docling-layout-heron",
            options: new LayoutRequestOptions(CreateOrphanClusters: true, KeepEmptyClusters: false, SkipCellAssignment: false),
            pages: new[] { payload });

            var runs = new List<LayoutPerfRunTelemetry>(options.Runs);
            for (var iteration = 0; iteration < options.Runs; iteration++)
            {
                var stopwatch = Stopwatch.StartNew();
                try
                {
                    _ = await service.DetectAsync(request, CancellationToken.None).ConfigureAwait(false);
                }
                finally
                {
                    stopwatch.Stop();
                }

                LayoutSdkProfilingSnapshot? snapshot = null;
                if (profilingSource is not null)
                {
                    var telemetry = profilingSource.ConsumeProfilingTelemetry();
                    if (telemetry.Count > 0)
                    {
                        snapshot = telemetry[^1].Snapshot;
                    }
                }

                runs.Add(new LayoutPerfRunTelemetry(stopwatch.Elapsed.TotalMilliseconds, snapshot));
            }

            var totals = TrimWarmup(runs.Select(run => run.TotalMilliseconds), options.Discard).ToList();
            var profiling = TrimWarmup(runs.Select(run => run.Profiling), options.Discard)
                .Where(p => p.HasValue)
                .Select(p => p!.Value)
                .ToList();

            var sample = new LayoutPerfSampleResult(
                Path.GetFullPath(input),
                payload.Width,
                payload.Height,
                payload.Dpi,
                options.Runs,
                options.Discard,
                options.AdvancedNonMaxSuppression ?? true,
                totals.Count > 0 ? totals.Average() : 0d,
                profiling.Count > 0 ? profiling.Average(p => p.PersistMilliseconds) : 0d,
                profiling.Count > 0 ? profiling.Average(p => p.InferenceMilliseconds) : 0d,
                profiling.Count > 0 ? profiling.Average(p => p.PostprocessMilliseconds) : 0d,
                totals,
                runs);

            results.Add(sample);
        }

        var output = new LayoutPerfOutput(
            Runtime: detectionOptions.Runtime.ToString(),
            Runs: options.Runs,
            Discard: options.Discard,
            AdvancedNonMaxSuppression: options.AdvancedNonMaxSuppression ?? true,
            Samples: results);

        var payloadJson = JsonSerializer.Serialize(output, JsonOptions);
        if (!string.IsNullOrWhiteSpace(options.OutputPath))
        {
            await File.WriteAllTextAsync(options.OutputPath, payloadJson).ConfigureAwait(false);
        }

        Console.WriteLine(payloadJson);
        return 0;
    }

    private static LayoutPagePayload CreatePayload(string path, int pageNumber)
    {
        var bytes = File.ReadAllBytes(path);
        using var stream = new SKMemoryStream(bytes);
        using var codec = SKCodec.Create(stream);
        ArgumentNullException.ThrowIfNull(codec);
        var width = codec.Info.Width;
        var height = codec.Info.Height;
        var dpi = 300d;
        var artifactId = Path.GetFileNameWithoutExtension(path) ?? $"page_{pageNumber:0000}";
        var metadata = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["source_path"] = Path.GetFullPath(path),
            ["page_number"] = pageNumber.ToString(CultureInfo.InvariantCulture),
        };

        return new LayoutPagePayload(
            new PageReference(pageNumber, dpi),
            artifactId,
            DetectMediaType(path),
            dpi,
            width,
            height,
            metadata,
            bytes);
    }

    private static string DetectMediaType(string path)
    {
        return Path.GetExtension(path).ToUpperInvariant() switch
        {
            ".JPG" or ".JPEG" => "image/jpeg",
            ".BMP" => "image/bmp",
            ".GIF" => "image/gif",
            ".WEBP" => "image/webp",
            ".TIF" or ".TIFF" => "image/tiff",
            _ => "image/png",
        };
    }

    private static bool TryParseArgs(string[] args, out RunnerOptions options, out string? error)
    {
        options = new RunnerOptions();
        error = null;

        for (var i = 0; i < args.Length; i++)
        {
            var token = args[i];
            switch (token)
            {
                case "--input":
                    if (!TryReadValue(args, ref i, out var inputPath))
                    {
                        error = "--input requires a path.";
                        return false;
                    }

                    options.Inputs.Add(inputPath);
                    break;

                case "--runs":
                    if (!TryReadValue(args, ref i, out var runsRaw) ||
                        !int.TryParse(runsRaw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var runs) ||
                        runs <= 0)
                    {
                        error = "--runs expects a positive integer.";
                        return false;
                    }

                    options.Runs = runs;
                    break;

                case "--discard":
                    if (!TryReadValue(args, ref i, out var discardRaw) ||
                        !int.TryParse(discardRaw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var discard) ||
                        discard < 0)
                    {
                        error = "--discard expects a non-negative integer.";
                        return false;
                    }

                    options.Discard = discard;
                    break;

                case "--output":
                    if (!TryReadValue(args, ref i, out var outputPath))
                    {
                        error = "--output requires a path.";
                        return false;
                    }

                    options.OutputPath = outputPath;
                    break;

                case "--advanced-nms":
                    if (!TryReadValue(args, ref i, out var nmsRaw) ||
                        !bool.TryParse(nmsRaw, out var nmsValue))
                    {
                        error = "--advanced-nms expects 'true' or 'false'.";
                        return false;
                    }

                    options.AdvancedNonMaxSuppression = nmsValue;
                    break;

                case "--enable-advanced-nms":
                    options.AdvancedNonMaxSuppression = true;
                    break;

                case "--disable-advanced-nms":
                    options.AdvancedNonMaxSuppression = false;
                    break;

                default:
                    error = $"Unrecognised argument '{token}'.";
                    return false;
            }
        }

        if (options.Discard >= options.Runs)
        {
            error = "--discard must be less than --runs.";
            return false;
        }

        return true;
    }

    private static IEnumerable<T> TrimWarmup<T>(IEnumerable<T> source, int discard)
    {
        return source.Skip(discard);
    }

    private static bool TryReadValue(string[] args, ref int index, out string value)
    {
        if (index + 1 >= args.Length)
        {
            value = string.Empty;
            return false;
        }

        value = args[++index];
        return true;
    }

    private static void PrintUsage()
    {
#pragma warning disable CA1303
        Console.WriteLine("Usage: dotnet run --project tools/LayoutPerfRunner -- --input <path> [--runs <n>] [--discard <n>] [--output <file>] [--advanced-nms <true|false>] [--enable-advanced-nms] [--disable-advanced-nms]");
#pragma warning restore CA1303
    }
}
