using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Collections.Generic;
using System.Linq;
using LayoutSdk;
using LayoutSdk.Configuration;
using SkiaSharp;

var parameters = ParseArgs(args);

if (!parameters.TryGetValue("--variant-name", out var variant))
{
    Console.Error.WriteLine("--variant-name is required");
    return;
}

var compare = parameters.ContainsKey("--compare");
if (compare)
{
    Console.Error.WriteLine("--compare is not supported because the ORT runtime is unavailable in this build.");
    return;
}

var runtime = LayoutRuntime.Onnx;
if (parameters.TryGetValue("--runtime", out var runtimeValue))
{
    runtime = Enum.Parse<LayoutRuntime>(runtimeValue, ignoreCase: true);
    if (runtime != LayoutRuntime.Onnx)
    {
        Console.Error.WriteLine($"Runtime '{runtime}' is not supported. Use 'onnx'.");
        return;
    }
}

var imagesDir = parameters.GetValueOrDefault("--images", "./dataset");
var outputRoot = parameters.GetValueOrDefault("--output", "results");
int warmup = int.Parse(parameters.GetValueOrDefault("--warmup", "1"), CultureInfo.InvariantCulture);
int runsPerImage = int.Parse(parameters.GetValueOrDefault("--runs-per-image", "1"), CultureInfo.InvariantCulture);
int targetH = int.Parse(parameters.GetValueOrDefault("--target-h", "640"), CultureInfo.InvariantCulture);
int targetW = int.Parse(parameters.GetValueOrDefault("--target-w", "640"), CultureInfo.InvariantCulture);

var images = CollectImages(imagesDir).ToList();
if (images.Count == 0)
{
    using var bmp = new SKBitmap(targetW, targetH);
    using var img = SKImage.FromBitmap(bmp);
    var tmp = Path.GetTempFileName() + ".png";
    using var data = img.Encode(SKEncodedImageFormat.Png, 90);
    using var fs = File.OpenWrite(tmp);
    data.SaveTo(fs);
    images.Add(tmp);
}

var onnxModelPath = parameters.GetValueOrDefault("--onnx-model", "models/heron-optimized.onnx");

var options = new LayoutSdkOptions(
    onnxModelPath,
    defaultLanguage: DocumentLanguage.English,
    validateModelPaths: parameters.ContainsKey("--validate-models"));

var timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
var runDirectory = Path.Combine(outputRoot, variant, $"run-{timestamp}");
Directory.CreateDirectory(runDirectory);

RunBenchmark(runtime, options, images, runDirectory, warmup, runsPerImage, targetH, targetW);

Console.WriteLine($"OK: {runDirectory}");

static Dictionary<string, string> ParseArgs(string[] args)
{
    var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    for (int i = 0; i < args.Length; i++)
    {
        var token = args[i];
        if (!token.StartsWith("--", StringComparison.Ordinal))
        {
            continue;
        }

        var key = token;
        string value = (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            ? args[++i]
            : "true";
        dict[key] = value;
    }

    return dict;
}

static string ResizeToTemp(string path, int w, int h)
{
    using var bmp = SKBitmap.Decode(path) ?? throw new InvalidOperationException($"Unable to decode image: {path}");
    using var resized = bmp.Resize(new SKImageInfo(w, h), BenchmarkDefaults.HighQualitySampling);
    var tmp = Path.GetTempFileName() + ".png";
    var source = resized ?? bmp;
    using var img = SKImage.FromBitmap(source);
    using var data = img.Encode(SKEncodedImageFormat.Png, 90);
    using var fs = File.OpenWrite(tmp);
    data.SaveTo(fs);
    return tmp;
}

static IReadOnlyList<string> CollectImages(string directory)
{
    if (!Directory.Exists(directory))
    {
        return Array.Empty<string>();
    }

    return Directory.GetFiles(directory)
        .Where(f => f.EndsWith(".jpg", true, CultureInfo.InvariantCulture)
                  || f.EndsWith(".jpeg", true, CultureInfo.InvariantCulture)
                  || f.EndsWith(".png", true, CultureInfo.InvariantCulture)
                  || f.EndsWith(".bmp", true, CultureInfo.InvariantCulture)
                  || f.EndsWith(".tif", true, CultureInfo.InvariantCulture)
                  || f.EndsWith(".tiff", true, CultureInfo.InvariantCulture))
        .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
        .ToList();
}

static double Percentile(List<double> seq, double percentile)
{
    if (seq.Count == 0)
    {
        return double.NaN;
    }

    var ordered = seq.OrderBy(x => x).ToList();
    var index = (int)Math.Ceiling(percentile / 100.0 * ordered.Count) - 1;
    index = Math.Clamp(index, 0, ordered.Count - 1);
    return ordered[index];
}

static string Sha256Of(string path)
{
    using var sha = SHA256.Create();
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(sha.ComputeHash(stream));
}

static BenchmarkArtifacts RunBenchmark(
    LayoutRuntime runtime,
    LayoutSdkOptions options,
    IReadOnlyList<string> imageFiles,
    string outputDir,
    int warmup,
    int runsPerImage,
    int targetH,
    int targetW)
{
    Directory.CreateDirectory(outputDir);

    using var sdk = new LayoutSdk.LayoutSdk(options);

    var timings = new List<double>();

    var warmSource = imageFiles[0];
    var warmResized = ResizeToTemp(warmSource, targetW, targetH);
    for (int i = 0; i < warmup; i++)
    {
        sdk.Process(warmResized, overlay: false, runtime);
    }

    using (var csv = new StreamWriter(Path.Combine(outputDir, "timings.csv")))
    {
        csv.WriteLine("filename,ms");
        foreach (var file in imageFiles)
        {
            var prepPath = ResizeToTemp(file, targetW, targetH);
            for (int run = 0; run < runsPerImage; run++)
            {
                var result = sdk.Process(prepPath, overlay: false, runtime);
                var ms = result.Metrics.TotalDuration.TotalMilliseconds;
                csv.WriteLine($"{Path.GetFileName(file)},{ms:F3}");
                timings.Add(ms);
            }
        }
    }

    var summary = new BenchmarkSummary(
        Count: timings.Count,
        MeanMs: timings.Count > 0 ? timings.Average() : double.NaN,
        MedianMs: Percentile(timings, 50),
        P95Ms: Percentile(timings, 95));

    File.WriteAllText(
        Path.Combine(outputDir, "summary.json"),
        JsonSerializer.Serialize(summary, BenchmarkDefaults.PrettyJson));

    object modelInfo = runtime switch
    {
        LayoutRuntime.Onnx => new
        {
            runtime = "onnx",
            model_path = options.OnnxModelPath,
            model_size_bytes = File.Exists(options.OnnxModelPath) ? new FileInfo(options.OnnxModelPath).Length : 0L,
            device = "CPU",
            precision = options.OnnxModelPath.Contains("fp16", StringComparison.OrdinalIgnoreCase) ? "fp16" : "fp32"
        },
        _ => throw new NotSupportedException($"Runtime {runtime} is not supported. Only ONNX runtime metadata is available.")
    };

    File.WriteAllText(
        Path.Combine(outputDir, "model_info.json"),
        JsonSerializer.Serialize(modelInfo, BenchmarkDefaults.PrettyJson));

    var env = new
    {
        dotnet = Environment.Version.ToString(),
        os = Environment.OSVersion.ToString()
    };
    File.WriteAllText(Path.Combine(outputDir, "env.json"), JsonSerializer.Serialize(env, BenchmarkDefaults.PrettyJson));

    File.WriteAllText(
        Path.Combine(outputDir, "config.json"),
        JsonSerializer.Serialize(new
        {
            runtime = runtime.ToString(),
            warmup,
            runs_per_image = runsPerImage,
            target_h = targetH,
            target_w = targetW
        }, BenchmarkDefaults.PrettyJson));

    var files = BenchmarkDefaults.ManifestFiles
        .Select(f => new { file = f, sha256 = Sha256Of(Path.Combine(outputDir, f)) })
        .ToList();

    File.WriteAllText(
        Path.Combine(outputDir, "manifest.json"),
        JsonSerializer.Serialize(new { files }, BenchmarkDefaults.PrettyJson));

    File.WriteAllText(Path.Combine(outputDir, "logs.txt"), $"RUN {runtime} ok, N={timings.Count}\n");
    return new BenchmarkArtifacts(runtime, outputDir, summary);
}

internal static class BenchmarkDefaults
{
    public static readonly JsonSerializerOptions PrettyJson = new() { WriteIndented = true };
    public static readonly SKSamplingOptions HighQualitySampling = new(SKFilterMode.Linear, SKMipmapMode.Linear);
    public static readonly IReadOnlyList<string> ManifestFiles = new[]
    {
        "timings.csv",
        "summary.json",
        "model_info.json",
        "env.json",
        "config.json"
    };
}

internal sealed record BenchmarkSummary(int Count, double MeanMs, double MedianMs, double P95Ms);

internal sealed record BenchmarkArtifacts(LayoutRuntime Runtime, string OutputDirectory, BenchmarkSummary Summary);
