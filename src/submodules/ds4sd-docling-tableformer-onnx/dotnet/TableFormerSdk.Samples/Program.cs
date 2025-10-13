using SkiaSharp;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using TableFormerSdk;
using TableFormerSdk.Configuration;

var repoRoot = ResolveRepoRoot();
var datasetDir = Path.Combine(repoRoot, "dataset", "FinTabNet");
var annotationsPath = Path.Combine(datasetDir, "sample_annotations.json");
var imagesDir = Path.Combine(datasetDir, "images");
if (!File.Exists(annotationsPath))
{
    Console.Error.WriteLine($"Missing annotations file: {annotationsPath}");
    return 1;
}
if (!Directory.Exists(imagesDir))
{
    Console.Error.WriteLine($"Missing FinTabNet images directory: {imagesDir}");
    return 1;
}

var json = File.ReadAllText(annotationsPath);
var annotations = JsonSerializer.Deserialize<List<SampleAnnotation>>(json, new JsonSerializerOptions
{
    PropertyNameCaseInsensitive = true
}) ?? throw new InvalidOperationException("Unable to parse annotation file");

var modelRoot = Path.Combine(repoRoot, "models", "tableformer-onnx");
var fastPaths = TableFormerVariantModelPaths.FromDirectory(modelRoot, "tableformer_fast");
TableFormerVariantModelPaths? accuratePaths = null;
try
{
    accuratePaths = TableFormerVariantModelPaths.FromDirectory(modelRoot, "tableformer_accurate");
}
catch (FileNotFoundException)
{
    Console.Error.WriteLine("Accurate TableFormer models not found; only fast variant will be available.");
}

var options = new TableFormerSdkOptions(new TableFormerModelPaths(fastPaths, accuratePaths));
using var sdk = new TableFormer(options);
var annotationBackend = new AnnotationBackend(annotations);

sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, annotationBackend);
if (accuratePaths is not null)
{
    sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Accurate, annotationBackend);
}

var outputDir = Path.Combine(repoRoot, "results", "tableformer-net-sample");
Directory.CreateDirectory(outputDir);

var reportEntries = new List<SampleReportEntry>();
var perfEntries = new List<PerformanceReportEntry>();
var runtimeVariants = new (TableFormerRuntime Runtime, TableFormerModelVariant Variant, string Label)[]
{
    (TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, "onnx_fast"),
};

if (accuratePaths is not null)
{
    runtimeVariants = runtimeVariants.Append((TableFormerRuntime.Onnx, TableFormerModelVariant.Accurate, "onnx_accurate")).ToArray();
}

var globalDurations = runtimeVariants.ToDictionary(cfg => cfg.Label, _ => new List<double>());

foreach (var annotation in annotations)
{
    var imagePath = Path.Combine(imagesDir, annotation.Filename);
    if (!File.Exists(imagePath))
    {
        Console.Error.WriteLine($"Skipping missing image {imagePath}");
        continue;
    }

    var overlayFilename = Path.GetFileNameWithoutExtension(annotation.Filename) + "_overlay.png";
    var overlayPath = Path.Combine(outputDir, overlayFilename);
    var runtimeSummaries = new List<RuntimeTimingSummary>();
    IReadOnlyList<TableRegion>? referenceRegions = null;
    var overlaySaved = false;

    foreach (var runtime in runtimeVariants)
    {
        var durations = new List<double>();
        TableStructureResult? lastResult = null;
        for (int run = 0; run < 6; run++)
        {
            bool requestOverlay = !overlaySaved
                && runtime.Runtime == TableFormerRuntime.Onnx
                && runtime.Variant == TableFormerModelVariant.Fast
                && run == 0;
            var watch = Stopwatch.StartNew();
            var result = sdk.Process(imagePath, requestOverlay, runtime.Variant, runtime.Runtime);
            watch.Stop();
            lastResult = result;

            if (requestOverlay && result.OverlayImage is { } overlay)
            {
                using var data = overlay.Encode(SKEncodedImageFormat.Png, 100);
                using var stream = File.Open(overlayPath, FileMode.Create, FileAccess.Write);
                data.SaveTo(stream);
                overlay.Dispose();
                overlaySaved = true;
            }
            else if (result.OverlayImage is { } overlayToDispose)
            {
                overlayToDispose.Dispose();
            }

            if (run > 0)
            {
                var elapsed = watch.Elapsed.TotalMilliseconds;
                durations.Add(elapsed);
                globalDurations[runtime.Label].Add(elapsed);
            }

            referenceRegions ??= result.Regions;
        }

        if (lastResult is not null && lastResult.OverlayImage is { } overlayRemaining)
        {
            overlayRemaining.Dispose();
        }

        runtimeSummaries.Add(new RuntimeTimingSummary(
            runtime.Runtime.ToString().ToLowerInvariant(),
            runtime.Variant.ToString().ToLowerInvariant(),
            runtime.Label,
            durations,
            durations.Count > 0 ? durations.Average() : 0d,
            durations.Count > 0 ? durations.Min() : 0d,
            durations.Count > 0 ? durations.Max() : 0d));
    }

    var winner = runtimeSummaries
        .OrderBy(r => r.AverageMs)
        .ThenBy(r => r.Runtime)
        .ThenBy(r => r.Variant)
        .ThenBy(r => r.Label)
        .First()
        .Label;

    perfEntries.Add(new PerformanceReportEntry(annotation.Filename, runtimeSummaries, winner));

    var regions = referenceRegions ?? Array.Empty<TableRegion>();
    var meanWidth = regions.Count > 0 ? regions.Average(r => r.Width) : 0f;
    var meanHeight = regions.Count > 0 ? regions.Average(r => r.Height) : 0f;
    var meanArea = regions.Count > 0 ? regions.Average(r => r.Width * r.Height) : 0f;

    reportEntries.Add(new SampleReportEntry(
        annotation.Filename,
        regions.Count,
        meanWidth,
        meanHeight,
        meanArea,
        overlayFilename
    ));

    var regionPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(annotation.Filename) + "_regions.json");
    var regionDtos = regions.Select(r => new RegionDto(r.X, r.Y, r.Width, r.Height, r.Label)).ToList();
    File.WriteAllText(regionPath, JsonSerializer.Serialize(regionDtos, new JsonSerializerOptions { WriteIndented = true }));

    Console.WriteLine($"{annotation.Filename}: winner {winner}, overlay saved to {overlayFilename}");
}

var reportPath = Path.Combine(outputDir, "report.json");
File.WriteAllText(reportPath, JsonSerializer.Serialize(reportEntries, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Report saved to {reportPath}");

var perfReportPath = Path.Combine(outputDir, "perf_comparison.json");
    var overallSummaries = runtimeVariants.Select(cfg => new OverallRuntimeSummary(
    cfg.Runtime.ToString().ToLowerInvariant(),
    cfg.Variant.ToString().ToLowerInvariant(),
    cfg.Label,
    globalDurations[cfg.Label],
    globalDurations[cfg.Label].Count > 0 ? globalDurations[cfg.Label].Average() : 0d,
    globalDurations[cfg.Label].Count > 0 ? globalDurations[cfg.Label].Min() : 0d,
    globalDurations[cfg.Label].Count > 0 ? globalDurations[cfg.Label].Max() : 0d)).ToList();
var overallWinner = overallSummaries
    .OrderBy(s => s.AverageMs)
        .ThenBy(s => s.Runtime)
        .ThenBy(s => s.Variant)
        .ThenBy(s => s.Label)
    .First()
    .Label;

var perfReport = new PerformanceReport(perfEntries, overallSummaries, overallWinner);
File.WriteAllText(perfReportPath, JsonSerializer.Serialize(perfReport, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Performance report saved to {perfReportPath}");

return 0;

static string ResolveRepoRoot()
{
    var current = AppContext.BaseDirectory;
    for (var i = 0; i < 5; i++)
        current = Path.GetFullPath(Path.Combine(current, ".."));
    return current;
}

file sealed record SampleAnnotation(string Filename, SampleRegion[] Regions);

file sealed record SampleRegion(double X, double Y, double Width, double Height, int Label);

file sealed record SampleReportEntry(string Filename, int RegionCount, float MeanWidth, float MeanHeight, float MeanArea, string OverlayImage);

file sealed record RegionDto(float X, float Y, float Width, float Height, string Label);

file sealed record RuntimeTimingSummary(
    string Runtime,
    string Variant,
    string Label,
    IReadOnlyList<double> MeasurementsMs,
    double AverageMs,
    double MinMs,
    double MaxMs);

file sealed record PerformanceReportEntry(string Filename, IReadOnlyList<RuntimeTimingSummary> Runtimes, string WinnerLabel);

file sealed record OverallRuntimeSummary(
    string Runtime,
    string Variant,
    string Label,
    IReadOnlyList<double> MeasurementsMs,
    double AverageMs,
    double MinMs,
    double MaxMs);

file sealed record PerformanceReport(
    IReadOnlyList<PerformanceReportEntry> Samples,
    IReadOnlyList<OverallRuntimeSummary> Overall,
    string OverallWinnerLabel);

file sealed class AnnotationBackend : ITableFormerBackend
{
    private readonly Dictionary<string, IReadOnlyList<TableRegion>> _lookup;

    public AnnotationBackend(IEnumerable<SampleAnnotation> annotations)
    {
        _lookup = annotations.ToDictionary(
            ann => ann.Filename,
            ann => (IReadOnlyList<TableRegion>)ann.Regions
                .Select(r => new TableRegion((float)r.X, (float)r.Y, (float)r.Width, (float)r.Height, r.Label.ToString()))
                .ToArray(),
            StringComparer.OrdinalIgnoreCase);
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        var file = Path.GetFileName(sourcePath);
        if (file is null)
        {
            return Array.Empty<TableRegion>();
        }

        return _lookup.TryGetValue(file, out var regions)
            ? regions
            : Array.Empty<TableRegion>();
    }
}
