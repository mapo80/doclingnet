using System;
using System.IO;
using LayoutSdk;
using LayoutSdk.Configuration;
using SkiaSharp;

Console.WriteLine("Docling Layout SDK (NuGet package demo)");
Console.WriteLine($"Models packaged under: {LayoutSdkBundledModels.ModelsRoot}");

LayoutSdkBundledModels.EnsureAllFilesExist();
Console.WriteLine("All bundled model files were found.");

var options = LayoutSdkBundledModels.CreateOptions(validateModelPaths: true);
options.EnsureModelPaths();

var imagePath = ResolveDatasetImage("gazette_de_france.jpg");
if (imagePath is null)
{
    Console.WriteLine("Sample image not found in the repo. A blank 640x640 page will be generated.");
}
else
{
    Console.WriteLine($"Using sample image: {imagePath}");
}

var preparedImagePath = PrepareInputImage(imagePath);
Console.WriteLine($"Running inference with input: {preparedImagePath}");

using var sdk = new LayoutSdk.LayoutSdk(options);

Console.WriteLine();
Console.WriteLine("=== ONNX Runtime ===");
Console.WriteLine($"Model source: {LayoutSdkBundledModels.GetOptimizedOnnxPath()}");

var result = sdk.Process(preparedImagePath, overlay: false, LayoutRuntime.Onnx);

Console.WriteLine($"Detected {result.Boxes.Count} layout elements.");
Console.WriteLine($"Preprocess ms: {result.Metrics.PreprocessDuration.TotalMilliseconds:F2}");
Console.WriteLine($"Inference ms: {result.Metrics.InferenceDuration.TotalMilliseconds:F2}");
Console.WriteLine($"Total ms: {result.Metrics.TotalDuration.TotalMilliseconds:F2}");

static string PrepareInputImage(string? sourceImage)
{
    var tempPath = Path.Combine(Path.GetTempPath(), "docling-layout-sample.png");
    if (sourceImage is null)
    {
        using var blank = new SKBitmap(new SKImageInfo(640, 640));
        using (var canvas = new SKCanvas(blank))
        {
            canvas.Clear(SKColors.White);
        }

        using var blankImage = SKImage.FromBitmap(blank);
        using var blankData = blankImage.Encode(SKEncodedImageFormat.Png, 90);
        using var blankStream = File.Open(tempPath, FileMode.Create, FileAccess.Write);
        blankData.SaveTo(blankStream);
        return tempPath;
    }

    using var bitmap = SKBitmap.Decode(sourceImage)
                      ?? throw new InvalidOperationException($"Unable to decode image at {sourceImage}");
    var sampling = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear);
    using var resized = bitmap.Resize(new SKImageInfo(640, 640), sampling)
                      ?? throw new InvalidOperationException("Failed to resize sample image to 640x640.");
    using var image = SKImage.FromBitmap(resized);
    using var data = image.Encode(SKEncodedImageFormat.Png, 95);
    using var stream = File.Open(tempPath, FileMode.Create, FileAccess.Write);
    data.SaveTo(stream);
    return tempPath;
}

static string? ResolveDatasetImage(string fileName)
{
    var current = AppContext.BaseDirectory;
    for (var i = 0; i < 8; i++)
    {
        var candidate = Path.Combine(current, "dataset", fileName);
        if (File.Exists(candidate))
        {
            return candidate;
        }

        current = Path.GetFullPath(Path.Combine(current, ".."));
    }

    return null;
}
