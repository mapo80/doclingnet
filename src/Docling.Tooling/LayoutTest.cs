using System;
using System.IO;
using LayoutSdk;
using LayoutSdk.Configuration;
using Serilog;

namespace Docling.Tooling;

public static class LayoutTest
{
    public static void Run(string imagePath)
    {
        Log.Information("Starting layout detection test");
        Log.Information("Image path: {ImagePath}", imagePath);

        if (!File.Exists(imagePath))
        {
            Log.Error("Image file not found: {ImagePath}", imagePath);
            return;
        }

        // Path al modello ONNX
        var modelPath = Path.GetFullPath(
            Path.Combine(
                AppContext.BaseDirectory,
                "..", "..", "..", "..", "..",
                "src", "submodules", "ds4sd-docling-layout-heron-onnx", "models", "heron-converted.onnx"));

        Log.Information("Model path: {ModelPath}", modelPath);

        if (!File.Exists(modelPath))
        {
            Log.Error("Model file not found: {ModelPath}", modelPath);
            return;
        }

        // Crea le opzioni per il layout SDK
        var options = new LayoutSdkOptions(
            onnxModelPath: modelPath,
            defaultLanguage: DocumentLanguage.English,
            validateModelPaths: true);

        try
        {
            options.EnsureModelPaths();
            Log.Information("Model paths validated successfully");
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to validate model paths");
            return;
        }

        // Crea il SDK e processa l'immagine
        using var sdk = new LayoutSdk.LayoutSdk(options);

        Log.Information("Processing image with LayoutSdk...");
        var result = sdk.Process(imagePath, overlay: false, LayoutRuntime.Onnx);

        Log.Information("Layout detection completed!");
        Log.Information("Detected {Count} layout elements", result.Boxes.Count);
        Log.Information("Preprocessing time: {Ms:F2} ms", result.Metrics.PreprocessDuration.TotalMilliseconds);
        Log.Information("Inference time: {Ms:F2} ms", result.Metrics.InferenceDuration.TotalMilliseconds);
        Log.Information("Total time: {Ms:F2} ms", result.Metrics.TotalDuration.TotalMilliseconds);

        // Stampa i dettagli dei box rilevati
        Log.Information("");
        Log.Information("Detected layout elements:");
        foreach (var box in result.Boxes)
        {
            Log.Information("  - {Label}: X={X:F2}, Y={Y:F2}, W={Width:F2}, H={Height:F2}, Confidence={Confidence:F3}",
                box.Label, box.X, box.Y, box.Width, box.Height, box.Confidence);
        }
    }
}
