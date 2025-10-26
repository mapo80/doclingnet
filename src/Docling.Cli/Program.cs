using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using LayoutSdk;
using LayoutSdk.Configuration;
using Serilog;
using SkiaSharp;
using TableFormerTorchSharpSdk.Artifacts;

namespace Docling.Cli;

internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.Console(formatProvider: CultureInfo.InvariantCulture)
            .CreateLogger();

        try
        {
            if (args.Length == 0)
            {
                Log.Error("Usage: Docling.Cli <image-path>");
                return 1;
            }

            var imagePath = args[0];
            Log.Information("Docling Layout Detection CLI");
            Log.Information("Image path: {ImagePath}", imagePath);

            if (!File.Exists(imagePath))
            {
                Log.Error("Image file not found: {ImagePath}", imagePath);
                return 1;
            }

            // Path al modello ONNX
            var modelPath = FindModelPath();
            if (modelPath == null)
            {
                Log.Error("Model file not found. Expected location: src/submodules/ds4sd-docling-layout-heron-onnx/models/heron-converted.onnx");
                return 1;
            }

            Log.Information("Model path: {ModelPath}", modelPath);

            // Crea le opzioni per il layout SDK
            var options = new LayoutSdkOptions(
                onnxModelPath: modelPath,
                defaultLanguage: DocumentLanguage.English,
                validateModelPaths: true);

            options.EnsureModelPaths();
            Log.Information("Model paths validated successfully");

            // Crea il SDK e processa l'immagine
            using var sdk = new LayoutSdk.LayoutSdk(options);

            Log.Information("Processing image with LayoutSdk...");
            var result = sdk.Process(imagePath, overlay: false, LayoutRuntime.Onnx);

            Log.Information("");
            Log.Information("=== LAYOUT DETECTION RESULTS ===");
            Log.Information("Detected {Count} layout elements", result.Boxes.Count);
            Log.Information("Preprocessing time: {Ms:F2} ms", result.Metrics.PreprocessDuration.TotalMilliseconds);
            Log.Information("Inference time: {Ms:F2} ms", result.Metrics.InferenceDuration.TotalMilliseconds);
            Log.Information("Total time: {Ms:F2} ms", result.Metrics.TotalDuration.TotalMilliseconds);

            // Stampa i dettagli dei box rilevati
            Log.Information("");
            Log.Information("Detected layout elements:");
            foreach (var box in result.Boxes)
            {
                Log.Information("  - {Label,-15} X={X,7:F2}, Y={Y,7:F2}, W={Width,7:F2}, H={Height,7:F2}, Conf={Confidence:F3}",
                    box.Label, box.X, box.Y, box.Width, box.Height, box.Confidence);
            }

            // Estrai le tabelle rilevate
            var tables = result.Boxes.Where(b => b.Label.Equals("Table", StringComparison.OrdinalIgnoreCase)).ToList();
            if (tables.Count > 0)
            {
                Log.Information("");
                Log.Information("=== TABLE EXTRACTION ===");
                Log.Information("Found {Count} table(s)", tables.Count);

                // Carica l'immagine originale per estrarre le tabelle
                using var originalImage = SKBitmap.Decode(imagePath);
                if (originalImage == null)
                {
                    Log.Error("Failed to load original image for table extraction");
                    return 1;
                }

                // Crea una directory per salvare le tabelle estratte
                var outputDir = Path.Combine(Path.GetDirectoryName(imagePath) ?? ".", "extracted_tables");
                Directory.CreateDirectory(outputDir);

                for (int i = 0; i < tables.Count; i++)
                {
                    var table = tables[i];
                    var tableFileName = Path.GetFileNameWithoutExtension(imagePath) + $"_table_{i + 1}.png";
                    var tablePath = Path.Combine(outputDir, tableFileName);

                    ExtractAndSaveRegion(originalImage, table, tablePath);
                    Log.Information("Extracted table {Index}/{Total} to: {Path}", i + 1, tables.Count, tablePath);
                }

                // Processa le tabelle estratte con TableFormer
                Log.Information("");
                Log.Information("=== TABLE STRUCTURE EXTRACTION (TableFormer) ===");
                await ProcessTablesWithTableFormer(outputDir).ConfigureAwait(false);
            }
            else
            {
                Log.Information("");
                Log.Information("No tables found in the image");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error processing image");
            return 1;
        }
        finally
        {
            await Log.CloseAndFlushAsync().ConfigureAwait(false);
        }
    }

    private static void ExtractAndSaveRegion(SKBitmap sourceImage, BoundingBox region, string outputPath)
    {
        // Calcola le coordinate per il crop
        var x = (int)Math.Max(0, region.X);
        var y = (int)Math.Max(0, region.Y);
        var width = (int)Math.Min(region.Width, sourceImage.Width - x);
        var height = (int)Math.Min(region.Height, sourceImage.Height - y);

        // Crea un subset dell'immagine
        var cropRect = new SKRectI(x, y, x + width, y + height);
        using var croppedImage = new SKBitmap(width, height);

        if (!sourceImage.ExtractSubset(croppedImage, cropRect))
        {
            // Se ExtractSubset fallisce, usa un approccio manuale
            using var surface = SKSurface.Create(new SKImageInfo(width, height));
            using var canvas = surface.Canvas;
            canvas.Clear(SKColors.White);

            var srcRect = new SKRect(x, y, x + width, y + height);
            var destRect = new SKRect(0, 0, width, height);
            canvas.DrawBitmap(sourceImage, srcRect, destRect);

            using var image = surface.Snapshot();
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            using var stream = File.OpenWrite(outputPath);
            data.SaveTo(stream);
            return;
        }

        // Salva l'immagine croppata
        using var img = SKImage.FromBitmap(croppedImage);
        using var encoded = img.Encode(SKEncodedImageFormat.Png, 100);
        using var fileStream = File.OpenWrite(outputPath);
        encoded.SaveTo(fileStream);
    }

    private static async Task ProcessTablesWithTableFormer(string tablesDirectory)
    {
        try
        {
            // Setup artifacts directory
            var artifactsRoot = new DirectoryInfo(Path.Combine(Directory.GetCurrentDirectory(), "artifacts"));
            Log.Information("TableFormer artifacts directory: {Path}", artifactsRoot.FullName);

            // Bootstrap TableFormer (scarica i modelli se necessario)
            Log.Information("Initializing TableFormer (downloading models from Hugging Face if needed)...");
            Log.Information("This may take a while on first run (downloading ~100MB of models)...");

            using var bootstrapper = new TableFormerArtifactBootstrapper(artifactsRoot, variant: "fast");

            var bootstrapResult = await bootstrapper.EnsureArtifactsAsync().ConfigureAwait(false);
            Log.Information("TableFormer models downloaded successfully!");
            Log.Information("Model directory: {Path}", bootstrapResult.ModelDirectory.FullName);

            // Initialize predictor
            Log.Information("Loading model weights...");
            var initSnapshot = await bootstrapResult.InitializePredictorAsync().ConfigureAwait(false);
            Log.Information("Model initialized successfully!");

            Log.Information("");
            Log.Information("TableFormer is ready to process tables!");
            Log.Information("Note: Full table structure extraction requires access to internal APIs.");
            Log.Information("The models have been successfully downloaded and initialized.");
            Log.Information("Extracted table images are available in: {Path}", tablesDirectory);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error initializing TableFormer");
        }
    }

    private static string? FindModelPath()
    {
        // Cerca il modello partendo dalla directory corrente
        var currentDir = Directory.GetCurrentDirectory();

        // Prova diversi path relativi
        var candidates = new[]
        {
            Path.Combine(currentDir, "src", "submodules", "ds4sd-docling-layout-heron-onnx", "models", "heron-converted.onnx"),
            Path.Combine(currentDir, "..", "submodules", "ds4sd-docling-layout-heron-onnx", "models", "heron-converted.onnx"),
            Path.Combine(currentDir, "..", "..", "src", "submodules", "ds4sd-docling-layout-heron-onnx", "models", "heron-converted.onnx"),
        };

        foreach (var candidate in candidates)
        {
            var fullPath = Path.GetFullPath(candidate);
            if (File.Exists(fullPath))
            {
                return fullPath;
            }
        }

        return null;
    }
}
