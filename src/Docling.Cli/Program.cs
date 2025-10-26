using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using EasyOcrNet;
using EasyOcrNet.Assets;
using EasyOcrNet.Models;
using LayoutSdk;
using LayoutSdk.Configuration;
using Serilog;
using SkiaSharp;
using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.Model;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Results;
using TableFormerTorchSharpSdk.Tensorization;

namespace Docling.Cli;

internal static class Program
{
    private static readonly System.Text.Json.JsonSerializerOptions TableResultSerializerOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase,
    };

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

            var imageBaseName = Path.GetFileNameWithoutExtension(imagePath);
            if (string.IsNullOrEmpty(imageBaseName))
            {
                imageBaseName = "image";
            }

            using var originalImage = SKBitmap.Decode(imagePath);
            if (originalImage is null)
            {
                Log.Error("Failed to load original image for OCR and table extraction");
                return 1;
            }

            var ocrOutputDir = Path.Combine(Path.GetDirectoryName(imagePath) ?? ".", "ocr_results");
            await ProcessLayoutItemsWithEasyOcrAsync(result.Boxes, originalImage, ocrOutputDir, imageBaseName).ConfigureAwait(false);

            // Estrai le tabelle rilevate
            var tables = result.Boxes.Where(b => b.Label.Equals("Table", StringComparison.OrdinalIgnoreCase)).ToList();
            if (tables.Count > 0)
            {
                Log.Information("");
                Log.Information("=== TABLE EXTRACTION ===");
                Log.Information("Found {Count} table(s)", tables.Count);

                // Crea una directory per salvare le tabelle estratte
                var outputDir = Path.Combine(Path.GetDirectoryName(imagePath) ?? ".", "extracted_tables");
                Directory.CreateDirectory(outputDir);

                for (int i = 0; i < tables.Count; i++)
                {
                    var table = tables[i];
                    var tableFileName = imageBaseName + $"_table_{i + 1}.png";
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

    private static void ExtractAndSaveRegion(SKBitmap sourceImage, LayoutSdk.BoundingBox region, string outputPath)
    {
        using var croppedImage = CropRegion(sourceImage, region);
        if (croppedImage is null)
        {
            Log.Warning("Unable to crop region '{Label}' for output '{OutputPath}'", region.Label, outputPath);
            return;
        }

        using var image = SKImage.FromBitmap(croppedImage);
        using var encoded = image.Encode(SKEncodedImageFormat.Png, 100);
        using var fileStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.None);
        encoded.SaveTo(fileStream);
    }

    private static SKBitmap? CropRegion(SKBitmap sourceImage, LayoutSdk.BoundingBox region)
    {
        var left = (int)Math.Floor(region.X);
        var top = (int)Math.Floor(region.Y);
        var right = (int)Math.Ceiling(region.X + region.Width);
        var bottom = (int)Math.Ceiling(region.Y + region.Height);

        left = Math.Clamp(left, 0, sourceImage.Width);
        top = Math.Clamp(top, 0, sourceImage.Height);
        right = Math.Clamp(right, 0, sourceImage.Width);
        bottom = Math.Clamp(bottom, 0, sourceImage.Height);

        if (right <= left || bottom <= top)
        {
            return null;
        }

        var width = right - left;
        var height = bottom - top;
        var cropRect = new SKRectI(left, top, right, bottom);
        var croppedImage = new SKBitmap(width, height);

        if (sourceImage.ExtractSubset(croppedImage, cropRect))
        {
            return croppedImage;
        }

        croppedImage.Dispose();

        using var surface = SKSurface.Create(new SKImageInfo(width, height));
        var canvas = surface.Canvas;
        canvas.Clear(SKColors.White);

        var srcRect = new SKRect(left, top, right, bottom);
        var destRect = new SKRect(0, 0, width, height);
        canvas.DrawBitmap(sourceImage, srcRect, destRect);

        using var snapshot = surface.Snapshot();
        return SKBitmap.FromImage(snapshot);
    }

    private static async Task ProcessLayoutItemsWithEasyOcrAsync(
        IEnumerable<LayoutSdk.BoundingBox> layoutBoxes,
        SKBitmap originalImage,
        string outputDirectory,
        string imageBaseName)
    {
        var candidates = layoutBoxes
            .Where(box => !box.Label.Equals("Table", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (candidates.Count == 0)
        {
            Log.Information("");
            Log.Information("No non-table layout elements to process with EasyOCR.");
            return;
        }

        Log.Information("");
        Log.Information("=== OCR EXTRACTION (EasyOCR) ===");

        try
        {
            Directory.CreateDirectory(outputDirectory);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to create EasyOCR output directory {Directory}", outputDirectory);
            return;
        }

        try
        {
            using var ocrEngine = await CreateEasyOcrAsync().ConfigureAwait(false);
            var ocrResults = new List<Dictionary<string, object?>>();

            foreach (var box in candidates)
            {
                try
                {
                    using var crop = CropRegion(originalImage, box);
                    if (crop is null)
                    {
                        Log.Warning("Skipping OCR for '{Label}' due to invalid crop bounds", box.Label);
                        continue;
                    }

                    // Process the cropped image with EasyOCR
                    var results = await ocrEngine.ProcessImageAsync(crop).ConfigureAwait(false);

                    var combinedText = results.Count > 0
                        ? string.Join(" ", results.Select(r => r.Text)).Trim()
                        : string.Empty;

                    if (string.IsNullOrWhiteSpace(combinedText))
                    {
                        Log.Information("  - {Label,-15} OCR => <no text>", box.Label);
                    }
                    else
                    {
                        Log.Information("  - {Label,-15} OCR => {Text}", box.Label, combinedText);
                    }

                    var segments = results.Select(r => new Dictionary<string, object?>(StringComparer.Ordinal)
                    {
                        ["text"] = r.Text,
                        ["confidence"] = r.Confidence,
                        ["bounding_box"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                        {
                            ["min_x"] = r.BoundingBox.MinX,
                            ["min_y"] = r.BoundingBox.MinY,
                            ["max_x"] = r.BoundingBox.MaxX,
                            ["max_y"] = r.BoundingBox.MaxY,
                            ["width"] = r.BoundingBox.Width,
                            ["height"] = r.BoundingBox.Height,
                        },
                    }).ToList();

                    ocrResults.Add(new Dictionary<string, object?>(StringComparer.Ordinal)
                    {
                        ["label"] = box.Label,
                        ["confidence"] = box.Confidence,
                        ["bounding_box"] = new Dictionary<string, double>(StringComparer.Ordinal)
                        {
                            ["x"] = box.X,
                            ["y"] = box.Y,
                            ["width"] = box.Width,
                            ["height"] = box.Height,
                        },
                        ["segment_count"] = results.Count,
                        ["segments"] = segments,
                        ["combined_text"] = combinedText,
                    });
                }
                catch (Exception ex)
                {
                    Log.Error(ex, "EasyOCR failed for layout element '{Label}'", box.Label);
                }
            }

            if (ocrResults.Count == 0)
            {
                Log.Warning("EasyOCR did not produce results for any layout elements.");
                return;
            }

            var payload = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["image"] = imageBaseName,
                ["items"] = ocrResults,
                ["processed_at_utc"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            };

            var outputPath = Path.Combine(outputDirectory, $"{imageBaseName}_ocr_results.json");
            using var stream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.None);
            System.Text.Json.JsonSerializer.Serialize(stream, payload, TableResultSerializerOptions);
            await stream.FlushAsync().ConfigureAwait(false);
            Log.Information("EasyOCR results saved to: {OutputPath}", outputPath);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Failed to initialise EasyOCR");
        }
    }

    private static async Task<OcrEngine> CreateEasyOcrAsync()
    {
        var modelDirectory = ResolveEasyOcrModelDirectory();
        Log.Information("Using EasyOCR model directory: {Directory}", modelDirectory);

        var detectorPath = Path.Combine(modelDirectory, "detection.onnx");
        var recognizerPath = Path.Combine(modelDirectory, "english_g2_rec.onnx");

        // Ensure models are downloaded if they don't exist
        var options = new GithubReleaseOptions(); // Use default repository and tag (v2025.09.19)
        await OcrReleaseDownloader.EnsureModelAsync(detectorPath, options, msg => Log.Information(msg)).ConfigureAwait(false);
        await OcrReleaseDownloader.EnsureModelAsync(recognizerPath, options, msg => Log.Information(msg)).ConfigureAwait(false);

        var config = new OcrConfig(Language: "en");
        return new OcrEngine(detectorPath, recognizerPath, "en", config, Path.Combine(modelDirectory, "character"));
    }

    private static string ResolveEasyOcrModelDirectory()
    {
        // Check for environment variable override
        var overrideDirectory = Environment.GetEnvironmentVariable("EASYOCR_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(overrideDirectory))
        {
            var resolvedOverride = Path.GetFullPath(overrideDirectory);
            if (ContainsEasyOcrDetectionModel(resolvedOverride))
            {
                return resolvedOverride;
            }

            // If override is set but doesn't contain models, create it and use it
            Log.Information("EasyOCR override directory '{Directory}' does not contain models yet. Will download them.", resolvedOverride);
            Directory.CreateDirectory(resolvedOverride);
            return resolvedOverride;
        }

        var baseDirectory = AppContext.BaseDirectory;

        // Check if models already exist in base directory
        if (ContainsEasyOcrDetectionModel(baseDirectory))
        {
            return baseDirectory;
        }

        // Check common candidate directories
        var candidates = new[]
        {
            Path.Combine(baseDirectory, "contentFiles", "any", "any", "models", "easyocr"),
            Path.Combine(baseDirectory, "contentFiles", "any", "any", "models"),
            Path.Combine(baseDirectory, "models", "easyocr"),
            Path.Combine(baseDirectory, "models", "onnx"),
            Path.Combine(baseDirectory, "models"),
        };

        foreach (var candidate in candidates)
        {
            if (ContainsEasyOcrDetectionModel(candidate))
            {
                return candidate;
            }
        }

        // No existing models found - create default directory for auto-download
        var defaultModelDir = Path.Combine(baseDirectory, "models");
        Log.Information("No existing EasyOCR models found. Creating directory '{Directory}' for auto-download.", defaultModelDir);
        Directory.CreateDirectory(defaultModelDir);
        return defaultModelDir;
    }

    private static bool ContainsEasyOcrDetectionModel(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
        {
            return false;
        }

        var detectionFiles = new[]
        {
            "detection.onnx",
            "detection.xml",
            "EasyOCRDetector.onnx",
        };

        foreach (var file in detectionFiles)
        {
            if (File.Exists(Path.Combine(directory, file)))
            {
                return true;
            }
        }

        return false;
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

            using var bootstrapper = new TableFormerArtifactBootstrapper(artifactsRoot, variant: TableFormerModelVariant.Fast);

            var bootstrapResult = await bootstrapper.EnsureArtifactsAsync().ConfigureAwait(false);
            Log.Information("TableFormer models downloaded successfully!");
            Log.Information("Model directory: {Path}", bootstrapResult.ModelDirectory.FullName);

            // Initialize predictor
            Log.Information("Loading model weights...");
            var initSnapshot = await bootstrapResult.InitializePredictorAsync().ConfigureAwait(false);
            Log.Information("Model initialized successfully!");

            // Create neural model and supporting components
            Log.Information("Creating TableFormer processing pipeline...");
            using var neuralModel = new TableFormerNeuralModel(
                bootstrapResult.ConfigSnapshot,
                initSnapshot,
                bootstrapResult.ModelDirectory);

            var decoder = new TableFormerSequenceDecoder(initSnapshot);
            var cellMatcher = new TableFormerCellMatcher(bootstrapResult.ConfigSnapshot);
            var preparer = new TableFormerPageInputPreparer();
            var cropper = new TableFormerTableCropper();
            var tensorizer = TableFormerImageTensorizer.FromConfig(bootstrapResult.ConfigSnapshot);
            var postProcessor = new TableFormerMatchingPostProcessor();
            var assembler = new TableFormerDoclingResponseAssembler();

            // Get all table image files
            var tableFiles = Directory.GetFiles(tablesDirectory, "*.png")
                .Concat(Directory.GetFiles(tablesDirectory, "*.jpg"))
                .Concat(Directory.GetFiles(tablesDirectory, "*.jpeg"))
                .ToArray();

            if (tableFiles.Length == 0)
            {
                Log.Warning("No table image files found in {Directory}", tablesDirectory);
                return;
            }

            Log.Information("Found {Count} table image(s) to process", tableFiles.Length);

            var allTables = new List<Dictionary<string, object?>>();

            // Process each table image
            for (int i = 0; i < tableFiles.Length; i++)
            {
                var tableFile = new FileInfo(tableFiles[i]);
                Log.Information("Processing table {Index}/{Total}: {FileName}", i + 1, tableFiles.Length, tableFile.Name);

                try
                {
                    var tableResult = ProcessSingleTable(
                        tableFile,
                        neuralModel,
                        decoder,
                        cellMatcher,
                        preparer,
                        cropper,
                        tensorizer,
                        postProcessor,
                        assembler);

                    if (tableResult != null)
                    {
                        allTables.Add(tableResult);
                        var cellCount = 0;
                        if (tableResult.TryGetValue("cell_count", out var cellCountValue)
                            && cellCountValue is int resolvedCellCount)
                        {
                            cellCount = resolvedCellCount;
                        }

                        Log.Information("  ✓ Table processed successfully - {CellCount} cells detected", cellCount);
                    }
                    else
                    {
                        Log.Warning("  ✗ Failed to process table: {FileName}", tableFile.Name);
                    }
                }
                catch (Exception ex)
                {
                    Log.Error(ex, "Error processing table {FileName}", tableFile.Name);
                }
            }

            // Save results
            if (allTables.Count > 0)
            {
                var resultsPath = Path.Combine(tablesDirectory, "table_structure_results.json");
                var results = new Dictionary<string, object?>
                {
                    ["num_tables"] = allTables.Count,
                    ["tables"] = allTables,
                    ["processing_timestamp"] = DateTime.UtcNow.ToString("O")
                };

                using var fileStream = new FileStream(resultsPath, FileMode.Create, FileAccess.Write, FileShare.None);
                await System.Text.Json.JsonSerializer.SerializeAsync(
                    fileStream,
                    results,
                    TableResultSerializerOptions).ConfigureAwait(false);

                Log.Information("");
                Log.Information("=== TABLE STRUCTURE EXTRACTION COMPLETED ===");
                Log.Information("Processed {ProcessedCount}/{TotalCount} tables successfully", allTables.Count, tableFiles.Length);
                Log.Information("Results saved to: {ResultsPath}", resultsPath);

                // Print summary of each table
                Log.Information("");
                Log.Information("Table Summary:");
                for (int i = 0; i < allTables.Count; i++)
                {
                    var table = allTables[i];
                    var summaryCellCount = 0;
                    if (table.TryGetValue("cell_count", out var cellCountValue)
                        && cellCountValue is int resolvedCellCount)
                    {
                        summaryCellCount = resolvedCellCount;
                    }

                    Log.Information("  Table {Index}: {CellCount} cells", i + 1, summaryCellCount);
                }
            }
            else
            {
                Log.Warning("No tables were successfully processed");
            }
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error in TableFormer processing pipeline");
        }
    }

    private static Dictionary<string, object?>? ProcessSingleTable(
        FileInfo tableFile,
        TableFormerNeuralModel neuralModel,
        TableFormerSequenceDecoder decoder,
        TableFormerCellMatcher cellMatcher,
        TableFormerPageInputPreparer preparer,
        TableFormerTableCropper cropper,
        TableFormerImageTensorizer tensorizer,
        TableFormerMatchingPostProcessor postProcessor,
        TableFormerDoclingResponseAssembler assembler)
    {
        try
        {
            // Decode the table image
            var decodedImage = TableFormerDecodedPageImage.Decode(tableFile);

            // Prepare page input (even though it's a single table, we need the page structure)
            var pageSnapshot = preparer.PreparePageInput(decodedImage);

            // Since this is already a cropped table image, we'll treat the entire image as one table
            var tableBoundingBoxes = new List<TableFormerBoundingBox>
            {
                new TableFormerBoundingBox(0, 0, decodedImage.Width, decodedImage.Height)
            };

            var cropSnapshot = cropper.PrepareTableCrops(decodedImage, tableBoundingBoxes);

            if (cropSnapshot.TableCrops.Count == 0)
            {
                return null;
            }

            var tables = new List<Dictionary<string, object?>>();
            var cellCount = 0;

            // Process the table crop (there should be only one)
            foreach (var crop in cropSnapshot.TableCrops)
            {
                // Create tensor from the cropped table image
                using var tensorSnapshot = tensorizer.CreateTensor(crop);

                // Run neural inference
                using var prediction = neuralModel.Predict(tensorSnapshot.Tensor);

                // Decode the sequence predictions
                var decoded = decoder.Decode(prediction);

                // Match cells to table structure
                var matchingResult = cellMatcher.MatchCells(pageSnapshot, crop, decoded);
                var matchingDetails = matchingResult.ToMatchingDetails();

                // Post-process the matching results
                var processed = pageSnapshot.Tokens.Count > 0
                    ? postProcessor.Process(matchingDetails.ToMutable(), correctOverlappingCells: false)
                    : matchingDetails;

                // Assemble final table structure
                var assembled = assembler.Assemble(processed, decoded, sortRowColIndexes: true);

                tables.Add(assembled.ToDictionary());
                cellCount += assembled.TfResponses.Count;
            }

            return new Dictionary<string, object?>
            {
                ["filename"] = tableFile.Name,
                ["num_tables"] = tables.Count,
                ["tables"] = tables,
                ["cell_count"] = cellCount
            };
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Error processing single table {FileName}", tableFile.Name);
            return null;
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
