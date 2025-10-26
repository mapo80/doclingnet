using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Serialization;
using EasyOcrNet;
using EasyOcrNet.Assets;
using EasyOcrNet.Models;
using LayoutSdk;
using LayoutSdk.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using TableFormerTorchSharpSdk.Artifacts;
using TableFormerTorchSharpSdk.Configuration;
using TableFormerTorchSharpSdk.Decoding;
using TableFormerTorchSharpSdk.Matching;
using TableFormerTorchSharpSdk.Model;
using TableFormerTorchSharpSdk.PagePreparation;
using TableFormerTorchSharpSdk.Results;
using TableFormerTorchSharpSdk.Tensorization;

namespace DoclingNetSdk;

/// <summary>
/// Main entry point for DoclingNet SDK.
/// Provides unified API for converting images to structured documents and Markdown.
/// </summary>
public sealed class DoclingConverter : IDisposable
{
    private readonly DoclingConfiguration _config;
    private readonly ILogger _logger;
    private readonly LayoutSdk.LayoutSdk _layoutSdk;
    private bool _disposed;

    /// <summary>
    /// Creates a new instance of DoclingConverter with the specified configuration.
    /// </summary>
    public DoclingConverter(DoclingConfiguration config, ILogger? logger = null)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _logger = logger ?? NullLogger.Instance;

        // Validate configuration
        config.Validate();

        // Find layout model automatically
        var layoutModelPath = FindLayoutModel();
        if (string.IsNullOrEmpty(layoutModelPath))
        {
            throw new FileNotFoundException(
                "Layout model not found. Please ensure the model is available in one of these locations:\n" +
                "  1. src/submodules/ds4sd-docling-layout-heron-onnx/models/heron-converted.onnx\n" +
                "  2. models/heron-converted.onnx\n" +
                "  3. [AppDirectory]/models/heron-converted.onnx");
        }

        _logger.LogInformation("Using layout model: {ModelPath}", layoutModelPath);

        // Initialize Layout SDK
        var layoutOptions = new LayoutSdkOptions(
            onnxModelPath: layoutModelPath,
            defaultLanguage: DocumentLanguage.English,
            validateModelPaths: true);

        _layoutSdk = new LayoutSdk.LayoutSdk(layoutOptions);

        _logger.LogInformation("DoclingConverter initialized successfully");
    }

    private static string? FindLayoutModel()
    {
        var candidates = new[]
        {
            Path.Combine(Directory.GetCurrentDirectory(), "src", "submodules", "ds4sd-docling-layout-heron-onnx", "models", "heron-converted.onnx"),
            Path.Combine(Directory.GetCurrentDirectory(), "models", "heron-converted.onnx"),
            Path.Combine(AppContext.BaseDirectory, "models", "heron-converted.onnx"),
        };

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    /// <summary>
    /// Converts an image to a structured document with Markdown export.
    /// </summary>
    public async Task<DoclingConversionResult> ConvertImageAsync(
        string imagePath,
        CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DoclingConverter));
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image file not found: {imagePath}", imagePath);
        }

        _logger.LogInformation("Converting image: {ImagePath}", imagePath);

        // Step 1: Layout Detection
        _logger.LogInformation("Step 1: Running layout detection...");
        var layoutResult = _layoutSdk.Process(imagePath, overlay: false, LayoutRuntime.Onnx);
        _logger.LogInformation("Detected {Count} layout elements", layoutResult.Boxes.Count);

        // Load original image for OCR and table processing
        using var originalImage = SKBitmap.Decode(imagePath);
        if (originalImage is null)
        {
            throw new InvalidOperationException($"Failed to decode image: {imagePath}");
        }

        // Step 2: OCR Extraction (for non-table elements)
        var ocrTexts = new Dictionary<string, string>();
        if (_config.EnableOcr)
        {
            _logger.LogInformation("Step 2: Running OCR extraction...");
            ocrTexts = await ExtractOcrTextAsync(layoutResult.Boxes, originalImage, cancellationToken);
            _logger.LogInformation("Extracted text from {Count} elements", ocrTexts.Count);
        }

        // Step 3: Table Structure Recognition
        var tableStructures = new Dictionary<string, TableStructureInfo>();
        if (_config.EnableTableRecognition)
        {
            _logger.LogInformation("Step 3: Running table structure recognition...");
            tableStructures = await ProcessTablesAsync(layoutResult.Boxes, originalImage, cancellationToken);
            _logger.LogInformation("Processed {Count} tables", tableStructures.Count);
        }

        // Step 4: Build Document
        _logger.LogInformation("Step 4: Building document structure...");
        var document = BuildDocument(imagePath, layoutResult.Boxes, ocrTexts, tableStructures);

        // Step 5: Export to Markdown
        _logger.LogInformation("Step 5: Exporting to markdown...");
        var markdown = ExportToMarkdown(document);

        var result = new DoclingConversionResult(
            document,
            markdown,
            layoutResult.Boxes.Count,
            ocrTexts.Count,
            tableStructures.Count);

        _logger.LogInformation("Conversion completed successfully");
        return result;
    }

    /// <summary>
    /// Converts multiple images in parallel.
    /// </summary>
    public async Task<Dictionary<string, DoclingConversionResult>> ConvertImagesAsync(
        IEnumerable<string> imagePaths,
        CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DoclingConverter));
        }

        ArgumentNullException.ThrowIfNull(imagePaths);

        var results = new Dictionary<string, DoclingConversionResult>();
        foreach (var imagePath in imagePaths)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var result = await ConvertImageAsync(imagePath, cancellationToken);
            results[imagePath] = result;
        }

        return results;
    }

    private async Task<Dictionary<string, string>> ExtractOcrTextAsync(
        IReadOnlyList<LayoutSdk.BoundingBox> layoutBoxes,
        SKBitmap originalImage,
        CancellationToken cancellationToken)
    {
        var candidates = layoutBoxes
            .Where(box => !box.Label.Equals("Table", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (candidates.Count == 0)
        {
            return new Dictionary<string, string>();
        }

        var results = new Dictionary<string, string>();

        try
        {
            using var ocrEngine = await CreateEasyOcrAsync(cancellationToken);

            foreach (var box in candidates)
            {
                cancellationToken.ThrowIfCancellationRequested();

                try
                {
                    using var crop = CropRegion(originalImage, box);
                    if (crop is null)
                    {
                        _logger.LogWarning("Skipping OCR for '{Label}' due to invalid crop bounds", box.Label);
                        continue;
                    }

                    var ocrResults = await ocrEngine.ProcessImageAsync(crop);
                    var combinedText = ocrResults.Count > 0
                        ? string.Join(" ", ocrResults.Select(r => r.Text)).Trim()
                        : string.Empty;

                    if (!string.IsNullOrWhiteSpace(combinedText))
                    {
                        results[$"{box.Label}_{box.X}_{box.Y}"] = combinedText;
                        _logger.LogDebug("OCR result for {Label}: {Text}", box.Label, combinedText);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "EasyOCR failed for layout element '{Label}'", box.Label);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize EasyOCR");
        }

        return results;
    }

    private async Task<OcrEngine> CreateEasyOcrAsync(CancellationToken cancellationToken)
    {
        var modelDirectory = _config.ArtifactsPath;
        Directory.CreateDirectory(modelDirectory);

        var detectorPath = Path.Combine(modelDirectory, "detection.onnx");
        var recognizerPath = Path.Combine(modelDirectory, "english_g2_rec.onnx");

        // Ensure models are downloaded if they don't exist
        var options = new GithubReleaseOptions();
        await OcrReleaseDownloader.EnsureModelAsync(detectorPath, options, msg => _logger.LogInformation(msg), cancellationToken);
        await OcrReleaseDownloader.EnsureModelAsync(recognizerPath, options, msg => _logger.LogInformation(msg), cancellationToken);

        var config = new OcrConfig(Language: _config.OcrLanguage);
        return new OcrEngine(detectorPath, recognizerPath, _config.OcrLanguage, config, Path.Combine(modelDirectory, "character"));
    }

    private async Task<Dictionary<string, TableStructureInfo>> ProcessTablesAsync(
        IReadOnlyList<LayoutSdk.BoundingBox> layoutBoxes,
        SKBitmap originalImage,
        CancellationToken cancellationToken)
    {
        var tables = layoutBoxes
            .Where(b => b.Label.Equals("Table", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (tables.Count == 0)
        {
            return new Dictionary<string, TableStructureInfo>();
        }

        var results = new Dictionary<string, TableStructureInfo>();

        try
        {
            // Setup TableFormer
            var artifactsRoot = new DirectoryInfo(_config.ArtifactsPath);
            var variant = _config.TableFormerVariant switch
            {
                DoclingNetSdk.TableFormerVariant.Fast => TableFormerModelVariant.Fast,
                DoclingNetSdk.TableFormerVariant.Base => TableFormerModelVariant.Accurate, // Map Base to Accurate
                DoclingNetSdk.TableFormerVariant.Accurate => TableFormerModelVariant.Accurate,
                _ => TableFormerModelVariant.Fast
            };
            using var bootstrapper = new TableFormerArtifactBootstrapper(artifactsRoot, variant: variant);

            var bootstrapResult = await bootstrapper.EnsureArtifactsAsync(cancellationToken);
            var initSnapshot = await bootstrapResult.InitializePredictorAsync(cancellationToken);

            // Create neural model and supporting components
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

            // Process each table
            for (int i = 0; i < tables.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var table = tables[i];
                try
                {
                    using var tableCrop = CropRegion(originalImage, table);
                    if (tableCrop is null)
                    {
                        _logger.LogWarning("Skipping table {Index} due to invalid crop bounds", i);
                        continue;
                    }

                    var tableStructure = ProcessSingleTable(
                        tableCrop,
                        neuralModel,
                        decoder,
                        cellMatcher,
                        preparer,
                        cropper,
                        tensorizer,
                        postProcessor,
                        assembler);

                    if (tableStructure != null)
                    {
                        results[$"Table_{i}_{table.X}_{table.Y}"] = tableStructure;
                        _logger.LogDebug("Table {Index} processed successfully - {CellCount} cells", i, tableStructure.CellCount);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing table {Index}", i);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in TableFormer processing pipeline");
        }

        return results;
    }

    private TableStructureInfo? ProcessSingleTable(
        SKBitmap tableImage,
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
            // Create a temporary file for the table image (TableFormer needs a file path)
            var tempFile = Path.GetTempFileName();
            try
            {
                using (var image = SKImage.FromBitmap(tableImage))
                using (var encoded = image.Encode(SKEncodedImageFormat.Png, 100))
                using (var stream = File.OpenWrite(tempFile))
                {
                    encoded.SaveTo(stream);
                }

                // Decode the table image
                var decodedImage = TableFormerDecodedPageImage.Decode(new FileInfo(tempFile));

                // Prepare page input
                var pageSnapshot = preparer.PreparePageInput(decodedImage);

                // Treat the entire image as one table
                var tableBoundingBoxes = new List<TableFormerBoundingBox>
                {
                    new TableFormerBoundingBox(0, 0, decodedImage.Width, decodedImage.Height)
                };

                var cropSnapshot = cropper.PrepareTableCrops(decodedImage, tableBoundingBoxes);

                if (cropSnapshot.TableCrops.Count == 0)
                {
                    return null;
                }

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

                    return new TableStructureInfo(
                        Rows: assembled.PredictDetails.NumRows,
                        Columns: assembled.PredictDetails.NumCols,
                        CellCount: assembled.TfResponses.Count,
                        Cells: assembled.TfResponses);
                }

                return null;
            }
            finally
            {
                try
                {
                    if (File.Exists(tempFile))
                    {
                        File.Delete(tempFile);
                    }
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing single table");
            return null;
        }
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

    private DoclingDocument BuildDocument(
        string imagePath,
        IReadOnlyList<LayoutSdk.BoundingBox> layoutBoxes,
        Dictionary<string, string> ocrTexts,
        Dictionary<string, TableStructureInfo> tableStructures)
    {
        // Create document with page
        var imageSize = SKBitmap.DecodeBounds(imagePath);
        var pageRef = new PageReference(pageNumber: 0, dpi: 72.0);
        var pages = new List<PageReference> { pageRef };

        var document = new DoclingDocument(
            sourceId: imagePath,
            pages: pages,
            documentId: Path.GetFileNameWithoutExtension(imagePath));

        var builder = new DoclingDocumentBuilder(document);

        // Add layout elements as document items
        foreach (var box in layoutBoxes)
        {
            var bbox = new Docling.Core.Geometry.BoundingBox(
                left: box.X,
                top: box.Y,
                right: box.X + box.Width,
                bottom: box.Y + box.Height);

            var key = $"{box.Label}_{box.X}_{box.Y}";

            // Check if this is a table
            if (box.Label.Equals("Table", StringComparison.OrdinalIgnoreCase))
            {
                // For now, add as text item since TableItem requires cells
                if (tableStructures.TryGetValue(key, out var tableInfo))
                {
                    var tableText = $"[Table: {tableInfo.Rows}x{tableInfo.Columns}, {tableInfo.CellCount} cells]";
                    builder.AddItem(new ParagraphItem(pageRef, bbox, tableText));
                }
                else
                {
                    builder.AddItem(new ParagraphItem(pageRef, bbox, "[Table]"));
                }
            }
            else
            {
                // Add text from OCR if available
                if (ocrTexts.TryGetValue(key, out var text))
                {
                    builder.AddItem(new ParagraphItem(pageRef, bbox, text));
                }
                else
                {
                    builder.AddItem(new ParagraphItem(pageRef, bbox, $"[{box.Label}]"));
                }
            }
        }

        return builder.Build();
    }

    private string ExportToMarkdown(DoclingDocument document)
    {
        var serializer = new MarkdownDocSerializer();
        var result = serializer.Serialize(document);
        return result.Markdown;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _layoutSdk?.Dispose();
        _disposed = true;
    }

    private sealed record TableStructureInfo(
        int Rows,
        int Columns,
        int CellCount,
        IReadOnlyList<TableFormerDoclingCellResponse> Cells);
}
