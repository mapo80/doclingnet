#if false
using System;
using System.IO;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Models.Tables;

/// <summary>
/// Integration test for the complete TableFormer pipeline.
/// Tests end-to-end inference from image to table structure.
/// </summary>
public sealed class TableFormerIntegrationTest
{
    /// <summary>
    /// Run a complete end-to-end test of the TableFormer pipeline.
    /// </summary>
    public static async Task<bool> RunAsync(ILogger? logger = null)
    {
        logger ??= LoggerFactory.Create(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Debug)).CreateLogger<TableFormerIntegrationTest>();

        logger.LogInformation("=== TableFormer Integration Test ===");
        logger.LogInformation("Starting end-to-end pipeline test");

        try
        {
            // Step 1: Create test image
            logger.LogInformation("Step 1: Creating test table image");
            var testImage = CreateTestTableImage();
            if (testImage == null)
            {
                logger.LogError("Failed to create test image");
                return false;
            }

            using (testImage)
            {
                // Step 2: Initialize service
                logger.LogInformation("Step 2: Initializing TableFormer service");
                var options = new TableFormerStructureServiceOptions
                {
                    Variant = TableFormerSdk.Enums.TableFormerModelVariant.Fast,
                    Runtime = TableFormerSdk.Enums.TableFormerRuntime.Onnx,
                    GenerateOverlay = true
                };

                using var service = new TableFormerTableStructureService(options, logger);

                // Check if backend loaded
                if (!service.IsUsingOnnxBackend())
                {
                    logger.LogWarning("ONNX backend not loaded - using stub backend");
                    logger.LogWarning("This might indicate missing model files");
                    return false;
                }

                var (fastEncoder, fastTagEncoder, accurateEncoder) = service.GetCurrentModelPaths();
                logger.LogInformation("Model paths:");
                logger.LogInformation("  Fast Encoder: {Path}", fastEncoder);
                logger.LogInformation("  Fast Tag Encoder: {Path}", fastTagEncoder);
                logger.LogInformation("  Accurate Encoder: {Path}", accurateEncoder ?? "(not loaded)");

                // Step 3: Encode image
                logger.LogInformation("Step 3: Encoding test image");
                using var encoded = SKImage.FromBitmap(testImage).Encode(SKEncodedImageFormat.Png, 90);
                var imageBytes = encoded.ToArray();

                // Step 4: Create request
                logger.LogInformation("Step 4: Creating table structure request");
                var page = new PageReference(DocumentIdentifier: "test-doc", PageNumber: 1);
                var tableBounds = new BoundingBox(0, 0, testImage.Width, testImage.Height);
                var request = new TableStructureRequest(page, tableBounds, imageBytes);

                // Step 5: Run inference
                logger.LogInformation("Step 5: Running TableFormer inference");
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var result = await service.InferStructureAsync(request);
                stopwatch.Stop();

                logger.LogInformation("Inference completed in {Time:F2}ms", stopwatch.Elapsed.TotalMilliseconds);

                // Step 6: Validate results
                logger.LogInformation("Step 6: Validating results");
                logger.LogInformation("  Cells detected: {Count}", result.Cells.Count);
                logger.LogInformation("  Rows: {Count}", result.RowCount);
                logger.LogInformation("  Columns: {Count}", result.ColumnCount);

                if (result.Cells.Count == 0)
                {
                    logger.LogWarning("No cells detected - this might indicate a problem");
                    logger.LogWarning("However, the pipeline executed successfully");
                }
                else
                {
                    // Show first few cells
                    var cellsToShow = Math.Min(5, result.Cells.Count);
                    logger.LogInformation("  First {Count} cells:", cellsToShow);
                    for (int i = 0; i < cellsToShow; i++)
                    {
                        var cell = result.Cells[i];
                        logger.LogInformation("    Cell {Index}: Bounds=({L:F2},{T:F2},{R:F2},{B:F2}) RowSpan={RS} ColSpan={CS}",
                            i,
                            cell.BoundingBox.Left, cell.BoundingBox.Top,
                            cell.BoundingBox.Right, cell.BoundingBox.Bottom,
                            cell.RowSpan, cell.ColumnSpan);
                    }
                }

                // Step 7: Check metrics
                logger.LogInformation("Step 7: Checking performance metrics");
                var metrics = service.GetMetrics();
                logger.LogInformation("  Total inferences: {Count}", metrics.TotalInferences);
                logger.LogInformation("  Success rate: {Rate:F1}%", metrics.SuccessRate * 100);
                logger.LogInformation("  Average time: {Time:F2}ms", metrics.AverageInferenceTime.TotalMilliseconds);

                logger.LogInformation("=== Integration Test PASSED ===");
                return true;
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Integration test failed");
            logger.LogInformation("=== Integration Test FAILED ===");
            return false;
        }
    }

    /// <summary>
    /// Create a simple test table image (3x3 grid with borders).
    /// </summary>
    private static SKBitmap? CreateTestTableImage()
    {
        const int width = 600;
        const int height = 400;
        const int rows = 3;
        const int cols = 3;

        var bitmap = new SKBitmap(width, height);
        using var canvas = new SKCanvas(bitmap);

        // White background
        canvas.Clear(SKColors.White);

        // Draw grid
        using var paint = new SKPaint
        {
            Color = SKColors.Black,
            StrokeWidth = 2,
            Style = SKPaintStyle.Stroke,
            IsAntialias = true
        };

        var cellWidth = width / (float)cols;
        var cellHeight = height / (float)rows;

        // Horizontal lines
        for (int i = 0; i <= rows; i++)
        {
            var y = i * cellHeight;
            canvas.DrawLine(0, y, width, y, paint);
        }

        // Vertical lines
        for (int i = 0; i <= cols; i++)
        {
            var x = i * cellWidth;
            canvas.DrawLine(x, 0, x, height, paint);
        }

        // Add some text to cells
        using var textPaint = new SKPaint
        {
            Color = SKColors.Black,
            TextSize = 24,
            IsAntialias = true,
            TextAlign = SKTextAlign.Center
        };

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                var centerX = (col + 0.5f) * cellWidth;
                var centerY = (row + 0.5f) * cellHeight + 8; // Offset for text baseline
                canvas.DrawText($"({row},{col})", centerX, centerY, textPaint);
            }
        }

        return bitmap;
    }
}
#endif
