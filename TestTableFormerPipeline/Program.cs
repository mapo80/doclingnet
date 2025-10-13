using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Models.Tables;
using SkiaSharp;

class Program
{
    static async Task<int> Main(string[] args)
    {
        Console.WriteLine("=" + new string('=', 79));
        Console.WriteLine(".NET PIPELINE TEST");
        Console.WriteLine("=" + new string('=', 79));

        var imagePath = "../dataset/2305.03393v1-pg9-img.png";
        if (!File.Exists(imagePath))
        {
            Console.WriteLine($"Error: Image not found at {imagePath}");
            return -1;
        }

        var fileInfo = new FileInfo(imagePath);
        Console.WriteLine($"Image: {imagePath}");
        Console.WriteLine($"Size: {fileInfo.Length / 1024.0:F1} KB");

        // Load image
        using var stream = File.OpenRead(imagePath);
        using var bitmap = SKBitmap.Decode(stream);
        Console.WriteLine($"Image dimensions: {bitmap.Width}x{bitmap.Height}");

        // Step 1: Layout detection
        Console.WriteLine("\n" + new string('-', 80));
        Console.WriteLine("STEP 1: Layout Detection");
        Console.WriteLine(new string('-', 80));

        var layoutSw = Stopwatch.StartNew();
        var layoutService = new LayoutDetectionService();
        var layoutRequest = new LayoutDetectionRequest(bitmap, DpiResolution: 300);
        var layoutResult = await layoutService.DetectAsync(layoutRequest);
        layoutSw.Stop();

        Console.WriteLine($"Layout detection time: {layoutSw.Elapsed.TotalSeconds:F3}s");
        Console.WriteLine($"Total cells detected: {layoutResult.Cells.Count}");

        // Find table cells
        var tableCells = layoutResult.Cells.Where(c => c.Label == "table").ToList();
        Console.WriteLine($"Table cells found: {tableCells.Count}");

        if (tableCells.Count == 0)
        {
            Console.WriteLine("No tables found in the image!");
            return 0;
        }

        // Step 2: TableFormer on each table
        Console.WriteLine("\n" + new string('-', 80));
        Console.WriteLine("STEP 2: TableFormer Structure Recognition");
        Console.WriteLine(new string('-', 80));

        var tableService = new TableFormerTableStructureService();
        var totalTableTime = 0.0;

        for (int idx = 0; idx < tableCells.Count; idx++)
        {
            var tableCell = tableCells[idx];
            var bbox = tableCell.BoundingBox;

            Console.WriteLine($"\nTable {idx + 1}:");
            Console.WriteLine($"  BBox: ({bbox.Left:F1}, {bbox.Top:F1}, {bbox.Right:F1}, {bbox.Bottom:F1})");
            Console.WriteLine($"  Size: {bbox.Width:F1}x{bbox.Height:F1}");

            // Crop table region
            using var tableBitmap = new SKBitmap((int)bbox.Width, (int)bbox.Height);
            using var canvas = new SKCanvas(tableBitmap);
            canvas.DrawBitmap(bitmap,
                SKRect.Create(bbox.Left, bbox.Top, bbox.Width, bbox.Height),
                SKRect.Create(0, 0, bbox.Width, bbox.Height));

            // Encode to PNG
            using var image = SKImage.FromBitmap(tableBitmap);
            using var encoded = image.Encode(SKEncodedImageFormat.Png, 90);
            var imageBytes = encoded.ToArray();

            // Create TableFormer request
            var pageRef = new PageReference(PageNumber: 1, DpiResolution: 300);
            var tableRequest = new TableStructureRequest(pageRef, bbox, imageBytes);

            // Run TableFormer
            var tfSw = Stopwatch.StartNew();
            var tableStructure = await tableService.InferStructureAsync(tableRequest);
            tfSw.Stop();

            totalTableTime += tfSw.Elapsed.TotalSeconds;

            Console.WriteLine($"  TableFormer time: {tfSw.Elapsed.TotalSeconds:F3}s");
            Console.WriteLine($"  Rows detected: {tableStructure.RowCount}");
            Console.WriteLine($"  Columns detected: {tableStructure.ColumnCount}");
            Console.WriteLine($"  Cells detected: {tableStructure.Cells.Count}");

            // Show first few cells
            if (tableStructure.Cells.Count > 0)
            {
                Console.WriteLine($"  First 3 cells:");
                for (int i = 0; i < Math.Min(3, tableStructure.Cells.Count); i++)
                {
                    var cell = tableStructure.Cells[i];
                    var cellBbox = cell.BoundingBox;
                    Console.WriteLine($"    Cell {i + 1}: BBox=({cellBbox.Left:F1}, {cellBbox.Top:F1}, " +
                                    $"{cellBbox.Right:F1}, {cellBbox.Bottom:F1}), " +
                                    $"RowSpan={cell.RowSpan}, ColSpan={cell.ColumnSpan}");
                }
            }
        }

        // Get metrics
        var metrics = tableService.GetMetrics();

        // Summary
        Console.WriteLine("\n" + new string('=', 80));
        Console.WriteLine(".NET SUMMARY");
        Console.WriteLine(new string('=', 80));
        Console.WriteLine($"Total layout time: {layoutSw.Elapsed.TotalSeconds:F3}s");
        Console.WriteLine($"Total TableFormer time: {totalTableTime:F3}s");
        Console.WriteLine($"Total tables processed: {tableCells.Count}");
        Console.WriteLine($"Total time: {(layoutSw.Elapsed.TotalSeconds + totalTableTime):F3}s");
        Console.WriteLine();
        Console.WriteLine("TableFormer Metrics:");
        Console.WriteLine($"  Total inferences: {metrics.TotalInferences}");
        Console.WriteLine($"  Successful: {metrics.SuccessfulInferences}");
        Console.WriteLine($"  Failed: {metrics.FailedInferences}");
        Console.WriteLine($"  Total cells detected: {metrics.TotalCellsDetected}");
        Console.WriteLine($"  Average inference time: {metrics.AverageInferenceTime.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Backend: {string.Join(", ", metrics.BackendUsage.Select(kv => $"{kv.Key}={kv.Value}"))}");

        tableService.Dispose();

        return 0;
    }
}
