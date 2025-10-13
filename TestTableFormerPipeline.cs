#!/usr/bin/env dotnet-script
#r "nuget: SkiaSharp, 3.119.1"
#r "nuget: Microsoft.Extensions.Logging.Abstractions, 9.0.9"
#r "nuget: Microsoft.ML.OnnxRuntime, 1.20.1"

using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SkiaSharp;

// Load assemblies from bin directory
var binPath = Path.Combine(Environment.CurrentDirectory, "src", "Docling.Models", "bin", "Debug", "net9.0");
System.Reflection.Assembly.LoadFrom(Path.Combine(binPath, "Docling.Core.dll"));
System.Reflection.Assembly.LoadFrom(Path.Combine(binPath, "LayoutSdk.dll"));
System.Reflection.Assembly.LoadFrom(Path.Combine(binPath, "TableFormerSdk.dll"));
var modelsAssembly = System.Reflection.Assembly.LoadFrom(Path.Combine(binPath, "Docling.Models.dll"));

dynamic CreateLayoutService()
{
    var layoutServiceType = modelsAssembly.GetType("Docling.Models.Layout.LayoutDetectionService");
    return Activator.CreateInstance(layoutServiceType);
}

dynamic CreateTableService()
{
    var tableServiceType = modelsAssembly.GetType("Docling.Models.Tables.TableFormerTableStructureService");
    return Activator.CreateInstance(tableServiceType);
}

Console.WriteLine("=" + new string('=', 79));
Console.WriteLine(".NET PIPELINE TEST");
Console.WriteLine("=" + new string('=', 79));

var imagePath = "dataset/2305.03393v1-pg9-img.png";
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
var layoutService = CreateLayoutService();
var layoutRequest = Activator.CreateInstance(
    modelsAssembly.GetType("Docling.Models.Layout.LayoutDetectionRequest"),
    new object[] { bitmap, 300.0f });

var layoutTask = layoutService.DetectAsync(layoutRequest);
layoutTask.Wait();
var layoutResult = layoutTask.Result;
layoutSw.Stop();

var cells = (System.Collections.IList)layoutResult.Cells;
Console.WriteLine($"Layout detection time: {layoutSw.Elapsed.TotalSeconds:F3}s");
Console.WriteLine($"Total cells detected: {cells.Count}");

// Find table cells
var tableCells = new System.Collections.Generic.List<dynamic>();
foreach (var cell in cells)
{
    if (cell.Label == "table")
    {
        tableCells.Add(cell);
    }
}

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

var tableService = CreateTableService();
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
    var pageRef = Activator.CreateInstance(
        modelsAssembly.GetType("Docling.Core.Primitives.PageReference"),
        new object[] { 1, 300.0f });

    var tableRequest = Activator.CreateInstance(
        modelsAssembly.GetType("Docling.Models.Tables.TableStructureRequest"),
        new object[] { pageRef, bbox, imageBytes });

    // Run TableFormer
    var tfSw = Stopwatch.StartNew();
    var tableTask = tableService.InferStructureAsync(tableRequest);
    tableTask.Wait();
    var tableStructure = tableTask.Result;
    tfSw.Stop();

    totalTableTime += tfSw.Elapsed.TotalSeconds;

    var structureCells = (System.Collections.IList)tableStructure.Cells;
    Console.WriteLine($"  TableFormer time: {tfSw.Elapsed.TotalSeconds:F3}s");
    Console.WriteLine($"  Rows detected: {tableStructure.RowCount}");
    Console.WriteLine($"  Columns detected: {tableStructure.ColumnCount}");
    Console.WriteLine($"  Cells detected: {structureCells.Count}");

    // Show first few cells
    if (structureCells.Count > 0)
    {
        Console.WriteLine($"  First 3 cells:");
        for (int i = 0; i < Math.Min(3, structureCells.Count); i++)
        {
            var cell = structureCells[i];
            var cellBbox = cell.BoundingBox;
            Console.WriteLine($"    Cell {i + 1}: BBox=({cellBbox.Left:F1}, {cellBbox.Top:F1}, {cellBbox.Right:F1}, {cellBbox.Bottom:F1}), " +
                            $"RowSpan={cell.RowSpan}, ColSpan={cell.ColumnSpan}");
        }
    }
}

// Summary
Console.WriteLine("\n" + new string('=', 80));
Console.WriteLine(".NET SUMMARY");
Console.WriteLine(new string('=', 80));
Console.WriteLine($"Total layout time: {layoutSw.Elapsed.TotalSeconds:F3}s");
Console.WriteLine($"Total TableFormer time: {totalTableTime:F3}s");
Console.WriteLine($"Total tables processed: {tableCells.Count}");
Console.WriteLine($"Total time: {(layoutSw.Elapsed.TotalSeconds + totalTableTime):F3}s");

return 0;
