using System.Diagnostics;
using DoclingNetSdk;
using Microsoft.Extensions.Logging;

// Parse arguments
if (args.Length == 0)
{
    Console.WriteLine("Docling CLI - Document to Markdown Converter");
    Console.WriteLine();
    Console.WriteLine("Usage: docling-cli <image-path> [options]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --output <path>          Output markdown file path (default: same as input with .md extension)");
    Console.WriteLine("  --artifacts <path>       Artifacts directory (default: ./artifacts)");
    Console.WriteLine("  --language <code>        OCR language code (default: en)");
    Console.WriteLine("  --no-ocr                 Disable OCR");
    Console.WriteLine("  --no-tables              Disable table recognition");
    Console.WriteLine("  --tableformer <variant>  TableFormer variant: Fast, Base, Accurate (default: Fast)");
    Console.WriteLine("  --verbose                Enable verbose logging");
    Console.WriteLine();
    Console.WriteLine("Example:");
    Console.WriteLine("  docling-cli document.png");
    Console.WriteLine("  docling-cli document.png --output result.md --verbose");
    return 1;
}

var imagePath = args[0];
string? outputPath = null;
var artifactsPath = "./artifacts";
var language = "en";
var enableOcr = true;
var enableTables = true;
var variant = TableFormerVariant.Fast;
var verbose = false;

// Parse options
for (int i = 1; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--output":
            if (i + 1 < args.Length)
                outputPath = args[++i];
            break;
        case "--artifacts":
            if (i + 1 < args.Length)
                artifactsPath = args[++i];
            break;
        case "--language":
            if (i + 1 < args.Length)
                language = args[++i];
            break;
        case "--no-ocr":
            enableOcr = false;
            break;
        case "--no-tables":
            enableTables = false;
            break;
        case "--tableformer":
            if (i + 1 < args.Length)
            {
                var v = args[++i];
                variant = v.ToLowerInvariant() switch
                {
                    "fast" => TableFormerVariant.Fast,
                    "base" => TableFormerVariant.Base,
                    "accurate" => TableFormerVariant.Accurate,
                    _ => TableFormerVariant.Fast
                };
            }
            break;
        case "--verbose":
            verbose = true;
            break;
    }
}

// Validate input
if (!File.Exists(imagePath))
{
    Console.Error.WriteLine($"Error: File not found: {imagePath}");
    return 1;
}

// Set output path if not specified
outputPath ??= Path.ChangeExtension(imagePath, ".md");

// Setup logging
using var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(verbose ? LogLevel.Debug : LogLevel.Information);
});
var logger = loggerFactory.CreateLogger<Program>();

try
{
    Console.WriteLine("Docling CLI - Document to Markdown Converter");
    Console.WriteLine("==============================================");
    Console.WriteLine();
    Console.WriteLine($"Input:       {imagePath}");
    Console.WriteLine($"Output:      {outputPath}");
    Console.WriteLine($"Artifacts:   {artifactsPath}");
    Console.WriteLine($"OCR:         {(enableOcr ? $"Enabled ({language})" : "Disabled")}");
    Console.WriteLine($"Tables:      {(enableTables ? $"Enabled ({variant})" : "Disabled")}");
    Console.WriteLine();

    // Create configuration
    var config = new DoclingConfiguration
    {
        ArtifactsPath = artifactsPath,
        OcrLanguage = language,
        EnableOcr = enableOcr,
        EnableTableRecognition = enableTables,
        TableFormerVariant = variant
    };

    // Create converter
    using var converter = new DoclingConverter(config, logger);

    // Convert
    Console.WriteLine("Converting...");
    var sw = Stopwatch.StartNew();
    var result = await converter.ConvertImageAsync(imagePath);
    sw.Stop();

    // Save output
    await File.WriteAllTextAsync(outputPath, result.Markdown);

    // Print statistics
    Console.WriteLine();
    Console.WriteLine("Conversion completed successfully!");
    Console.WriteLine("==================================");
    Console.WriteLine($"Time elapsed:    {sw.Elapsed.TotalSeconds:F2}s");
    Console.WriteLine($"Layout elements: {result.LayoutElementCount}");
    Console.WriteLine($"OCR elements:    {result.OcrElementCount}");
    Console.WriteLine($"Tables:          {result.TableCount}");
    Console.WriteLine($"Total items:     {result.TotalItems}");
    Console.WriteLine($"Markdown size:   {result.Markdown.Length} characters");
    Console.WriteLine();
    Console.WriteLine($"Output saved to: {outputPath}");

    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine();
    Console.Error.WriteLine($"Error: {ex.Message}");
    if (verbose)
    {
        Console.Error.WriteLine();
        Console.Error.WriteLine("Stack trace:");
        Console.Error.WriteLine(ex.ToString());
    }
    return 1;
}
