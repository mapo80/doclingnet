using DoclingNetSdk;

// Example: Basic usage of DoclingNetSdk

Console.WriteLine("DoclingNetSdk - Basic Usage Example");
Console.WriteLine("====================================\n");

// 1. Check if we have a test image
var testImage = args.Length > 0 ? args[0] : "../../dataset/2305.03393v1-pg9-img.png";

if (!File.Exists(testImage))
{
    Console.WriteLine($"Error: Test image not found: {testImage}");
    Console.WriteLine("\nUsage: dotnet run <path-to-image>");
    return 1;
}

Console.WriteLine($"Input image: {testImage}\n");

try
{
    // 2. Create configuration with auto-detection
    Console.WriteLine("Creating configuration...");
    var config = DoclingConfiguration.CreateDefault();
    Console.WriteLine($"  Artifacts: {config.ArtifactsPath}");
    Console.WriteLine($"  OCR language: {config.OcrLanguage}\n");

    // 3. Create converter
    Console.WriteLine("Initializing DoclingConverter...");
    using var converter = new DoclingConverter(config);
    Console.WriteLine("  ✓ Converter initialized\n");

    // 4. Convert image to markdown
    Console.WriteLine("Converting image to markdown...");
    Console.WriteLine("  (This may take 5-10 seconds on first run due to model downloads)\n");

    var result = await converter.ConvertImageAsync(testImage).ConfigureAwait(false);

    // 5. Display results
    Console.WriteLine("\n✓ Conversion completed!\n");
    Console.WriteLine("=== STATISTICS ===");
    Console.WriteLine($"  Layout elements: {result.LayoutElementCount}");
    Console.WriteLine($"  OCR elements: {result.OcrElementCount}");
    Console.WriteLine($"  Tables: {result.TableCount}");
    Console.WriteLine($"  Total items: {result.TotalItems}");
    Console.WriteLine($"  Markdown length: {result.Markdown.Length} characters\n");

    // 6. Show document structure
    Console.WriteLine("=== DOCUMENT STRUCTURE ===");
    foreach (var item in result.Document.Items.Take(10))
    {
        Console.WriteLine($"  [{item.Kind}] ID: {item.Id}");
    }
    if (result.TotalItems > 10)
    {
        Console.WriteLine($"  ... and {result.TotalItems - 10} more items\n");
    }

    // 7. Save markdown to file
    var outputPath = Path.ChangeExtension(testImage, ".md");
    await File.WriteAllTextAsync(outputPath, result.Markdown).ConfigureAwait(false);
    Console.WriteLine($"\n✓ Markdown saved to: {outputPath}\n");

    // 8. Display markdown preview
    Console.WriteLine("=== MARKDOWN PREVIEW (first 500 chars) ===");
    var preview = result.Markdown.Length > 500
        ? string.Concat(result.Markdown.AsSpan(0, 500), "...")
        : result.Markdown;
    Console.WriteLine(preview);
    Console.WriteLine("\n===========================================\n");

    return 0;
}
catch (Exception ex)
{
    Console.WriteLine($"\n✗ Error: {ex.Message}");
    Console.WriteLine($"  Type: {ex.GetType().Name}");
    if (ex.InnerException != null)
    {
        Console.WriteLine($"  Inner: {ex.InnerException.Message}");
    }
    return 1;
}
