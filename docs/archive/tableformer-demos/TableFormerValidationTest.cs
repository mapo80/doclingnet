#if false
using System;
using System.IO;
using System.Linq;
using Docling.Core.Models.Tables;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("🚀 TableFormer Validation Suite - Phase 8 Test");
        Console.WriteLine("==============================================");

        try
        {
            // Test markdown parsing first
            await TestMarkdownParsing();

            // Test validation suite
            await TestValidationSuite();

            Console.WriteLine("✅ All validation tests completed successfully!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Test failed: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }

    static async Task TestMarkdownParsing()
    {
        Console.WriteLine("\n📄 Testing Markdown Table Parser");
        Console.WriteLine("-------------------------------");

        // Read the golden markdown file
        var markdownPath = "dataset/golden/v0.12.0/2305.03393v1-pg9/python-cli/docling.md";
        if (!File.Exists(markdownPath))
        {
            Console.WriteLine($"❌ Golden markdown file not found: {markdownPath}");
            return;
        }

        var markdownContent = File.ReadAllText(markdownPath);
        Console.WriteLine($"✅ Loaded markdown content: {markdownContent.Length} characters");

        // Parse tables from markdown
        var tables = MarkdownTableParser.ParseMarkdownTables(markdownContent);

        Console.WriteLine($"✅ Found {tables.Count} table(s) in markdown");

        foreach (var (table, index) in tables.Select((t, i) => (t, i)))
        {
            Console.WriteLine($"   Table {index + 1}: {table.TotalRows} rows, {table.TotalColumns} columns, {table.Cells.Count} cells");
        }
    }

    static async Task TestValidationSuite()
    {
        Console.WriteLine("\n🔬 Testing Validation Suite");
        Console.WriteLine("--------------------------");

        // Create validation suite
        var validationSuite = new TableFormerValidationSuite("src/submodules/ds4sd-docling-tableformer-onnx/models");
        Console.WriteLine("✅ Validation suite initialized");

        // Run complete validation
        var report = await validationSuite.RunCompleteValidationAsync();

        // Display results
        Console.WriteLine("\n📊 VALIDATION RESULTS:");
        Console.WriteLine($"   Total Tests: {report.OverallMetrics.TotalTests}");
        Console.WriteLine($"   Successful Tests: {report.OverallMetrics.SuccessfulTests}");

        if (report.OverallMetrics.SuccessfulTests > 0)
        {
            Console.WriteLine($"   Average TEDS: {report.OverallMetrics.AverageTEDS:F3}");
            Console.WriteLine($"   Average mAP: {report.OverallMetrics.AverageMAP:F3}");
            Console.WriteLine($"   Average Precision: {report.OverallMetrics.AveragePrecision:F3}");
            Console.WriteLine($"   Average Recall: {report.OverallMetrics.AverageRecall:F3}");
            Console.WriteLine($"   Average F1 Score: {report.OverallMetrics.AverageF1Score:F3}");

            Console.WriteLine("\n💡 RECOMMENDATIONS:");
            foreach (var recommendation in report.Recommendations)
            {
                Console.WriteLine($"   {recommendation}");
            }
        }

        Console.WriteLine("\n✅ Validation suite test completed");
    }
}
#endif
