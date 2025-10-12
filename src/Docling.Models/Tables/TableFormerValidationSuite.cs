using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Docling.Core.Primitives;
using Docling.Core.Geometry;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Comprehensive validation suite for TableFormer table structure recognition.
/// Tests against golden datasets, calculates quality metrics, and generates validation reports.
/// </summary>
internal sealed class TableFormerValidationSuite
{
    private readonly string _modelsDirectory;
    private readonly TableFormerBenchmark _benchmark;

    public TableFormerValidationSuite(string modelsDirectory)
    {
        _modelsDirectory = modelsDirectory ?? throw new ArgumentNullException(nameof(modelsDirectory));
        _benchmark = new TableFormerBenchmark(modelsDirectory);
    }

    /// <summary>
    /// Run complete validation suite against all test documents.
    /// </summary>
    public async Task<ValidationReport> RunCompleteValidationAsync()
    {
        Console.WriteLine("🚀 Starting Complete TableFormer Validation Suite");
        Console.WriteLine($"Models Directory: {_modelsDirectory}");
        Console.WriteLine(new string('-', 80));

        var report = new ValidationReport
        {
            Timestamp = DateTime.UtcNow,
            ModelsDirectory = _modelsDirectory,
            TestResults = new List<TestCaseResult>()
        };

        try
        {
            // Run validation on test dataset
            report.TestResults = await RunGoldenDatasetTestsAsync();

            // Calculate overall metrics
            report.OverallMetrics = CalculateOverallMetrics(report.TestResults);

            // Generate recommendations
            report.Recommendations = GenerateRecommendations(report.OverallMetrics);

            // Print summary
            PrintValidationSummary(report);

            return report;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Validation suite failed: {ex.Message}");
            throw;
        }
    }

    private async Task<List<TestCaseResult>> RunGoldenDatasetTestsAsync()
    {
        var testCases = GetTestCases();
        var results = new List<TestCaseResult>();

        Console.WriteLine($"📊 Running validation on {testCases.Count} test cases...");

        foreach (var testCase in testCases)
        {
            Console.WriteLine($"Testing: {testCase.DocumentName}");

            try
            {
                var result = await RunSingleTestCaseAsync(testCase);
                results.Add(result);

                Console.WriteLine($"  ✅ {testCase.DocumentName}: TEDS={result.TEDS:F3}, mAP={result.MAP:F3}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ❌ {testCase.DocumentName}: Failed - {ex.Message}");

                results.Add(new TestCaseResult
                {
                    DocumentName = testCase.DocumentName,
                    Success = false,
                    Error = ex.Message
                });
            }
        }

        return results;
    }

    private async Task<TestCaseResult> RunSingleTestCaseAsync(TestCase testCase)
    {
        // Load test image and ground truth
        var image = LoadTestImage(testCase.ImagePath);
        var groundTruth = LoadGroundTruth(testCase);

        // Run TableFormer inference
        var service = new TableFormerTableStructureService();
        var request = new TableStructureRequest
        {
            Page = new PageReference(1),
            RasterizedImage = image,
            BoundingBox = testCase.TableBounds
        };

        var predictedStructure = await service.InferStructureAsync(request);

        // Calculate quality metrics
        var qualityMetrics = new QualityMetrics.QualityAssessment();
        if (groundTruth != null)
        {
            qualityMetrics.TEDS = QualityMetrics.CalculateTEDS(
                predictedStructure.Cells, groundTruth, testCase.ImageWidth, testCase.ImageHeight);

            qualityMetrics.MAP = QualityMetrics.CalculateMAP(
                predictedStructure.Cells, groundTruth,
                Enumerable.Repeat(0.9, predictedStructure.Cells.Count).ToArray());

            qualityMetrics.CellAccuracy = QualityMetrics.CalculateCellAccuracy(
                predictedStructure.Cells, groundTruth);
        }

        return new TestCaseResult
        {
            DocumentName = testCase.DocumentName,
            Success = true,
            TEDS = qualityMetrics.TEDS,
            MAP = qualityMetrics.MAP,
            CellPrecision = qualityMetrics.CellAccuracy.Precision,
            CellRecall = qualityMetrics.CellAccuracy.Recall,
            CellF1Score = qualityMetrics.CellAccuracy.F1Score,
            PredictedCells = predictedStructure.Cells.Count,
            GroundTruthCells = groundTruth?.Count ?? 0
        };
    }

    private List<TestCase> GetTestCases()
    {
        // Define test cases - load from golden dataset
        return new List<TestCase>
        {
            new TestCase
            {
                DocumentName = "2305.03393v1-pg9",
                ImagePath = "dataset/2305.03393v1-pg9-img.png",
                GoldenMarkdownPath = "dataset/golden/v0.12.0/2305.03393v1-pg9/python-cli/docling.md",
                TableBounds = new BoundingBox(0, 0, 1200, 800),
                ImageWidth = 1200,
                ImageHeight = 800,
                ExpectedCells = 45
            },
            // Add more test cases as needed
        };
    }

    private byte[] LoadTestImage(string imagePath)
    {
        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Test image not found: {imagePath}");
        }

        return File.ReadAllBytes(imagePath);
    }

    private IReadOnlyList<OtslParser.TableCell>? LoadGroundTruth(TestCase testCase)
    {
        if (string.IsNullOrEmpty(testCase.GoldenMarkdownPath) || !File.Exists(testCase.GoldenMarkdownPath))
        {
            Console.WriteLine($"Warning: Golden markdown not found: {testCase.GoldenMarkdownPath}");
            return null;
        }

        try
        {
            // Read markdown content
            var markdownContent = File.ReadAllText(testCase.GoldenMarkdownPath);

            // Parse tables from markdown
            var tableStructures = MarkdownTableParser.ParseMarkdownTables(markdownContent);

            if (tableStructures.Count == 0)
            {
                Console.WriteLine($"Warning: No tables found in markdown: {testCase.GoldenMarkdownPath}");
                return null;
            }

            // Convert to OTSL cells - use the first table for now
            var otslCells = MarkdownTableParser.ConvertToOtslCells(tableStructures[0]);

            // Calculate spans
            var cellsWithSpans = MarkdownTableParser.CalculateSpans(otslCells);

            Console.WriteLine($"✅ Loaded ground truth: {cellsWithSpans.Count} cells from {testCase.GoldenMarkdownPath}");
            return cellsWithSpans;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading ground truth from {testCase.GoldenMarkdownPath}: {ex.Message}");
            return null;
        }
    }

    private OverallMetrics CalculateOverallMetrics(List<TestCaseResult> results)
    {
        var successfulTests = results.Where(r => r.Success).ToList();

        if (successfulTests.Count == 0)
        {
            return new OverallMetrics
            {
                TotalTests = results.Count,
                SuccessfulTests = 0,
                AverageTEDS = 0.0,
                AverageMAP = 0.0,
                AveragePrecision = 0.0,
                AverageRecall = 0.0,
                AverageF1Score = 0.0
            };
        }

        return new OverallMetrics
        {
            TotalTests = results.Count,
            SuccessfulTests = successfulTests.Count,
            AverageTEDS = successfulTests.Average(r => r.TEDS),
            AverageMAP = successfulTests.Average(r => r.MAP),
            AveragePrecision = successfulTests.Average(r => r.CellPrecision),
            AverageRecall = successfulTests.Average(r => r.CellRecall),
            AverageF1Score = successfulTests.Average(r => r.CellF1Score)
        };
    }

    private List<string> GenerateRecommendations(OverallMetrics metrics)
    {
        var recommendations = new List<string>();

        // TEDS recommendations
        if (metrics.AverageTEDS < 0.7)
        {
            recommendations.Add("⚠️ Low TEDS score - consider using Accurate variant for better structure recognition");
        }
        else if (metrics.AverageTEDS > 0.9)
        {
            recommendations.Add("✅ Excellent TEDS score - current configuration is optimal");
        }

        // Precision/Recall balance
        if (metrics.AveragePrecision > metrics.AverageRecall + 0.1)
        {
            recommendations.Add("⚠️ High precision, low recall - consider lowering confidence threshold");
        }
        else if (metrics.AverageRecall > metrics.AveragePrecision + 0.1)
        {
            recommendations.Add("⚠️ High recall, low precision - consider raising confidence threshold");
        }

        // F1 score recommendations
        if (metrics.AverageF1Score < 0.8)
        {
            recommendations.Add("⚠️ Low F1 score - review table bounds detection and image preprocessing");
        }

        if (recommendations.Count == 0)
        {
            recommendations.Add("✅ All metrics are within acceptable ranges - system is performing well");
        }

        return recommendations;
    }

    private void PrintValidationSummary(ValidationReport report)
    {
        Console.WriteLine("\n" + new string('=', 80));
        Console.WriteLine("📈 VALIDATION REPORT SUMMARY");
        Console.WriteLine(new string('=', 80));

        Console.WriteLine($"Timestamp: {report.Timestamp}");
        Console.WriteLine($"Models Directory: {report.ModelsDirectory}");
        Console.WriteLine($"Total Tests: {report.OverallMetrics.TotalTests}");
        Console.WriteLine($"Successful Tests: {report.OverallMetrics.SuccessfulTests}");

        if (report.OverallMetrics.SuccessfulTests > 0)
        {
            Console.WriteLine("\n📊 QUALITY METRICS:");
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

        Console.WriteLine(new string('=', 80));
    }

    /// <summary>
    /// Test case definition for validation.
    /// </summary>
    private sealed class TestCase
    {
        public string DocumentName { get; set; } = "";
        public string ImagePath { get; set; } = "";
        public string GroundTruthPath { get; set; } = "";
        public string GoldenMarkdownPath { get; set; } = "";
        public BoundingBox TableBounds { get; set; } = new BoundingBox(0, 0, 0, 0);
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public int ExpectedCells { get; set; }
    }

    /// <summary>
    /// Result of a single test case.
    /// </summary>
    public sealed class TestCaseResult
    {
        public string DocumentName { get; set; } = "";
        public bool Success { get; set; }
        public double TEDS { get; set; }
        public double MAP { get; set; }
        public double CellPrecision { get; set; }
        public double CellRecall { get; set; }
        public double CellF1Score { get; set; }
        public int PredictedCells { get; set; }
        public int GroundTruthCells { get; set; }
        public string? Error { get; set; }
    }

    /// <summary>
    /// Overall validation metrics.
    /// </summary>
    public sealed class OverallMetrics
    {
        public int TotalTests { get; set; }
        public int SuccessfulTests { get; set; }
        public double AverageTEDS { get; set; }
        public double AverageMAP { get; set; }
        public double AveragePrecision { get; set; }
        public double AverageRecall { get; set; }
        public double AverageF1Score { get; set; }
    }

    /// <summary>
    /// Complete validation report.
    /// </summary>
    public sealed class ValidationReport
    {
        public DateTime Timestamp { get; set; }
        public string ModelsDirectory { get; set; } = "";
        public List<TestCaseResult> TestResults { get; set; } = new();
        public OverallMetrics OverallMetrics { get; set; } = new();
        public List<string> Recommendations { get; set; } = new();
    }
}