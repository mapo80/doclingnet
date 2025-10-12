using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Image;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Imaging;
using Docling.Export.Serialization;
using Docling.Models.Layout;
using Docling.Models.Ocr;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Assembly;
using Docling.Pipelines.Export;
using Docling.Pipelines.Internal;
using Docling.Pipelines.Layout;
using Docling.Pipelines.Ocr;
using Docling.Pipelines.Options;
using Docling.Pipelines.Serialization;
using Docling.Pipelines.Preprocessing;
using Docling.Pipelines.Tables;
using Docling.Tests.Regression;
using Docling.Tooling.Parity;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using TableFormerSdk.Enums;
using Xunit;
using Xunit.Sdk;

namespace Docling.Tests.Tooling;

public sealed class RegressionParityTests
{
    public static TheoryData<RegressionDatasetId> DatasetCases { get; } = new()
    {
        { RegressionDatasetId.AmtHandbookSample },
        { RegressionDatasetId.Arxiv230503393Page9 },
    };

    [Theory]
    [MemberData(nameof(DatasetCases))]
    public async Task DotNetPipelineMatchesPythonGoldensAsync(RegressionDatasetId datasetId)
    {
        if (UsingStubModels())
        {
            return;
        }

        var dataset = RegressionDatasets.Load(datasetId);
        var golden = ParityGoldenCatalog.TryResolve(dataset);
        Assert.NotNull(golden);

        await using var harness = new RegressionParityHarness(dataset, golden!);
        var run = await harness.ExecuteAsync(CancellationToken.None).ConfigureAwait(false);

        var goldenMarkdown = await File.ReadAllTextAsync(golden!.MarkdownPath, Encoding.UTF8, CancellationToken.None).ConfigureAwait(false);
        var comparison = ParityComparison.Compare(goldenMarkdown, golden.Manifest, run.Markdown, run.Snapshot);
        if (!comparison.HasDifferences)
        {
            await ParityDiffReporter.CleanupAsync(golden.DiffsDirectory).ConfigureAwait(false);
            return;
        }

        var reportPath = await ParityDiffReporter.WriteReportAsync(
            golden,
            dataset,
            comparison,
            run,
            CancellationToken.None).ConfigureAwait(false);

        var failure = new StringBuilder();
        failure.AppendLine($"Parity regression detected for dataset '{dataset.Name}'.");
        failure.AppendLine($"Golden manifest: {golden.ManifestPath}");
        failure.AppendLine($"Diff report: {reportPath}");

        if (!comparison.Markdown.AreEquivalent)
        {
            failure.AppendLine("- Markdown differences detected.");
        }

        if (comparison.AssetDifferences.Count > 0)
        {
            failure.AppendLine("- Asset differences detected.");
        }

        if (comparison.MetadataDifferences.Count > 0)
        {
            failure.AppendLine("- Metadata differences detected.");
        }

        failure.AppendLine();
        failure.Append(comparison.ToDiagnosticString());

        Assert.True(false, failure.ToString());
    }

    private sealed class RegressionParityHarness : IAsyncDisposable
    {
        private readonly RegressionDataset _dataset;
        private readonly string _workingDirectory;
        private readonly string _tableFormerDirectory;

        public RegressionParityHarness(RegressionDataset dataset, ParityGoldenCase golden)
        {
            _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
            ArgumentNullException.ThrowIfNull(golden);
            _workingDirectory = Path.Combine(Path.GetTempPath(), $"docling-parity-{Guid.NewGuid():N}");
            _tableFormerDirectory = Path.Combine(Path.GetTempPath(), $"docling-tableformer-{Guid.NewGuid():N}");
            Directory.CreateDirectory(_workingDirectory);
        }

        public async Task<ParityRunResult> ExecuteAsync(CancellationToken cancellationToken)
        {
            var inputPath = _dataset.AssetPath;
            var extension = Path.GetExtension(inputPath);
            var isPdf = string.Equals(extension, ".pdf", StringComparison.OrdinalIgnoreCase);

            EnsureEasyOcrModels();
            var easyOcrModelsDirectory = ResolveEasyOcrModelDirectory();

            using var store = new PageImageStore();
            var pages = await LoadDocumentAsync(inputPath, store, isPdf, cancellationToken).ConfigureAwait(false);

            var preprocessingOptions = new PreprocessingOptions
            {
                TargetDpi = 144,
                ColorMode = PageColorMode.Preserve,
                EnableDeskew = false,
                NormalizeContrast = true,
            };

            var pipelineOptions = new PdfPipelineOptions
            {
                GeneratePageImages = true,
                GeneratePictureImages = true,
                GenerateImageDebugArtifacts = false,
                Layout = new LayoutOptions
                {
                    Model = LayoutModelConfiguration.DoclingLayoutEgretMedium,
                    CreateOrphanClusters = false,
                    KeepEmptyClusters = true,
                    GenerateDebugArtifacts = false,
                },
                TableStructure = new TableStructureOptions
                {
                    Mode = TableFormerMode.Accurate,
                },
                Ocr = new EasyOcrOptions
                {
                    Languages = new[] { "en" },
                    BitmapAreaThreshold = 0.0005,
                    ForceFullPageOcr = true,
                    ModelStorageDirectory = easyOcrModelsDirectory,
                    DownloadEnabled = false,
                },
            };

            var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
            context.Set(PipelineContextKeys.PageImageStore, store);
            context.Set(PipelineContextKeys.PageSequence, pages);
            context.Set(PipelineContextKeys.DocumentId, _dataset.Name);

            var preprocessor = new DefaultPagePreprocessor(preprocessingOptions, NullLogger<DefaultPagePreprocessor>.Instance);
            var preprocessingStage = new PagePreprocessingStage(preprocessor, NullLogger<PagePreprocessingStage>.Instance);

            using var layoutService = new LayoutSdkDetectionService(
                new LayoutSdkDetectionOptions
                {
                    ValidateModelFiles = true,
                    MaxDegreeOfParallelism = 1,
                },
                NullLogger<LayoutSdkDetectionService>.Instance);

            var layoutStage = new LayoutAnalysisStage(
                layoutService,
                pipelineOptions.Layout,
                NullLogger<LayoutAnalysisStage>.Instance);

            using var tableStructureService = new TableFormerTableStructureService(
                new TableFormerStructureServiceOptions
                {
                    Runtime = TableFormerRuntime.Onnx,
                    Variant = TableFormerModelVariant.Accurate,
                    WorkingDirectory = _tableFormerDirectory,
                },
                NullLogger<TableFormerTableStructureService>.Instance);

            var tableStructureStage = new TableStructureInferenceStage(
                tableStructureService,
                pipelineOptions,
                NullLogger<TableStructureInferenceStage>.Instance);

            var ocrFactory = new OcrServiceFactory();
            var ocrStage = new OcrStage(ocrFactory, pipelineOptions, NullLogger<OcrStage>.Instance);

            var assemblyStage = new PageAssemblyStage(NullLogger<PageAssemblyStage>.Instance);
            using var imageCropService = new ImageCropService();
            var imageExportStage = new ImageExportStage(imageCropService, pipelineOptions, NullLogger<ImageExportStage>.Instance);
            var markdownStage = new MarkdownSerializationStage(new MarkdownDocSerializer(new MarkdownSerializerOptions
            {
                AssetsPath = "assets",
            }));

            var pipeline = new ConvertPipelineBuilder()
                .AddStage(preprocessingStage)
                .AddStage(layoutStage)
                .AddStage(tableStructureStage)
                .AddStage(ocrStage)
                .AddStage(assemblyStage)
                .AddStage(imageExportStage)
                .AddStage(markdownStage)
                .Build(NullLogger<ConvertPipeline>.Instance);

            await pipeline.ExecuteAsync(context, cancellationToken).ConfigureAwait(false);

            var document = context.GetRequired<DoclingDocument>(PipelineContextKeys.Document);
            var serializationResult = context.GetRequired<MarkdownSerializationResult>(PipelineContextKeys.MarkdownSerializationResult);

            var markdownPath = Path.Combine(_workingDirectory, "docling.md");
            await File.WriteAllTextAsync(markdownPath, serializationResult.Markdown, Encoding.UTF8, cancellationToken).ConfigureAwait(false);

            var extractionOptions = new ParityExtractionOptions { BaseDirectory = _workingDirectory };
            var snapshot = ParityResultExtractor.Extract(markdownPath, serializationResult, document, extractionOptions);

            return new ParityRunResult(markdownPath, serializationResult.Markdown, snapshot, _workingDirectory);
        }

        public async ValueTask DisposeAsync()
        {
            try
            {
                if (Directory.Exists(_tableFormerDirectory))
                {
                    Directory.Delete(_tableFormerDirectory, recursive: true);
                }

                if (Directory.Exists(_workingDirectory))
                {
                    Directory.Delete(_workingDirectory, recursive: true);
                }
            }
            catch
            {
                await Task.CompletedTask;
            }
        }

        private static async Task<IReadOnlyList<PageReference>> LoadDocumentAsync(
            string inputPath,
            PageImageStore store,
            bool isPdf,
            CancellationToken cancellationToken)
        {
            if (isPdf)
            {
                return await LoadPdfAsync(inputPath, store, cancellationToken).ConfigureAwait(false);
            }

            return await LoadImageAsync(inputPath, store, cancellationToken).ConfigureAwait(false);
        }

        private static async Task<IReadOnlyList<PageReference>> LoadPdfAsync(
            string inputPath,
            PageImageStore store,
            CancellationToken cancellationToken)
        {
            const int targetDpi = 144;
            const int layoutInputSize = 640;

            var backend = new PdfBackend(
                new PdfToImageRenderer(),
                new PdfBackendOptions
                {
                    StreamFactory = _ => Task.FromResult<Stream>(File.OpenRead(inputPath)),
                    DocumentId = Path.GetFileNameWithoutExtension(inputPath),
                    SourceName = "dataset",
                    RenderSettings = new PdfRenderSettings
                    {
                        Dpi = targetDpi,
                        WithAspectRatio = true,
                        BackgroundColor = SKColors.White,
                    },
                });

            var pages = new List<PageReference>();
            await foreach (var pageImage in backend.LoadAsync(cancellationToken).ConfigureAwait(false))
            {
                using (pageImage)
                {
                    var layoutCompatible = CreateLayoutCompatiblePage(pageImage, layoutInputSize);
                    pages.Add(layoutCompatible.Page);
                    store.Add(layoutCompatible);
                }
            }

            return new ReadOnlyCollection<PageReference>(pages);
        }

        private static async Task<IReadOnlyList<PageReference>> LoadImageAsync(
            string inputPath,
            PageImageStore store,
            CancellationToken cancellationToken)
        {
            const int datasetDpi = 144;
            const int layoutInputSize = 640;

            var resized = LoadLayoutCompatibleImage(inputPath, layoutInputSize);
            var descriptor = new ImageSourceDescriptor
            {
                Identifier = Path.GetFileNameWithoutExtension(inputPath),
                FileName = Path.GetFileName(inputPath),
                MediaType = "image/png",
                Dpi = datasetDpi,
                StreamFactory = _ => Task.FromResult<Stream>(new MemoryStream(resized, writable: false)),
            };

            var backend = new ImageBackend(
                new ImageBackendOptions
                {
                    Sources = new[] { descriptor },
                    DocumentId = descriptor.Identifier,
                    SourceName = "dataset",
                    DefaultDpi = datasetDpi,
                },
                NullLogger<ImageBackend>.Instance);

            var pages = new List<PageReference>();
            await foreach (var page in backend.LoadAsync(cancellationToken).ConfigureAwait(false))
            {
                pages.Add(page.Page);
                store.Add(page);
            }

            return new ReadOnlyCollection<PageReference>(pages);
        }
    }

    private static PageImage CreateLayoutCompatiblePage(PageImage source, int targetSize)
    {
        ArgumentNullException.ThrowIfNull(source);

        var info = new SKImageInfo(targetSize, targetSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        var squareBitmap = new SKBitmap(info);
        try
        {
            using var canvas = new SKCanvas(squareBitmap);
            canvas.Clear(SKColors.White);

            var scale = Math.Min((float)targetSize / source.Width, (float)targetSize / source.Height);
            var scaledWidth = Math.Max(1, (int)Math.Round(source.Width * scale));
            var scaledHeight = Math.Max(1, (int)Math.Round(source.Height * scale));
            var offsetX = (targetSize - scaledWidth) / 2f;
            var offsetY = (targetSize - scaledHeight) / 2f;
            var destination = SKRect.Create(offsetX, offsetY, scaledWidth, scaledHeight);

            var scaledInfo = new SKImageInfo(scaledWidth, scaledHeight, source.Bitmap.ColorType, source.Bitmap.AlphaType);
            using var scaled = source.Bitmap.Resize(scaledInfo, new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None))
                ?? throw new InvalidOperationException("Failed to resize PDF page to layout model input.");

            canvas.DrawBitmap(scaled, destination);
            canvas.Flush();

            var clone = squareBitmap.Copy()
                ?? throw new InvalidOperationException("Failed to clone layout-compatible bitmap.");

            return new PageImage(source.Page, clone, source.Metadata);
        }
        finally
        {
            squareBitmap.Dispose();
            source.Dispose();
        }
    }

    private static byte[] LoadLayoutCompatibleImage(string imagePath, int targetSize)
    {
        using var original = SKBitmap.Decode(imagePath)
            ?? throw new InvalidOperationException($"Unable to decode dataset image '{imagePath}'.");

        var info = new SKImageInfo(targetSize, targetSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var squareBitmap = new SKBitmap(info);
        using var canvas = new SKCanvas(squareBitmap);
        canvas.Clear(SKColors.White);

        var scale = Math.Min((float)targetSize / original.Width, (float)targetSize / original.Height);
        var scaledWidth = Math.Max(1, (int)Math.Round(original.Width * scale));
        var scaledHeight = Math.Max(1, (int)Math.Round(original.Height * scale));
        var offsetX = (targetSize - scaledWidth) / 2f;
        var offsetY = (targetSize - scaledHeight) / 2f;
        var destination = SKRect.Create(offsetX, offsetY, scaledWidth, scaledHeight);

        var scaledInfo = new SKImageInfo(scaledWidth, scaledHeight, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var scaled = original.Resize(scaledInfo, new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None))
            ?? throw new InvalidOperationException("Failed to resize dataset image to layout model input.");

        canvas.DrawBitmap(scaled, destination);
        canvas.Flush();

        using var image = SKImage.FromBitmap(squareBitmap);
        using var encoded = image.Encode(SKEncodedImageFormat.Png, 100)
            ?? throw new InvalidOperationException("Failed to encode resized dataset image.");

        return encoded.ToArray();
    }

    private static void EnsureEasyOcrModels()
    {
        var targetRoots = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "contentFiles", "any", "any", "models"),
            Path.Combine(AppContext.BaseDirectory, "models", "easyocr"),
        };

        foreach (var target in targetRoots)
        {
            if (Directory.Exists(target) && Directory.EnumerateFiles(target, "*", SearchOption.AllDirectories).Any())
            {
                return;
            }
        }

        var assemblyLocation = typeof(EasyOcrService).Assembly.Location;
        var assemblyDirectory = Path.GetDirectoryName(assemblyLocation)
            ?? throw new InvalidOperationException("Unable to determine the Docling.Models assembly directory.");

        var sourceRoots = new[]
        {
            Path.Combine(assemblyDirectory, "contentFiles", "any", "any", "models"),
            Path.Combine(assemblyDirectory, "models", "easyocr"),
            Path.Combine(assemblyDirectory, "models"),
        };

        var sourceRoot = sourceRoots.FirstOrDefault(Directory.Exists);
        if (sourceRoot is null)
        {
            throw new DirectoryNotFoundException($"EasyOCR model directory not found under '{assemblyDirectory}'. Ensure the package is restored.");
        }

        var destination = targetRoots[^1];
        var normalizedSource = Path.GetFullPath(sourceRoot);
        var normalizedDestination = Path.GetFullPath(destination);

        if (!string.Equals(normalizedSource, normalizedDestination, StringComparison.OrdinalIgnoreCase))
        {
            if (!Directory.Exists(destination) ||
                !Directory.EnumerateFiles(destination, "*", SearchOption.AllDirectories).Any())
            {
                CopyDirectoryRecursively(sourceRoot, destination);
            }
        }

        var compatibilityRoot = targetRoots[0];
        var normalizedCompatibility = Path.GetFullPath(compatibilityRoot);
        if (!string.Equals(normalizedSource, normalizedCompatibility, StringComparison.OrdinalIgnoreCase))
        {
            if (!Directory.Exists(compatibilityRoot) ||
                !Directory.EnumerateFiles(compatibilityRoot, "*", SearchOption.AllDirectories).Any())
            {
                CopyDirectoryRecursively(sourceRoot, compatibilityRoot);
            }
        }

        var legacyOnnx = Path.Combine(compatibilityRoot, "onnx");
        var normalizedLegacy = Path.GetFullPath(legacyOnnx);
        if (!string.Equals(normalizedDestination, normalizedLegacy, StringComparison.OrdinalIgnoreCase))
        {
            if (!Directory.Exists(legacyOnnx) ||
                !Directory.EnumerateFiles(legacyOnnx, "*", SearchOption.AllDirectories).Any())
            {
                CopyDirectoryRecursively(destination, legacyOnnx);
            }
        }
    }

    private static string ResolveEasyOcrModelDirectory()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "contentFiles", "any", "any", "models"),
            Path.Combine(AppContext.BaseDirectory, "models", "easyocr"),
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        return candidates[^1];
    }

    private static bool UsingStubModels()
    {
        var assemblyDirectory = Path.GetDirectoryName(typeof(LayoutSdkDetectionService).Assembly.Location);
        if (string.IsNullOrEmpty(assemblyDirectory))
        {
            return true;
        }

        var sentinels = new[]
        {
            Path.Combine(assemblyDirectory, "models", "easyocr", "detection.onnx"),
            Path.Combine(assemblyDirectory, "models", "tableformer", "encoder.onnx"),
        };

        foreach (var sentinel in sentinels)
        {
            if (!File.Exists(sentinel))
            {
                continue;
            }

            if (new FileInfo(sentinel).Length == 0)
            {
                return true;
            }
        }

        return false;
    }

    private static void CopyDirectoryRecursively(string source, string destination)
    {
        foreach (var directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
        {
            var relativeDirectory = Path.GetRelativePath(source, directory);
            Directory.CreateDirectory(Path.Combine(destination, relativeDirectory));
        }

        foreach (var file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
        {
            var relativeFile = Path.GetRelativePath(source, file);
            var targetPath = Path.Combine(destination, relativeFile);
            Directory.CreateDirectory(Path.GetDirectoryName(targetPath)!);
            File.Copy(file, targetPath, overwrite: true);
        }
    }
}

internal sealed record ParityRunResult(string MarkdownPath, string Markdown, ParityExtractionResult Snapshot, string WorkingDirectory);

internal static class ParityComparison
{
    public static ParityComparisonReport Compare(
        string expectedMarkdown,
        ParityGoldenManifest manifest,
        string actualMarkdown,
        ParityExtractionResult actual)
    {
        var markdownDiff = CompareMarkdown(expectedMarkdown, actualMarkdown);
        var assetDiffs = CompareAssets(manifest, actual);
        var metadataDiffs = CompareMetadata(manifest, actual);
        return new ParityComparisonReport(markdownDiff, assetDiffs, metadataDiffs);
    }

    private static MarkdownDiffResult CompareMarkdown(string expected, string actual)
    {
        var options = MarkdownComparisonOptions.Default;
        var expectedLines = NormalizeMarkdown(expected ?? string.Empty, options);
        var actualLines = NormalizeMarkdown(actual ?? string.Empty, options);

        var differences = new List<MarkdownDifference>();
        var max = Math.Max(expectedLines.Count, actualLines.Count);
        for (var index = 0; index < max; index++)
        {
            var expectedLine = index < expectedLines.Count ? expectedLines[index] : null;
            var actualLine = index < actualLines.Count ? actualLines[index] : null;
            if (!string.Equals(expectedLine, actualLine, StringComparison.Ordinal))
            {
                differences.Add(new MarkdownDifference(index + 1, expectedLine, actualLine));
            }
        }

        var expectedNormalized = string.Join("\n", expectedLines);
        var actualNormalized = string.Join("\n", actualLines);
        return new MarkdownDiffResult(differences.Count == 0, differences, expectedNormalized, actualNormalized);
    }

    private static IReadOnlyList<string> NormalizeMarkdown(string text, MarkdownComparisonOptions options)
    {
        if (string.IsNullOrEmpty(text))
        {
            return Array.Empty<string>();
        }

        var normalized = text
            .Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace('\r', '\n');

        var lines = normalized.Split('\n');
        var builder = new List<string>(lines.Length);
        foreach (var rawLine in lines)
        {
            var line = options.IgnoreTrailingWhitespace ? rawLine.TrimEnd() : rawLine;

            if (options.CollapseSequentialBlankLines && line.Length == 0)
            {
                if (builder.Count > 0 && builder[^1].Length == 0)
                {
                    continue;
                }
            }

            builder.Add(line);
        }

        if (options.TrimFinalBlankLines)
        {
            while (builder.Count > 0 && builder[^1].Length == 0)
            {
                builder.RemoveAt(builder.Count - 1);
            }
        }

        return new ReadOnlyCollection<string>(builder);
    }

    private static IReadOnlyList<string> CompareAssets(ParityGoldenManifest manifest, ParityExtractionResult actual)
    {
        var differences = new List<string>();

        var goldenAssets = manifest.Assets.ToDictionary(asset => NormalizePath(asset.RelativePath), StringComparer.Ordinal);
        var actualAssets = actual.Assets.ToDictionary(asset => NormalizePath(asset.RelativePath), StringComparer.Ordinal);

        foreach (var (path, golden) in goldenAssets)
        {
            if (!actualAssets.TryGetValue(path, out var actualAsset))
            {
                differences.Add($"Missing asset '{path}' in .NET output.");
                continue;
            }

            if (!string.Equals(golden.Kind, actualAsset.Kind, StringComparison.OrdinalIgnoreCase))
            {
                differences.Add($"Asset '{path}': expected kind '{golden.Kind}' but found '{actualAsset.Kind}'.");
            }

            if (!string.Equals(golden.Checksum, actualAsset.Checksum, StringComparison.OrdinalIgnoreCase))
            {
                differences.Add($"Asset '{path}': checksum mismatch (expected {golden.Checksum}, actual {actualAsset.Checksum}).");
            }

            if (golden.Width != actualAsset.Width || golden.Height != actualAsset.Height)
            {
                differences.Add($"Asset '{path}': expected size {golden.Width}x{golden.Height} but found {actualAsset.Width}x{actualAsset.Height}.");
            }

            if (!NearlyEquals(golden.Dpi, actualAsset.Dpi))
            {
                differences.Add($"Asset '{path}': expected DPI {golden.Dpi.ToString(CultureInfo.InvariantCulture)} but found {actualAsset.Dpi.ToString(CultureInfo.InvariantCulture)}.");
            }

            if (Math.Abs(golden.PageNumber - actualAsset.PageNumber) > 0)
            {
                differences.Add($"Asset '{path}': expected page {golden.PageNumber} but found {actualAsset.PageNumber}.");
            }

            if (!BoundingBoxesMatch(golden.BoundingBox, actualAsset.BoundingBox))
            {
                differences.Add($"Asset '{path}': bounding box mismatch (expected {golden.BoundingBox}, actual {actualAsset.BoundingBox}).");
            }
        }

        foreach (var path in actualAssets.Keys)
        {
            if (!goldenAssets.ContainsKey(path))
            {
                differences.Add($"Unexpected asset '{path}' emitted by .NET pipeline.");
            }
        }

        return differences;
    }

    private static IReadOnlyList<string> CompareMetadata(ParityGoldenManifest manifest, ParityExtractionResult actual)
    {
        var differences = new List<string>();

        if (!string.Equals(manifest.MarkdownSha256, actual.MarkdownSha256, StringComparison.OrdinalIgnoreCase))
        {
            differences.Add($"Markdown checksum mismatch (expected {manifest.MarkdownSha256}, actual {actual.MarkdownSha256}).");
        }

        if (manifest.Pages.Count != actual.Pages.Count)
        {
            differences.Add($"Page count mismatch (expected {manifest.Pages.Count}, actual {actual.Pages.Count}).");
        }

        var goldenPages = manifest.Pages.OrderBy(page => page.PageNumber).ToList();
        var actualPages = actual.Pages.OrderBy(page => page.PageNumber).ToList();
        var count = Math.Min(goldenPages.Count, actualPages.Count);
        for (var index = 0; index < count; index++)
        {
            var golden = goldenPages[index];
            var actualPage = actualPages[index];
            if (golden.PageNumber != actualPage.PageNumber)
            {
                differences.Add($"Page index {index}: expected page number {golden.PageNumber} but found {actualPage.PageNumber}.");
            }

            if (!NearlyEquals(golden.Dpi, actualPage.Dpi))
            {
                differences.Add($"Page {golden.PageNumber}: expected DPI {golden.Dpi.ToString(CultureInfo.InvariantCulture)} but found {actualPage.Dpi.ToString(CultureInfo.InvariantCulture)}.");
            }
        }

        return differences;
    }

    private static bool BoundingBoxesMatch(ParityGoldenBoundingBox expected, ParityBoundingBox actual)
    {
        return NearlyEquals(expected.Left, actual.Left)
            && NearlyEquals(expected.Top, actual.Top)
            && NearlyEquals(expected.Right, actual.Right)
            && NearlyEquals(expected.Bottom, actual.Bottom);
    }

    private static bool NearlyEquals(double expected, double actual)
    {
        if (double.IsNaN(expected) && double.IsNaN(actual))
        {
            return true;
        }

        return Math.Abs(expected - actual) <= 0.05;
    }

    private static string NormalizePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        var normalized = path.Replace('\\', '/');
        if (normalized.StartsWith("./", StringComparison.Ordinal))
        {
            normalized = normalized[2..];
        }

        return normalized;
    }
}

internal sealed record ParityComparisonReport(
    MarkdownDiffResult Markdown,
    IReadOnlyList<string> AssetDifferences,
    IReadOnlyList<string> MetadataDifferences)
{
    public bool HasDifferences => !Markdown.AreEquivalent || AssetDifferences.Count > 0 || MetadataDifferences.Count > 0;

    public string ToDiagnosticString()
    {
        var builder = new StringBuilder();
        if (!Markdown.AreEquivalent)
        {
            builder.AppendLine("# Markdown differences");
            foreach (var diff in Markdown.Differences)
            {
                builder.AppendLine($"Line {diff.LineNumber}: expected='{diff.Expected}' actual='{diff.Actual}'");
            }

            builder.AppendLine();
        }

        if (AssetDifferences.Count > 0)
        {
            builder.AppendLine("# Asset differences");
            foreach (var diff in AssetDifferences)
            {
                builder.AppendLine("- " + diff);
            }

            builder.AppendLine();
        }

        if (MetadataDifferences.Count > 0)
        {
            builder.AppendLine("# Metadata differences");
            foreach (var diff in MetadataDifferences)
            {
                builder.AppendLine("- " + diff);
            }

            builder.AppendLine();
        }

        return builder.ToString();
    }
}

internal static class ParityDiffReporter
{
    private const string ReportFileName = "dotnet-vs-python.md";

    public static async Task<string> WriteReportAsync(
        ParityGoldenCase golden,
        RegressionDataset dataset,
        ParityComparisonReport comparison,
        ParityRunResult run,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(golden.DiffsDirectory);
        var reportPath = Path.Combine(golden.DiffsDirectory, ReportFileName);
        using var stream = File.Create(reportPath);
        await using var writer = new StreamWriter(stream, Encoding.UTF8, leaveOpen: false);

        await writer.WriteLineAsync($"# Regression report – {dataset.Name}").ConfigureAwait(false);
        await writer.WriteLineAsync($"- Golden manifest: {golden.ManifestPath}").ConfigureAwait(false);
        await writer.WriteLineAsync($"- .NET markdown: {run.MarkdownPath}").ConfigureAwait(false);
        await writer.WriteLineAsync();

        if (!comparison.Markdown.AreEquivalent)
        {
            await writer.WriteLineAsync("## Markdown differences").ConfigureAwait(false);
            await writer.WriteLineAsync("```diff").ConfigureAwait(false);
            foreach (var diff in comparison.Markdown.Differences)
            {
                var expected = diff.Expected ?? string.Empty;
                var actual = diff.Actual ?? string.Empty;
                await writer.WriteLineAsync($"- {diff.LineNumber}: {expected}").ConfigureAwait(false);
                await writer.WriteLineAsync($"+ {diff.LineNumber}: {actual}").ConfigureAwait(false);
            }

            await writer.WriteLineAsync("```").ConfigureAwait(false);
            await writer.WriteLineAsync();
        }

        if (comparison.AssetDifferences.Count > 0)
        {
            await writer.WriteLineAsync("## Asset differences").ConfigureAwait(false);
            foreach (var diff in comparison.AssetDifferences)
            {
                await writer.WriteLineAsync("- " + diff).ConfigureAwait(false);
            }

            await writer.WriteLineAsync();
        }

        if (comparison.MetadataDifferences.Count > 0)
        {
            await writer.WriteLineAsync("## Metadata differences").ConfigureAwait(false);
            foreach (var diff in comparison.MetadataDifferences)
            {
                await writer.WriteLineAsync("- " + diff).ConfigureAwait(false);
            }

            await writer.WriteLineAsync();
        }

        await writer.FlushAsync().ConfigureAwait(false);
        return reportPath;
    }

    public static Task CleanupAsync(string diffsDirectory)
    {
        if (!Directory.Exists(diffsDirectory))
        {
            return Task.CompletedTask;
        }

        var reportPath = Path.Combine(diffsDirectory, ReportFileName);
        if (File.Exists(reportPath))
        {
            File.Delete(reportPath);
        }

        if (!Directory.EnumerateFileSystemEntries(diffsDirectory).Any())
        {
            Directory.Delete(diffsDirectory);
        }

        return Task.CompletedTask;
    }
}

internal sealed record ParityGoldenCase(
    string CaseId,
    string MarkdownPath,
    string ManifestPath,
    string DiffsDirectory,
    ParityGoldenManifest Manifest);

internal static class ParityGoldenCatalog
{
    private const string DefaultGoldenRoot = "dataset/golden";
    private const string ManifestFileName = "manifest.json";
    private const string MarkdownFileName = "docling.md";

    public static ParityGoldenCase? TryResolve(RegressionDataset dataset)
    {
        ArgumentNullException.ThrowIfNull(dataset);

        var goldenRoot = Environment.GetEnvironmentVariable("DOCLING_PARITY_GOLDEN_ROOT");
        if (string.IsNullOrWhiteSpace(goldenRoot))
        {
            goldenRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", DefaultGoldenRoot));
        }

        if (!Directory.Exists(goldenRoot))
        {
            return null;
        }

        var assetFileName = Path.GetFileName(dataset.AssetPath);
        foreach (var manifestPath in Directory.EnumerateFiles(goldenRoot, ManifestFileName, SearchOption.AllDirectories))
        {
            var pythonCliDirectory = Path.GetDirectoryName(manifestPath);
            if (pythonCliDirectory is null)
            {
                continue;
            }

            var caseDirectory = Directory.GetParent(pythonCliDirectory)?.FullName;
            if (caseDirectory is null)
            {
                continue;
            }

            var sourceDirectory = Path.Combine(caseDirectory, "source");
            if (!Directory.Exists(sourceDirectory))
            {
                continue;
            }

            var matchesAsset = Directory.EnumerateFiles(sourceDirectory, "*", SearchOption.TopDirectoryOnly)
                .Any(file => string.Equals(Path.GetFileName(file), assetFileName, StringComparison.OrdinalIgnoreCase));
            if (!matchesAsset)
            {
                continue;
            }

            var markdownPath = Path.Combine(pythonCliDirectory, MarkdownFileName);
            if (!File.Exists(markdownPath))
            {
                continue;
            }

            var manifest = ParityGoldenManifest.Load(manifestPath);
            var diffsDirectory = Path.Combine(caseDirectory, "diffs");
            return new ParityGoldenCase(manifest.CaseId, markdownPath, manifestPath, diffsDirectory, manifest);
        }

        return null;
    }
}

internal sealed class ParityGoldenManifest
{
    private ParityGoldenManifest(
        string caseId,
        string markdownSha256,
        IReadOnlyList<ParityGoldenAsset> assets,
        IReadOnlyList<ParityGoldenPage> pages)
    {
        CaseId = caseId;
        MarkdownSha256 = markdownSha256;
        Assets = assets;
        Pages = pages;
    }

    public string CaseId { get; }

    public string MarkdownSha256 { get; }

    public IReadOnlyList<ParityGoldenAsset> Assets { get; }

    public IReadOnlyList<ParityGoldenPage> Pages { get; }

    public static ParityGoldenManifest Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        var caseId = root.TryGetProperty("caseId", out var caseIdProperty)
            ? caseIdProperty.GetString() ?? string.Empty
            : string.Empty;
        var markdownSha = root.TryGetProperty("markdownSha256", out var markdownProperty)
            ? markdownProperty.GetString() ?? string.Empty
            : string.Empty;

        var assets = new List<ParityGoldenAsset>();
        if (root.TryGetProperty("assets", out var assetsArray) && assetsArray.ValueKind == JsonValueKind.Array)
        {
            foreach (var element in assetsArray.EnumerateArray())
            {
                assets.Add(ParseAsset(element));
            }
        }

        var pages = new List<ParityGoldenPage>();
        if (root.TryGetProperty("pages", out var pagesArray) && pagesArray.ValueKind == JsonValueKind.Array)
        {
            foreach (var element in pagesArray.EnumerateArray())
            {
                pages.Add(ParsePage(element));
            }
        }

        return new ParityGoldenManifest(caseId, markdownSha, assets, pages);
    }

    private static ParityGoldenAsset ParseAsset(JsonElement element)
    {
        var relativePath = element.TryGetProperty("path", out var pathProperty)
            ? pathProperty.GetString()
            : element.TryGetProperty("fileName", out var fileNameProperty)
                ? fileNameProperty.GetString()
                : null;

        relativePath ??= string.Empty;

        var kind = element.TryGetProperty("kind", out var kindProperty)
            ? kindProperty.GetString() ?? string.Empty
            : string.Empty;

        var checksum = element.TryGetProperty("sha256", out var checksumProperty)
            ? checksumProperty.GetString() ?? string.Empty
            : string.Empty;

        var pageNumber = element.TryGetProperty("page", out var pageProperty)
            ? pageProperty.GetInt32()
            : element.TryGetProperty("pageNumber", out var pageNumberProperty)
                ? pageNumberProperty.GetInt32()
                : 0;

        var width = element.TryGetProperty("widthPx", out var widthProperty)
            ? widthProperty.GetInt32()
            : element.TryGetProperty("width", out var widthAlternative)
                ? widthAlternative.GetInt32()
                : 0;

        var height = element.TryGetProperty("heightPx", out var heightProperty)
            ? heightProperty.GetInt32()
            : element.TryGetProperty("height", out var heightAlternative)
                ? heightAlternative.GetInt32()
                : 0;

        var dpi = element.TryGetProperty("dpi", out var dpiProperty)
            ? dpiProperty.GetDouble()
            : 0d;

        var boundingBox = element.TryGetProperty("boundingBox", out var boundingBoxProperty)
            ? ParseBoundingBox(boundingBoxProperty)
            : new ParityGoldenBoundingBox(0, 0, 0, 0);

        return new ParityGoldenAsset(
            relativePath,
            kind,
            checksum,
            pageNumber,
            width,
            height,
            dpi,
            boundingBox);
    }

    private static ParityGoldenPage ParsePage(JsonElement element)
    {
        var pageNumber = element.TryGetProperty("pageNumber", out var pageNumberProperty)
            ? pageNumberProperty.GetInt32()
            : element.TryGetProperty("page", out var pageProperty)
                ? pageProperty.GetInt32()
                : 0;

        var dpi = element.TryGetProperty("dpi", out var dpiProperty)
            ? dpiProperty.GetDouble()
            : 0d;

        return new ParityGoldenPage(pageNumber, dpi);
    }

    private static ParityGoldenBoundingBox ParseBoundingBox(JsonElement element)
    {
        double ReadDouble(string propertyName)
        {
            return element.TryGetProperty(propertyName, out var property) && property.TryGetDouble(out var value)
                ? value
                : 0d;
        }

        return new ParityGoldenBoundingBox(
            ReadDouble("left"),
            ReadDouble("top"),
            ReadDouble("right"),
            ReadDouble("bottom"));
    }
}

internal sealed record ParityGoldenAsset(
    string RelativePath,
    string Kind,
    string Checksum,
    int PageNumber,
    int Width,
    int Height,
    double Dpi,
    ParityGoldenBoundingBox BoundingBox);

internal sealed record ParityGoldenPage(int PageNumber, double Dpi);

internal readonly record struct ParityGoldenBoundingBox(double Left, double Top, double Right, double Bottom);
