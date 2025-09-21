using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Image;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Layout;
using Docling.Models.Ocr;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Internal;
using Docling.Pipelines.Layout;
using Docling.Pipelines.Ocr;
using Docling.Pipelines.Options;
using Docling.Pipelines.Preprocessing;
using Docling.Pipelines.Tables;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using TableFormerSdk.Enums;

namespace Docling.Tests.Integration;

public sealed class PipelineIntegrationTests
{
    [Fact]
    public async Task ConvertPipelineProcessesDatasetImage()
    {
        var imagePath = GetDatasetAssetPath("2305.03393v1-pg9-img.png");
        File.Exists(imagePath).Should().BeTrue("the dataset PNG must be copied next to the test binaries");

        EnsureEasyOcrModels();
        var easyOcrModelDirectory = Path.Combine(AppContext.BaseDirectory, "contentFiles", "any", "any", "models", "onnx");
        Directory.Exists(easyOcrModelDirectory).Should().BeTrue("EasyOCR models must be available for the integration test");

        const int layoutInputSize = 640;
        const int datasetDpi = 144;
        var resizedImage = LoadLayoutCompatibleImage(imagePath, layoutInputSize);

        var descriptor = new ImageSourceDescriptor
        {
            Identifier = "dataset-page-0",
            FileName = Path.GetFileName(imagePath),
            MediaType = "image/png",
            Dpi = datasetDpi,
            StreamFactory = _ => Task.FromResult<Stream>(new MemoryStream(resizedImage, writable: false)),
        };

        var backend = new ImageBackend(
            new ImageBackendOptions
            {
                Sources = new[] { descriptor },
                DocumentId = "dataset-doc",
                SourceName = "dataset",
                DefaultDpi = datasetDpi,
                Metadata = new Dictionary<string, string>
                {
                    ["origin"] = "integration-test",
                },
            },
            NullLogger<ImageBackend>.Instance);

        using var store = new PageImageStore();
        var pages = new List<PageReference>();

        await foreach (var page in backend.LoadAsync(CancellationToken.None))
        {
            pages.Add(page.Page);
            store.Add(page);
        }

        pages.Should().HaveCount(1);
        var originalPage = pages[0];

        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, pages);
        context.Set(PipelineContextKeys.DocumentId, "dataset-integration");

        var preprocessingOptions = new PreprocessingOptions
        {
            TargetDpi = datasetDpi,
            ColorMode = PageColorMode.Preserve,
            EnableDeskew = false,
            NormalizeContrast = true,
        };

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
            new LayoutOptions
            {
                Model = LayoutModelConfiguration.DoclingLayoutEgretMedium,
                CreateOrphanClusters = false,
                KeepEmptyClusters = true,
            },
            NullLogger<LayoutAnalysisStage>.Instance);

        var ocrFactory = new OcrServiceFactory();

        var tableFormerWorkingDirectory = Path.Combine(Path.GetTempPath(), $"docling-tableformer-{Guid.NewGuid():N}");
        var tableFormerOptions = new TableFormerStructureServiceOptions
        {
            Runtime = TableFormerRuntime.Onnx,
            Variant = TableFormerModelVariant.Accurate,
            WorkingDirectory = tableFormerWorkingDirectory,
        };

        var pdfOptions = new PdfPipelineOptions
        {
            Ocr = new EasyOcrOptions
            {
                Languages = new[] { "en" },
                BitmapAreaThreshold = 0.0005,
                ForceFullPageOcr = true,
                ModelStorageDirectory = easyOcrModelDirectory,
                DownloadEnabled = false,
            },
        };

        using var tableStructureService = new TableFormerTableStructureService(
            tableFormerOptions,
            NullLogger<TableFormerTableStructureService>.Instance);
        var tableStructureStage = new TableStructureInferenceStage(
            tableStructureService,
            pdfOptions,
            NullLogger<TableStructureInferenceStage>.Instance);

        var ocrStage = new OcrStage(ocrFactory, pdfOptions, NullLogger<OcrStage>.Instance);

        var pipeline = new ConvertPipelineBuilder()
            .AddStage(preprocessingStage)
            .AddStage(layoutStage)
            .AddStage(tableStructureStage)
            .AddStage(ocrStage)
            .Build(NullLogger<ConvertPipeline>.Instance);

        await pipeline.ExecuteAsync(context, CancellationToken.None);

        context.GetRequired<bool>(PipelineContextKeys.PreprocessingCompleted).Should().BeTrue();

        PageReference normalizedReference;
        double normalizedWidth;
        double normalizedHeight;
        using (var normalized = store.Rent(originalPage))
        {
            normalizedReference = normalized.Page;
            normalizedWidth = normalized.Width;
            normalizedHeight = normalized.Height;
            normalized.Page.Dpi.Should().BeApproximately(preprocessingOptions.TargetDpi, 0.01);
            normalized.Metadata.Properties.Should().ContainKey(PageImageMetadataKeys.NormalizedDpi);
            normalized.Metadata.Properties.Should().ContainKey(PageImageMetadataKeys.ScaleFactor);
            normalized.Metadata.Properties.Should().ContainKey(PageImageMetadataKeys.ColorMode);
        }

        pages[0] = normalizedReference;
        var normalizedPageBounds = BoundingBox.FromSize(0, 0, normalizedWidth, normalizedHeight);

        context.GetRequired<bool>(PipelineContextKeys.LayoutAnalysisCompleted).Should().BeTrue();
        var layoutItems = context.GetRequired<IReadOnlyList<LayoutItem>>(PipelineContextKeys.LayoutItems);
        layoutItems.Should().NotBeNull();
        if (layoutItems.Count > 0)
        {
            foreach (var item in layoutItems)
            {
                item.Page.Should().Be(normalizedReference);
                item.BoundingBox.Width.Should().BeGreaterThan(0);
                item.BoundingBox.Height.Should().BeGreaterThan(0);
                item.BoundingBox.Left.Should().BeGreaterThanOrEqualTo(0);
                item.BoundingBox.Top.Should().BeGreaterThanOrEqualTo(0);
                item.BoundingBox.Right.Should().BeLessThanOrEqualTo(normalizedWidth);
                item.BoundingBox.Bottom.Should().BeLessThanOrEqualTo(normalizedHeight);
            }

            layoutItems.Should().Contain(item => item.Kind == LayoutItemKind.Text);
            layoutItems.Should().Contain(item => item.Kind == LayoutItemKind.Table);
        }

        var tableStructures = context.GetRequired<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures);
        var tableLayoutItems = layoutItems.Where(item => item.Kind == LayoutItemKind.Table).ToList();
        if (tableLayoutItems.Count > 0)
        {
            tableStructures.Should().NotBeEmpty("TableFormer should emit table structures when layout finds table regions");
            tableStructures.Count.Should().Be(tableLayoutItems.Count);

            for (var i = 0; i < tableLayoutItems.Count; i++)
            {
                var tableItem = tableLayoutItems[i];
                var structure = tableStructures[i];

                structure.Page.Should().Be(tableItem.Page);
                structure.Cells.Should().NotBeEmpty("TableFormer must return cell polygons for the detected table");
                structure.RowCount.Should().BeGreaterThan(0);
                structure.ColumnCount.Should().BeGreaterThan(0);

                foreach (var cell in structure.Cells)
                {
                    cell.BoundingBox.Left.Should().BeGreaterThanOrEqualTo(tableItem.BoundingBox.Left);
                    cell.BoundingBox.Top.Should().BeGreaterThanOrEqualTo(tableItem.BoundingBox.Top);
                    cell.BoundingBox.Right.Should().BeLessThanOrEqualTo(tableItem.BoundingBox.Right);
                    cell.BoundingBox.Bottom.Should().BeLessThanOrEqualTo(tableItem.BoundingBox.Bottom);
                }
            }
        }
        else
        {
            tableStructures.Should().BeEmpty();
        }

        context.GetRequired<bool>(PipelineContextKeys.OcrCompleted).Should().BeTrue();
        var ocrResult = context.GetRequired<OcrDocumentResult>(PipelineContextKeys.OcrResults);
        ocrResult.Blocks.Should().NotBeEmpty("OCR should return recognised text for the dataset page");

        var recognisedLines = ocrResult.Blocks.SelectMany(block => block.Lines).ToList();
        recognisedLines.Should().NotBeEmpty();
        recognisedLines.Should().OnlyContain(line => !string.IsNullOrWhiteSpace(line.Text));

        foreach (var block in ocrResult.Blocks)
        {
            block.Region.Width.Should().BeGreaterThan(0);
            block.Region.Height.Should().BeGreaterThan(0);
            block.Kind.Should().BeOneOf(OcrRegionKind.LayoutBlock, OcrRegionKind.FullPage);

            if (block.Kind == OcrRegionKind.LayoutBlock)
            {
                layoutItems.Should().NotBeEmpty();
                layoutItems.Should().Contain(item => item.BoundingBox.Intersects(block.Region));
            }
            else
            {
                block.Region.Should().Be(normalizedPageBounds);
            }
        }
    }

    [Fact]
    public async Task ConvertPipelineProcessesDatasetPdf()
    {
        var pdfPath = GetDatasetAssetPath("amt_handbook_sample.pdf");
        File.Exists(pdfPath).Should().BeTrue("the dataset PDF must be copied next to the test binaries");

        EnsureEasyOcrModels();
        var easyOcrModelDirectory = Path.Combine(AppContext.BaseDirectory, "contentFiles", "any", "any", "models", "onnx");
        Directory.Exists(easyOcrModelDirectory).Should().BeTrue("EasyOCR models must be available for the integration test");

        const int targetDpi = 144;
        const int layoutInputSize = 640;
        var backend = new PdfBackend(
            new PdfToImageRenderer(),
            new PdfBackendOptions
            {
                StreamFactory = _ => Task.FromResult<Stream>(File.OpenRead(pdfPath)),
                DocumentId = "dataset-pdf",
                SourceName = "dataset",
                RenderSettings = new PdfRenderSettings
                {
                    Dpi = targetDpi,
                    WithAspectRatio = true,
                    BackgroundColor = SKColors.White,
                },
                Metadata = new Dictionary<string, string>
                {
                    ["origin"] = "integration-test",
                },
            });

        using var store = new PageImageStore();
        var pages = new List<PageReference>();

        await foreach (var pageImage in backend.LoadAsync(CancellationToken.None))
        {
            using (pageImage)
            {
                var layoutCompatible = CreateLayoutCompatiblePage(pageImage, layoutInputSize);
                pages.Add(layoutCompatible.Page);
                store.Add(layoutCompatible);
            }
        }

        pages.Should().NotBeEmpty("the dataset PDF must contain at least one page");

        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, pages);
        context.Set(PipelineContextKeys.DocumentId, "dataset-pdf-integration");

        var preprocessingOptions = new PreprocessingOptions
        {
            TargetDpi = targetDpi,
            ColorMode = PageColorMode.Preserve,
            EnableDeskew = false,
            NormalizeContrast = true,
        };

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
            new LayoutOptions
            {
                Model = LayoutModelConfiguration.DoclingLayoutEgretMedium,
                CreateOrphanClusters = false,
                KeepEmptyClusters = true,
            },
            NullLogger<LayoutAnalysisStage>.Instance);

        var ocrFactory = new OcrServiceFactory();

        var tableFormerWorkingDirectory = Path.Combine(Path.GetTempPath(), $"docling-tableformer-{Guid.NewGuid():N}");
        var tableFormerOptions = new TableFormerStructureServiceOptions
        {
            Runtime = TableFormerRuntime.Onnx,
            Variant = TableFormerModelVariant.Accurate,
            WorkingDirectory = tableFormerWorkingDirectory,
        };

        var pipelineOptions = new PdfPipelineOptions
        {
            Ocr = new EasyOcrOptions
            {
                Languages = new[] { "en" },
                BitmapAreaThreshold = 0.0005,
                ForceFullPageOcr = true,
                ModelStorageDirectory = easyOcrModelDirectory,
                DownloadEnabled = false,
            },
        };

        using var tableStructureService = new TableFormerTableStructureService(
            tableFormerOptions,
            NullLogger<TableFormerTableStructureService>.Instance);
        var tableStructureStage = new TableStructureInferenceStage(
            tableStructureService,
            pipelineOptions,
            NullLogger<TableStructureInferenceStage>.Instance);

        var ocrStage = new OcrStage(ocrFactory, pipelineOptions, NullLogger<OcrStage>.Instance);

        var pipeline = new ConvertPipelineBuilder()
            .AddStage(preprocessingStage)
            .AddStage(layoutStage)
            .AddStage(tableStructureStage)
            .AddStage(ocrStage)
            .Build(NullLogger<ConvertPipeline>.Instance);

        await pipeline.ExecuteAsync(context, CancellationToken.None);

        context.GetRequired<bool>(PipelineContextKeys.PreprocessingCompleted).Should().BeTrue();

        var normalizedPages = new Dictionary<int, (PageReference Reference, BoundingBox Bounds)>(pages.Count);
        for (var i = 0; i < pages.Count; i++)
        {
            var originalReference = pages[i];
            using var normalized = store.Rent(originalReference);
            normalized.Page.Dpi.Should().BeApproximately(preprocessingOptions.TargetDpi, 0.01);
            normalized.Metadata.Properties.Should().ContainKey(PageImageMetadataKeys.NormalizedDpi);
            normalized.Metadata.Properties.Should().ContainKey(PageImageMetadataKeys.ScaleFactor);
            normalized.Metadata.Properties.Should().ContainKey(PageImageMetadataKeys.ColorMode);

            var normalizedReference = normalized.Page;
            var bounds = BoundingBox.FromSize(0, 0, normalized.Width, normalized.Height);

            pages[i] = normalizedReference;
            normalizedPages[normalizedReference.PageNumber] = (normalizedReference, bounds);
        }

        normalizedPages.Should().HaveCount(pages.Count);

        context.GetRequired<bool>(PipelineContextKeys.LayoutAnalysisCompleted).Should().BeTrue();
        var layoutItems = context.GetRequired<IReadOnlyList<LayoutItem>>(PipelineContextKeys.LayoutItems);
        layoutItems.Should().NotBeNull();

        foreach (var item in layoutItems)
        {
            normalizedPages.Should().ContainKey(item.Page.PageNumber);
            var normalized = normalizedPages[item.Page.PageNumber];
            item.Page.Should().Be(normalized.Reference);

            item.BoundingBox.Width.Should().BeGreaterThan(0);
            item.BoundingBox.Height.Should().BeGreaterThan(0);
            item.BoundingBox.Left.Should().BeGreaterThanOrEqualTo(normalized.Bounds.Left);
            item.BoundingBox.Top.Should().BeGreaterThanOrEqualTo(normalized.Bounds.Top);
            item.BoundingBox.Right.Should().BeLessThanOrEqualTo(normalized.Bounds.Right);
            item.BoundingBox.Bottom.Should().BeLessThanOrEqualTo(normalized.Bounds.Bottom);
        }

        if (layoutItems.Count > 0)
        {
            layoutItems.Should().Contain(item => item.Kind == LayoutItemKind.Text);
        }

        var tableStructures = context.GetRequired<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures);
        var tableLayoutItems = layoutItems.Where(item => item.Kind == LayoutItemKind.Table).ToList();
        if (tableLayoutItems.Count > 0)
        {
            tableStructures.Should().NotBeEmpty("TableFormer should emit table structures when layout finds table regions");
            tableStructures.Count.Should().Be(tableLayoutItems.Count);

            for (var i = 0; i < tableLayoutItems.Count; i++)
            {
                var tableItem = tableLayoutItems[i];
                var structure = tableStructures[i];

                structure.Page.Should().Be(tableItem.Page);
                structure.Cells.Should().NotBeEmpty("TableFormer must return cell polygons for the detected table");
                structure.RowCount.Should().BeGreaterThan(0);
                structure.ColumnCount.Should().BeGreaterThan(0);

                foreach (var cell in structure.Cells)
                {
                    cell.BoundingBox.Left.Should().BeGreaterThanOrEqualTo(tableItem.BoundingBox.Left);
                    cell.BoundingBox.Top.Should().BeGreaterThanOrEqualTo(tableItem.BoundingBox.Top);
                    cell.BoundingBox.Right.Should().BeLessThanOrEqualTo(tableItem.BoundingBox.Right);
                    cell.BoundingBox.Bottom.Should().BeLessThanOrEqualTo(tableItem.BoundingBox.Bottom);
                }
            }
        }
        else
        {
            tableStructures.Should().BeEmpty();
        }

        context.GetRequired<bool>(PipelineContextKeys.OcrCompleted).Should().BeTrue();
        var ocrResult = context.GetRequired<OcrDocumentResult>(PipelineContextKeys.OcrResults);
        ocrResult.Blocks.Should().NotBeEmpty("OCR should return recognised text for the dataset PDF");

        if (layoutItems.Count == 0)
        {
            ocrResult.Blocks.Should().Contain(block => block.Kind == OcrRegionKind.FullPage);
        }

        var recognisedLines = ocrResult.Blocks.SelectMany(block => block.Lines).ToList();
        recognisedLines.Should().NotBeEmpty();
        recognisedLines.Should().OnlyContain(line => !string.IsNullOrWhiteSpace(line.Text));

        foreach (var block in ocrResult.Blocks)
        {
            normalizedPages.Should().ContainKey(block.Page.PageNumber);
            var normalized = normalizedPages[block.Page.PageNumber];

            block.Page.Should().Be(normalized.Reference);
            block.Region.Width.Should().BeGreaterThan(0);
            block.Region.Height.Should().BeGreaterThan(0);
            block.Kind.Should().BeOneOf(OcrRegionKind.LayoutBlock, OcrRegionKind.FullPage, OcrRegionKind.TableCell);

            if (block.Kind == OcrRegionKind.LayoutBlock)
            {
                layoutItems.Should().Contain(item =>
                    item.Page.PageNumber == block.Page.PageNumber && item.BoundingBox.Intersects(block.Region));
            }
            else if (block.Kind == OcrRegionKind.FullPage)
            {
                block.Region.Should().Be(normalized.Bounds);
            }
            else if (block.Kind == OcrRegionKind.TableCell)
            {
                tableLayoutItems.Should().NotBeEmpty();
                tableLayoutItems.Should().Contain(item =>
                    item.Page.PageNumber == block.Page.PageNumber && item.BoundingBox.Contains(block.Region));
            }
        }
    }

    private static string GetDatasetAssetPath(string fileName)
        => Path.Combine(AppContext.BaseDirectory, "Assets", fileName);

    private static void EnsureEasyOcrModels()
    {
        var targetRoot = Path.Combine(AppContext.BaseDirectory, "contentFiles", "any", "any", "models");
        if (Directory.Exists(targetRoot) && Directory.EnumerateFiles(targetRoot, "*", SearchOption.AllDirectories).Any())
        {
            return;
        }

        var assemblyLocation = typeof(EasyOcrService).Assembly.Location;
        var assemblyDirectory = Path.GetDirectoryName(assemblyLocation)
            ?? throw new InvalidOperationException("Unable to determine the Docling.Models assembly directory.");
        var sourceRoot = Path.Combine(assemblyDirectory, "models");
        if (!Directory.Exists(sourceRoot))
        {
            throw new DirectoryNotFoundException($"EasyOCR model directory '{sourceRoot}' not found. Ensure the package is restored.");
        }

        CopyDirectoryRecursively(sourceRoot, targetRoot);
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
                ?? throw new InvalidOperationException("Failed to resize dataset PDF page to layout model input.");

            canvas.DrawBitmap(scaled, destination);
            canvas.Flush();

            var clone = squareBitmap.Copy()
                ?? throw new InvalidOperationException("Failed to clone layout-compatible bitmap.");
            return new PageImage(source.Page, clone, source.Metadata);
        }
        finally
        {
            squareBitmap.Dispose();
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

}
