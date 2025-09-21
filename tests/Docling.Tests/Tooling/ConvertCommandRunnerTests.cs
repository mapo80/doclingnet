using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Abstractions;
using Docling.Backends.Image;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Imaging;
using Docling.Export.Serialization;
using Docling.Models.Layout;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Options;
using Docling.Tooling.Commands;
using Docling.Tooling.Runtime;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using Xunit;

namespace Docling.Tests.Tooling;

public sealed class ConvertCommandRunnerTests
{
    [Fact]
    public async Task ExecuteAsyncWritesMarkdownAssetsAndMetadata()
    {
        var inputPath = CreateTemporaryImage();
        var outputDirectory = Path.Combine(Path.GetTempPath(), $"docling-runner-{Guid.NewGuid():N}");
        Directory.CreateDirectory(outputDirectory);

        try
        {
            var parse = ConvertCommandOptions.Parse(new[]
            {
                "--input", inputPath,
                "--output", outputDirectory,
                "--markdown", "document.md",
                "--assets", "assets",
                "--layout-debug",
                "--image-debug",
                "--table-debug",
            });
            parse.Options.Should().NotBeNull();
            var options = parse.Options!;

            using var backend = FakeImageBackend.Create();
            var services = new ServiceCollection();
            services.AddLogging(builder => builder.AddProvider(NullLoggerProvider.Instance));
            services.AddSingleton(options);
            services.AddSingleton(new PdfPipelineOptions());
            services.AddSingleton(new MarkdownSerializerOptions { AssetsPath = options.AssetsDirectoryName });
            services.AddSingleton<IImageBackend>(backend);
            services.AddSingleton<IPipelineStage>(new StubStage());
            services.AddSingleton<ConvertCommandRunner>();

            using var provider = services.BuildServiceProvider();
            var runner = provider.GetRequiredService<ConvertCommandRunner>();
            var result = await runner.ExecuteAsync(CancellationToken.None);
            result.Should().Be(0);

            var markdownPath = Path.Combine(outputDirectory, "document.md");
            File.Exists(markdownPath).Should().BeTrue();
            var markdown = await File.ReadAllTextAsync(markdownPath);
            markdown.Should().Contain("# Sample");

            var assetPath = Path.Combine(outputDirectory, "assets", "img-1.png");
            File.Exists(assetPath).Should().BeTrue();
            var assetBytes = await File.ReadAllBytesAsync(assetPath);
            assetBytes.Should().BeEquivalentTo(new byte[] { 10, 20, 30 });

            var layoutDebugPath = Path.Combine(outputDirectory, "debug", "layout", "layout_page_0001.png");
            File.Exists(layoutDebugPath).Should().BeTrue();

            var imageDebugOverlay = Path.Combine(outputDirectory, "debug", "image_exports", "image_debug_page_0001.png");
            var imageDebugManifest = Path.Combine(outputDirectory, "debug", "image_exports", "image_debug_page_0001.json");
            File.Exists(imageDebugOverlay).Should().BeTrue();
            File.Exists(imageDebugManifest).Should().BeTrue();

            var tableDebugPath = Path.Combine(outputDirectory, "debug", "tables", "table_debug_page_0001_001.png");
            File.Exists(tableDebugPath).Should().BeTrue();

            var metadataPath = Path.Combine(outputDirectory, ConvertCommandOptions.MetadataFileName);
            File.Exists(metadataPath).Should().BeTrue();

            using var document = JsonDocument.Parse(await File.ReadAllTextAsync(metadataPath));
            var root = document.RootElement;
            root.GetProperty("documentId").GetString().Should().Be(options.DocumentId);
            root.GetProperty("markdownPath").GetString().Should().Be("document.md");
            root.GetProperty("assets").GetArrayLength().Should().Be(1);
            var assetEntry = root.GetProperty("assets")[0];
            assetEntry.GetProperty("path").GetString().Should().Be("assets/img-1.png");
            assetEntry.GetProperty("kind").GetString().Should().Be("picture");
            root.GetProperty("layoutDebugOverlays").GetArrayLength().Should().Be(1);
            root.GetProperty("imageDebugArtifacts").GetArrayLength().Should().Be(1);
            root.GetProperty("tableDebugImages").GetArrayLength().Should().Be(1);
            root.GetProperty("exportedImageCount").GetInt32().Should().Be(1);
        }
        finally
        {
            Directory.Delete(outputDirectory, recursive: true);
            File.Delete(inputPath);
        }
    }

    private static string CreateTemporaryImage()
    {
        var path = Path.Combine(Path.GetTempPath(), $"docling-input-{Guid.NewGuid():N}.png");
        using var bitmap = new SKBitmap(8, 8);
        using var surface = new SKCanvas(bitmap);
        surface.Clear(SKColors.LightGray);
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 90);
        File.WriteAllBytes(path, data.ToArray());
        return path;
    }

    private sealed class FakeImageBackend : IImageBackend, IDisposable
    {
        private readonly SKBitmap _bitmap;
        private readonly PageReference _page;
        private readonly PageImageMetadata _metadata;

        private FakeImageBackend(SKBitmap bitmap)
        {
            _bitmap = bitmap;
            _page = new PageReference(1, 200);
            _metadata = new PageImageMetadata("doc", "source", "image/png", null);
        }

        public static FakeImageBackend Create()
        {
            var bitmap = new SKBitmap(16, 16);
            using var canvas = new SKCanvas(bitmap);
            canvas.Clear(SKColors.White);
            using var paint = new SKPaint { Color = SKColors.Blue };
            canvas.DrawRect(SKRect.Create(0, 0, 16, 16), paint);
            return new FakeImageBackend(bitmap);
        }

        public async IAsyncEnumerable<PageImage> LoadAsync([System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var clone = _bitmap.Copy();
            if (clone is null)
            {
                throw new InvalidOperationException("Failed to clone bitmap for fake backend.");
            }

            yield return new PageImage(_page, clone, _metadata);
            await Task.CompletedTask;
        }

        public void Dispose() => _bitmap.Dispose();
    }

    private sealed class StubStage : IPipelineStage
    {
        public string Name => "stub";

        public Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
        {
            var page = new PageReference(1, 200);
            var imageRef = new ImageRef(
                "img-1",
                page,
                BoundingBox.FromSize(0, 0, 50, 50),
                "image/png",
                new byte[] { 10, 20, 30 },
                50,
                50,
                200,
                "checksum");

            var asset = new MarkdownAsset("assets/img-1.png", ImageExportKind.Picture, imageRef, "item-1");
            var result = new MarkdownSerializationResult("# Sample\n", new[] { asset }, new Dictionary<string, string>
            {
                ["source"] = "unit-test",
            });

            context.Set(PipelineContextKeys.MarkdownSerializationResult, result);
            context.Set(PipelineContextKeys.ImageExports, new[] { new ImageExportArtifact(ImageExportKind.Picture, imageRef, "item-1") });
            context.Set(PipelineContextKeys.LayoutDebugArtifacts, new[] { new LayoutDebugOverlay(page, new byte[] { 1, 2, 3 }) });

            var manifest = new ImageExportDebugManifest("doc", page.PageNumber, new[]
            {
                new ImageExportDebugEntry(
                    "item-1",
                    imageRef.Id,
                    ImageExportKind.Picture,
                    ImageExportDebugBounds.FromBoundingBox(imageRef.SourceRegion),
                    ImageExportDebugBounds.FromBoundingBox(imageRef.SourceRegion),
                    imageRef.MediaType,
                    imageRef.Width,
                    imageRef.Height,
                    imageRef.Dpi,
                    imageRef.Checksum),
            });
            context.Set(PipelineContextKeys.ImageExportDebugArtifacts, new[]
            {
                new ImageExportDebugArtifact(page, new byte[] { 5, 4, 3 }, manifest),
            });

            var tableStructure = new TableStructure(
                page,
                new[] { new TableCell(imageRef.SourceRegion, 1, 1, null) },
                1,
                1,
                new TableStructureDebugArtifact(page, new byte[] { 7, 7, 7 }));
            context.Set(PipelineContextKeys.TableStructures, new[] { tableStructure });

            return Task.CompletedTask;
        }
    }
}
