using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Imaging;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Export;
using Docling.Pipelines.Options;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using Xunit;
using System.Text.Json;

namespace Docling.Tests.Pipelines.Export;

public sealed class ImageExportStageTests
{
    [Fact]
    public async Task ExecuteAsyncExportsPictureWhenEnabled()
    {
        var options = new PdfPipelineOptions
        {
            GeneratePictureImages = true,
        };
        using var cropper = new ImageCropService();
        var stage = new ImageExportStage(cropper, options, NullLogger<ImageExportStage>.Instance);
        using var store = new PageImageStore();
        var page = new PageReference(1, 144);
        AddPageImage(store, page);

        var document = new DoclingDocument("doc", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(10, 10, 40, 40));
        document.AddItem(picture);

        var context = CreateContext(store, document, page);

        await stage.ExecuteAsync(context, CancellationToken.None);

        picture.Image.Should().NotBeNull();
        picture.Image!.MediaType.Should().Be("image/png");
        context.GetRequired<bool>(PipelineContextKeys.ImageExportCompleted).Should().BeTrue();

        var exports = context.GetRequired<IReadOnlyList<ImageExportArtifact>>(PipelineContextKeys.ImageExports);
        exports.Should().ContainSingle();
        exports[0].Kind.Should().Be(ImageExportKind.Picture);
        exports[0].TargetItemId.Should().Be(picture.Id);

        context.TryGet<IReadOnlyList<ImageExportDebugArtifact>>(PipelineContextKeys.ImageExportDebugArtifacts, out var debugArtifacts)
            .Should().BeTrue();
        debugArtifacts.Should().BeEmpty();
    }

    [Fact]
    public async Task ExecuteAsyncSkipsWhenDisabled()
    {
        var options = new PdfPipelineOptions();
        using var cropper = new ImageCropService();
        var stage = new ImageExportStage(cropper, options, NullLogger<ImageExportStage>.Instance);
        using var store = new PageImageStore();
        var page = new PageReference(1, 144);
        AddPageImage(store, page);

        var document = new DoclingDocument("doc", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(10, 10, 40, 40));
        document.AddItem(picture);

        var context = CreateContext(store, document, page);

        await stage.ExecuteAsync(context, CancellationToken.None);

        picture.Image.Should().BeNull();
        context.GetRequired<bool>(PipelineContextKeys.ImageExportCompleted).Should().BeTrue();
        var exports = context.GetRequired<IReadOnlyList<ImageExportArtifact>>(PipelineContextKeys.ImageExports);
        exports.Should().BeEmpty();

        context.TryGet<IReadOnlyList<ImageExportDebugArtifact>>(PipelineContextKeys.ImageExportDebugArtifacts, out var debugArtifacts)
            .Should().BeTrue();
        debugArtifacts.Should().BeEmpty();
    }

    [Fact]
    public async Task ExecuteAsyncExportsTablePreviewWhenEnabled()
    {
#pragma warning disable CS0618 // GenerateTableImages retained for parity with Python options surface
        var options = new PdfPipelineOptions
        {
            GenerateTableImages = true,
        };
#pragma warning restore CS0618
        using var cropper = new ImageCropService();
        var stage = new ImageExportStage(cropper, options, NullLogger<ImageExportStage>.Instance);
        using var store = new PageImageStore();
        var page = new PageReference(2, 200);
        AddPageImage(store, page);

        var document = new DoclingDocument("doc", new[] { page });
        var table = new TableItem(page, BoundingBox.FromSize(5, 5, 80, 60), Array.Empty<TableCellItem>(), 0, 0);
        document.AddItem(table);

        var context = CreateContext(store, document, page);

        await stage.ExecuteAsync(context, CancellationToken.None);

        table.PreviewImage.Should().NotBeNull();
        table.PreviewImage!.Width.Should().BeGreaterThan(0);

        var exports = context.GetRequired<IReadOnlyList<ImageExportArtifact>>(PipelineContextKeys.ImageExports);
        exports.Should().ContainSingle(exp => exp.Kind == ImageExportKind.Table && exp.TargetItemId == table.Id);
    }

    [Fact]
    public async Task ExecuteAsyncDeduplicatesImagesByChecksum()
    {
        var options = new PdfPipelineOptions
        {
            GeneratePictureImages = true,
        };

        using var cropper = new ImageCropService();
        var stage = new ImageExportStage(cropper, options, NullLogger<ImageExportStage>.Instance);
        using var store = new PageImageStore();
        var page = new PageReference(5, 144);
        AddPageImage(store, page);

        var document = new DoclingDocument("doc", new[] { page });
        var bounds = BoundingBox.FromSize(10, 10, 50, 60);
        var picture1 = new PictureItem(page, bounds);
        var picture2 = new PictureItem(page, bounds);
        document.AddItem(picture1);
        document.AddItem(picture2);

        var context = CreateContext(store, document, page);

        await stage.ExecuteAsync(context, CancellationToken.None);

        picture1.Image.Should().NotBeNull();
        picture2.Image.Should().NotBeNull();
        picture2.Image.Should().BeSameAs(picture1.Image);

        var checksum = picture1.Image!.Checksum;
        checksum.Should().NotBeNull();
        checksum.Should().NotBeEmpty();
        checksum.Should().Be(ComputeChecksum(picture1.Image!.Data.Span));

        var exports = context.GetRequired<IReadOnlyList<ImageExportArtifact>>(PipelineContextKeys.ImageExports);
        exports.Should().ContainSingle();
        exports.Single().Image.Should().BeSameAs(picture1.Image);
    }

    [Fact]
    public async Task ExecuteAsyncEmitsDebugArtifactsWhenEnabled()
    {
        var options = new PdfPipelineOptions
        {
            GeneratePictureImages = true,
            GenerateImageDebugArtifacts = true,
        };

        using var cropper = new ImageCropService();
        var stage = new ImageExportStage(cropper, options, NullLogger<ImageExportStage>.Instance);
        using var store = new PageImageStore();
        var page = new PageReference(7, 180);
        AddPageImage(store, page);

        var document = new DoclingDocument("doc", new[] { page });
        var picture = new PictureItem(page, BoundingBox.FromSize(12, 18, 36, 28));
        document.AddItem(picture);

        var context = CreateContext(store, document, page);

        await stage.ExecuteAsync(context, CancellationToken.None);

        picture.Image.Should().NotBeNull();

        context.TryGet<IReadOnlyList<ImageExportDebugArtifact>>(PipelineContextKeys.ImageExportDebugArtifacts, out var debugArtifacts)
            .Should().BeTrue();

        debugArtifacts.Should().ContainSingle();
        var artifact = debugArtifacts[0];

        artifact.Page.Should().Be(page);
        artifact.OverlayImage.Length.Should().BeGreaterThan(0);
        artifact.ManifestContent.Length.Should().BeGreaterThan(0);

        using var manifest = JsonDocument.Parse(artifact.ManifestContent.ToArray());
        var root = manifest.RootElement;
        root.GetProperty("document_id").GetString().Should().Be(document.Id);
        root.GetProperty("page_number").GetInt32().Should().Be(page.PageNumber);

        var items = root.GetProperty("items");
        items.GetArrayLength().Should().Be(1);
        var entry = items[0];
        entry.GetProperty("target_item_id").GetString().Should().Be(picture.Id);
        entry.GetProperty("image_id").GetString().Should().Be(picture.Image!.Id);
        entry.GetProperty("kind").GetString().Should().Be("picture");
        entry.GetProperty("media_type").GetString().Should().Be("image/png");
        entry.GetProperty("checksum").GetString().Should().Be(picture.Image!.Checksum);

        var cropBounds = entry.GetProperty("crop_bounds");
        cropBounds.GetProperty("left").GetDouble().Should().BeGreaterThanOrEqualTo(0);
        cropBounds.GetProperty("top").GetDouble().Should().BeGreaterThanOrEqualTo(0);
    }

    private static PipelineContext CreateContext(PageImageStore store, DoclingDocument document, params PageReference[] pages)
    {
        var services = new ServiceCollection().BuildServiceProvider();
        var context = new PipelineContext(services);
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.Document, document);
        context.Set(PipelineContextKeys.PageSequence, pages);
        return context;
    }

    private static void AddPageImage(PageImageStore store, PageReference page)
    {
        using var bitmap = new SKBitmap(120, 160);
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
        }

        using var image = new PageImage(page, bitmap);
        store.Add(image.Clone());
    }

    private static string ComputeChecksum(ReadOnlySpan<byte> data)
    {
        if (data.IsEmpty)
        {
            return string.Empty;
        }

        var hash = SHA256.HashData(data);
        return Convert.ToHexString(hash);
    }
}
