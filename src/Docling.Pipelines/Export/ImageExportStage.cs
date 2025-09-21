using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Export.Abstractions;
using Docling.Export.Imaging;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Options;
using Microsoft.Extensions.Logging;
using SkiaSharp;

namespace Docling.Pipelines.Export;

/// <summary>
/// Pipeline stage that crops page bitmaps to generate reusable image artefacts (figures, tables, pages).
/// </summary>
public sealed partial class ImageExportStage : IPipelineStage
{
    private readonly IImageCropService _cropService;
    private readonly PdfPipelineOptions _options;
    private readonly ILogger<ImageExportStage> _logger;
    private static readonly float[] DebugDashPattern = { 8f, 5f };

    public ImageExportStage(IImageCropService cropService, PdfPipelineOptions options, ILogger<ImageExportStage> logger)
    {
        _cropService = cropService ?? throw new ArgumentNullException(nameof(cropService));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    private static void RecordDebugEntry(
        Dictionary<int, DebugPageBuilder> debugPages,
        DocItem item,
        ImageExportKind kind,
        BoundingBox originalBounds,
        BoundingBox cropBounds,
        ImageRef image)
    {
        if (!debugPages.TryGetValue(item.Page.PageNumber, out var page))
        {
            page = new DebugPageBuilder(item.Page);
            debugPages[item.Page.PageNumber] = page;
        }

        page.Entries.Add(new ImageExportDebugEntry(
            item.Id,
            image.Id,
            kind,
            ImageExportDebugBounds.FromBoundingBox(originalBounds),
            ImageExportDebugBounds.FromBoundingBox(cropBounds),
            image.MediaType,
            image.Width,
            image.Height,
            image.Dpi,
            image.Checksum));
    }

    public string Name => "image_export";

    public async Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        context.Set(PipelineContextKeys.ImageExportDebugArtifacts, Array.Empty<ImageExportDebugArtifact>());

        if (!ShouldExportAnyImages())
        {
            StageLogger.StageDisabled(_logger);
            context.Set(PipelineContextKeys.ImageExports, Array.Empty<ImageExportArtifact>());
            context.Set(PipelineContextKeys.ImageExportCompleted, true);
            return;
        }

        if (!context.TryGet<PageImageStore>(PipelineContextKeys.PageImageStore, out var store))
        {
            throw new InvalidOperationException("Pipeline context does not contain a page image store.");
        }

        var document = context.GetRequired<DoclingDocument>(PipelineContextKeys.Document);
        var exports = new List<ImageExportArtifact>();
        var dedupe = new Dictionary<string, List<ImageRef>>(StringComparer.Ordinal);
        var debugPages = _options.GenerateImageDebugArtifacts
            ? new Dictionary<int, DebugPageBuilder>()
            : null;

        if (_options.GeneratePageImages &&
            context.TryGet<IReadOnlyList<PageReference>>(PipelineContextKeys.PageSequence, out var pages))
        {
            foreach (var page in pages)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (!store.TryRent(page, out var pageImage))
                {
                    StageLogger.PageImageMissing(_logger, page.PageNumber);
                    continue;
                }

                using (pageImage)
                {
                    var buffer = EncodeToPng(pageImage.Bitmap);
                    var checksum = ComputeChecksum(buffer);
                    var pageRegion = pageImage.BoundingBox.Normalized();

                    var (imageRef, added) = RegisterImage(
                        dedupe,
                        checksum,
                        page,
                        pageRegion,
                        "image/png",
                        pageImage.Width,
                        pageImage.Height,
                        page.Dpi,
                        () => new ImageRef(
                            BuildPageImageId(document, page),
                            page,
                            pageRegion,
                            "image/png",
                            buffer,
                            pageImage.Width,
                            pageImage.Height,
                            page.Dpi,
                            checksum));

                    if (added)
                    {
                        exports.Add(new ImageExportArtifact(ImageExportKind.Page, imageRef, null));
                    }
                }
            }
        }

        if (_options.GeneratePictureImages)
        {
            foreach (var picture in document.Items.OfType<PictureItem>())
            {
                cancellationToken.ThrowIfCancellationRequested();

                var image = await ExportItemAsync(store, document, picture, "picture", ImageExportKind.Picture, dedupe, exports, debugPages, cancellationToken).ConfigureAwait(false);
                if (image is null)
                {
                    continue;
                }

                picture.SetImage(image);
            }
        }

#pragma warning disable CS0618 // GenerateTableImages retained for parity with Python options surface
        if (_options.GenerateTableImages)
#pragma warning restore CS0618
        {
            foreach (var table in document.Items.OfType<TableItem>())
            {
                cancellationToken.ThrowIfCancellationRequested();

                var image = await ExportItemAsync(store, document, table, "table", ImageExportKind.Table, dedupe, exports, debugPages, cancellationToken).ConfigureAwait(false);
                if (image is null)
                {
                    continue;
                }

                table.SetPreviewImage(image);
            }
        }

        context.Set(PipelineContextKeys.ImageExports, new ReadOnlyCollection<ImageExportArtifact>(exports));
        context.Set(PipelineContextKeys.ImageExportCompleted, true);

        EmitDebugArtifacts(context, store, document, debugPages, cancellationToken);

        StageLogger.StageCompleted(_logger, exports.Count);
    }

    private bool ShouldExportAnyImages()
    {
#pragma warning disable CS0618 // GenerateTableImages retained for parity with Python options surface
        return _options.GeneratePictureImages || _options.GeneratePageImages || _options.GenerateTableImages;
#pragma warning restore CS0618
    }

    private async Task<ImageRef?> ExportItemAsync(
        PageImageStore store,
        DoclingDocument document,
        DocItem item,
        string prefix,
        ImageExportKind kind,
        Dictionary<string, List<ImageRef>> dedupe,
        List<ImageExportArtifact> exports,
        Dictionary<int, DebugPageBuilder>? debugPages,
        CancellationToken cancellationToken)
    {
        if (item.BoundingBox.IsEmpty)
        {
            StageLogger.EmptyBoundingBox(_logger, item.Id);
            return null;
        }

        if (!store.TryRent(item.Page, out var pageImage))
        {
            StageLogger.ItemImageMissing(_logger, item.Id, item.Page.PageNumber);
            return null;
        }

        using (pageImage)
        {
            var originalBounds = item.BoundingBox.Normalized();
            var region = ScaleBoundingBox(item.BoundingBox);
            using var crop = await _cropService.CropAsync(pageImage, region, cancellationToken).ConfigureAwait(false);
            var buffer = EncodeToPng(crop.Bitmap);
            var checksum = ComputeChecksum(buffer);

            var (image, added) = RegisterImage(
                dedupe,
                checksum,
                item.Page,
                crop.SourceRegion.Normalized(),
                "image/png",
                crop.Bitmap.Width,
                crop.Bitmap.Height,
                item.Page.Dpi,
                () => new ImageRef(
                    BuildItemImageId(document, prefix, item.Id),
                    item.Page,
                    crop.SourceRegion,
                    "image/png",
                    buffer,
                    crop.Bitmap.Width,
                    crop.Bitmap.Height,
                    item.Page.Dpi,
                    checksum));

            if (added)
            {
                exports.Add(new ImageExportArtifact(kind, image, item.Id));
            }
            else
            {
                StageLogger.ImageDeduplicated(_logger, item.Id, checksum);
            }

            if (debugPages is not null)
            {
                RecordDebugEntry(debugPages, item, kind, originalBounds, crop.SourceRegion, image);
            }

            return image;
        }
    }

    private void EmitDebugArtifacts(
        PipelineContext context,
        PageImageStore store,
        DoclingDocument document,
        Dictionary<int, DebugPageBuilder>? debugPages,
        CancellationToken cancellationToken)
    {
        if (debugPages is null || debugPages.Count == 0)
        {
            return;
        }

        var artifacts = new List<ImageExportDebugArtifact>(debugPages.Count);
        foreach (var pair in debugPages.OrderBy(entry => entry.Key))
        {
            cancellationToken.ThrowIfCancellationRequested();

            var pageData = pair.Value;
            if (pageData.Entries.Count == 0)
            {
                continue;
            }

            PageImage pageImage;
            try
            {
                pageImage = store.Rent(pageData.Page);
            }
            catch (KeyNotFoundException)
            {
                StageLogger.PageImageMissing(_logger, pageData.Page.PageNumber);
                continue;
            }

            using (pageImage)
            {
                var entries = pageData.Entries.ToArray();
                var overlayBuffer = CreateDebugOverlay(pageImage, entries);
                var manifest = new ImageExportDebugManifest(document.Id, pageData.Page.PageNumber, Array.AsReadOnly(entries));
                artifacts.Add(new ImageExportDebugArtifact(pageData.Page, overlayBuffer, manifest));
            }
        }

        if (artifacts.Count == 0)
        {
            return;
        }

        context.Set(
            PipelineContextKeys.ImageExportDebugArtifacts,
            new ReadOnlyCollection<ImageExportDebugArtifact>(artifacts));
        StageLogger.DebugArtifactsGenerated(_logger, artifacts.Count);
    }

    private static byte[] CreateDebugOverlay(PageImage pageImage, IReadOnlyList<ImageExportDebugEntry> entries)
    {
        using var surface = SKSurface.Create(new SKImageInfo(pageImage.Width, pageImage.Height, SKColorType.Rgba8888, SKAlphaType.Premul));
        var canvas = surface.Canvas;
        canvas.Clear(SKColors.Transparent);
        canvas.DrawBitmap(pageImage.Bitmap, 0, 0);

        using var shade = new SKPaint
        {
            Style = SKPaintStyle.Fill,
            Color = new SKColor(0, 0, 0, 80),
        };
        canvas.DrawRect(SKRect.Create(pageImage.Width, pageImage.Height), shade);

        using var fillPaint = new SKPaint { Style = SKPaintStyle.Fill, IsAntialias = true };
        using var strokePaint = new SKPaint { Style = SKPaintStyle.Stroke, IsAntialias = true, StrokeWidth = 3f };
        using var dashEffect = SKPathEffect.CreateDash(DebugDashPattern, 0f);
        using var dashedStroke = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            IsAntialias = true,
            StrokeWidth = 3f,
            PathEffect = dashEffect,
        };
        using var textPaint = new SKPaint { IsAntialias = true, Color = SKColors.White };
        using var font = new SKFont(SKTypeface.Default, 16f);

        foreach (var entry in entries)
        {
            var color = SelectDebugColor(entry.Kind);
            fillPaint.Color = color.WithAlpha(72);
            strokePaint.Color = color;
            dashedStroke.Color = color;

            var cropRect = ToRect(entry.CropBounds);
            var originalRect = ToRect(entry.OriginalBounds);

            if (fillPaint.Color.Alpha > 0)
            {
                canvas.DrawRect(cropRect, fillPaint);
            }

            canvas.DrawRect(cropRect, strokePaint);

            if (!cropRect.Equals(originalRect))
            {
                canvas.DrawRect(originalRect, dashedStroke);
            }

            var label = string.IsNullOrWhiteSpace(entry.TargetItemId)
                ? entry.KindName
                : $"{entry.KindName} • {entry.TargetItemId}";
            var textX = cropRect.Left + 6f;
            var textY = cropRect.Top + 6f + font.Size;
            canvas.DrawText(label, textX, textY, SKTextAlign.Left, font, textPaint);
        }

        using var snapshot = surface.Snapshot();
        using var encoded = snapshot.Encode(SKEncodedImageFormat.Png, 90);
        if (encoded is null || encoded.Size == 0)
        {
            throw new InvalidOperationException("Failed to encode image export debug overlay.");
        }

        return encoded.ToArray();
    }

    private static SKRect ToRect(ImageExportDebugBounds bounds)
    {
        var width = (float)(bounds.Right - bounds.Left);
        var height = (float)(bounds.Bottom - bounds.Top);
        return SKRect.Create((float)bounds.Left, (float)bounds.Top, Math.Max(width, 1f), Math.Max(height, 1f));
    }

    private static SKColor SelectDebugColor(ImageExportKind kind) => kind switch
    {
        ImageExportKind.Picture => new SKColor(251, 140, 0),
        ImageExportKind.Table => new SKColor(30, 136, 229),
        ImageExportKind.Page => new SKColor(56, 142, 60),
        _ => SKColors.Magenta,
    };

    private BoundingBox ScaleBoundingBox(BoundingBox box)
    {
        var scale = _options.ImagesScale;
        if (Math.Abs(scale - 1d) < 1e-6)
        {
            return box.Normalized();
        }

        try
        {
            var horizontal = box.Width * (scale - 1d) / 2d;
            var vertical = box.Height * (scale - 1d) / 2d;
            return box.Inflate(horizontal, vertical).Normalized();
        }
        catch (ArgumentOutOfRangeException)
        {
            return box.Normalized();
        }
    }

    private static byte[] EncodeToPng(SKBitmap bitmap)
    {
        using var image = SKImage.FromBitmap(bitmap) ?? throw new InvalidOperationException("Failed to encode bitmap.");
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        if (data is null)
        {
            throw new InvalidOperationException("Failed to encode bitmap to PNG.");
        }

        return data.ToArray();
    }

    private static string BuildPageImageId(DoclingDocument document, PageReference page)
        => $"{document.Id}-page-{page.PageNumber:000}";

    private static string BuildItemImageId(DoclingDocument document, string prefix, string itemId)
        => $"{document.Id}-{prefix}-{itemId}";

    private static string ComputeChecksum(ReadOnlySpan<byte> buffer)
    {
        if (buffer.IsEmpty)
        {
            return string.Empty;
        }

        var hash = SHA256.HashData(buffer);
        return Convert.ToHexString(hash);
    }

    private static (ImageRef Image, bool Added) RegisterImage(
        Dictionary<string, List<ImageRef>> dedupe,
        string checksum,
        PageReference page,
        BoundingBox sourceRegion,
        string mediaType,
        int width,
        int height,
        double dpi,
        Func<ImageRef> factory)
    {
        if (dedupe.TryGetValue(checksum, out var images))
        {
            foreach (var existing in images)
            {
                if (IsEquivalent(existing, page, sourceRegion, mediaType, width, height, dpi))
                {
                    return (existing, false);
                }
            }
        }

        var created = factory();
        if (!string.Equals(created.Checksum, checksum, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Image checksum mismatch while registering export artefact.");
        }

        images ??= new List<ImageRef>();
        images.Add(created);
        dedupe[checksum] = images;

        return (created, true);
    }

    private static bool IsEquivalent(
        ImageRef existing,
        PageReference page,
        BoundingBox sourceRegion,
        string mediaType,
        int width,
        int height,
        double dpi)
    {
        return existing.Page.Equals(page) &&
               existing.SourceRegion.Equals(sourceRegion) &&
               existing.MediaType.Equals(mediaType, StringComparison.OrdinalIgnoreCase) &&
               existing.Width == width &&
               existing.Height == height &&
               Math.Abs(existing.Dpi - dpi) < 1e-6;
    }

    private static partial class StageLogger
    {
        [LoggerMessage(EventId = 6000, Level = LogLevel.Information, Message = "Image export disabled by configuration.")]
        public static partial void StageDisabled(ILogger logger);

        [LoggerMessage(EventId = 6001, Level = LogLevel.Warning, Message = "Missing cached page image for page {PageNumber}, skipping export.")]
        public static partial void PageImageMissing(ILogger logger, int pageNumber);

        [LoggerMessage(EventId = 6002, Level = LogLevel.Warning, Message = "Skipping item {ItemId} because its bounding box is empty.")]
        public static partial void EmptyBoundingBox(ILogger logger, string itemId);

        [LoggerMessage(EventId = 6003, Level = LogLevel.Warning, Message = "Skipping item {ItemId} on page {PageNumber} because the page image is unavailable.")]
        public static partial void ItemImageMissing(ILogger logger, string itemId, int pageNumber);

        [LoggerMessage(EventId = 6004, Level = LogLevel.Information, Message = "Exported {Count} image artefacts.")]
        public static partial void StageCompleted(ILogger logger, int count);

        [LoggerMessage(EventId = 6005, Level = LogLevel.Debug, Message = "Reused existing image for item {ItemId} with checksum {Checksum}.")]
        public static partial void ImageDeduplicated(ILogger logger, string itemId, string checksum);

        [LoggerMessage(EventId = 6006, Level = LogLevel.Debug, Message = "Produced {Count} image export debug artifacts.")]
        public static partial void DebugArtifactsGenerated(ILogger logger, int count);
    }

    private sealed class DebugPageBuilder
    {
        public DebugPageBuilder(PageReference page)
        {
            Page = page;
            Entries = new List<ImageExportDebugEntry>();
        }

        public PageReference Page { get; }

        public List<ImageExportDebugEntry> Entries { get; }
    }
}
