using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Abstractions;
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
using Docling.Pipelines.Export;
using Docling.Pipelines.Internal;
using Docling.Pipelines.Options;
using Docling.Tooling.Commands;
using Microsoft.Extensions.Logging;

namespace Docling.Tooling.Runtime;

internal sealed partial class ConvertCommandRunner
{
    private readonly ConvertCommandOptions _options;
    private readonly IServiceProvider _services;
    private readonly List<IPipelineStage> _stages;
    private readonly List<IPipelineObserver> _observers;
    private readonly ILogger<ConvertCommandRunner> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private readonly PipelineTelemetryObserver _telemetryObserver;
    private readonly IPdfBackend? _pdfBackend;
    private readonly IImageBackend? _imageBackend;
    private readonly WorkflowDebugObserver? _workflowDebugObserver;
    private static readonly JsonSerializerOptions MetadataSerializerOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    public ConvertCommandRunner(
        ConvertCommandOptions options,
        IServiceProvider services,
        IEnumerable<IPipelineStage> stages,
        IEnumerable<IPipelineObserver> observers,
        PdfPipelineOptions pipelineOptions,
        MarkdownSerializerOptions serializerOptions,
        ILogger<ConvertCommandRunner> logger,
        ILoggerFactory loggerFactory,
        IPdfBackend? pdfBackend = null,
        IImageBackend? imageBackend = null,
        PipelineTelemetryObserver? telemetryObserver = null,
        WorkflowDebugObserver? workflowDebugObserver = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _services = services ?? throw new ArgumentNullException(nameof(services));
        _stages = stages?.ToList() ?? throw new ArgumentNullException(nameof(stages));
        if (_stages.Count == 0)
        {
            throw new ArgumentException("At least one pipeline stage must be registered.", nameof(stages));
        }

        _observers = observers?.ToList() ?? throw new ArgumentNullException(nameof(observers));

        _ = pipelineOptions ?? throw new ArgumentNullException(nameof(pipelineOptions));
        _ = serializerOptions ?? throw new ArgumentNullException(nameof(serializerOptions));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        _pdfBackend = pdfBackend;
        _imageBackend = imageBackend;
        _telemetryObserver = telemetryObserver ?? new PipelineTelemetryObserver();
        _workflowDebugObserver = workflowDebugObserver;
    }

    public async Task<int> ExecuteAsync(CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        RunnerLogger.ConversionStarted(
            _logger,
            _options.InputPath,
            _options.DocumentId,
            _options.PreprocessingDpi.ToString("F0", CultureInfo.InvariantCulture),
            string.Join(',', _options.OcrLanguages),
            _options.TableMode.ToString(),
            _options.ImageMode.ToString(),
            _options.GeneratePageImages,
            _options.GeneratePictureImages);

        var configurationSnapshot = CreateConfigurationSnapshot();
        var configurationJson = JsonSerializer.Serialize(configurationSnapshot, MetadataSerializerOptions);
        RunnerLogger.ConfigurationSnapshot(_logger, Environment.NewLine, configurationJson);

        Directory.CreateDirectory(_options.OutputDirectory);

        using var store = new PageImageStore();
        var pages = await LoadDocumentAsync(store, cancellationToken).ConfigureAwait(false);
        RunnerLogger.PagesLoaded(_logger, pages.Count);

        var context = new PipelineContext(_services);
        context.Set(PipelineContextKeys.PageImageStore, store);
        context.Set(PipelineContextKeys.PageSequence, pages);
        context.Set(PipelineContextKeys.DocumentId, _options.DocumentId);

        var builder = new ConvertPipelineBuilder();
        foreach (var stage in _stages)
        {
            builder.AddStage(stage);
        }

        _telemetryObserver.Reset();
        builder.AddObserver(_telemetryObserver);
        foreach (var observer in _observers)
        {
            if (!ReferenceEquals(observer, _telemetryObserver))
            {
                builder.AddObserver(observer);
            }
        }

        var pipeline = builder.Build(_loggerFactory.CreateLogger<ConvertPipeline>());
        await pipeline.ExecuteAsync(context, cancellationToken).ConfigureAwait(false);

        var serializationResult = context.GetRequired<MarkdownSerializationResult>(PipelineContextKeys.MarkdownSerializationResult);
        context.TryGet<IReadOnlyList<ImageExportArtifact>>(PipelineContextKeys.ImageExports, out var imageExports);

        var markdownPath = await WriteMarkdownAsync(serializationResult.Markdown, cancellationToken).ConfigureAwait(false);
        var assetMetadata = await WriteAssetsAsync(serializationResult.Assets, cancellationToken).ConfigureAwait(false);
        var layoutDebugMetadata = await WriteLayoutDebugArtifactsAsync(context, cancellationToken).ConfigureAwait(false);
        var imageDebugMetadata = await WriteImageDebugArtifactsAsync(context, cancellationToken).ConfigureAwait(false);
        var tableDebugMetadata = await WriteTableDebugArtifactsAsync(context, cancellationToken).ConfigureAwait(false);
        var workflowDebugMetadata = await WriteWorkflowDebugArtifactsAsync(cancellationToken).ConfigureAwait(false);

        var metadata = new CliOutputMetadata(
            _options.DocumentId,
            _options.InputPath,
            NormalizeRelativePath(markdownPath),
            assetMetadata,
            new Dictionary<string, string>(serializationResult.Metadata, StringComparer.OrdinalIgnoreCase),
            layoutDebugMetadata,
            imageDebugMetadata,
            tableDebugMetadata,
            workflowDebugMetadata,
            imageExports?.Count ?? 0,
            DateTimeOffset.UtcNow);

        await WriteMetadataAsync(metadata, cancellationToken).ConfigureAwait(false);

        stopwatch.Stop();
        var elapsedMilliseconds = stopwatch.ElapsedMilliseconds;

        context.TryGet(PipelineContextKeys.Document, out DoclingDocument? document);
        var stageTelemetry = _telemetryObserver.CreateSnapshot();
        _telemetryObserver.Reset();
        var telemetrySnapshot = CreateTelemetrySnapshot(
            metadata,
            pages,
            document,
            serializationResult,
            imageExports,
            elapsedMilliseconds,
            stageTelemetry);
        var telemetryJson = JsonSerializer.Serialize(telemetrySnapshot, MetadataSerializerOptions);
        RunnerLogger.TelemetrySummary(_logger, Environment.NewLine, telemetryJson);

        RunnerLogger.ConversionCompleted(_logger, elapsedMilliseconds, metadata.MarkdownPath, metadata.Assets.Count);

        return 0;
    }

    private async Task<List<PageReference>> LoadDocumentAsync(PageImageStore store, CancellationToken cancellationToken)
    {
        var pages = new List<PageReference>();
        switch (_options.InputKind)
        {
            case DocumentInputKind.Pdf:
            {
                if (_pdfBackend is null)
                {
                    throw new InvalidOperationException("PDF backend is not configured.");
                }

                await foreach (var pageImage in _pdfBackend.LoadAsync(cancellationToken).ConfigureAwait(false))
                {
                    pages.Add(pageImage.Page);
                    store.Add(pageImage);
                }

                break;
            }

            case DocumentInputKind.Image:
            {
                if (_imageBackend is null)
                {
                    throw new InvalidOperationException("Image backend is not configured.");
                }

                await foreach (var pageImage in _imageBackend.LoadAsync(cancellationToken).ConfigureAwait(false))
                {
                    pages.Add(pageImage.Page);
                    store.Add(pageImage);
                }

                break;
            }

            default:
                throw new InvalidOperationException($"Unsupported input kind '{_options.InputKind}'.");
        }

        return pages;
    }

    private async Task<string> WriteMarkdownAsync(string markdown, CancellationToken cancellationToken)
    {
        var path = Path.Combine(_options.OutputDirectory, _options.MarkdownFileName);
        await File.WriteAllTextAsync(path, markdown, cancellationToken).ConfigureAwait(false);
        return path;
    }

    private async Task<List<CliAssetMetadata>> WriteAssetsAsync(IReadOnlyList<MarkdownAsset> assets, CancellationToken cancellationToken)
    {
        var metadata = new List<CliAssetMetadata>(assets.Count);
        foreach (var asset in assets)
        {
            var relativePath = asset.RelativePath.Replace('/', Path.DirectorySeparatorChar);
            var fullPath = Path.Combine(_options.OutputDirectory, relativePath);
            var directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            await File.WriteAllBytesAsync(fullPath, asset.Image.Data.ToArray(), cancellationToken).ConfigureAwait(false);

            metadata.Add(new CliAssetMetadata(
                NormalizeRelativePath(fullPath),
                GetAssetKindSlug(asset.Kind),
                asset.TargetItemId,
                asset.Image.MediaType,
                asset.Image.Width,
                asset.Image.Height,
                asset.Image.Dpi,
                asset.Image.Checksum));
        }

        return metadata;
    }

    private async Task<List<string>> WriteLayoutDebugArtifactsAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        if (!_options.GenerateLayoutDebugArtifacts)
        {
            return new List<string>();
        }

        if (!context.TryGet<IReadOnlyList<LayoutDebugOverlay>>(PipelineContextKeys.LayoutDebugArtifacts, out var overlays) ||
            overlays is null || overlays.Count == 0)
        {
            return new List<string>();
        }

        var output = new List<string>(overlays.Count);
        var root = Path.Combine(_options.OutputDirectory, "debug", "layout");
        Directory.CreateDirectory(root);

        foreach (var overlay in overlays)
        {
            var fileName = string.Create(CultureInfo.InvariantCulture, $"layout_page_{overlay.Page.PageNumber:0000}.png");
            var fullPath = Path.Combine(root, fileName);
            await File.WriteAllBytesAsync(fullPath, overlay.ImageContent.ToArray(), cancellationToken).ConfigureAwait(false);
            output.Add(NormalizeRelativePath(fullPath));
        }

        return output;
    }

    private async Task<List<CliImageDebugMetadata>> WriteImageDebugArtifactsAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        if (!_options.GenerateImageDebugArtifacts)
        {
            return new List<CliImageDebugMetadata>();
        }

        if (!context.TryGet<IReadOnlyList<ImageExportDebugArtifact>>(PipelineContextKeys.ImageExportDebugArtifacts, out var artifacts) ||
            artifacts is null || artifacts.Count == 0)
        {
            return new List<CliImageDebugMetadata>();
        }

        var output = new List<CliImageDebugMetadata>(artifacts.Count);
        var root = Path.Combine(_options.OutputDirectory, "debug", "image_exports");
        Directory.CreateDirectory(root);

        foreach (var artifact in artifacts)
        {
            var baseName = string.Create(CultureInfo.InvariantCulture, $"image_debug_page_{artifact.Page.PageNumber:0000}");
            var overlayPath = Path.Combine(root, baseName + ".png");
            var manifestPath = Path.Combine(root, baseName + ".json");

            await File.WriteAllBytesAsync(overlayPath, artifact.OverlayImage.ToArray(), cancellationToken).ConfigureAwait(false);
            await File.WriteAllBytesAsync(manifestPath, artifact.ManifestContent.ToArray(), cancellationToken).ConfigureAwait(false);

            output.Add(new CliImageDebugMetadata(NormalizeRelativePath(overlayPath), NormalizeRelativePath(manifestPath)));
        }

        return output;
    }

    private async Task<List<string>> WriteTableDebugArtifactsAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        if (!_options.GenerateTableDebugArtifacts)
        {
            return new List<string>();
        }

        if (!context.TryGet<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures, out var structures) ||
            structures is null || structures.Count == 0)
        {
            return new List<string>();
        }

        var output = new List<string>();
        var root = Path.Combine(_options.OutputDirectory, "debug", "tables");
        Directory.CreateDirectory(root);

        var index = 0;
        foreach (var structure in structures)
        {
            if (structure.DebugArtifact is null || structure.DebugArtifact.ImageContent.IsEmpty)
            {
                continue;
            }

            index++;
            var fileName = string.Create(CultureInfo.InvariantCulture, $"table_debug_page_{structure.Page.PageNumber:0000}_{index:000}.png");
            var fullPath = Path.Combine(root, fileName);
            await File.WriteAllBytesAsync(fullPath, structure.DebugArtifact.ImageContent.ToArray(), cancellationToken).ConfigureAwait(false);
            output.Add(NormalizeRelativePath(fullPath));
        }

        return output;
    }

    private Task<List<string>> WriteWorkflowDebugArtifactsAsync(CancellationToken cancellationToken)
    {
        _ = cancellationToken;

        if (_workflowDebugObserver is null)
        {
            return Task.FromResult(new List<string>());
        }

        var artifacts = _workflowDebugObserver.GetCreatedArtifacts();
        if (artifacts.Count == 0)
        {
            return Task.FromResult(new List<string>());
        }

        var output = new List<string>(artifacts.Count);
        foreach (var path in artifacts)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                continue;
            }

            output.Add(NormalizeRelativePath(path));
        }

        return Task.FromResult(output);
    }

    private async Task WriteMetadataAsync(CliOutputMetadata metadata, CancellationToken cancellationToken)
    {
        var path = Path.Combine(_options.OutputDirectory, ConvertCommandOptions.MetadataFileName);
        using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            4096,
            FileOptions.Asynchronous | FileOptions.SequentialScan);

        await JsonSerializer.SerializeAsync(stream, metadata, MetadataSerializerOptions, cancellationToken).ConfigureAwait(false);
    }

    private string NormalizeRelativePath(string fullPath)
    {
        var relative = Path.GetRelativePath(_options.OutputDirectory, fullPath);
        return relative.Replace(Path.DirectorySeparatorChar, '/');
    }

    private static string GetAssetKindSlug(ImageExportKind kind)
        => kind switch
        {
            ImageExportKind.Page => "page",
            ImageExportKind.Picture => "picture",
            ImageExportKind.Table => "table",
            _ => kind.ToString(),
        };

    private CliConfigurationSnapshot CreateConfigurationSnapshot()
    {
        return new CliConfigurationSnapshot(
            _options.DocumentId,
            _options.InputPath,
            _options.InputKind.ToString(),
            _options.SourceName,
            _options.OutputDirectory,
            _options.MarkdownFileName,
            _options.AssetsDirectoryName,
            _options.PreprocessingDpi,
            _options.RenderDpi,
            _options.ForceFullPageOcr,
            _options.TableMode.ToString(),
            _options.ImageMode.ToString(),
            _options.OcrLanguages.ToArray(),
            _options.GeneratePageImages,
            _options.GeneratePictureImages,
            _options.GenerateLayoutDebugArtifacts,
            _options.GenerateImageDebugArtifacts,
            _options.GenerateTableDebugArtifacts,
            _stages.Select(stage => stage.Name).ToArray());
    }

    private static CliTelemetrySnapshot CreateTelemetrySnapshot(
        CliOutputMetadata metadata,
        List<PageReference> pages,
        DoclingDocument? document,
        MarkdownSerializationResult serializationResult,
        IReadOnlyList<ImageExportArtifact>? imageExports,
        long elapsedMilliseconds,
        IReadOnlyList<PipelineStageTelemetry> stageTelemetry)
    {
        var paragraphCount = GetItemCount(document, DocItemKind.Paragraph);
        var headingCount = GetItemCount(document, DocItemKind.Heading);
        var pictureCount = GetItemCount(document, DocItemKind.Picture);
        var tableCount = GetItemCount(document, DocItemKind.Table);
        var captionCount = GetItemCount(document, DocItemKind.Caption);

        var stageSnapshots = stageTelemetry
            .Select(entry => new CliStageTelemetry(entry.Stage, entry.DurationMilliseconds))
            .ToArray();

        return new CliTelemetrySnapshot(
            metadata.DocumentId,
            metadata.SourcePath,
            pages.Count,
            document?.Items.Count ?? 0,
            paragraphCount,
            headingCount,
            pictureCount,
            tableCount,
            captionCount,
            document?.Properties.Count ?? 0,
            serializationResult.Metadata.Count,
            serializationResult.Markdown.Length,
            metadata.Assets.Count,
            metadata.ExportedImageCount,
            metadata.LayoutDebugOverlays.Count,
            metadata.ImageDebugArtifacts.Count,
            metadata.TableDebugImages.Count,
            metadata.WorkflowDebugFiles.Count,
            imageExports?.Count ?? 0,
            elapsedMilliseconds,
            metadata.GeneratedAt,
            stageSnapshots);
    }

    private static int GetItemCount(DoclingDocument? document, DocItemKind kind)
    {
        if (document is null)
        {
            return 0;
        }

        return document.GetItemsOfKind(kind).Count;
    }

    private static partial class RunnerLogger
    {
        [LoggerMessage(EventId = 3800, Level = LogLevel.Information, Message = "Starting Docling conversion for {Input} (document id: {DocumentId}). Configuration: DPI={Dpi}, OCR languages={Languages}, TableMode={TableMode}, ImageMode={ImageMode}, PageImages={PageImages}, PictureImages={PictureImages}.")]
        public static partial void ConversionStarted(
            ILogger logger,
            string input,
            string documentId,
            string dpi,
            string languages,
            string tableMode,
            string imageMode,
            bool pageImages,
            bool pictureImages);

        [LoggerMessage(EventId = 3801, Level = LogLevel.Information, Message = "Loaded {PageCount} page(s) from input document.")]
        public static partial void PagesLoaded(ILogger logger, int pageCount);

        [LoggerMessage(EventId = 3802, Level = LogLevel.Information, Message = "Conversion completed in {Elapsed} ms. Markdown written to {MarkdownPath} with {AssetCount} asset(s).")]
        public static partial void ConversionCompleted(ILogger logger, long elapsed, string markdownPath, int assetCount);

        [LoggerMessage(EventId = 3803, Level = LogLevel.Information, Message = "Configuration snapshot:{NewLine}{ConfigurationJson}")]
        public static partial void ConfigurationSnapshot(ILogger logger, string newLine, string configurationJson);

        [LoggerMessage(EventId = 3804, Level = LogLevel.Information, Message = "Telemetry summary:{NewLine}{TelemetryJson}")]
        public static partial void TelemetrySummary(ILogger logger, string newLine, string telemetryJson);
    }

    private sealed record CliAssetMetadata(
        string Path,
        string Kind,
        string? TargetItemId,
        string MediaType,
        int Width,
        int Height,
        double Dpi,
        string? Checksum);

    private sealed record CliImageDebugMetadata(string OverlayPath, string ManifestPath);

    private sealed record CliOutputMetadata(
        string DocumentId,
        string SourcePath,
        string MarkdownPath,
        IReadOnlyList<CliAssetMetadata> Assets,
        IReadOnlyDictionary<string, string> Properties,
        IReadOnlyList<string> LayoutDebugOverlays,
        IReadOnlyList<CliImageDebugMetadata> ImageDebugArtifacts,
        IReadOnlyList<string> TableDebugImages,
        IReadOnlyList<string> WorkflowDebugFiles,
        int ExportedImageCount,
        DateTimeOffset GeneratedAt);

    private sealed record CliConfigurationSnapshot(
        string DocumentId,
        string SourcePath,
        string InputKind,
        string SourceName,
        string OutputDirectory,
        string MarkdownFileName,
        string AssetsDirectoryName,
        double PreprocessingDpi,
        int RenderDpi,
        bool ForceFullPageOcr,
        string TableMode,
        string ImageMode,
        IReadOnlyList<string> OcrLanguages,
        bool GeneratePageImages,
        bool GeneratePictureImages,
        bool GenerateLayoutDebugArtifacts,
        bool GenerateImageDebugArtifacts,
        bool GenerateTableDebugArtifacts,
        IReadOnlyList<string> PipelineStages);

    private sealed record CliTelemetrySnapshot(
        string DocumentId,
        string SourcePath,
        int PageCount,
        int DocumentItemCount,
        int ParagraphCount,
        int HeadingCount,
        int PictureCount,
        int TableCount,
        int CaptionCount,
        int DocumentPropertyCount,
        int MetadataPropertyCount,
        int MarkdownLength,
        int AssetCount,
        int ExportedImageCount,
        int LayoutDebugCount,
        int ImageDebugCount,
        int TableDebugCount,
        int WorkflowDebugCount,
        int ImageExportCount,
        long ElapsedMilliseconds,
        DateTimeOffset GeneratedAt,
        IReadOnlyList<CliStageTelemetry> Stages);

    private sealed record CliStageTelemetry(string Stage, long DurationMilliseconds);
}
