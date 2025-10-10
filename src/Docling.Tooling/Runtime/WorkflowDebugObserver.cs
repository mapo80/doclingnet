using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
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
using Docling.Tooling.Commands;

namespace Docling.Tooling.Runtime;

internal sealed class WorkflowDebugObserver : IPipelineObserver
{
    private readonly ConvertCommandOptions _options;
    private readonly JsonSerializerOptions _serializerOptions;
    private readonly Dictionary<IPipelineStage, int> _stageOrder = new(ReferenceEqualityComparer.Instance);
    private readonly List<string> _createdFiles = new();
    private readonly object _gate = new();
    private int _nextStageIndex;

    public WorkflowDebugObserver(ConvertCommandOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _serializerOptions = new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };
    }

    private string RootDirectory => Path.Combine(_options.OutputDirectory, "debug", "workflow");

    public Task OnStageStartingAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (!_options.GenerateWorkflowDebugArtifacts)
        {
            return Task.CompletedTask;
        }

        lock (_gate)
        {
            if (!_stageOrder.ContainsKey(context.Stage))
            {
                _nextStageIndex++;
                _stageOrder[context.Stage] = _nextStageIndex;
            }
        }

        return Task.CompletedTask;
    }

    public async Task OnStageCompletedAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (!_options.GenerateWorkflowDebugArtifacts)
        {
            return;
        }

        int index;
        lock (_gate)
        {
            if (!_stageOrder.TryGetValue(context.Stage, out index))
            {
                _nextStageIndex++;
                index = _nextStageIndex;
                _stageOrder[context.Stage] = index;
            }
        }

        var snapshot = CreateSnapshot(context);
        if (snapshot is null)
        {
            return;
        }

        var path = GetStageFilePath(index, context.Stage.Name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            4096,
            FileOptions.Asynchronous | FileOptions.SequentialScan);

        await JsonSerializer.SerializeAsync(stream, snapshot, snapshot.GetType(), _serializerOptions, cancellationToken)
            .ConfigureAwait(false);

        lock (_gate)
        {
            _createdFiles.Add(Path.GetFullPath(path));
        }
    }

    public IReadOnlyList<string> GetCreatedArtifacts()
    {
        lock (_gate)
        {
            return _createdFiles.ToArray();
        }
    }

    private static WorkflowStageSnapshotBase? CreateSnapshot(PipelineStageExecutionContext context)
    {
        var stageName = string.IsNullOrWhiteSpace(context.Stage.Name)
            ? "stage"
            : context.Stage.Name;
        var timestamp = DateTimeOffset.UtcNow;
        var pipelineContext = context.PipelineContext;

        return stageName switch
        {
            "page_preprocessing" => CreatePreprocessingSnapshot(stageName, timestamp, pipelineContext),
            "layout_analysis" => CreateLayoutAnalysisSnapshot(stageName, timestamp, pipelineContext),
            "table_structure" => CreateTableStructureSnapshot(stageName, timestamp, pipelineContext),
            "ocr" => CreateOcrSnapshot(stageName, timestamp, pipelineContext),
            "page_assembly" => CreatePageAssemblySnapshot(stageName, timestamp, pipelineContext),
            "image_export" => CreateImageExportSnapshot(stageName, timestamp, pipelineContext),
            "markdown_serialization" => CreateMarkdownSerializationSnapshot(stageName, timestamp, pipelineContext),
            _ => new WorkflowStageSnapshot<GenericStagePayload>(stageName, timestamp, new GenericStagePayload("No snapshot available for this stage.")),
        };
    }

    private static WorkflowStageSnapshot<PagePreprocessingSnapshot> CreatePreprocessingSnapshot(string stage, DateTimeOffset timestamp, PipelineContext context)
    {
        var pages = context.TryGet<IReadOnlyList<PageReference>>(PipelineContextKeys.PageSequence, out var sequence) && sequence is not null
            ? sequence
            : Array.Empty<PageReference>();
        _ = context.TryGet<PageImageStore>(PipelineContextKeys.PageImageStore, out var store);

        var entries = new List<PagePreprocessingPageSnapshot>(pages.Count);
        foreach (var page in pages)
        {
            var cached = store?.Contains(page) ?? false;
            PageImage? snapshot = null;
            try
            {
                if (store is not null && store.TryRent(page, out var rented))
                {
                    snapshot = rented;
                }

                var width = snapshot?.Width;
                var height = snapshot?.Height;
                string? scale = null;
                string? colorMode = null;
                string? deskew = null;

                if (snapshot is not null)
                {
                    var properties = snapshot.Metadata.Properties;
                    if (properties.TryGetValue(PageImageMetadataKeys.ScaleFactor, out var scaleValue))
                    {
                        scale = scaleValue;
                    }

                    if (properties.TryGetValue(PageImageMetadataKeys.ColorMode, out var colorValue))
                    {
                        colorMode = colorValue;
                    }

                    if (properties.TryGetValue(PageImageMetadataKeys.DeskewAngle, out var deskewValue))
                    {
                        deskew = deskewValue;
                    }
                }

                entries.Add(new PagePreprocessingPageSnapshot(
                    page.PageNumber,
                    page.Dpi,
                    cached,
                    width,
                    height,
                    scale,
                    colorMode,
                    deskew));
            }
            finally
            {
                snapshot?.Dispose();
            }
        }

        var completed = context.TryGet<bool>(PipelineContextKeys.PreprocessingCompleted, out var flag) && flag;

        return new WorkflowStageSnapshot<PagePreprocessingSnapshot>(
            stage,
            timestamp,
            new PagePreprocessingSnapshot(completed, entries.Count, entries));
    }

    private static WorkflowStageSnapshot<LayoutAnalysisSnapshot> CreateLayoutAnalysisSnapshot(string stage, DateTimeOffset timestamp, PipelineContext context)
    {
        var items = context.TryGet<IReadOnlyList<LayoutItem>>(PipelineContextKeys.LayoutItems, out var layoutItems) && layoutItems is not null
            ? layoutItems
            : Array.Empty<LayoutItem>();
        var overlays = context.TryGet<IReadOnlyList<LayoutDebugOverlay>>(PipelineContextKeys.LayoutDebugArtifacts, out var debugOverlays) && debugOverlays is not null
            ? debugOverlays
            : Array.Empty<LayoutDebugOverlay>();
        var normalisations = context.TryGet<IReadOnlyList<LayoutNormalizationTelemetry>>(PipelineContextKeys.LayoutNormalisationMetadata, out var telemetry) && telemetry is not null
            ? telemetry
            : Array.Empty<LayoutNormalizationTelemetry>();
        var completed = context.TryGet<bool>(PipelineContextKeys.LayoutAnalysisCompleted, out var flag) && flag;

        var snapshots = new List<LayoutItemSnapshot>(items.Count);
        foreach (var item in items)
        {
            snapshots.Add(CreateLayoutItemSnapshot(item));
        }

        var overlaySnapshots = overlays.Select(overlay => new LayoutDebugOverlaySnapshot(overlay.Page.PageNumber)).ToList();
        var normalisationSnapshots = normalisations.Select(CreateLayoutNormalisationSnapshot).ToList();
        var failure = context.TryGet<string>(PipelineContextKeys.LayoutAnalysisError, out var error) ? error : null;

        return new WorkflowStageSnapshot<LayoutAnalysisSnapshot>(
            stage,
            timestamp,
            new LayoutAnalysisSnapshot(
                completed,
                snapshots.Count,
                overlaySnapshots.Count,
                normalisationSnapshots.Count,
                snapshots,
                overlaySnapshots,
                normalisationSnapshots,
                string.IsNullOrWhiteSpace(failure) ? null : failure));
    }

    private static WorkflowStageSnapshot<TableStructureStageSnapshot> CreateTableStructureSnapshot(string stage, DateTimeOffset timestamp, PipelineContext context)
    {
        var structures = context.TryGet<IReadOnlyList<TableStructure>>(PipelineContextKeys.TableStructures, out var tableStructures) && tableStructures is not null
            ? tableStructures
            : Array.Empty<TableStructure>();

        var snapshots = new List<TableStructureSnapshot>(structures.Count);
        foreach (var structure in structures)
        {
            snapshots.Add(CreateTableStructureSnapshot(structure));
        }

        return new WorkflowStageSnapshot<TableStructureStageSnapshot>(
            stage,
            timestamp,
            new TableStructureStageSnapshot(snapshots.Count, snapshots));
    }

    private static WorkflowStageSnapshot<OcrStageSnapshot> CreateOcrSnapshot(string stage, DateTimeOffset timestamp, PipelineContext context)
    {
        var completed = context.TryGet<bool>(PipelineContextKeys.OcrCompleted, out var flag) && flag;
        var result = context.TryGet<OcrDocumentResult>(PipelineContextKeys.OcrResults, out var ocrResults) && ocrResults is not null
            ? ocrResults
            : new OcrDocumentResult(Array.Empty<OcrBlockResult>());

        var blocks = new List<OcrBlockSnapshot>(result.Blocks.Count);
        foreach (var block in result.Blocks)
        {
            blocks.Add(CreateOcrBlockSnapshot(block));
        }

        return new WorkflowStageSnapshot<OcrStageSnapshot>(
            stage,
            timestamp,
            new OcrStageSnapshot(completed, blocks.Count, blocks));
    }

    private static WorkflowStageSnapshot<PageAssemblyStageSnapshot> CreatePageAssemblySnapshot(string stage, DateTimeOffset timestamp, PipelineContext context)
    {
        var completed = context.TryGet<bool>(PipelineContextKeys.DocumentAssemblyCompleted, out var flag) && flag;
        if (!context.TryGet(PipelineContextKeys.Document, out DoclingDocument? document) || document is null)
        {
            return new WorkflowStageSnapshot<PageAssemblyStageSnapshot>(
                stage,
                timestamp,
                new PageAssemblyStageSnapshot(
                    completed,
                    string.Empty,
                    string.Empty,
                    DateTimeOffset.MinValue,
                    0,
                    Array.Empty<PageReferenceSnapshot>(),
                    new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
                    0,
                    Array.Empty<DocItemSnapshot>()));
        }

        var pages = document.Pages
            .OrderBy(page => page.PageNumber)
            .Select(page => new PageReferenceSnapshot(page.PageNumber, page.Dpi))
            .ToList();
        var properties = document.Properties.ToDictionary(kvp => kvp.Key, kvp => kvp.Value, StringComparer.OrdinalIgnoreCase);
        var items = document.Items.Select(CreateDocItemSnapshot).ToList();

        return new WorkflowStageSnapshot<PageAssemblyStageSnapshot>(
            stage,
            timestamp,
            new PageAssemblyStageSnapshot(
                completed,
                document.Id,
                document.SourceId,
                document.CreatedAt,
                pages.Count,
                pages,
                properties,
                items.Count,
                items));
    }

    private static WorkflowStageSnapshot<ImageExportStageSnapshot> CreateImageExportSnapshot(string stage, DateTimeOffset timestamp, PipelineContext context)
    {
        var completed = context.TryGet<bool>(PipelineContextKeys.ImageExportCompleted, out var flag) && flag;
        var exports = context.TryGet<IReadOnlyList<ImageExportArtifact>>(PipelineContextKeys.ImageExports, out var artifacts) && artifacts is not null
            ? artifacts
            : Array.Empty<ImageExportArtifact>();
        var debugArtifacts = context.TryGet<IReadOnlyList<ImageExportDebugArtifact>>(PipelineContextKeys.ImageExportDebugArtifacts, out var debug) && debug is not null
            ? debug
            : Array.Empty<ImageExportDebugArtifact>();

        var exportSnapshots = exports.Select(CreateImageExportSnapshot).ToList();
        var debugSnapshots = debugArtifacts.Select(CreateImageExportDebugSnapshot).ToList();

        return new WorkflowStageSnapshot<ImageExportStageSnapshot>(
            stage,
            timestamp,
            new ImageExportStageSnapshot(
                completed,
                exportSnapshots.Count,
                exportSnapshots,
                debugSnapshots.Count,
                debugSnapshots));
    }

    private static WorkflowStageSnapshot<MarkdownSerializationStageSnapshot> CreateMarkdownSerializationSnapshot(string stage, DateTimeOffset timestamp, PipelineContext context)
    {
        var completed = context.TryGet<bool>(PipelineContextKeys.MarkdownSerializationCompleted, out var flag) && flag;
        if (!context.TryGet(PipelineContextKeys.MarkdownSerializationResult, out MarkdownSerializationResult? serializationResult) || serializationResult is null)
        {
            return new WorkflowStageSnapshot<MarkdownSerializationStageSnapshot>(
                stage,
                timestamp,
                new MarkdownSerializationStageSnapshot(
                    completed,
                    0,
                    0,
                    new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
                    string.Empty,
                    Array.Empty<MarkdownAssetSnapshot>()));
        }

        var metadata = serializationResult.Metadata.ToDictionary(kvp => kvp.Key, kvp => kvp.Value, StringComparer.OrdinalIgnoreCase);
        var assets = serializationResult.Assets.Select(CreateMarkdownAssetSnapshot).ToList();

        return new WorkflowStageSnapshot<MarkdownSerializationStageSnapshot>(
            stage,
            timestamp,
            new MarkdownSerializationStageSnapshot(
                completed,
                serializationResult.Markdown.Length,
                assets.Count,
                metadata,
                serializationResult.Markdown,
                assets));
    }

    private static LayoutItemSnapshot CreateLayoutItemSnapshot(LayoutItem item)
    {
        var polygons = item.Polygons?.Count > 0
            ? item.Polygons.Select(ToPolygonSnapshot).ToList()
            : new List<PolygonSnapshot>();

        return new LayoutItemSnapshot(
            item.Page.PageNumber,
            item.Kind.ToString(),
            ToBoundingBoxSnapshot(item.BoundingBox),
            polygons);
    }

    private static LayoutNormalisationSnapshot CreateLayoutNormalisationSnapshot(LayoutNormalizationTelemetry telemetry)
    {
        var metadata = telemetry.Metadata;
        return new LayoutNormalisationSnapshot(
            telemetry.Page.PageNumber,
            metadata.OriginalWidth,
            metadata.OriginalHeight,
            metadata.ScaledWidth,
            metadata.ScaledHeight,
            metadata.Scale,
            metadata.OffsetX,
            metadata.OffsetY);
    }

    private static TableStructureSnapshot CreateTableStructureSnapshot(TableStructure structure)
    {
        var cells = structure.Cells?.Count > 0
            ? structure.Cells.Select(CreateTableCellSnapshot).ToList()
            : new List<TableCellSnapshot>();
        var boundingBox = ComputeBoundingBox(structure.Cells);
        var hasDebug = structure.DebugArtifact is TableStructureDebugArtifact artifact && !artifact.ImageContent.IsEmpty;

        return new TableStructureSnapshot(
            structure.Page.PageNumber,
            structure.RowCount,
            structure.ColumnCount,
            cells.Count,
            boundingBox,
            hasDebug,
            cells);
    }

    private static BoundingBoxSnapshot? ComputeBoundingBox(IReadOnlyList<TableCell>? cells)
    {
        if (cells is null || cells.Count == 0)
        {
            return null;
        }

        var aggregate = cells[0].BoundingBox;
        for (var i = 1; i < cells.Count; i++)
        {
            aggregate = aggregate.ExpandToInclude(cells[i].BoundingBox);
        }

        return ToBoundingBoxSnapshot(aggregate);
    }

    private static TableCellSnapshot CreateTableCellSnapshot(TableCell cell)
    {
        return new TableCellSnapshot(
            cell.RowSpan,
            cell.ColumnSpan,
            ToBoundingBoxSnapshot(cell.BoundingBox),
            cell.Text ?? string.Empty);
    }

    private static OcrBlockSnapshot CreateOcrBlockSnapshot(OcrBlockResult block)
    {
        var metadata = block.Metadata.ToDictionary(kvp => kvp.Key, kvp => kvp.Value, StringComparer.OrdinalIgnoreCase);
        var lines = block.Lines?.Count > 0
            ? block.Lines.Select(CreateOcrLineSnapshot).ToList()
            : new List<OcrLineSnapshot>();

        return new OcrBlockSnapshot(
            block.Page.PageNumber,
            block.Kind.ToString(),
            ToBoundingBoxSnapshot(block.Region),
            metadata,
            lines.Count,
            lines);
    }

    private static OcrLineSnapshot CreateOcrLineSnapshot(OcrLine line)
    {
        return new OcrLineSnapshot(line.Text, line.Confidence, ToBoundingBoxSnapshot(line.BoundingBox));
    }

    private static DocItemSnapshot CreateDocItemSnapshot(DocItem item)
    {
        var tags = item.Tags.Count > 0 ? item.Tags.ToArray() : Array.Empty<string>();
        var provenance = item.Provenance?.Count > 0
            ? item.Provenance.Select(CreateProvenanceSnapshot).ToList()
            : new List<DocItemProvenanceSnapshot>();
        var content = CreateDocItemContent(item);

        return new DocItemSnapshot(
            item.Id,
            item.Kind.ToString(),
            item.Page.PageNumber,
            ToBoundingBoxSnapshot(item.BoundingBox),
            tags,
            provenance,
            content);
    }

    private static object? CreateDocItemContent(DocItem item)
    {
        return item switch
        {
            ParagraphItem paragraph => new ParagraphContentSnapshot(paragraph.Text),
            CaptionItem caption => new CaptionContentSnapshot(caption.Text, caption.TargetItemId),
            PictureItem picture => new PictureContentSnapshot(picture.Description, ToImageRefSnapshot(picture.Image)),
            TableItem table => new TableContentSnapshot(
                table.RowCount,
                table.ColumnCount,
                table.Cells.Count,
                table.Cells.Select(CreateTableCellContentSnapshot).ToList(),
                ToImageRefSnapshot(table.PreviewImage)),
            _ => null,
        };
    }

    private static TableCellContentSnapshot CreateTableCellContentSnapshot(TableCellItem cell)
    {
        return new TableCellContentSnapshot(
            cell.RowIndex,
            cell.ColumnIndex,
            cell.RowSpan,
            cell.ColumnSpan,
            ToBoundingBoxSnapshot(cell.BoundingBox),
            cell.Text ?? string.Empty);
    }

    private static DocItemProvenanceSnapshot CreateProvenanceSnapshot(DocItemProvenance provenance)
    {
        return new DocItemProvenanceSnapshot(
            provenance.PageNumber,
            ToBoundingBoxSnapshot(provenance.BoundingBox),
            provenance.CharStart,
            provenance.CharEnd);
    }

    private static ImageExportSnapshot CreateImageExportSnapshot(ImageExportArtifact artifact)
    {
        return new ImageExportSnapshot(
            artifact.Kind.ToString(),
            artifact.Image.Id,
            artifact.TargetItemId,
            ToImageRefSnapshot(artifact.Image)!);
    }

    private static ImageExportDebugSnapshot CreateImageExportDebugSnapshot(ImageExportDebugArtifact artifact)
    {
        var items = artifact.Manifest.Items.Select(CreateImageExportDebugEntrySnapshot).ToList();
        return new ImageExportDebugSnapshot(
            artifact.Manifest.PageNumber,
            items.Count,
            items);
    }

    private static ImageExportDebugEntrySnapshot CreateImageExportDebugEntrySnapshot(ImageExportDebugEntry entry)
    {
        return new ImageExportDebugEntrySnapshot(
            entry.TargetItemId,
            entry.ImageId,
            entry.KindName,
            ToDebugBoundsSnapshot(entry.OriginalBounds),
            ToDebugBoundsSnapshot(entry.CropBounds),
            entry.MediaType,
            entry.Width,
            entry.Height,
            entry.Dpi,
            entry.Checksum);
    }

    private static MarkdownAssetSnapshot CreateMarkdownAssetSnapshot(MarkdownAsset asset)
    {
        return new MarkdownAssetSnapshot(
            asset.RelativePath,
            asset.Kind.ToString(),
            asset.TargetItemId,
            ToImageRefSnapshot(asset.Image)!);
    }

    private static ImageRefSnapshot? ToImageRefSnapshot(ImageRef? image)
    {
        if (image is null)
        {
            return null;
        }

        return new ImageRefSnapshot(
            image.Id,
            image.MediaType,
            image.Width,
            image.Height,
            image.Dpi,
            image.Checksum,
            image.Page.PageNumber,
            image.Page.Dpi,
            ToBoundingBoxSnapshot(image.SourceRegion));
    }

    private static PolygonSnapshot ToPolygonSnapshot(Polygon polygon)
    {
        var points = new List<PointSnapshot>(polygon.Count);
        for (var i = 0; i < polygon.Count; i++)
        {
            points.Add(new PointSnapshot(polygon[i].X, polygon[i].Y));
        }

        return new PolygonSnapshot(ToBoundingBoxSnapshot(polygon.BoundingBox), points);
    }

    private static BoundingBoxSnapshot ToBoundingBoxSnapshot(BoundingBox box)
    {
        return new BoundingBoxSnapshot(
            box.Left,
            box.Top,
            box.Right,
            box.Bottom,
            box.Width,
            box.Height);
    }

    private static ImageExportDebugBoundsSnapshot ToDebugBoundsSnapshot(ImageExportDebugBounds bounds)
    {
        return new ImageExportDebugBoundsSnapshot(bounds.Left, bounds.Top, bounds.Right, bounds.Bottom);
    }

    private string GetStageFilePath(int index, string? stageName)
    {
        var name = string.IsNullOrWhiteSpace(stageName) ? "stage" : stageName;
        var sanitized = Sanitize(name);
        return Path.Combine(RootDirectory, string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{index:00}_{sanitized}.json"));
    }

    private static string Sanitize(string value)
    {
        Span<char> buffer = value.Length <= 64 ? stackalloc char[value.Length] : new char[value.Length];
        for (var i = 0; i < value.Length; i++)
        {
            var ch = value[i];
            buffer[i] = char.IsLetterOrDigit(ch) || ch is '_' or '-' ? ch : '_';
        }

        return new string(buffer);
    }

    private sealed record WorkflowStageSnapshot<T>(string Stage, DateTimeOffset CompletedAt, T Data) : WorkflowStageSnapshotBase(Stage, CompletedAt);

    private abstract record WorkflowStageSnapshotBase(string Stage, DateTimeOffset CompletedAt);

    private sealed record GenericStagePayload(string Message);

    private sealed record PagePreprocessingSnapshot(bool Completed, int PageCount, IReadOnlyList<PagePreprocessingPageSnapshot> Pages);

    private sealed record PagePreprocessingPageSnapshot(
        int PageNumber,
        double Dpi,
        bool Cached,
        int? Width,
        int? Height,
        string? ScaleFactor,
        string? ColorMode,
        string? DeskewAngle);

    private sealed record LayoutAnalysisSnapshot(
        bool Completed,
        int ItemCount,
        int DebugOverlayCount,
        int NormalisationCount,
        IReadOnlyList<LayoutItemSnapshot> Items,
        IReadOnlyList<LayoutDebugOverlaySnapshot> DebugOverlays,
        IReadOnlyList<LayoutNormalisationSnapshot> Normalisations,
        string? Failure);

    private sealed record LayoutItemSnapshot(
        int PageNumber,
        string Kind,
        BoundingBoxSnapshot BoundingBox,
        IReadOnlyList<PolygonSnapshot> Polygons);

    private sealed record LayoutDebugOverlaySnapshot(int PageNumber);

    private sealed record LayoutNormalisationSnapshot(
        int PageNumber,
        int OriginalWidth,
        int OriginalHeight,
        int ScaledWidth,
        int ScaledHeight,
        double Scale,
        double OffsetX,
        double OffsetY);

    private sealed record TableStructureStageSnapshot(int TableCount, IReadOnlyList<TableStructureSnapshot> Tables);

    private sealed record TableStructureSnapshot(
        int PageNumber,
        int RowCount,
        int ColumnCount,
        int CellCount,
        BoundingBoxSnapshot? BoundingBox,
        bool HasDebugArtifact,
        IReadOnlyList<TableCellSnapshot> Cells);

    private sealed record TableCellSnapshot(int RowSpan, int ColumnSpan, BoundingBoxSnapshot BoundingBox, string Text);

    private sealed record OcrStageSnapshot(bool Completed, int BlockCount, IReadOnlyList<OcrBlockSnapshot> Blocks);

    private sealed record OcrBlockSnapshot(
        int PageNumber,
        string Kind,
        BoundingBoxSnapshot Region,
        IReadOnlyDictionary<string, string> Metadata,
        int LineCount,
        IReadOnlyList<OcrLineSnapshot> Lines);

    private sealed record OcrLineSnapshot(string Text, double Confidence, BoundingBoxSnapshot BoundingBox);

    private sealed record PageAssemblyStageSnapshot(
        bool Completed,
        string DocumentId,
        string SourceId,
        DateTimeOffset CreatedAt,
        int PageCount,
        IReadOnlyList<PageReferenceSnapshot> Pages,
        IReadOnlyDictionary<string, string> Properties,
        int ItemCount,
        IReadOnlyList<DocItemSnapshot> Items);

    private sealed record PageReferenceSnapshot(int PageNumber, double Dpi);

    private sealed record DocItemSnapshot(
        string Id,
        string Kind,
        int PageNumber,
        BoundingBoxSnapshot BoundingBox,
        IReadOnlyList<string> Tags,
        IReadOnlyList<DocItemProvenanceSnapshot> Provenance,
        object? Content);

    private sealed record DocItemProvenanceSnapshot(int PageNumber, BoundingBoxSnapshot BoundingBox, int? CharStart, int? CharEnd);

    private sealed record ParagraphContentSnapshot(string Text);

    private sealed record CaptionContentSnapshot(string Text, string? TargetItemId);

    private sealed record PictureContentSnapshot(string Description, ImageRefSnapshot? Image);

    private sealed record TableContentSnapshot(
        int RowCount,
        int ColumnCount,
        int CellCount,
        IReadOnlyList<TableCellContentSnapshot> Cells,
        ImageRefSnapshot? PreviewImage);

    private sealed record TableCellContentSnapshot(
        int RowIndex,
        int ColumnIndex,
        int RowSpan,
        int ColumnSpan,
        BoundingBoxSnapshot BoundingBox,
        string Text);

    private sealed record ImageExportStageSnapshot(
        bool Completed,
        int ExportCount,
        IReadOnlyList<ImageExportSnapshot> Exports,
        int DebugArtifactCount,
        IReadOnlyList<ImageExportDebugSnapshot> DebugArtifacts);

    private sealed record ImageExportSnapshot(string Kind, string ImageId, string? TargetItemId, ImageRefSnapshot Image);

    private sealed record ImageExportDebugSnapshot(int PageNumber, int ItemCount, IReadOnlyList<ImageExportDebugEntrySnapshot> Items);

    private sealed record ImageExportDebugEntrySnapshot(
        string TargetItemId,
        string ImageId,
        string Kind,
        ImageExportDebugBoundsSnapshot OriginalBounds,
        ImageExportDebugBoundsSnapshot CropBounds,
        string MediaType,
        int Width,
        int Height,
        double Dpi,
        string? Checksum);

    private sealed record ImageExportDebugBoundsSnapshot(double Left, double Top, double Right, double Bottom);

    private sealed record MarkdownSerializationStageSnapshot(
        bool Completed,
        int MarkdownLength,
        int AssetCount,
        IReadOnlyDictionary<string, string> Metadata,
        string Markdown,
        IReadOnlyList<MarkdownAssetSnapshot> Assets);

    private sealed record MarkdownAssetSnapshot(string RelativePath, string Kind, string? TargetItemId, ImageRefSnapshot Image);

    private sealed record ImageRefSnapshot(
        string Id,
        string MediaType,
        int Width,
        int Height,
        double Dpi,
        string? Checksum,
        int PageNumber,
        double PageDpi,
        BoundingBoxSnapshot SourceRegion);

    private sealed record PolygonSnapshot(BoundingBoxSnapshot BoundingBox, IReadOnlyList<PointSnapshot> Points);

    private sealed record PointSnapshot(double X, double Y);

    private sealed record BoundingBoxSnapshot(double Left, double Top, double Right, double Bottom, double Width, double Height);
}
