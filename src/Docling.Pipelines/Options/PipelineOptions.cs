using System;

namespace Docling.Pipelines.Options;

/// <summary>
/// Base configuration shared by all Docling pipeline implementations.
/// </summary>
public abstract class PipelineOptions
{
    private TimeSpan? _documentTimeout;
    private string? _artifactsPath;
    private AcceleratorOptions _accelerator = new();

    /// <summary>
    /// Hard timeout for processing an entire document. <c>null</c> disables the watchdog.
    /// </summary>
    public TimeSpan? DocumentTimeout
    {
        get => _documentTimeout;
        init
        {
            if (value is { } timeout && timeout <= TimeSpan.Zero)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Timeout must be greater than zero when specified.");
            }

            _documentTimeout = value;
        }
    }

    /// <summary>
    /// Controls accelerator selection and runtime tuning.
    /// </summary>
    public AcceleratorOptions Accelerator
    {
        get => _accelerator;
        init => _accelerator = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>
    /// When true remote services (OCR, layout, table structure) may be invoked.
    /// </summary>
    public bool EnableRemoteServices { get; init; }

    /// <summary>
    /// When true plugins originating outside the trusted bundle are allowed to execute.
    /// </summary>
    public bool AllowExternalPlugins { get; init; }

    /// <summary>
    /// Optional directory where generated artefacts (debug overlays, crops, etc.) should be emitted.
    /// </summary>
    public string? ArtifactsPath
    {
        get => _artifactsPath;
        init => _artifactsPath = string.IsNullOrWhiteSpace(value) ? null : value;
    }

    /// <summary>
    /// Validates the configuration state throwing an <see cref="InvalidOperationException"/> when inconsistencies are detected.
    /// </summary>
    public virtual void Validate()
    {
        // No-op for now, derived types add concrete rules.
    }
}

/// <summary>
/// Base class for pipelines that convert a document into markdown or similar artefacts.
/// </summary>
public abstract class ConvertPipelineOptions : PipelineOptions
{
    private PictureDescriptionOptions _pictureDescription = new();

    /// <summary>
    /// When enabled the pipeline will run picture classification on figure crops.
    /// </summary>
    public bool DoPictureClassification { get; init; }

    /// <summary>
    /// When enabled the pipeline generates natural language descriptions for figures.
    /// </summary>
    public bool DoPictureDescription { get; init; }

    /// <summary>
    /// Controls how picture descriptions are generated (inline VLM or remote API).
    /// </summary>
    public PictureDescriptionOptions PictureDescription
    {
        get => _pictureDescription;
        init => _pictureDescription = value ?? throw new ArgumentNullException(nameof(value));
    }

    public override void Validate()
    {
        base.Validate();
        PictureDescription.EnsureValid();
    }
}

/// <summary>
/// Base class for pipelines that operate on paginated sources such as PDFs.
/// </summary>
public abstract class PaginatedPipelineOptions : ConvertPipelineOptions
{
    private double _imagesScale = 1d;

    /// <summary>
    /// Scale factor applied when generating preview images.
    /// </summary>
    public double ImagesScale
    {
        get => _imagesScale;
        init => _imagesScale = value > 0d
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Image scale must be positive.");
    }

    /// <summary>
    /// Emit full page images alongside the converted output.
    /// </summary>
    public bool GeneratePageImages { get; init; }

    /// <summary>
    /// Emit cropped images for detected pictures.
    /// </summary>
    public bool GeneratePictureImages { get; init; }
}

/// <summary>
/// Options for the standard PDF pipeline.
/// </summary>
public class PdfPipelineOptions : PaginatedPipelineOptions
{
    private TableStructureOptions _tableStructure = new();
    private OcrOptions _ocr = new EasyOcrOptions();
    private LayoutOptions _layout = new();

    public bool DoTableStructure { get; init; } = true;

    public bool DoOcr { get; init; } = true;

    public bool DoCodeEnrichment { get; init; }

    public bool DoFormulaEnrichment { get; init; }

    public bool ForceBackendText { get; init; }

    public TableStructureOptions TableStructure
    {
        get => _tableStructure;
        init => _tableStructure = value ?? throw new ArgumentNullException(nameof(value));
    }

    public OcrOptions Ocr
    {
        get => _ocr;
        init => _ocr = value ?? throw new ArgumentNullException(nameof(value));
    }

    public LayoutOptions Layout
    {
        get => _layout;
        init => _layout = value ?? throw new ArgumentNullException(nameof(value));
    }

    [Obsolete("Use GeneratePageImages and TableItem.GetImage instead. Included for Python parity.")]
    public bool GenerateTableImages { get; init; }

    public bool GenerateParsedPages { get; init; }

    public override void Validate()
    {
        base.Validate();
        _ = TableStructure ?? throw new InvalidOperationException("Table structure options must not be null.");
        _ = Ocr ?? throw new InvalidOperationException("OCR options must not be null.");
        _ = Layout ?? throw new InvalidOperationException("Layout options must not be null.");
    }
}

/// <summary>
/// Options for the threaded PDF pipeline variant which orchestrates batched processing.
/// </summary>
public sealed class ThreadedPdfPipelineOptions : PdfPipelineOptions
{
    private int _ocrBatchSize = 4;
    private int _layoutBatchSize = 4;
    private int _tableBatchSize = 4;
    private TimeSpan _batchTimeout = TimeSpan.FromSeconds(2);
    private int _queueMaxSize = 100;

    public int OcrBatchSize
    {
        get => _ocrBatchSize;
        init => _ocrBatchSize = EnsurePositive(value, nameof(OcrBatchSize));
    }

    public int LayoutBatchSize
    {
        get => _layoutBatchSize;
        init => _layoutBatchSize = EnsurePositive(value, nameof(LayoutBatchSize));
    }

    public int TableBatchSize
    {
        get => _tableBatchSize;
        init => _tableBatchSize = EnsurePositive(value, nameof(TableBatchSize));
    }

    public TimeSpan BatchTimeout
    {
        get => _batchTimeout;
        init
        {
            if (value <= TimeSpan.Zero)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Batch timeout must be greater than zero.");
            }

            _batchTimeout = value;
        }
    }

    public int QueueMaxSize
    {
        get => _queueMaxSize;
        init => _queueMaxSize = EnsurePositive(value, nameof(QueueMaxSize));
    }

    public override void Validate()
    {
        base.Validate();
        _ = OcrBatchSize;
        _ = LayoutBatchSize;
        _ = TableBatchSize;
        _ = QueueMaxSize;
    }

    private static int EnsurePositive(int value, string parameterName)
    {
        return value > 0
            ? value
            : throw new ArgumentOutOfRangeException(parameterName, value, "Value must be positive.");
    }
}
