using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Docling.Pipelines.Options;

/// <summary>
/// Supported OCR engine identifiers.
/// </summary>
public enum OcrEngine
{
    EasyOcr,
    RapidOcr,
    TesseractCli,
    Tesseract,
    OcrMac,
}

/// <summary>
/// Base abstraction describing common OCR configuration surface.
/// </summary>
public abstract class OcrOptions
{
    private ReadOnlyCollection<string> _languages;
    private double _bitmapAreaThreshold = 0.05d;

    protected OcrOptions(OcrEngine engine, IReadOnlyList<string> defaultLanguages)
    {
        Engine = engine;
        _languages = CloneLanguages(defaultLanguages);
    }

    /// <summary>
    /// Identifies the OCR engine that should be instantiated.
    /// </summary>
    public OcrEngine Engine { get; }

    /// <summary>
    /// Languages the OCR engine should load. At least one language must be supplied.
    /// </summary>
    public IReadOnlyList<string> Languages
    {
        get => _languages;
        init => _languages = CloneLanguages(value);
    }

    /// <summary>
    /// If enabled the pipeline will always perform a full-page OCR even when programmatic text is available.
    /// </summary>
    public bool ForceFullPageOcr { get; init; }

    /// <summary>
    /// Fraction of bitmap area that must be exceeded before triggering OCR on a raster region.
    /// </summary>
    public double BitmapAreaThreshold
    {
        get => _bitmapAreaThreshold;
        init => _bitmapAreaThreshold = ValidateRatio(value, nameof(value));
    }

    private static ReadOnlyCollection<string> CloneLanguages(IReadOnlyList<string>? source)
    {
        ArgumentNullException.ThrowIfNull(source);

        if (source.Count == 0)
        {
            throw new ArgumentException("At least one language must be specified.", nameof(source));
        }

        var clone = new string[source.Count];
        for (var i = 0; i < source.Count; i++)
        {
            var language = source[i];
            if (string.IsNullOrWhiteSpace(language))
            {
                throw new ArgumentException("Language identifiers cannot be null or whitespace.", nameof(source));
            }

            clone[i] = language;
        }

        return new ReadOnlyCollection<string>(clone);
    }

    private static double ValidateRatio(double value, string paramName)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0d || value > 1d)
        {
            throw new ArgumentOutOfRangeException(paramName, value, "The value must be between 0 (exclusive) and 1 (inclusive).");
        }

        return value;
    }
}

/// <summary>
/// Configuration for the EasyOCR engine.
/// </summary>
public sealed class EasyOcrOptions : OcrOptions
{
    private static readonly IReadOnlyList<string> DefaultLanguages = new[] { "fr", "de", "es", "en" };

    public EasyOcrOptions()
        : base(OcrEngine.EasyOcr, DefaultLanguages)
    {
    }

    public bool? UseGpu { get; init; }

    public double ConfidenceThreshold { get; init; } = 0.5d;

    public string? ModelStorageDirectory { get; init; }

    public string? RecognitionNetwork { get; init; } = "standard";

    public bool DownloadEnabled { get; init; } = true;

    public bool SuppressMpsWarnings { get; init; } = true;
}

/// <summary>
/// Backend providers supported by RapidOCR.
/// </summary>
public enum RapidOcrBackend
{
    OnnxRuntime,
    OpenVino,
    Paddle,
    Torch,
}

/// <summary>
/// Configuration for the RapidOCR engine.
/// </summary>
public sealed class RapidOcrOptions : OcrOptions
{
    private static readonly IReadOnlyList<string> DefaultLanguages = new[] { "english", "chinese" };
    private IReadOnlyDictionary<string, object?> _parameters = CreateReadOnlyDictionary(new Dictionary<string, object?>());

    public RapidOcrOptions()
        : base(OcrEngine.RapidOcr, DefaultLanguages)
    {
    }

    public RapidOcrBackend Backend { get; init; } = RapidOcrBackend.OnnxRuntime;

    public double TextScore { get; init; } = 0.5d;

    public bool? UseDetector { get; init; }

    public bool? UseClassifier { get; init; }

    public bool? UseRecognizer { get; init; }

    public bool PrintVerbose { get; init; }

    public string? DetectorModelPath { get; init; }

    public string? ClassifierModelPath { get; init; }

    public string? RecognizerModelPath { get; init; }

    public string? RecognizerKeysPath { get; init; }

    public string? RecognizerFontPath { get; init; }

    public IReadOnlyDictionary<string, object?> Parameters
    {
        get => _parameters;
        init => _parameters = CreateReadOnlyDictionary(value ?? throw new ArgumentNullException(nameof(value)));
    }

    private static ReadOnlyDictionary<string, object?> CreateReadOnlyDictionary(IEnumerable<KeyValuePair<string, object?>> dictionary)
    {
        var copy = new Dictionary<string, object?>();
        foreach (var pair in dictionary)
        {
            copy[pair.Key] = pair.Value;
        }

        return new ReadOnlyDictionary<string, object?>(copy);
    }
}
