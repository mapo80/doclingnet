using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Pipelines.Options;
using EasyOcrNet;
using EasyOcrNet.Languages;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using EasyOcrConfigurationOptions = EasyOcrNet.Configuration.OcrOptions;

namespace Docling.Models.Ocr;

/// <summary>
/// EasyOCR implementation of <see cref="IOcrService"/> backed by the EasyOcrNet package.
/// </summary>
public sealed class EasyOcrService : IOcrService
{
    private static readonly Dictionary<string, OcrLanguage> LanguageMap = CreateLanguageMap();
    private static readonly Action<ILogger, InferenceBackend, string, OcrLanguage, string, Exception?> LogInitialised =
        LoggerMessage.Define<InferenceBackend, string, OcrLanguage, string>(
            LogLevel.Information,
            new EventId(1, nameof(EasyOcrService) + "Initialised"),
            "Initialised EasyOCR.NET backend {Backend} ({Device}) with language {Language} using models in {Directory}.");

    private static readonly Action<ILogger, string, OcrLanguage, InferenceBackend, string, Exception?> LogConfiguration =
        LoggerMessage.Define<string, OcrLanguage, InferenceBackend, string>(
            LogLevel.Debug,
            new EventId(2, nameof(EasyOcrService) + "Configured"),
            "Configuring EasyOCR.NET: requested languages {Languages} resolved to {Language}, backend {Backend}, device {Device}.");

    private static readonly Action<ILogger, string, Exception?> LogMissingLanguage =
        LoggerMessage.Define<string>(
            LogLevel.Warning,
            new EventId(3, nameof(EasyOcrService) + "LanguageFallback"),
            "None of the configured EasyOCR languages ({Languages}) are supported. Falling back to English.");

    private readonly ILogger<EasyOcrService> _logger;
    private readonly IEasyOcrEngine _engine;
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private readonly string _modelDirectory;
    private readonly OcrLanguage _language;
    private readonly InferenceBackend _backend;
    private readonly string _device;
    private bool _disposed;

    public EasyOcrService(EasyOcrOptions options, ILogger<EasyOcrService>? logger = null)
        : this(EnsureOptions(options),
            logger ?? NullLogger<EasyOcrService>.Instance,
            CreateEngine(EnsureOptions(options), logger ?? NullLogger<EasyOcrService>.Instance, out var modelDirectory, out var language, out var backend, out var device),
            modelDirectory,
            language,
            backend,
            device)
    {
    }

    internal EasyOcrService(
        EasyOcrOptions options,
        ILogger<EasyOcrService> logger,
        IEasyOcrEngine engine,
        string modelDirectory,
        OcrLanguage language,
        InferenceBackend backend,
        string device)
    {
        ArgumentNullException.ThrowIfNull(options);
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _modelDirectory = modelDirectory ?? throw new ArgumentNullException(nameof(modelDirectory));
        _language = language;
        _backend = backend;
        _device = device ?? throw new ArgumentNullException(nameof(device));

        LogInitialised(_logger, _backend, _device, _language, _modelDirectory, null);
    }

    public async IAsyncEnumerable<OcrLine> RecognizeAsync(OcrRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Image);

        if (request.Region.IsEmpty)
        {
            throw new ArgumentException("The OCR region cannot be empty.", nameof(request));
        }

        cancellationToken.ThrowIfCancellationRequested();

        using var crop = CreateCrop(request.Image, request.Region);
        IReadOnlyList<OcrResult> results;

        await _semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            results = await Task.Run(() => _engine.Read(crop.Bitmap), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _semaphore.Release();
        }

        foreach (var result in results)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var normalizedText = OcrTextNormalizer.Normalize(result.Text);
            if (string.IsNullOrEmpty(normalizedText))
            {
                continue;
            }

            var translated = BoundingBox.FromSize(
                crop.OffsetX + result.BoundingBox.Left,
                crop.OffsetY + result.BoundingBox.Top,
                result.BoundingBox.Width,
                result.BoundingBox.Height);

            if (translated.IsEmpty)
            {
                continue;
            }

            yield return new OcrLine(normalizedText, translated, Confidence: 1.0d);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _engine.Dispose();
        _semaphore.Dispose();
    }

    private static EasyOcrEngine CreateEngine(
        EasyOcrOptions options,
        ILogger logger,
        out string modelDirectory,
        out OcrLanguage language,
        out InferenceBackend backend,
        out string device)
    {
        language = ResolveLanguage(options, logger);
        backend = ResolveBackend(options);
        device = ResolveDevice(options);
        modelDirectory = ResolveModelDirectory(options);

        LogConfiguration(logger, string.Join(",", options.Languages), language, backend, device, null);

        var easyOptions = new EasyOcrConfigurationOptions(modelDirectory, language, backend, device);
        return new EasyOcrEngine(new EasyOcrNet.EasyOcr(easyOptions));
    }

    private static CropContext CreateCrop(SKBitmap source, BoundingBox region)
    {
        ArgumentNullException.ThrowIfNull(source);

        var left = (int)Math.Floor(region.Left);
        var top = (int)Math.Floor(region.Top);
        var right = (int)Math.Ceiling(region.Right);
        var bottom = (int)Math.Ceiling(region.Bottom);

        left = Math.Clamp(left, 0, source.Width);
        top = Math.Clamp(top, 0, source.Height);
        right = Math.Clamp(right, 0, source.Width);
        bottom = Math.Clamp(bottom, 0, source.Height);

        if (right <= left || bottom <= top)
        {
            throw new ArgumentException("The requested OCR region does not intersect the source image.", nameof(region));
        }

        var rect = new SKRectI(left, top, right, bottom);
        var subset = new SKBitmap();

        if (!source.ExtractSubset(subset, rect))
        {
            subset.Dispose();
            throw new InvalidOperationException("Failed to copy the OCR crop from the source bitmap.");
        }

        return new CropContext(subset, left, top);
    }

    private static string ResolveModelDirectory(EasyOcrOptions options)
    {
        var directory = options.ModelStorageDirectory;
        if (!string.IsNullOrWhiteSpace(directory))
        {
            directory = Path.GetFullPath(directory);
            if (!Directory.Exists(directory))
            {
                throw new DirectoryNotFoundException($"EasyOCR model directory '{directory}' does not exist.");
            }

            return directory;
        }

        var defaultDirectory = Path.Combine(AppContext.BaseDirectory, "contentFiles", "any", "any", "models");
        if (!Directory.Exists(defaultDirectory))
        {
            throw new DirectoryNotFoundException($"Default EasyOCR model directory '{defaultDirectory}' not found. Ensure the EasyOcrNet package is restored.");
        }

        return defaultDirectory;
    }

    private static OcrLanguage ResolveLanguage(EasyOcrOptions options, ILogger logger)
    {
        foreach (var language in options.Languages)
        {
            if (LanguageMap.TryGetValue(language, out var mapped))
            {
                return mapped;
            }
        }

        LogMissingLanguage(logger, string.Join(",", options.Languages), null);
        return OcrLanguage.English;
    }

    private static InferenceBackend ResolveBackend(EasyOcrOptions options)
    {
        if (options.UseGpu == true)
        {
            return InferenceBackend.OpenVino;
        }

        return InferenceBackend.Onnx;
    }

    private static string ResolveDevice(EasyOcrOptions options)
    {
        if (options.UseGpu == true)
        {
            return "GPU";
        }

        return "CPU";
    }

    private static Dictionary<string, OcrLanguage> CreateLanguageMap()
    {
        var map = new Dictionary<string, OcrLanguage>(StringComparer.OrdinalIgnoreCase)
        {
            ["af"] = OcrLanguage.Afrikaans,
            ["afk"] = OcrLanguage.Afrikaans,
            ["sq"] = OcrLanguage.Albanian,
            ["eu"] = OcrLanguage.Basque,
            ["ca"] = OcrLanguage.Catalan,
            ["hr"] = OcrLanguage.Croatian,
            ["cs"] = OcrLanguage.Czech,
            ["da"] = OcrLanguage.Danish,
            ["nl"] = OcrLanguage.Dutch,
            ["en"] = OcrLanguage.English,
            ["eng"] = OcrLanguage.English,
            ["et"] = OcrLanguage.Estonian,
            ["fil"] = OcrLanguage.Filipino,
            ["fi"] = OcrLanguage.Finnish,
            ["fr"] = OcrLanguage.French,
            ["gl"] = OcrLanguage.Galician,
            ["de"] = OcrLanguage.German,
            ["hu"] = OcrLanguage.Hungarian,
            ["is"] = OcrLanguage.Icelandic,
            ["id"] = OcrLanguage.Indonesian,
            ["ga"] = OcrLanguage.Irish,
            ["it"] = OcrLanguage.Italian,
            ["ku"] = OcrLanguage.Kurdish,
            ["la"] = OcrLanguage.Latin,
            ["lv"] = OcrLanguage.Latvian,
            ["lt"] = OcrLanguage.Lithuanian,
            ["mi"] = OcrLanguage.Maori,
            ["ms"] = OcrLanguage.Malay,
            ["mt"] = OcrLanguage.Maltese,
            ["no"] = OcrLanguage.Norwegian,
            ["nb"] = OcrLanguage.Norwegian,
            ["nn"] = OcrLanguage.Norwegian,
            ["pl"] = OcrLanguage.Polish,
            ["pt"] = OcrLanguage.Portuguese,
            ["pt-br"] = OcrLanguage.Portuguese,
            ["ro"] = OcrLanguage.Romanian,
            ["sr-latn"] = OcrLanguage.SerbianLatin,
            ["sk"] = OcrLanguage.Slovak,
            ["sl"] = OcrLanguage.Slovenian,
            ["es"] = OcrLanguage.Spanish,
            ["sw"] = OcrLanguage.Swahili,
            ["sv"] = OcrLanguage.Swedish,
            ["tr"] = OcrLanguage.Turkish,
            ["uz"] = OcrLanguage.Uzbek,
            ["vi"] = OcrLanguage.Vietnamese,
            ["zh"] = OcrLanguage.SimplifiedChinese,
            ["zh-cn"] = OcrLanguage.SimplifiedChinese,
            ["ja"] = OcrLanguage.Japanese,
            ["ko"] = OcrLanguage.Korean,
            ["th"] = OcrLanguage.Thai,
        };

        return map;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, nameof(EasyOcrService));
    }

    private static EasyOcrOptions EnsureOptions(EasyOcrOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return options;
    }

    private sealed class CropContext : IDisposable
    {
        public CropContext(SKBitmap bitmap, double offsetX, double offsetY)
        {
            Bitmap = bitmap ?? throw new ArgumentNullException(nameof(bitmap));
            OffsetX = offsetX;
            OffsetY = offsetY;
        }

        public SKBitmap Bitmap { get; }

        public double OffsetX { get; }

        public double OffsetY { get; }

        public void Dispose() => Bitmap.Dispose();
    }

    internal interface IEasyOcrEngine : IDisposable
    {
        IReadOnlyList<OcrResult> Read(SKBitmap image);
    }

    private sealed class EasyOcrEngine : IEasyOcrEngine
    {
        private readonly EasyOcrNet.EasyOcr _engine;

        public EasyOcrEngine(EasyOcrNet.EasyOcr engine)
        {
            _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        }

        public IReadOnlyList<OcrResult> Read(SKBitmap image) => _engine.Read(image);

        public void Dispose() => _engine.Dispose();
    }
}
