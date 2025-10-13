using LayoutSdk.Configuration;
using LayoutSdk.Factories;
using LayoutSdk.Metrics;
using LayoutSdk.Processing;
using LayoutSdk.Rendering;
using SkiaSharp;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;

namespace LayoutSdk;

public sealed class LayoutSdk : IDisposable
{
    private readonly ConcurrentDictionary<LayoutRuntime, LayoutPipeline> _pipelines = new();
    private readonly LayoutSdkOptions _options;
    private readonly ILayoutBackendFactory _backendFactory;
    private readonly IImageOverlayRenderer _overlayRenderer;
    private readonly IImagePreprocessor _preprocessor;

    public LayoutSdk(
        LayoutSdkOptions options,
        ILayoutBackendFactory? backendFactory = null,
        IImageOverlayRenderer? overlayRenderer = null,
        IImagePreprocessor? preprocessor = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _backendFactory = backendFactory ?? new LayoutBackendFactory(_options);
        _overlayRenderer = overlayRenderer ?? new ImageOverlayRenderer();
        _preprocessor = preprocessor ?? new SkiaImagePreprocessor();
    }

    public LayoutResult Process(string imagePath, bool overlay, LayoutRuntime runtime)
    {
        ValidateImagePath(imagePath);

        using var bitmap = SKBitmap.Decode(imagePath)
                           ?? throw new InvalidOperationException(LayoutDefaults.ImageDecodeFailureMessage);

        var pipeline = _pipelines.GetOrAdd(
            runtime,
            r =>
            {
                var backend = _backendFactory.Create(r);
                var postprocessor = new LayoutPostprocessor(LayoutPostprocessOptions.CreateDefault());
                return new LayoutPipeline(backend, _preprocessor, postprocessor);
            });
        var pipelineResult = pipeline.Execute(bitmap);

        var overlayDuration = TimeSpan.Zero;
        SKBitmap? overlayImage = null;
        if (overlay)
        {
            var overlayWatch = Stopwatch.StartNew();
            overlayImage = _overlayRenderer.CreateOverlay(bitmap, pipelineResult.Boxes);
            overlayWatch.Stop();
            overlayDuration = overlayWatch.Elapsed;
        }

        var metrics = new LayoutExecutionMetrics(
            pipelineResult.Metrics.PreprocessDuration,
            pipelineResult.Metrics.InferenceDuration,
            overlayDuration)
        {
            PostprocessDuration = pipelineResult.Metrics.PostprocessDuration
        };

        return new LayoutResult(pipelineResult.Boxes, overlayImage, _options.DefaultLanguage, metrics);
    }

    private static void ValidateImagePath(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException(LayoutDefaults.EmptyImagePathMessage, nameof(imagePath));
        }

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException(LayoutDefaults.ImageNotFoundMessage, imagePath);
        }
    }

    public void Dispose()
    {
        foreach (var pipeline in _pipelines.Values)
        {
            pipeline.Dispose();
        }
    }
}
