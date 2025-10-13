using SkiaSharp;
using System;
using System.IO;
using TableFormerSdk.Backends;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;
using TableFormerSdk.Rendering;

namespace TableFormerSdk;

public sealed class TableFormer : IDisposable
{
    private readonly TableFormerSdkOptions _options;
    private readonly BackendRegistry _backendRegistry;
    private readonly OverlayRenderer _overlayRenderer;
    private readonly TableFormerPerformanceAdvisor _performanceAdvisor;

    public TableFormer(TableFormerSdkOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _backendRegistry = new BackendRegistry(new DefaultBackendFactory(_options));
        _overlayRenderer = new OverlayRenderer(_options.Visualization);
        _performanceAdvisor = new TableFormerPerformanceAdvisor(_options.Performance, _options.AvailableRuntimes);
    }

    public TableStructureResult Process(
        string imagePath,
        bool overlay,
        TableFormerModelVariant variant,
        TableFormerRuntime runtime = TableFormerRuntime.Auto,
        TableFormerLanguage? language = null)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path is empty", nameof(imagePath));
        }

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException("Image not found", imagePath);
        }

        using var bitmap = SKBitmap.Decode(imagePath) ?? throw new InvalidOperationException("Unable to decode image");
        var selectedLanguage = language ?? _options.DefaultLanguage;
        _options.EnsureLanguageIsSupported(selectedLanguage);

        var backendKey = _performanceAdvisor.ResolveBackend(variant, runtime);
        var backend = _backendRegistry.GetOrCreateBackend(backendKey.Runtime, backendKey.Variant);
        var stopwatch = ValueStopwatch.StartNew();
        var regions = backend.Infer(bitmap, imagePath);
        var elapsed = stopwatch.GetElapsedTime();
        var snapshot = _performanceAdvisor.Record(backendKey, elapsed);
        var overlayImage = overlay ? _overlayRenderer.CreateOverlay(bitmap, regions) : null;
        return new TableStructureResult(regions, overlayImage, selectedLanguage, backendKey.Runtime, elapsed, snapshot);
    }

    public void RegisterBackend(TableFormerRuntime runtime, TableFormerModelVariant variant, ITableFormerBackend backend)
        => _backendRegistry.RegisterBackend(runtime, variant, backend);

    public IReadOnlyList<TableFormerPerformanceSnapshot> GetPerformanceSnapshots(TableFormerModelVariant variant)
        => _performanceAdvisor.GetSnapshots(variant);

    public TableFormerPerformanceSnapshot? GetLatestSnapshot(TableFormerRuntime runtime, TableFormerModelVariant variant)
        => _performanceAdvisor.TryGetSnapshot(runtime, variant);

    public void Dispose()
    {
        _backendRegistry.Dispose();
    }
}
