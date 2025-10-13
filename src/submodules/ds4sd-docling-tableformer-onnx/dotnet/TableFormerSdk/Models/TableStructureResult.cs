using SkiaSharp;
using System;
using System.Collections.Generic;
using TableFormerSdk.Enums;
using TableFormerSdk.Performance;

namespace TableFormerSdk.Models;

public sealed class TableStructureResult
{
    public TableStructureResult(
        IReadOnlyList<TableRegion> regions,
        SKBitmap? overlay,
        TableFormerLanguage language,
        TableFormerRuntime runtime,
        TimeSpan inferenceTime,
        TableFormerPerformanceSnapshot performanceSnapshot)
    {
        Regions = regions ?? throw new ArgumentNullException(nameof(regions));
        OverlayImage = overlay;
        Language = language;
        Runtime = runtime;
        InferenceTime = inferenceTime;
        PerformanceSnapshot = performanceSnapshot ?? throw new ArgumentNullException(nameof(performanceSnapshot));
    }

    public IReadOnlyList<TableRegion> Regions { get; }

    public SKBitmap? OverlayImage { get; }

    public TableFormerLanguage Language { get; }

    public TableFormerRuntime Runtime { get; }

    public TimeSpan InferenceTime { get; }

    public TableFormerPerformanceSnapshot PerformanceSnapshot { get; }
}

public sealed record TableRegion(float X, float Y, float Width, float Height, string Label);
