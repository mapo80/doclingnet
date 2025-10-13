using System;

namespace LayoutSdk.Metrics;

public readonly record struct LayoutExecutionMetrics(
    TimeSpan PreprocessDuration,
    TimeSpan InferenceDuration,
    TimeSpan OverlayDuration)
{
    public TimeSpan TotalDuration => PreprocessDuration + InferenceDuration + OverlayDuration;

    public TimeSpan PostprocessDuration { get; init; } = TimeSpan.Zero;

    public TimeSpan FullTotalDuration => PreprocessDuration + InferenceDuration + PostprocessDuration + OverlayDuration;
}
