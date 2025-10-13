using System;
using System.Globalization;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Performance;

public sealed record TableFormerPerformanceSnapshot(
    TableFormerRuntime Runtime,
    TableFormerModelVariant Variant,
    int WindowSampleCount,
    int TotalSampleCount,
    double AverageLatencyMilliseconds,
    double BestLatencyMilliseconds,
    double LastLatencyMilliseconds)
{
    public override string ToString() => $"{Runtime}/{Variant}: n={WindowSampleCount}, avg={AverageLatencyMilliseconds:F3} ms, best={BestLatencyMilliseconds:F3} ms, last={LastLatencyMilliseconds:F3} ms";

    public TimeSpan AverageLatency => TimeSpan.FromMilliseconds(AverageLatencyMilliseconds);

    public TimeSpan BestLatency => TimeSpan.FromMilliseconds(BestLatencyMilliseconds);

    public TimeSpan LastLatency => TimeSpan.FromMilliseconds(LastLatencyMilliseconds);
}
