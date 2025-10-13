using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Performance;

public sealed class TableFormerPerformanceOptions
{
    public TableFormerPerformanceOptions(
        bool enableAdaptiveRuntimeSelection = true,
        int minimumSamples = 3,
        int slidingWindowSize = 20,
        TableFormerRuntime defaultRuntime = TableFormerRuntime.Onnx,
        IEnumerable<TableFormerRuntime>? runtimePriority = null)
    {
        if (minimumSamples < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumSamples), minimumSamples, "Minimum samples must be positive");
        }

        if (slidingWindowSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(slidingWindowSize), slidingWindowSize, "Sliding window size must be positive");
        }

        EnableAdaptiveRuntimeSelection = enableAdaptiveRuntimeSelection;
        MinimumSamples = minimumSamples;
        SlidingWindowSize = slidingWindowSize;
        DefaultRuntime = defaultRuntime;
        RuntimePriority = BuildRuntimePriority(runtimePriority);
    }

    public bool EnableAdaptiveRuntimeSelection { get; }

    public int MinimumSamples { get; }

    public int SlidingWindowSize { get; }

    public TableFormerRuntime DefaultRuntime { get; }

    public IReadOnlyList<TableFormerRuntime> RuntimePriority { get; }

    public static TableFormerPerformanceOptions CreateDefault() => new();

    private static IReadOnlyList<TableFormerRuntime> BuildRuntimePriority(IEnumerable<TableFormerRuntime>? runtimePriority)
    {
        var ordered = runtimePriority is null
            ? new List<TableFormerRuntime>()
            : runtimePriority.Where(r => r != TableFormerRuntime.Auto).Distinct().ToList();

        return new ReadOnlyCollection<TableFormerRuntime>(ordered);
    }
}
