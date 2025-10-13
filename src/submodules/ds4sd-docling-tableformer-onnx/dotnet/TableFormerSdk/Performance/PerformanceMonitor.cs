using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using TableFormerSdk.Backends;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Performance;

internal sealed class PerformanceMonitor
{
    private readonly ConcurrentDictionary<BackendKey, TimeWindowStatistics> _statistics = new();
    private readonly int _windowSize;

    public PerformanceMonitor(int windowSize)
    {
        _windowSize = windowSize;
    }

    public TableFormerPerformanceSnapshot Record(BackendKey backendKey, TimeSpan duration)
    {
        var stats = _statistics.GetOrAdd(backendKey, _ => new TimeWindowStatistics(_windowSize));
        return stats.Register(backendKey, duration);
    }

    public TableFormerPerformanceSnapshot? TryGetSnapshot(BackendKey backendKey)
    {
        if (_statistics.TryGetValue(backendKey, out var stats) && stats.TotalSampleCount > 0)
        {
            return stats.Snapshot(backendKey);
        }

        return null;
    }

    public IReadOnlyList<TableFormerPerformanceSnapshot> GetSnapshots(TableFormerModelVariant variant)
    {
        return _statistics
            .Where(kvp => kvp.Key.Variant == variant && kvp.Value.TotalSampleCount > 0)
            .Select(kvp => kvp.Value.Snapshot(kvp.Key))
            .OrderBy(snapshot => snapshot.AverageLatencyMilliseconds)
            .ToList();
    }

    public BackendKey? FindRuntimeNeedingSamples(TableFormerModelVariant variant, IEnumerable<TableFormerRuntime> runtimes, int minimumSamples)
    {
        foreach (var runtime in runtimes)
        {
            var key = new BackendKey(runtime, variant);
            if (!_statistics.TryGetValue(key, out var stats) || stats.TotalSampleCount < minimumSamples)
            {
                return key;
            }
        }

        return null;
    }

    public TableFormerPerformanceSnapshot? GetBestSnapshot(TableFormerModelVariant variant, IEnumerable<TableFormerRuntime> runtimes, int minimumSamples)
    {
        TableFormerPerformanceSnapshot? best = null;
        foreach (var runtime in runtimes)
        {
            var key = new BackendKey(runtime, variant);
            if (_statistics.TryGetValue(key, out var stats) && stats.WindowSampleCount >= minimumSamples)
            {
                var snapshot = stats.Snapshot(key);
                if (best is null || snapshot.AverageLatencyMilliseconds < best.AverageLatencyMilliseconds)
                {
                    best = snapshot;
                }
            }
        }

        return best;
    }
}
