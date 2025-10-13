using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using TableFormerSdk.Backends;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Performance;

internal sealed class TableFormerPerformanceAdvisor
{
    private readonly TableFormerPerformanceOptions _options;
    private readonly PerformanceMonitor _monitor;
    private readonly IReadOnlyList<TableFormerRuntime> _runtimeOrder;

    public TableFormerPerformanceAdvisor(TableFormerPerformanceOptions options, IEnumerable<TableFormerRuntime> availableRuntimes)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _monitor = new PerformanceMonitor(options.SlidingWindowSize);
        _runtimeOrder = BuildRuntimeOrder(options, availableRuntimes);
    }

    public BackendKey ResolveBackend(TableFormerModelVariant variant, TableFormerRuntime requestedRuntime)
    {
        if (requestedRuntime != TableFormerRuntime.Auto || !_options.EnableAdaptiveRuntimeSelection)
        {
            var runtime = requestedRuntime == TableFormerRuntime.Auto
                ? GetFallbackRuntime()
                : requestedRuntime;
            return new BackendKey(runtime, variant);
        }

        if (_runtimeOrder.Count == 0)
        {
            throw new InvalidOperationException("No runtime is available for adaptive selection");
        }

        var pending = _monitor.FindRuntimeNeedingSamples(variant, _runtimeOrder, _options.MinimumSamples);
        if (pending is BackendKey pendingKey)
        {
            return pendingKey;
        }

        var best = _monitor.GetBestSnapshot(variant, _runtimeOrder, _options.MinimumSamples);
        if (best is not null)
        {
            return new BackendKey(best.Runtime, variant);
        }

        return new BackendKey(GetFallbackRuntime(), variant);
    }

    public TableFormerPerformanceSnapshot Record(BackendKey backendKey, TimeSpan duration)
        => _monitor.Record(backendKey, duration);

    public IReadOnlyList<TableFormerPerformanceSnapshot> GetSnapshots(TableFormerModelVariant variant)
        => _monitor.GetSnapshots(variant);

    public TableFormerPerformanceSnapshot? TryGetSnapshot(TableFormerRuntime runtime, TableFormerModelVariant variant)
        => _monitor.TryGetSnapshot(new BackendKey(runtime, variant));

    private TableFormerRuntime GetFallbackRuntime()
    {
        if (_runtimeOrder.Count == 0)
        {
            return _options.DefaultRuntime;
        }

        return _runtimeOrder.Contains(_options.DefaultRuntime)
            ? _options.DefaultRuntime
            : _runtimeOrder[0];
    }

    private static IReadOnlyList<TableFormerRuntime> BuildRuntimeOrder(TableFormerPerformanceOptions options, IEnumerable<TableFormerRuntime> availableRuntimes)
    {
        var order = new List<TableFormerRuntime>();

        if (options.RuntimePriority.Count > 0)
        {
            foreach (var runtime in options.RuntimePriority)
            {
                if (!order.Contains(runtime))
                {
                    order.Add(runtime);
                }
            }
        }

        foreach (var runtime in availableRuntimes)
        {
            if (runtime == TableFormerRuntime.Auto)
            {
                continue;
            }

            if (!order.Contains(runtime))
            {
                order.Add(runtime);
            }
        }

        return new ReadOnlyCollection<TableFormerRuntime>(order);
    }
}
