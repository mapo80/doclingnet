using System;
using System.Collections.Generic;
using System.Linq;
using TableFormerSdk.Backends;

namespace TableFormerSdk.Performance;

internal sealed class TimeWindowStatistics
{
    private readonly int _capacity;
    private readonly Queue<double> _window = new();
    private double _windowSum;
    private double _lifetimeSum;
    private int _lifetimeCount;
    private double _last;
    private readonly object _sync = new();

    public TimeWindowStatistics(int capacity)
    {
        _capacity = capacity;
    }

    public TableFormerPerformanceSnapshot Register(BackendKey backendKey, TimeSpan duration)
    {
        lock (_sync)
        {
            Append(duration.TotalMilliseconds);
            return SnapshotInternal(backendKey);
        }
    }

    public TableFormerPerformanceSnapshot Snapshot(BackendKey backendKey)
    {
        lock (_sync)
        {
            return SnapshotInternal(backendKey);
        }
    }

    public int WindowSampleCount
    {
        get
        {
            lock (_sync)
            {
                return _window.Count;
            }
        }
    }

    public int TotalSampleCount
    {
        get
        {
            lock (_sync)
            {
                return _lifetimeCount;
            }
        }
    }

    private void Append(double value)
    {
        _last = value;
        _lifetimeSum += value;
        _lifetimeCount++;

        _window.Enqueue(value);
        _windowSum += value;
        if (_window.Count > _capacity)
        {
            _windowSum -= _window.Dequeue();
        }
    }

    private TableFormerPerformanceSnapshot SnapshotInternal(BackendKey backendKey)
    {
        if (_lifetimeCount == 0)
        {
            return new TableFormerPerformanceSnapshot(
                backendKey.Runtime,
                backendKey.Variant,
                0,
                0,
                double.PositiveInfinity,
                double.PositiveInfinity,
                double.PositiveInfinity);
        }

        var windowCount = _window.Count;
        var average = windowCount > 0 ? _windowSum / windowCount : double.PositiveInfinity;
        var best = windowCount > 0 ? _window.Min() : double.PositiveInfinity;

        return new TableFormerPerformanceSnapshot(
            backendKey.Runtime,
            backendKey.Variant,
            windowCount,
            _lifetimeCount,
            average,
            best,
            _last);
    }
}
