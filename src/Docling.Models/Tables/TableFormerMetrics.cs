using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace Docling.Models.Tables;

/// <summary>
/// Metrics tracker for TableFormer inference operations.
/// </summary>
internal sealed class TableFormerMetrics
{
    private readonly object _lock = new();
    private int _totalInferences;
    private int _successfulInferences;
    private int _failedInferences;
    private int _totalCellsDetected;
    private readonly ConcurrentDictionary<string, int> _backendUsage = new();
    private readonly List<TimeSpan> _inferenceTimes = new();

    public void RecordInference(bool success, TimeSpan inferenceTime, int cellsFound, int rowsDetected, int columnsDetected, string? backendType = null)
    {
        lock (_lock)
        {
            _totalInferences++;

            if (success)
            {
                _successfulInferences++;
                _totalCellsDetected += cellsFound;
                _inferenceTimes.Add(inferenceTime);
            }
            else
            {
                _failedInferences++;
            }

            if (!string.IsNullOrEmpty(backendType))
            {
                _backendUsage.AddOrUpdate(backendType, 1, (_, count) => count + 1);
            }
        }
    }

    public TableFormerMetricsSnapshot GetSnapshot()
    {
        lock (_lock)
        {
            var averageInferenceTime = _inferenceTimes.Count > 0
                ? TimeSpan.FromMilliseconds(_inferenceTimes.Average(t => t.TotalMilliseconds))
                : TimeSpan.Zero;

            var maxInferenceTime = _inferenceTimes.Count > 0
                ? _inferenceTimes.Max()
                : TimeSpan.Zero;

            var totalInferenceTime = _inferenceTimes.Count > 0
                ? TimeSpan.FromMilliseconds(_inferenceTimes.Sum(t => t.TotalMilliseconds))
                : TimeSpan.Zero;

            var successRate = _totalInferences > 0
                ? (double)_successfulInferences / _totalInferences
                : 0.0;

            return new TableFormerMetricsSnapshot(
                _totalInferences,
                _successfulInferences,
                _failedInferences,
                _totalCellsDetected,
                new Dictionary<string, int>(_backendUsage),
                averageInferenceTime,
                maxInferenceTime,
                totalInferenceTime,
                successRate);
        }
    }

    public void Reset()
    {
        lock (_lock)
        {
            _totalInferences = 0;
            _successfulInferences = 0;
            _failedInferences = 0;
            _totalCellsDetected = 0;
            _backendUsage.Clear();
            _inferenceTimes.Clear();
        }
    }
}

/// <summary>
/// Immutable snapshot of TableFormer metrics at a point in time.
/// </summary>
public sealed record TableFormerMetricsSnapshot(
    int TotalInferences,
    int SuccessfulInferences,
    int FailedInferences,
    int TotalCellsDetected,
    IReadOnlyDictionary<string, int> BackendUsage,
    TimeSpan AverageInferenceTime,
    TimeSpan MaxInferenceTime,
    TimeSpan TotalInferenceTime,
    double SuccessRate);
