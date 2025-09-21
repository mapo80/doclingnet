using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Docling.Pipelines.Abstractions;

namespace Docling.Tooling.Runtime;

/// <summary>
/// Collects per-stage execution timings so the CLI can emit telemetry summaries.
/// </summary>
internal sealed class PipelineTelemetryObserver : IPipelineObserver
{
    private readonly object _gate = new();
    private readonly List<PipelineStageTelemetry> _completedStages = new();
    private string? _currentStage;
    private Stopwatch? _stopwatch;

    public Task OnStageStartingAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        lock (_gate)
        {
            _currentStage = context.Stage.Name;
            _stopwatch = Stopwatch.StartNew();
        }

        return Task.CompletedTask;
    }

    public Task OnStageCompletedAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        lock (_gate)
        {
            if (_stopwatch is not null && _currentStage is not null)
            {
                _stopwatch.Stop();
                _completedStages.Add(new PipelineStageTelemetry(_currentStage, _stopwatch.ElapsedMilliseconds));
            }

            _stopwatch = null;
            _currentStage = null;
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Clears the collected timings so the observer can be reused for another run.
    /// </summary>
    public void Reset()
    {
        lock (_gate)
        {
            _completedStages.Clear();
            _stopwatch = null;
            _currentStage = null;
        }
    }

    /// <summary>
    /// Returns a snapshot of the collected stage timings.
    /// </summary>
    public IReadOnlyList<PipelineStageTelemetry> CreateSnapshot()
    {
        lock (_gate)
        {
            return _completedStages.ToArray();
        }
    }
}

/// <summary>
/// Represents the elapsed time for a pipeline stage execution.
/// </summary>
internal readonly record struct PipelineStageTelemetry(string Stage, long DurationMilliseconds);
