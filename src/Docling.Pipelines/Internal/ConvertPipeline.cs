using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Docling.Pipelines.Abstractions;
using Microsoft.Extensions.Logging;

namespace Docling.Pipelines.Internal;

/// <summary>
/// Coordinates sequential execution of registered pipeline stages.
/// </summary>
public sealed partial class ConvertPipeline
{
    private readonly IReadOnlyList<IPipelineStage> _stages;
    private readonly IReadOnlyList<IPipelineObserver> _observers;
    private readonly ILogger<ConvertPipeline> _logger;

    public ConvertPipeline(
        IReadOnlyList<IPipelineStage> stages,
        IReadOnlyList<IPipelineObserver> observers,
        ILogger<ConvertPipeline> logger)
    {
        _stages = stages ?? throw new ArgumentNullException(nameof(stages));
        _observers = observers ?? throw new ArgumentNullException(nameof(observers));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        if (stages.Count == 0)
        {
            throw new ArgumentException("At least one stage is required to build a pipeline.", nameof(stages));
        }
    }

    public async Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(context);

        foreach (var stage in _stages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var stageContext = new PipelineStageExecutionContext(context, stage);
            await NotifyAsync(observer => observer.OnStageStartingAsync(stageContext, cancellationToken)).ConfigureAwait(false);

            using var activity = new Activity(stage.Name);
            activity.Start();
            var stopwatch = Stopwatch.StartNew();
            try
            {
                PipelineLogger.StageStarting(_logger, stage.Name);
                await stage.ExecuteAsync(context, cancellationToken).ConfigureAwait(false);
                stopwatch.Stop();
                PipelineLogger.StageCompleted(_logger, stage.Name, stopwatch.ElapsedMilliseconds);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                stopwatch.Stop();
                PipelineLogger.StageFaulted(_logger, stage.Name, stopwatch.ElapsedMilliseconds, ex);
                throw;
            }
            finally
            {
                await NotifyAsync(observer => observer.OnStageCompletedAsync(stageContext, cancellationToken)).ConfigureAwait(false);
            }
        }
    }

    private async Task NotifyAsync(Func<IPipelineObserver, Task> callback)
    {
        foreach (var observer in _observers)
        {
            await callback(observer).ConfigureAwait(false);
        }
    }

    private static partial class PipelineLogger
    {
        [LoggerMessage(EventId = 1000, Level = LogLevel.Information, Message = "Starting pipeline stage {Stage}.")]
        public static partial void StageStarting(ILogger logger, string stage);

        [LoggerMessage(EventId = 1001, Level = LogLevel.Information, Message = "Completed pipeline stage {Stage} in {ElapsedMs} ms.")]
        public static partial void StageCompleted(ILogger logger, string stage, long elapsedMs);

        [LoggerMessage(EventId = 1002, Level = LogLevel.Error, Message = "Pipeline stage {Stage} failed after {ElapsedMs} ms.")]
        public static partial void StageFaulted(ILogger logger, string stage, long elapsedMs, Exception exception);
    }
}
