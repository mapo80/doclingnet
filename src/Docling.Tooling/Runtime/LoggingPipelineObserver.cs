using System;
using System.Diagnostics.CodeAnalysis;
using System.Threading;
using System.Threading.Tasks;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Internal;
using Microsoft.Extensions.Logging;

namespace Docling.Tooling.Runtime;

/// <summary>
/// Observer emitting structured logs for each pipeline stage transition.
/// </summary>
[SuppressMessage("Performance", "CA1812:Avoid uninstantiated internal classes", Justification = "Instantiated via dependency injection.")]
internal sealed partial class LoggingPipelineObserver : IPipelineObserver
{
    private readonly ILogger<LoggingPipelineObserver> _logger;

    public LoggingPipelineObserver(ILogger<LoggingPipelineObserver> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public Task OnStageStartingAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        ObserverLogger.StageStarting(_logger, context.Stage.Name);
        return Task.CompletedTask;
    }

    public Task OnStageCompletedAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        ObserverLogger.StageCompleted(_logger, context.Stage.Name);
        return Task.CompletedTask;
    }

    private static partial class ObserverLogger
    {
        [LoggerMessage(EventId = 3850, Level = LogLevel.Debug, Message = "Stage {Stage} starting.")]
        public static partial void StageStarting(ILogger logger, string stage);

        [LoggerMessage(EventId = 3851, Level = LogLevel.Debug, Message = "Stage {Stage} completed.")]
        public static partial void StageCompleted(ILogger logger, string stage);
    }
}
