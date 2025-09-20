using System;
using System.Threading;
using System.Threading.Tasks;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Internal;
using Microsoft.Extensions.Logging;

namespace Docling.Tooling.Runtime;

/// <summary>
/// Observer emitting structured logs for each pipeline stage transition.
/// </summary>
public sealed class LoggingPipelineObserver : IPipelineObserver
{
    private readonly ILogger<LoggingPipelineObserver> _logger;

    public LoggingPipelineObserver(ILogger<LoggingPipelineObserver> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public Task OnStageStartingAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
    {
        _logger.LogDebug("Stage {Stage} starting.", context.Stage.Name);
        return Task.CompletedTask;
    }

    public Task OnStageCompletedAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
    {
        _logger.LogDebug("Stage {Stage} completed.", context.Stage.Name);
        return Task.CompletedTask;
    }
}
