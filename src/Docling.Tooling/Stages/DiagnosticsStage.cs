using System;
using System.Threading;
using System.Threading.Tasks;
using Docling.Pipelines.Abstractions;
using Microsoft.Extensions.Logging;

namespace Docling.Tooling.Stages;

/// <summary>
/// Lightweight diagnostics stage proving the pipeline infrastructure.
/// </summary>
public sealed class DiagnosticsStage : IPipelineStage
{
    private readonly ILogger<DiagnosticsStage> _logger;

    public DiagnosticsStage(ILogger<DiagnosticsStage> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public string Name => "Diagnostics";

    public Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        context.Set("Diagnostics.Timestamp", DateTimeOffset.UtcNow);
        _logger.LogInformation("Diagnostics stage executed. Service provider type: {ProviderType}.", context.Services.GetType().Name);
        return Task.CompletedTask;
    }
}
