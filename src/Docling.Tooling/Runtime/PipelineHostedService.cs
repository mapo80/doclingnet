using System;
using System.Threading;
using System.Threading.Tasks;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Internal;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Docling.Tooling.Runtime;

/// <summary>
/// Minimal hosted service showcasing the conversion pipeline wiring.
/// </summary>
public sealed class PipelineHostedService : BackgroundService
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ConvertPipelineBuilder _builder;
    private readonly ILogger<PipelineHostedService> _logger;
    private readonly ILogger<ConvertPipeline> _pipelineLogger;

    public PipelineHostedService(
        IServiceProvider serviceProvider,
        ConvertPipelineBuilder builder,
        ILogger<PipelineHostedService> logger,
        ILogger<ConvertPipeline> pipelineLogger)
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _builder = builder ?? throw new ArgumentNullException(nameof(builder));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _pipelineLogger = pipelineLogger ?? throw new ArgumentNullException(nameof(pipelineLogger));
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Docling pipeline scaffolding bootstrapped.");
        var context = new PipelineContext(_serviceProvider);
        var pipeline = _builder.Build(_pipelineLogger);
        await pipeline.ExecuteAsync(context, stoppingToken).ConfigureAwait(false);
        _logger.LogInformation("Docling pipeline execution completed successfully.");
    }
}
