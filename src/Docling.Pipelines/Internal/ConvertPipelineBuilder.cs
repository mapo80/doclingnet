using System;
using System.Collections.Generic;
using Docling.Pipelines.Abstractions;
using Microsoft.Extensions.Logging;

namespace Docling.Pipelines.Internal;

/// <summary>
/// Fluent builder that composes a <see cref="ConvertPipeline"/>.
/// </summary>
public sealed class ConvertPipelineBuilder
{
    private readonly List<IPipelineStage> _stages = new();
    private readonly List<IPipelineObserver> _observers = new();

    public ConvertPipelineBuilder AddStage(IPipelineStage stage)
    {
        _stages.Add(stage ?? throw new ArgumentNullException(nameof(stage)));
        return this;
    }

    public ConvertPipelineBuilder AddObserver(IPipelineObserver observer)
    {
        _observers.Add(observer ?? throw new ArgumentNullException(nameof(observer)));
        return this;
    }

    public ConvertPipeline Build(ILogger<ConvertPipeline> logger)
    {
        ArgumentNullException.ThrowIfNull(logger);
        if (_stages.Count == 0)
        {
            throw new InvalidOperationException("Cannot build a pipeline without stages.");
        }

        return new ConvertPipeline(_stages.ToArray(), _observers.ToArray(), logger);
    }
}
