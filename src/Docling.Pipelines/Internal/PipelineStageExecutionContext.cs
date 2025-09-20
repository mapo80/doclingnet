using System;

namespace Docling.Pipelines.Abstractions;

/// <summary>
/// Provides rich context information for observer notifications.
/// </summary>
public sealed class PipelineStageExecutionContext
{
    public PipelineStageExecutionContext(PipelineContext pipelineContext, IPipelineStage stage)
    {
        PipelineContext = pipelineContext ?? throw new ArgumentNullException(nameof(pipelineContext));
        Stage = stage ?? throw new ArgumentNullException(nameof(stage));
    }

    public PipelineContext PipelineContext { get; }

    public IPipelineStage Stage { get; }
}
