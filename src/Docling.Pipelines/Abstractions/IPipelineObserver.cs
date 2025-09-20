using System.Threading;
using System.Threading.Tasks;

namespace Docling.Pipelines.Abstractions;

/// <summary>
/// Allows external systems to observe pipeline execution without coupling to stage implementations.
/// </summary>
public interface IPipelineObserver
{
    Task OnStageStartingAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken);

    Task OnStageCompletedAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken);
}
