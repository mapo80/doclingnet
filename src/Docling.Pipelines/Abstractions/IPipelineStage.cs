using System.Threading;
using System.Threading.Tasks;

namespace Docling.Pipelines.Abstractions;

/// <summary>
/// Represents a composable unit of work executed by the conversion pipeline.
/// </summary>
public interface IPipelineStage
{
    string Name { get; }

    Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken);
}
