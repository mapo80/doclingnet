using System.Threading.Tasks;
using Xunit;

namespace Docling.Tests;

public sealed class PipelineDetailedAnalysisTest
{
    [Fact(Skip = "Manual pipeline exploration disabled in automated test runs.")]
    public Task AnalyzePipelineStepByStep()
    {
        return Task.CompletedTask;
    }
}
