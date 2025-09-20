using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Internal;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace Docling.Tests.Pipelines;

public sealed class ConvertPipelineTests
{
    [Fact]
    public async Task ExecuteAsyncRunsStagesSequentially()
    {
        var executionOrder = new List<string>();
        var stageOne = new TestStage("StageOne", executionOrder);
        var stageTwo = new TestStage("StageTwo", executionOrder);
        var observer = new TestObserver();
        var pipeline = new ConvertPipeline(new[] { stageOne, stageTwo }, new[] { observer }, NullLogger<ConvertPipeline>.Instance);
        var context = new PipelineContext(new ServiceCollection().BuildServiceProvider());

        await pipeline.ExecuteAsync(context, CancellationToken.None);

        executionOrder.Should().Equal("StageOne", "StageTwo");
        observer.Started.Should().Equal("StageOne", "StageTwo");
        observer.Completed.Should().Equal("StageOne", "StageTwo");
    }

    private sealed class TestStage : IPipelineStage
    {
        private readonly IList<string> _executionOrder;

        public TestStage(string name, IList<string> executionOrder)
        {
            Name = name;
            _executionOrder = executionOrder;
        }

        public string Name { get; }

        public Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
        {
            _executionOrder.Add(Name);
            return Task.CompletedTask;
        }
    }

    private sealed class TestObserver : IPipelineObserver
    {
        public List<string> Started { get; } = new();

        public List<string> Completed { get; } = new();

        public Task OnStageStartingAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
        {
            Started.Add(context.Stage.Name);
            return Task.CompletedTask;
        }

        public Task OnStageCompletedAsync(PipelineStageExecutionContext context, CancellationToken cancellationToken)
        {
            Completed.Add(context.Stage.Name);
            return Task.CompletedTask;
        }
    }
}
