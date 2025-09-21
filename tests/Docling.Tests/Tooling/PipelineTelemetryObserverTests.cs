using System;
using System.Threading;
using System.Threading.Tasks;
using Docling.Pipelines.Abstractions;
using Docling.Tooling.Runtime;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace Docling.Tests.Tooling;

public sealed class PipelineTelemetryObserverTests
{
    [Fact]
    public async Task ObserverCapturesStageDurations()
    {
        var observer = new PipelineTelemetryObserver();
        using var provider = new ServiceCollection().BuildServiceProvider();
        var context = new PipelineStageExecutionContext(new PipelineContext(provider), new StubStage("stage-a"));

        await observer.OnStageStartingAsync(context, CancellationToken.None);
        await Task.Delay(TimeSpan.FromMilliseconds(5));
        await observer.OnStageCompletedAsync(context, CancellationToken.None);

        var snapshot = observer.CreateSnapshot();
        snapshot.Should().HaveCount(1);
        snapshot[0].Stage.Should().Be("stage-a");
        snapshot[0].DurationMilliseconds.Should().BeGreaterOrEqualTo(0);
    }

    [Fact]
    public async Task ObserverCanBeResetBetweenRuns()
    {
        var observer = new PipelineTelemetryObserver();
        using var provider = new ServiceCollection().BuildServiceProvider();
        var context = new PipelineStageExecutionContext(new PipelineContext(provider), new StubStage("stage-b"));

        await observer.OnStageStartingAsync(context, CancellationToken.None);
        await observer.OnStageCompletedAsync(context, CancellationToken.None);
        observer.CreateSnapshot().Should().HaveCount(1);

        observer.Reset();
        observer.CreateSnapshot().Should().BeEmpty();
    }

    private sealed class StubStage : IPipelineStage
    {
        public StubStage(string name)
        {
            Name = name;
        }

        public string Name { get; }

        public Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
        {
            return Task.CompletedTask;
        }
    }
}
