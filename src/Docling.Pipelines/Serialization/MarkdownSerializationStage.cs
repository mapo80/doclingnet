using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Documents;
using Docling.Export.Imaging;
using Docling.Export.Serialization;
using Docling.Pipelines.Abstractions;

namespace Docling.Pipelines.Serialization;

/// <summary>
/// Pipeline stage responsible for transforming a <see cref="DoclingDocument"/> into markdown output.
/// </summary>
public sealed class MarkdownSerializationStage : IPipelineStage
{
    private readonly MarkdownDocSerializer _serializer;

    public MarkdownSerializationStage(MarkdownDocSerializer serializer)
    {
        _serializer = serializer ?? throw new ArgumentNullException(nameof(serializer));
    }

    public string Name => "markdown_serialization";

    public Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        var document = context.GetRequired<DoclingDocument>(PipelineContextKeys.Document);
        context.TryGet<IReadOnlyList<ImageExportArtifact>>(PipelineContextKeys.ImageExports, out var exports);

        var result = _serializer.Serialize(document, exports);

        context.Set(PipelineContextKeys.MarkdownSerializationResult, result);
        context.Set(PipelineContextKeys.MarkdownSerializationCompleted, true);

        return Task.CompletedTask;
    }
}
