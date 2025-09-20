using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Storage;
using Docling.Core.Primitives;
using Docling.Pipelines.Abstractions;
using Microsoft.Extensions.Logging;

namespace Docling.Pipelines.Preprocessing;

/// <summary>
/// Pipeline stage that orchestrates <see cref="IPagePreprocessor"/> across all cached <see cref="PageImage"/> instances.
/// </summary>
public sealed partial class PagePreprocessingStage : IPipelineStage
{
    private readonly IPagePreprocessor _preprocessor;
    private readonly ILogger<PagePreprocessingStage> _logger;

    public PagePreprocessingStage(IPagePreprocessor preprocessor, ILogger<PagePreprocessingStage> logger)
    {
        _preprocessor = preprocessor ?? throw new ArgumentNullException(nameof(preprocessor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public string Name => "page_preprocessing";

    public async Task ExecuteAsync(PipelineContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (!context.TryGet<PageImageStore>(PipelineContextKeys.PageImageStore, out var store))
        {
            throw new InvalidOperationException("Pipeline context does not contain a page image store.");
        }

        var pages = context.GetRequired<IReadOnlyList<PageReference>>(PipelineContextKeys.PageSequence);
        if (pages.Count == 0)
        {
            StageLogger.NoPages(_logger);
            context.Set(PipelineContextKeys.PreprocessingCompleted, true);
            return;
        }

        foreach (var page in pages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var original = store.Rent(page);
            StageLogger.PageStarting(_logger, page.PageNumber, original.Metadata.MediaType, original.Page.Dpi);
            var stopwatch = Stopwatch.StartNew();
            try
            {
                var processed = await _preprocessor.PreprocessAsync(original, cancellationToken).ConfigureAwait(false);
                store.Add(processed, overwrite: true);
                stopwatch.Stop();
                StageLogger.PageCompleted(_logger, page.PageNumber, stopwatch.ElapsedMilliseconds);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                stopwatch.Stop();
                StageLogger.PageFailed(_logger, page.PageNumber, ex);
                throw;
            }
        }

        context.Set(PipelineContextKeys.PreprocessingCompleted, true);
    }

    private static partial class StageLogger
    {
        [LoggerMessage(EventId = 2000, Level = LogLevel.Debug, Message = "No pages available for preprocessing.")]
        public static partial void NoPages(ILogger logger);

        [LoggerMessage(EventId = 2001, Level = LogLevel.Information, Message = "Preprocessing page {PageNumber} (media: {MediaType}, dpi: {Dpi}).")]
        public static partial void PageStarting(ILogger logger, int pageNumber, string? mediaType, double dpi);

        [LoggerMessage(EventId = 2002, Level = LogLevel.Information, Message = "Completed preprocessing for page {PageNumber} in {ElapsedMs} ms.")]
        public static partial void PageCompleted(ILogger logger, int pageNumber, long elapsedMs);

        [LoggerMessage(EventId = 2003, Level = LogLevel.Error, Message = "Preprocessing failed for page {PageNumber}.")]
        public static partial void PageFailed(ILogger logger, int pageNumber, Exception exception);
    }
}
