using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using LayoutSdk;
using LayoutSdk.Configuration;
using LayoutSdk.Factories;
using LayoutSdk.Processing;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace Docling.Models.Layout;

internal interface ILayoutSdkRunner : IDisposable
{
    Task<LayoutSdkInferenceResult> InferAsync(ReadOnlyMemory<byte> imageContent, CancellationToken cancellationToken);
}

internal readonly record struct LayoutSdkInferenceResult(
    IReadOnlyList<LayoutSdk.BoundingBox> Boxes,
    LayoutSdkNormalisationMetadata? Normalisation);

public readonly record struct LayoutSdkNormalisationMetadata(
    int OriginalWidth,
    int OriginalHeight,
    int ScaledWidth,
    int ScaledHeight,
    double Scale,
    double OffsetX,
    double OffsetY);

[ExcludeFromCodeCoverage]
internal sealed partial class LayoutSdkRunner : ILayoutSdkRunner
{
    private const int ModelInputSize = 640;

    private readonly LayoutSdkDetectionOptions _options;
    private readonly LayoutSdk.LayoutSdk _sdk;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _semaphore;
    private readonly string _workingDirectory;
    private bool _disposed;

    private LayoutSdkRunner(LayoutSdkDetectionOptions options, ILogger logger)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.EnsureValid();
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _workingDirectory = ResolveWorkingDirectory(_options.WorkingDirectory);
        Directory.CreateDirectory(_workingDirectory);

        var sdkOptions = CreateSdkOptions(_options);
        var backendFactory = CreateBackendFactory(sdkOptions);

        _sdk = new LayoutSdk.LayoutSdk(
            sdkOptions,
            backendFactory,
            new Docling.Models.Layout.PassthroughOverlayRenderer(),
            new SkiaImagePreprocessor());
        _semaphore = new SemaphoreSlim(_options.MaxDegreeOfParallelism);
        RunnerLogger.Initialized(_logger, _options.Runtime.ToString(), _options.Language.ToString(), _workingDirectory);
    }

    public static ILayoutSdkRunner Create(LayoutSdkDetectionOptions options, ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new LayoutSdkRunner(options.Clone(), logger);
    }

    public async Task<LayoutSdkInferenceResult> InferAsync(ReadOnlyMemory<byte> imageContent, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();

        var persisted = await PersistAsync(imageContent, cancellationToken).ConfigureAwait(false);
        var path = persisted.Path;
        try
        {
            await _semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                var result = _sdk.Process(path, _options.GenerateOverlay, _options.Runtime);
                try
                {
                    var boxes = result.Boxes ?? Array.Empty<LayoutSdk.BoundingBox>();
                    RunnerLogger.RawDetections(_logger, boxes.Count);

                    // Throw exception if Layout SDK returned no detections
                    if (boxes.Count == 0)
                    {
                        throw new LayoutServiceException(
                            "Layout SDK failed to detect any layout elements. " +
                            "This usually indicates a problem with the ONNX model output parsing or the image preprocessing. " +
                            "Ensure the LayoutSdk backend (OnnxRuntimeBackend/OpenVinoBackend) correctly implements the ParseOutputs method.");
                    }

                    var projected = ReprojectBoundingBoxes(boxes, persisted.Normalisation);
                    if (persisted.Normalisation is LayoutSdkNormalisationMetadata normalisation)
                    {
                        RunnerLogger.DenormalisedDetections(
                            _logger,
                            normalisation.OriginalWidth,
                            normalisation.OriginalHeight,
                            normalisation.ScaledWidth,
                            normalisation.ScaledHeight,
                            normalisation.Scale,
                            normalisation.OffsetX,
                            normalisation.OffsetY,
                            projected.Count);
                    }
                    else
                    {
                        RunnerLogger.DenormalisationSkipped(_logger, projected.Count);
                    }

                    return new LayoutSdkInferenceResult(projected, persisted.Normalisation);
                }
                finally
                {
                    result.OverlayImage?.Dispose();
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new LayoutServiceException("The layout SDK failed to execute the Heron model.", ex);
            }
            finally
            {
                _semaphore.Release();
            }
        }
        finally
        {
            if (!_options.KeepTemporaryFiles)
            {
                TryDelete(path);
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _semaphore.Dispose();
        _sdk.Dispose();
    }

    private static LayoutSdkOptions CreateSdkOptions(LayoutSdkDetectionOptions options)
    {
        var preferOrt = options.Runtime == LayoutRuntime.Ort;
        var preferOpenVino = options.Runtime == LayoutRuntime.OpenVino;
        if (options.ValidateModelFiles)
        {
            LayoutSdkBundledModels.EnsureAllFilesExist();
        }

        var sdkOptions = LayoutSdkBundledModels.CreateOptions(options.Language, preferOrt, preferOpenVino);
        if (options.ValidateModelFiles)
        {
            sdkOptions.EnsureModelPaths();
        }

        return sdkOptions;
    }

    private static LayoutBackendFactory CreateBackendFactory(LayoutSdkOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        var assembly = typeof(LayoutRuntime).Assembly;
        var onnxBackendType = assembly.GetType("LayoutSdk.OnnxRuntimeBackend", throwOnError: true)!;
        var onnxFormatType = assembly.GetType("LayoutSdk.OnnxModelFormat", throwOnError: true)!;
        var onnxCtor = onnxBackendType.GetConstructor(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(string), onnxFormatType },
            modifiers: null) ?? throw new InvalidOperationException("Unable to locate OnnxRuntimeBackend constructor.");
        var onnxValue = Enum.Parse(onnxFormatType, "Onnx");
        var ortValue = Enum.Parse(onnxFormatType, "Ort");

        LayoutSdk.ILayoutBackend CreateOnnxBackend(string modelPath)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
            var backend = onnxCtor.Invoke(new[] { modelPath, onnxValue });
            return (LayoutSdk.ILayoutBackend)backend!;
        }

        LayoutSdk.ILayoutBackend CreateOrtBackend(string modelPath)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
            var backend = onnxCtor.Invoke(new[] { modelPath, ortValue });
            return (LayoutSdk.ILayoutBackend)backend!;
        }

        var openVinoBackendType = assembly.GetType("LayoutSdk.OpenVinoBackend", throwOnError: true)!;
        var openVinoExecutorType = assembly.GetType("LayoutSdk.OpenVinoBackend+OpenVinoExecutor", throwOnError: true)!;
        var openVinoExecutorCtor = openVinoExecutorType.GetConstructor(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(string), typeof(string) },
            modifiers: null) ?? throw new InvalidOperationException("Unable to locate OpenVINO executor constructor.");
        var openVinoCtor = openVinoBackendType.GetConstructor(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: new[] { openVinoExecutorType },
            modifiers: null) ?? throw new InvalidOperationException("Unable to locate OpenVINO backend constructor.");

        LayoutSdk.ILayoutBackend CreateOpenVinoBackend(string xmlPath, string binPath)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(xmlPath);
            ArgumentException.ThrowIfNullOrWhiteSpace(binPath);
            var executor = openVinoExecutorCtor.Invoke(new object[] { xmlPath, binPath });
            var backend = openVinoCtor.Invoke(new[] { executor });
            return (LayoutSdk.ILayoutBackend)backend!;
        }

        return new LayoutBackendFactory(options, CreateOnnxBackend, CreateOrtBackend, CreateOpenVinoBackend);
    }

    private static string ResolveWorkingDirectory(string? workingDirectory)
    {
        var root = string.IsNullOrWhiteSpace(workingDirectory)
            ? Path.Combine(Path.GetTempPath(), "docling-layout-sdk")
            : Path.GetFullPath(workingDirectory);
        return root;
    }

    private async Task<PersistedImage> PersistAsync(ReadOnlyMemory<byte> content, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(_workingDirectory);
        var fileName = Path.Combine(_workingDirectory, Guid.NewGuid().ToString("N")) + ".png";
        var stream = new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.Read, 4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
        var normalised = NormaliseForModel(content);
        try
        {
            if (normalised.UseOriginal || normalised.Image is null)
            {
                await stream.WriteAsync(content, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                await stream.WriteAsync(normalised.Image.AsMemory(), cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            await stream.DisposeAsync().ConfigureAwait(false);
        }

        return new PersistedImage(fileName, normalised.Metadata);
    }

    [SuppressMessage("Performance", "CA1508:Avoid dead conditional code", Justification = "SKSurface.Create can return null when Skia fails to allocate a surface.")]
    private NormalisationResult NormaliseForModel(ReadOnlyMemory<byte> content)
    {
        LayoutSdkNormalisationMetadata? metadata = null;

        using var data = SKData.CreateCopy(content.Span);
        using var bitmap = SKBitmap.Decode(data);
        if (bitmap is null)
        {
            RunnerLogger.PreprocessingFallback(_logger, "Unable to decode the input image.");
            return new NormalisationResult(true, null, metadata);
        }

        if (bitmap.Width <= 0 || bitmap.Height <= 0)
        {
            RunnerLogger.PreprocessingFallback(_logger, "The input image dimensions are invalid.");
            return new NormalisationResult(true, null, metadata);
        }

        metadata = new LayoutSdkNormalisationMetadata(
            bitmap.Width,
            bitmap.Height,
            bitmap.Width,
            bitmap.Height,
            Scale: 1.0,
            OffsetX: 0,
            OffsetY: 0);

        if (bitmap.Width == ModelInputSize && bitmap.Height == ModelInputSize)
        {
            return new NormalisationResult(true, null, metadata);
        }

        var scale = Math.Min((double)ModelInputSize / bitmap.Width, (double)ModelInputSize / bitmap.Height);
        if (scale <= 0)
        {
            RunnerLogger.PreprocessingFallback(_logger, "Failed to compute a valid scaling factor.");
            return new NormalisationResult(true, null, metadata);
        }

        var scaledWidth = Math.Clamp((float)(bitmap.Width * scale), 1f, ModelInputSize);
        var scaledHeight = Math.Clamp((float)(bitmap.Height * scale), 1f, ModelInputSize);
        var createdSurface = SKSurface.Create(new SKImageInfo(ModelInputSize, ModelInputSize, SKColorType.Rgba8888, SKAlphaType.Premul));
        if (createdSurface is null)
        {
            RunnerLogger.PreprocessingFallback(_logger, "Unable to allocate the normalised canvas.");
            return new NormalisationResult(true, null, metadata);
        }

        using var surface = createdSurface;
        var canvas = surface.Canvas;
        canvas.Clear(SKColors.Black);

        var offsetX = (ModelInputSize - scaledWidth) / 2f;
        var offsetY = (ModelInputSize - scaledHeight) / 2f;
        var destination = SKRect.Create(offsetX, offsetY, scaledWidth, scaledHeight);
        canvas.DrawBitmap(bitmap, destination);
        canvas.Flush();

        using var image = surface.Snapshot();
        using var encoded = image.Encode(SKEncodedImageFormat.Png, 100);
        if (encoded is null)
        {
            RunnerLogger.PreprocessingFallback(_logger, "Unable to encode the normalised image.");
            return new NormalisationResult(true, null, metadata);
        }

        var scaledWidthInt = Math.Clamp((int)Math.Round(scaledWidth, MidpointRounding.AwayFromZero), 1, ModelInputSize);
        var scaledHeightInt = Math.Clamp((int)Math.Round(scaledHeight, MidpointRounding.AwayFromZero), 1, ModelInputSize);
        metadata = new LayoutSdkNormalisationMetadata(
            bitmap.Width,
            bitmap.Height,
            scaledWidthInt,
            scaledHeightInt,
            Scale: scale,
            OffsetX: offsetX,
            OffsetY: offsetY);
        RunnerLogger.ImageNormalised(_logger, bitmap.Width, bitmap.Height, scaledWidthInt, scaledHeightInt);
        return new NormalisationResult(false, encoded.ToArray(), metadata);
    }


    internal static IReadOnlyList<LayoutSdk.BoundingBox> ReprojectBoundingBoxes(
        IReadOnlyList<LayoutSdk.BoundingBox> boxes,
        LayoutSdkNormalisationMetadata? metadata)
    {
        if (boxes.Count == 0)
        {
            return Array.Empty<LayoutSdk.BoundingBox>();
        }

        if (metadata is null || metadata.Value.Scale <= 0)
        {
            return boxes
                .Select(b => new LayoutSdk.BoundingBox(b.X, b.Y, b.Width, b.Height, b.Label))
                .ToArray();
        }

        var meta = metadata.Value;
        var projected = new List<LayoutSdk.BoundingBox>(boxes.Count);
        var contentMaxX = meta.OffsetX + meta.ScaledWidth;
        var contentMaxY = meta.OffsetY + meta.ScaledHeight;

        for (var i = 0; i < boxes.Count; i++)
        {
            var box = boxes[i];

            var left = Clamp(box.X, meta.OffsetX, contentMaxX) - meta.OffsetX;
            var top = Clamp(box.Y, meta.OffsetY, contentMaxY) - meta.OffsetY;
            var right = Clamp(box.X + box.Width, meta.OffsetX, contentMaxX) - meta.OffsetX;
            var bottom = Clamp(box.Y + box.Height, meta.OffsetY, contentMaxY) - meta.OffsetY;

            var originalLeft = ClampToPage(left / meta.Scale, meta.OriginalWidth);
            var originalTop = ClampToPage(top / meta.Scale, meta.OriginalHeight);
            var originalRight = ClampToPage(right / meta.Scale, meta.OriginalWidth);
            var originalBottom = ClampToPage(bottom / meta.Scale, meta.OriginalHeight);

            var width = originalRight - originalLeft;
            var height = originalBottom - originalTop;
            if (width <= 0 || height <= 0)
            {
                continue;
            }

            projected.Add(new LayoutSdk.BoundingBox(
                (float)originalLeft,
                (float)originalTop,
                (float)width,
                (float)height,
                box.Label));
        }

        return projected;
    }







    private static double Clamp(double value, double min, double max)
    {
        if (min > max)
        {
            (min, max) = (max, min);
        }

        return Math.Clamp(value, min, max);
    }

    private static double ClampToPage(double value, double max)
    {
        return Math.Clamp(value, 0, max);
    }

    private void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException ex)
        {
            RunnerLogger.DeletionFailed(_logger, path, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            RunnerLogger.DeletionFailed(_logger, path, ex);
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    private static partial class RunnerLogger
    {
        [LoggerMessage(EventId = 4100, Level = LogLevel.Information, Message = "Initialised layout SDK runner (runtime: {Runtime}, language: {Language}, workspace: {WorkingDirectory}).")]
        public static partial void Initialized(ILogger logger, string runtime, string language, string workingDirectory);

        [LoggerMessage(EventId = 4101, Level = LogLevel.Warning, Message = "Failed to delete temporary layout file {Path}.")]
        public static partial void DeletionFailed(ILogger logger, string path, Exception exception);

        [LoggerMessage(EventId = 4102, Level = LogLevel.Warning, Message = "Falling back to the original layout image: {Reason}")]
        public static partial void PreprocessingFallback(ILogger logger, string reason);

        [LoggerMessage(EventId = 4103, Level = LogLevel.Debug, Message = "Normalised layout input from {OriginalWidth}x{OriginalHeight} to {ResizedWidth}x{ResizedHeight}.")]
        public static partial void ImageNormalised(ILogger logger, int originalWidth, int originalHeight, int resizedWidth, int resizedHeight);

        [LoggerMessage(EventId = 4104, Level = LogLevel.Debug, Message = "Layout SDK returned {DetectionCount} raw detections before reprojection.")]
        public static partial void RawDetections(ILogger logger, int detectionCount);

        [LoggerMessage(EventId = 4109, Level = LogLevel.Debug, Message = "Reprojected {DetectionCount} layout detections to {OriginalWidth}x{OriginalHeight} (scaled {ScaledWidth}x{ScaledHeight}, scale {Scale:F3}, offsets {OffsetX:F2},{OffsetY:F2}).")]
        public static partial void DenormalisedDetections(
            ILogger logger,
            int originalWidth,
            int originalHeight,
            int scaledWidth,
            int scaledHeight,
            double scale,
            double offsetX,
            double offsetY,
            int detectionCount);

        [LoggerMessage(EventId = 4110, Level = LogLevel.Debug, Message = "Reprojection skipped because no normalisation metadata was available (kept {DetectionCount} detections as-is).")]
        public static partial void DenormalisationSkipped(ILogger logger, int detectionCount);
    }

    private readonly record struct PersistedImage(string Path, LayoutSdkNormalisationMetadata? Normalisation);

    private readonly record struct NormalisationResult(bool UseOriginal, byte[]? Image, LayoutSdkNormalisationMetadata? Metadata);

}
