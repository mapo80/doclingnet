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
    private const float FallbackScoreThreshold = 0.20f;
    private const double CoverageAugmentationThreshold = 0.55d;
    private const double EdgeAugmentationThreshold = 0.05d;
    private const double DuplicateIoUThreshold = 0.5d;
    private const double ContainmentSuppressionRatio = 8d;
    private const double ContainmentTolerance = 1e-2;

    private readonly LayoutSdkDetectionOptions _options;
    private readonly LayoutSdk.LayoutSdk _sdk;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _semaphore;
    private readonly string _workingDirectory;
    private readonly Lazy<InferenceSession> _fallbackSession;
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
        _fallbackSession = new Lazy<InferenceSession>(CreateFallbackSession, LazyThreadSafetyMode.ExecutionAndPublication);
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

                    var finalDetections = projected;
                    var fallbackReason = DetermineFallbackReason(projected, boxes.Count, persisted.Normalisation, out var coverageRatio);
                    if (fallbackReason != FallbackReason.None)
                    {
                        if (fallbackReason == FallbackReason.Augmentation && persisted.Normalisation is LayoutSdkNormalisationMetadata metadata)
                        {
                            RunnerLogger.FallbackAugmentationTriggered(
                                _logger,
                                metadata.OriginalWidth,
                                metadata.OriginalHeight,
                                coverageRatio);
                        }

                        var fallbackDetections = RunFallbackInference(path, persisted.Normalisation);
                        if (fallbackDetections.Count > 0)
                        {
                            if (fallbackReason == FallbackReason.Augmentation && projected.Count > 0)
                            {
                                finalDetections = MergeDetections(projected, fallbackDetections);
                                RunnerLogger.FallbackAugmented(_logger, projected.Count, fallbackDetections.Count, finalDetections.Count);
                            }
                            else
                            {
                                RunnerLogger.FallbackSucceeded(_logger, fallbackDetections.Count);
                                finalDetections = fallbackDetections;
                            }
                        }
                        else if (fallbackReason == FallbackReason.Augmentation)
                        {
                            RunnerLogger.FallbackAugmentationSkipped(_logger);
                        }
                    }

                    return new LayoutSdkInferenceResult(finalDetections, persisted.Normalisation);
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
        if (_fallbackSession.IsValueCreated)
        {
            _fallbackSession.Value.Dispose();
        }
        _sdk.Dispose();
    }

    private InferenceSession CreateFallbackSession()
    {
        var options = LayoutSdkBundledModels.CreateOptions(_options.Language, false, false);
        var modelPath = options.OnnxModelPath;
        RunnerLogger.FallbackSessionCreated(_logger, modelPath);
        var sessionOptions = new SessionOptions();
        try
        {
            return new InferenceSession(modelPath, sessionOptions);
        }
        finally
        {
            sessionOptions.Dispose();
        }
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

    [SuppressMessage("Design", "CA1031:Do not catch general exception types", Justification = "Fallback inference must not bubble failures.")]
    private IReadOnlyList<LayoutSdk.BoundingBox> RunFallbackInference(string path, LayoutSdkNormalisationMetadata? metadata)
    {
        if (metadata is null)
        {
            return Array.Empty<LayoutSdk.BoundingBox>();
        }

        try
        {
            using var bitmap = SKBitmap.Decode(path);
            if (bitmap is null)
            {
                RunnerLogger.FallbackSkipped(_logger, "Unable to decode normalised PNG for fallback inference.");
                return Array.Empty<LayoutSdk.BoundingBox>();
            }

            using var resized = EnsureModelSize(bitmap);
            var tensor = CreateInputTensor(resized);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("pixel_values", tensor) };
            try
            {
                using var results = _fallbackSession.Value.Run(inputs);
                var logits = results.FirstOrDefault(value => value.Name == "logits")?.AsTensor<float>();
                var predBoxes = results.FirstOrDefault(value => value.Name == "pred_boxes")?.AsTensor<float>();
                if (logits is null || predBoxes is null)
                {
                    RunnerLogger.FallbackSkipped(_logger, "Fallback inference did not return logits/pred_boxes outputs.");
                    return Array.Empty<LayoutSdk.BoundingBox>();
                }

                var decoded = DecodeDetections(logits, predBoxes);
                if (decoded.Count == 0)
                {
                    return Array.Empty<LayoutSdk.BoundingBox>();
                }

                var projected = ReprojectBoundingBoxes(decoded, metadata);
                if (metadata is LayoutSdkNormalisationMetadata normalisation)
                {
                    RunnerLogger.FallbackDenormalised(
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
                    RunnerLogger.FallbackDenormalisationSkipped(_logger, projected.Count);
                }

                return projected;
            }
            finally
            {
                // NamedOnnxValue does not implement IDisposable in this runtime.
            }
        }
        catch (Exception ex)
        {
            RunnerLogger.FallbackFailed(_logger, ex);
            return Array.Empty<LayoutSdk.BoundingBox>();
        }
    }

    [SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope", Justification = "Ownership of the resized bitmap is transferred to the caller; the local variable is cleared before leaving the method.")]
    private static SKBitmap EnsureModelSize(SKBitmap bitmap)
    {
        if (bitmap.Width == ModelInputSize && bitmap.Height == ModelInputSize)
        {
            return bitmap.Copy() ?? throw new InvalidOperationException("Failed to copy fallback bitmap.");
        }

        var info = new SKImageInfo(ModelInputSize, ModelInputSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        SKBitmap? resized = null;
        try
        {
            resized = new SKBitmap(info);
            using var surface = new SKCanvas(resized);
            surface.Clear(SKColors.Black);
            var sampling = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None);
            if (!bitmap.ScalePixels(resized, sampling))
            {
                throw new InvalidOperationException("Failed to resize bitmap for fallback inference.");
            }

            surface.Flush();
            var result = resized;
            resized = null;
            return result;
        }
        catch
        {
            resized?.Dispose();
            throw;
        }
    }

    private static DenseTensor<float> CreateInputTensor(SKBitmap bitmap)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });
        for (var y = 0; y < ModelInputSize; y++)
        {
            for (var x = 0; x < ModelInputSize; x++)
            {
                var pixel = bitmap.GetPixel(x, y);
                var r = pixel.Red / 255f;
                var g = pixel.Green / 255f;
                var b = pixel.Blue / 255f;
                tensor[0, 0, y, x] = r;
                tensor[0, 1, y, x] = g;
                tensor[0, 2, y, x] = b;
            }
        }

        return tensor;
    }

    private static IReadOnlyList<LayoutSdk.BoundingBox> DecodeDetections(Tensor<float> logits, Tensor<float> predBoxes)
    {
        if (logits.Dimensions.Length != 3 || predBoxes.Dimensions.Length != 3)
        {
            return Array.Empty<LayoutSdk.BoundingBox>();
        }

        var queryCount = logits.Dimensions[1];
        var classCount = logits.Dimensions[2];
        var boxes = new List<LayoutSdk.BoundingBox>(queryCount);

        for (var i = 0; i < queryCount; i++)
        {
            var maxScore = float.NegativeInfinity;
            for (var c = 1; c < classCount; c++)
            {
                var score = logits[0, i, c];
                if (score > maxScore)
                {
                    maxScore = score;
                }
            }

            var sum = 0f;
            for (var c = 0; c < classCount; c++)
            {
                sum += (float)Math.Exp(logits[0, i, c] - maxScore);
            }

            var bestClass = 0;
            var bestProbability = 0f;
            for (var c = 1; c < classCount; c++)
            {
                var probability = (float)Math.Exp(logits[0, i, c] - maxScore) / sum;
                if (probability > bestProbability)
                {
                    bestProbability = probability;
                    bestClass = c;
                }
            }

            if (bestProbability < FallbackScoreThreshold)
            {
                continue;
            }

            var cx = predBoxes[0, i, 0] * ModelInputSize;
            var cy = predBoxes[0, i, 1] * ModelInputSize;
            var width = predBoxes[0, i, 2] * ModelInputSize;
            var height = predBoxes[0, i, 3] * ModelInputSize;
            var left = Math.Max(0f, cx - width / 2f);
            var top = Math.Max(0f, cy - height / 2f);
            var clampedWidth = Math.Max(0f, Math.Min(ModelInputSize, left + width) - left);
            var clampedHeight = Math.Max(0f, Math.Min(ModelInputSize, top + height) - top);
            if (clampedWidth <= 1f || clampedHeight <= 1f)
            {
                continue;
            }

            var label = MapFallbackLabel(bestClass);
            boxes.Add(new LayoutSdk.BoundingBox(left, top, clampedWidth, clampedHeight, label));
        }

        return boxes;
    }

    private static string MapFallbackLabel(int bestClass) => "text";

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

    private static FallbackReason DetermineFallbackReason(
        IReadOnlyList<LayoutSdk.BoundingBox> projected,
        int rawCount,
        LayoutSdkNormalisationMetadata? metadata,
        out double coverageRatio)
    {
        coverageRatio = 0d;

        if (rawCount == 0)
        {
            return FallbackReason.Empty;
        }

        if (projected.Count == 0)
        {
            return FallbackReason.Clipped;
        }

        if (ShouldAugmentWithFallback(projected, metadata, out coverageRatio))
        {
            return FallbackReason.Augmentation;
        }

        return FallbackReason.None;
    }

    internal static bool ShouldAugmentWithFallback(
        IReadOnlyList<LayoutSdk.BoundingBox> projected,
        LayoutSdkNormalisationMetadata? metadata,
        out double coverageRatio)
    {
        coverageRatio = 0d;

        if (projected.Count == 0 || metadata is null)
        {
            return false;
        }

        var meta = metadata.Value;
        var pageArea = (double)meta.OriginalWidth * meta.OriginalHeight;
        if (pageArea <= 0d)
        {
            return false;
        }

        double sumArea = 0d;
        var minLeft = double.PositiveInfinity;
        var minTop = double.PositiveInfinity;

        for (var i = 0; i < projected.Count; i++)
        {
            var box = projected[i];
            var area = Math.Max(0d, box.Width) * Math.Max(0d, box.Height);
            sumArea += area;
            minLeft = Math.Min(minLeft, box.X);
            minTop = Math.Min(minTop, box.Y);
        }

        coverageRatio = sumArea / pageArea;
        if (coverageRatio < CoverageAugmentationThreshold)
        {
            return true;
        }

        if (minLeft > meta.OriginalWidth * EdgeAugmentationThreshold)
        {
            return true;
        }

        if (minTop > meta.OriginalHeight * EdgeAugmentationThreshold)
        {
            return true;
        }

        return false;
    }

    internal static IReadOnlyList<LayoutSdk.BoundingBox> MergeDetections(
        IReadOnlyList<LayoutSdk.BoundingBox> primary,
        IReadOnlyList<LayoutSdk.BoundingBox> fallback)
    {
        if (fallback.Count == 0)
        {
            return primary.ToArray();
        }

        if (primary.Count == 0)
        {
            return fallback.ToArray();
        }

        var merged = new List<LayoutSdk.BoundingBox>(primary);
        for (var i = 0; i < fallback.Count; i++)
        {
            var candidate = fallback[i];
            var bestIoU = 0d;
            for (var j = 0; j < merged.Count; j++)
            {
                if (!LabelsEqual(merged[j].Label, candidate.Label))
                {
                    continue;
                }

                var current = ComputeIoU(merged[j], candidate);
                if (current > bestIoU)
                {
                    bestIoU = current;
                }
            }

            if (bestIoU >= DuplicateIoUThreshold)
            {
                continue;
            }

            merged.Add(candidate);
        }

        return SuppressContainedDetections(merged);
    }

    private static IReadOnlyList<LayoutSdk.BoundingBox> SuppressContainedDetections(List<LayoutSdk.BoundingBox> boxes)
    {
        if (boxes.Count <= 1)
        {
            return boxes.ToArray();
        }

        var toRemove = new bool[boxes.Count];
        for (var i = 0; i < boxes.Count; i++)
        {
            if (toRemove[i])
            {
                continue;
            }

            var current = boxes[i];
            var currentArea = ComputeArea(current);
            if (currentArea <= 0d)
            {
                continue;
            }

            for (var j = 0; j < boxes.Count; j++)
            {
                if (i == j || toRemove[j])
                {
                    continue;
                }

                var candidate = boxes[j];
                if (!LabelsEqual(current.Label, candidate.Label))
                {
                    continue;
                }

                var candidateArea = ComputeArea(candidate);
                if (candidateArea <= 0d)
                {
                    toRemove[j] = true;
                    continue;
                }

                LayoutSdk.BoundingBox bigger;
                LayoutSdk.BoundingBox smaller;
                int smallerIndex;
                if (currentArea >= candidateArea)
                {
                    bigger = current;
                    smaller = candidate;
                    smallerIndex = j;
                }
                else
                {
                    bigger = candidate;
                    smaller = current;
                    smallerIndex = i;
                }

                var biggerArea = Math.Max(currentArea, candidateArea);
                var smallerArea = Math.Min(currentArea, candidateArea);
                if (biggerArea < smallerArea * ContainmentSuppressionRatio)
                {
                    continue;
                }

                if (Contains(bigger, smaller))
                {
                    toRemove[smallerIndex] = true;
                }
            }
        }

        var filtered = new List<LayoutSdk.BoundingBox>(boxes.Count);
        for (var i = 0; i < boxes.Count; i++)
        {
            if (!toRemove[i])
            {
                filtered.Add(boxes[i]);
            }
        }

        return filtered;
    }

    private static bool Contains(LayoutSdk.BoundingBox container, LayoutSdk.BoundingBox candidate)
    {
        var left = container.X - ContainmentTolerance;
        var top = container.Y - ContainmentTolerance;
        var right = container.X + container.Width + ContainmentTolerance;
        var bottom = container.Y + container.Height + ContainmentTolerance;

        var candidateRight = candidate.X + candidate.Width;
        var candidateBottom = candidate.Y + candidate.Height;

        return left <= candidate.X
            && top <= candidate.Y
            && right >= candidateRight
            && bottom >= candidateBottom;
    }

    private static bool LabelsEqual(string? first, string? second)
    {
        return string.Equals(first, second, StringComparison.OrdinalIgnoreCase);
    }

    private static double ComputeIoU(LayoutSdk.BoundingBox first, LayoutSdk.BoundingBox second)
    {
        var ax1 = first.X;
        var ay1 = first.Y;
        var ax2 = first.X + first.Width;
        var ay2 = first.Y + first.Height;

        var bx1 = second.X;
        var by1 = second.Y;
        var bx2 = second.X + second.Width;
        var by2 = second.Y + second.Height;

        var interLeft = Math.Max(ax1, bx1);
        var interTop = Math.Max(ay1, by1);
        var interRight = Math.Min(ax2, bx2);
        var interBottom = Math.Min(ay2, by2);

        var interWidth = Math.Max(0d, interRight - interLeft);
        var interHeight = Math.Max(0d, interBottom - interTop);
        var interArea = interWidth * interHeight;
        var union = ComputeArea(first) + ComputeArea(second) - interArea;

        if (union <= 0d)
        {
            return 0d;
        }

        return interArea / union;
    }

    private static double ComputeArea(LayoutSdk.BoundingBox box)
    {
        return Math.Max(0d, box.Width) * Math.Max(0d, box.Height);
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

        [LoggerMessage(EventId = 4105, Level = LogLevel.Information, Message = "Initialised fallback ONNX session using model {ModelPath}.")]
        public static partial void FallbackSessionCreated(ILogger logger, string modelPath);

        [LoggerMessage(EventId = 4106, Level = LogLevel.Debug, Message = "Fallback inference discarded: {Reason}")]
        public static partial void FallbackSkipped(ILogger logger, string reason);

        [LoggerMessage(EventId = 4107, Level = LogLevel.Information, Message = "Fallback ONNX inference produced {DetectionCount} detections.")]
        public static partial void FallbackSucceeded(ILogger logger, int detectionCount);

        [LoggerMessage(EventId = 4108, Level = LogLevel.Warning, Message = "Fallback ONNX inference failed.")]
        public static partial void FallbackFailed(ILogger logger, Exception exception);

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

        [LoggerMessage(EventId = 4111, Level = LogLevel.Debug, Message = "Fallback reprojection mapped {DetectionCount} detections to {OriginalWidth}x{OriginalHeight} (scaled {ScaledWidth}x{ScaledHeight}, scale {Scale:F3}, offsets {OffsetX:F2},{OffsetY:F2}).")]
        public static partial void FallbackDenormalised(
            ILogger logger,
            int originalWidth,
            int originalHeight,
            int scaledWidth,
            int scaledHeight,
            double scale,
            double offsetX,
            double offsetY,
            int detectionCount);

        [LoggerMessage(EventId = 4112, Level = LogLevel.Debug, Message = "Fallback reprojection skipped because no normalisation metadata was available (kept {DetectionCount} detections as-is).")]
        public static partial void FallbackDenormalisationSkipped(ILogger logger, int detectionCount);

        [LoggerMessage(EventId = 4113, Level = LogLevel.Information, Message = "Coverage-driven fallback triggered (page {OriginalWidth}x{OriginalHeight}, coverage {CoverageRatio:P1}).")]
        public static partial void FallbackAugmentationTriggered(ILogger logger, int originalWidth, int originalHeight, double coverageRatio);

        [LoggerMessage(EventId = 4114, Level = LogLevel.Information, Message = "Augmented layout detections with fallback results (primary {PrimaryCount}, fallback {FallbackCount}, merged {MergedCount}).")]
        public static partial void FallbackAugmented(ILogger logger, int primaryCount, int fallbackCount, int mergedCount);

        [LoggerMessage(EventId = 4115, Level = LogLevel.Debug, Message = "Fallback augmentation produced no additional detections.")]
        public static partial void FallbackAugmentationSkipped(ILogger logger);
    }

    private readonly record struct PersistedImage(string Path, LayoutSdkNormalisationMetadata? Normalisation);

    private readonly record struct NormalisationResult(bool UseOriginal, byte[]? Image, LayoutSdkNormalisationMetadata? Metadata);

    private enum FallbackReason
    {
        None,
        Empty,
        Clipped,
        Augmentation,
    }

}
