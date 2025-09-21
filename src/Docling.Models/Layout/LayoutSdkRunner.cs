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
using SkiaSharp;

namespace Docling.Models.Layout;

internal interface ILayoutSdkRunner : IDisposable
{
    Task<IReadOnlyList<LayoutSdk.BoundingBox>> InferAsync(ReadOnlyMemory<byte> imageContent, CancellationToken cancellationToken);
}

[ExcludeFromCodeCoverage]
internal sealed partial class LayoutSdkRunner : ILayoutSdkRunner
{
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

    public async Task<IReadOnlyList<LayoutSdk.BoundingBox>> InferAsync(ReadOnlyMemory<byte> imageContent, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();

        var path = await PersistAsync(imageContent, cancellationToken).ConfigureAwait(false);
        try
        {
            await _semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                var result = _sdk.Process(path, _options.GenerateOverlay, _options.Runtime);
                try
                {
                    var boxes = result.Boxes ?? Array.Empty<LayoutSdk.BoundingBox>();
                    if (boxes.Count == 0)
                    {
                        return Array.Empty<LayoutSdk.BoundingBox>();
                    }

                    return boxes.Select(b => new LayoutSdk.BoundingBox(b.X, b.Y, b.Width, b.Height, b.Label)).ToArray();
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

    private async Task<string> PersistAsync(ReadOnlyMemory<byte> content, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(_workingDirectory);
        var fileName = Path.Combine(_workingDirectory, Guid.NewGuid().ToString("N")) + ".png";
        var stream = new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.Read, 4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
        try
        {
            await stream.WriteAsync(content, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            await stream.DisposeAsync().ConfigureAwait(false);
        }

        return fileName;
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
    }

}
