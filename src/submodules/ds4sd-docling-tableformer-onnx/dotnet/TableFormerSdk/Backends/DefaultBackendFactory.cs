using System;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Backends;

internal sealed class DefaultBackendFactory : ITableFormerBackendFactory
{
    private readonly TableFormerSdkOptions _options;

    public DefaultBackendFactory(TableFormerSdkOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public ITableFormerBackend CreateBackend(TableFormerRuntime runtime, TableFormerModelVariant variant) => runtime switch
    {
        TableFormerRuntime.Auto => throw new ArgumentException("Auto runtime must be resolved before backend creation", nameof(runtime)),
        TableFormerRuntime.Onnx => new TableFormerOnnxBackend(_options.Onnx.GetModelPaths(variant)),
        TableFormerRuntime.Pipeline or TableFormerRuntime.OptimizedPipeline => throw new NotSupportedException(
            $"Runtime '{runtime}' is no longer supported. Use the ONNX backend instead."),
        _ => throw new ArgumentOutOfRangeException(nameof(runtime), runtime, "Unsupported runtime")
    };
}
