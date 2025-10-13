using System;
using LayoutSdk.Configuration;

namespace LayoutSdk.Factories;

public sealed class LayoutBackendFactory : ILayoutBackendFactory
{
   private readonly LayoutSdkOptions _options;
   private readonly Func<string, ILayoutBackend> _onnxFactory;

   public LayoutBackendFactory(
       LayoutSdkOptions options,
       Func<string, ILayoutBackend>? onnxFactory = null)
   {
       _options = options ?? throw new ArgumentNullException(nameof(options));
       _options.EnsureModelPaths();
       _onnxFactory = onnxFactory ?? (path => new OnnxRuntimeBackend(path));
   }

   public ILayoutBackend Create(LayoutRuntime runtime) => runtime switch
   {
       LayoutRuntime.Onnx => _onnxFactory(_options.OnnxModelPath),
       _ => throw new NotSupportedException($"Runtime {runtime} is not supported. Only ONNX runtime is available.")
   };
}
