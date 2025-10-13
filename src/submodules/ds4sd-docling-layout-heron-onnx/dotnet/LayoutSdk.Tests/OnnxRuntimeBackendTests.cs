using LayoutSdk;
using LayoutSdk.Processing;
using System;
using Xunit;

namespace LayoutSdk.Tests;

public class OnnxRuntimeBackendTests
{
   [Fact]
   public void Constructor_InvalidModel_Throws()
   {
       Assert.Throws<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => new OnnxRuntimeBackend("missing.onnx"));
   }
}
