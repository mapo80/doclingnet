using System.Reflection;
using Docling.Models.Layout;
using Xunit;

namespace Docling.Tests.Layout;

public static class LayoutSdkRunnerTests
{
    [Fact]
    public static void RunnerUsesNamespaceOverlayRenderer()
    {
        var nestedType = typeof(LayoutSdkRunner).GetNestedType("PassthroughOverlayRenderer", BindingFlags.NonPublic);
        Assert.Null(nestedType);

        var overlayType = typeof(LayoutSdkRunner).Assembly.GetType("Docling.Models.Layout.PassthroughOverlayRenderer");
        Assert.NotNull(overlayType);
        Assert.True(typeof(LayoutSdk.Rendering.IImageOverlayRenderer).IsAssignableFrom(overlayType));
    }
}
