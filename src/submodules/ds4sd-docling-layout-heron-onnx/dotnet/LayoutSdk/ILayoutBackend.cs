using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

public interface ILayoutBackend
{
    LayoutBackendResult Infer(ImageTensor tensor);
}
