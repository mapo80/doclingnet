using SkiaSharp;

namespace LayoutSdk.Processing;

public interface IImagePreprocessor
{
    ImageTensor Preprocess(SKBitmap image);
}
