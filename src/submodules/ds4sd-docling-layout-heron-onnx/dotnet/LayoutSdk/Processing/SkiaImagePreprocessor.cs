using SkiaSharp;
using System.Runtime.InteropServices;
using System;

namespace LayoutSdk.Processing;

public sealed class SkiaImagePreprocessor : IImagePreprocessor
{
    internal const int Channels = 3;
    internal const int ModelInputSize = 640;

    private static readonly SKSamplingOptions Sampling = new(SKFilterMode.Linear, SKMipmapMode.None);
    private static readonly ThreadLocal<SKBitmap> ResizeBuffer = new(() =>
        new SKBitmap(new SKImageInfo(ModelInputSize, ModelInputSize, SKColorType.Bgra8888, SKAlphaType.Premul)));

    public ImageTensor Preprocess(SKBitmap image)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        // Step 1: Resize to the model input dimensions (RT-DETR uses anisotropic resize without padding)
        var resizeTarget = ResizeBuffer.Value ?? throw new InvalidOperationException("Unable to allocate resize buffer.");
        if (!image.ScalePixels(resizeTarget, Sampling))
        {
            throw new InvalidOperationException("Failed to resize image to model input size.");
        }
        var pixelBytes = resizeTarget.GetPixelSpan();
        var pixels = MemoryMarshal.Cast<byte, SKColor>(pixelBytes);

        // Step 2: Convert to tensor with CLIP/ViT normalization (same as HuggingFace)
        var tensor = ImageTensor.RentPooled(Channels, ModelInputSize, ModelInputSize);
        var span = tensor.AsSpan();

        for (var i = 0; i < pixels.Length; i++)
        {
            var color = pixels[i];

            // Convert to float [0,1] range
            var r = color.Red / 255f;
            var g = color.Green / 255f;
            var b = color.Blue / 255f;

            // Apply CLIP/ViT normalization (same as HuggingFace docling-layout-heron)
            span[i] = r;
            span[i + pixels.Length] = g;
            span[i + 2 * pixels.Length] = b;
        }

        return tensor;
    }
}
