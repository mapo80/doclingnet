namespace TableFormerSdk.Enums;

/// <summary>
/// TableFormer model variants from HuggingFace asmud/ds4sd-docling-models-onnx
/// </summary>
public enum TableFormerModelVariant
{
    /// <summary>
    /// Fast variant - Optimized for speed (~0.7ms inference, 94% TEDS)
    /// </summary>
    Fast,

    /// <summary>
    /// Accurate variant - Optimized for accuracy (~1ms inference, 95.4% TEDS)
    /// </summary>
    Accurate
}
