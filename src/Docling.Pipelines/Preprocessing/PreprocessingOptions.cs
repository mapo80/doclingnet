using System;
using SkiaSharp;

namespace Docling.Pipelines.Preprocessing;

/// <summary>
/// Enumerates supported colour output modes for the preprocessing stage.
/// </summary>
public enum PageColorMode
{
    Preserve,
    Grayscale,
    Binary,
}

/// <summary>
/// Describes how page images should be normalised prior to downstream processing.
/// </summary>
public sealed class PreprocessingOptions
{
    private double _targetDpi = 300d;
    private double _deskewMaxAngle = 5d;
    private double _deskewMinimumAngle = 0.1d;
    private float _contrastAmount = 0.35f;
    private SKColor _backgroundColor = SKColors.White;

    /// <summary>
    /// Desired DPI for the normalised bitmap.
    /// </summary>
    public double TargetDpi
    {
        get => _targetDpi;
        init => _targetDpi = value > 0d
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Target DPI must be positive.");
    }

    /// <summary>
    /// Output colour mode for the processed bitmap.
    /// </summary>
    public PageColorMode ColorMode { get; init; } = PageColorMode.Preserve;

    /// <summary>
    /// When true a deskew routine will attempt to recover a horizontal baseline.
    /// </summary>
    public bool EnableDeskew { get; init; } = true;

    /// <summary>
    /// Maximum absolute rotation (in degrees) that the deskew algorithm may apply.
    /// </summary>
    public double DeskewMaxAngle
    {
        get => _deskewMaxAngle;
        init
        {
            if (value <= 0d || value > 45d)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Deskew maximum angle must be within (0, 45].");
            }

            _deskewMaxAngle = value;
        }
    }

    /// <summary>
    /// Minimum absolute rotation (in degrees) required to trigger deskewing.
    /// </summary>
    public double DeskewMinimumAngle
    {
        get => _deskewMinimumAngle;
        init
        {
            if (value < 0d)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Deskew minimum angle cannot be negative.");
            }

            _deskewMinimumAngle = value;
        }
    }

    /// <summary>
    /// When true a contrast normalisation pass is applied after colour conversion.
    /// </summary>
    public bool NormalizeContrast { get; init; } = true;

    /// <summary>
    /// Contrast adjustment intensity when <see cref="NormalizeContrast"/> is enabled. Range [0, 1].
    /// </summary>
    public float ContrastAmount
    {
        get => _contrastAmount;
        init
        {
            if (value is < 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Contrast amount must be within [0, 1].");
            }

            _contrastAmount = value;
        }
    }

    /// <summary>
    /// Background colour used when expanding pixels during rotation operations.
    /// </summary>
    public SKColor BackgroundColor
    {
        get => _backgroundColor;
        init => _backgroundColor = value;
    }

    /// <summary>
    /// Validates option invariants.
    /// </summary>
    public void EnsureValid()
    {
        if (DeskewMinimumAngle > DeskewMaxAngle)
        {
            throw new InvalidOperationException("Deskew minimum angle cannot exceed the maximum angle.");
        }
    }
}
