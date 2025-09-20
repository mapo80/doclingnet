using System;

namespace Docling.Pipelines.Options;

/// <summary>
/// Represents a strongly typed identifier for a layout detection model.
/// </summary>
public sealed record class LayoutModelConfiguration
{
    private LayoutModelConfiguration(string identifier)
    {
        Identifier = identifier;
    }

    /// <summary>
    /// Canonical identifier for the model preset.
    /// </summary>
    public string Identifier { get; }

    public static LayoutModelConfiguration DoclingLayoutHeron { get; } = new("docling-layout-heron");
    public static LayoutModelConfiguration DoclingLayoutEgretMedium { get; } = new("docling-layout-egret-medium");
    public static LayoutModelConfiguration DoclingLayoutEgretLarge { get; } = new("docling-layout-egret-large");
    public static LayoutModelConfiguration DoclingLayoutEgretXlarge { get; } = new("docling-layout-egret-xlarge");

    /// <summary>
    /// Creates a custom model configuration using a free-form identifier.
    /// </summary>
    public static LayoutModelConfiguration Custom(string identifier)
    {
        if (string.IsNullOrWhiteSpace(identifier))
        {
            throw new ArgumentException("The identifier must be a non-empty string.", nameof(identifier));
        }

        return new LayoutModelConfiguration(identifier);
    }
}

/// <summary>
/// Options influencing the layout analysis stage.
/// </summary>
public sealed class LayoutOptions
{
    public bool CreateOrphanClusters { get; init; } = true;

    public bool KeepEmptyClusters { get; init; }

    public LayoutModelConfiguration Model { get; init; } = LayoutModelConfiguration.DoclingLayoutHeron;

    public bool SkipCellAssignment { get; init; }

    public bool GenerateDebugArtifacts { get; init; }

    public float DebugOverlayOpacity { get; init; } = 0.25f;

    public float DebugStrokeWidth { get; init; } = 2.5f;
}
