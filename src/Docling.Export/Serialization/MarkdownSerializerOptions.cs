using System;

namespace Docling.Export.Serialization;

/// <summary>
/// Configures how the markdown serializer emits content and auxiliary assets.
/// </summary>
public sealed class MarkdownSerializerOptions
{
    private string _assetsPath = "assets";

    /// <summary>
    /// Optional provider invoked to enrich alternative text for figures and table previews.
    /// </summary>
    public IMarkdownAltTextProvider? AltTextProvider { get; init; }

    /// <summary>
    /// Controls how images are embedded in the generated markdown.
    /// </summary>
    public MarkdownImageMode ImageMode { get; init; } = MarkdownImageMode.Referenced;

    /// <summary>
    /// Optional relative path where image assets should be written when <see cref="ImageMode"/> is <see cref="MarkdownImageMode.Referenced"/>.
    /// Forward slashes are used regardless of the host platform to keep markdown links portable.
    /// </summary>
    public string AssetsPath
    {
        get => _assetsPath;
        init => _assetsPath = NormaliseAssetsPath(value);
    }

    /// <summary>
    /// Controls the caption template applied to pictures. "{0}" will be replaced with the figure index.
    /// </summary>
    public string FigureLabelFormat { get; init; } = "Figure {0}";

    /// <summary>
    /// Controls the caption template applied to tables. "{0}" will be replaced with the table index.
    /// </summary>
    public string TableLabelFormat { get; init; } = "Table {0}";

    internal string CombineAssetPath(string fileName)
    {
        if (string.IsNullOrEmpty(_assetsPath))
        {
            return fileName;
        }

        return _assetsPath.EndsWith('/')
            ? _assetsPath + fileName
            : _assetsPath + "/" + fileName;
    }

    private static string NormaliseAssetsPath(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        var trimmed = value.Trim();
        return trimmed.Replace("\\", "/", StringComparison.Ordinal);
    }
}
