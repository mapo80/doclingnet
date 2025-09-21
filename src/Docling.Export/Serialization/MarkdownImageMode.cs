namespace Docling.Export.Serialization;

/// <summary>
/// Controls how images are represented in the generated markdown.
/// </summary>
public enum MarkdownImageMode
{
    /// <summary>
    /// Emits textual placeholders referencing the original image content.
    /// </summary>
    Placeholder,

    /// <summary>
    /// Emits references to asset files alongside the markdown payload.
    /// </summary>
    Referenced,

    /// <summary>
    /// Embeds image bytes directly in the markdown using data URIs.
    /// </summary>
    Embedded,
}
