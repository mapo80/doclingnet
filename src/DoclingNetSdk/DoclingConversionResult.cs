using System;
using Docling.Core.Documents;

namespace DoclingNetSdk;

/// <summary>
/// Result of a document conversion operation.
/// </summary>
public sealed class DoclingConversionResult
{
    /// <summary>
    /// Creates a new conversion result.
    /// </summary>
    public DoclingConversionResult(
        DoclingDocument document,
        string markdown,
        int layoutElementCount,
        int ocrElementCount,
        int tableCount)
    {
        Document = document ?? throw new ArgumentNullException(nameof(document));
        Markdown = markdown ?? throw new ArgumentNullException(nameof(markdown));
        LayoutElementCount = layoutElementCount;
        OcrElementCount = ocrElementCount;
        TableCount = tableCount;
    }

    /// <summary>
    /// The structured document containing all extracted elements.
    /// </summary>
    public DoclingDocument Document { get; }

    /// <summary>
    /// The document exported as Markdown.
    /// </summary>
    public string Markdown { get; }

    /// <summary>
    /// Number of layout elements detected (text blocks, tables, figures, etc.).
    /// </summary>
    public int LayoutElementCount { get; }

    /// <summary>
    /// Number of OCR elements processed.
    /// </summary>
    public int OcrElementCount { get; }

    /// <summary>
    /// Number of tables detected and processed.
    /// </summary>
    public int TableCount { get; }

    /// <summary>
    /// Total number of document items in the structured output.
    /// </summary>
    public int TotalItems => Document.Items.Count;

    /// <summary>
    /// Returns a summary of the conversion result.
    /// </summary>
    public override string ToString()
    {
        return $"DoclingConversionResult: {TotalItems} items, {LayoutElementCount} layout elements, " +
               $"{OcrElementCount} OCR elements, {TableCount} tables, {Markdown.Length} chars markdown";
    }
}
