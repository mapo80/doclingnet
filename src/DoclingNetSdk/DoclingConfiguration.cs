using System;
using System.IO;

namespace DoclingNetSdk;

/// <summary>
/// Configuration for DoclingConverter.
/// </summary>
public sealed class DoclingConfiguration
{
    /// <summary>
    /// Path to the artifacts directory for caching models and intermediate files.
    /// Default: ./artifacts
    /// </summary>
    public string ArtifactsPath { get; set; } = Path.Combine(Directory.GetCurrentDirectory(), "artifacts");

    /// <summary>
    /// OCR language code (e.g., "en" for English, "it" for Italian).
    /// Default: "en"
    /// </summary>
    public string OcrLanguage { get; set; } = "en";

    /// <summary>
    /// Enable table structure recognition.
    /// Default: true
    /// </summary>
    public bool EnableTableRecognition { get; set; } = true;

    /// <summary>
    /// Enable OCR text extraction.
    /// Default: true
    /// </summary>
    public bool EnableOcr { get; set; } = true;

    /// <summary>
    /// TableFormer model variant (Fast, Base, or Accurate).
    /// Default: Fast
    /// </summary>
    public TableFormerVariant TableFormerVariant { get; set; } = TableFormerVariant.Fast;

    /// <summary>
    /// Creates a default configuration.
    /// </summary>
    /// <returns>Configuration with default settings.</returns>
    public static DoclingConfiguration CreateDefault()
    {
        var config = new DoclingConfiguration();

        // Create artifacts directory
        Directory.CreateDirectory(config.ArtifactsPath);

        return config;
    }

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when configuration is invalid.</exception>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(ArtifactsPath))
        {
            throw new InvalidOperationException("ArtifactsPath cannot be empty.");
        }

        // Create artifacts directory if it doesn't exist
        Directory.CreateDirectory(ArtifactsPath);
    }
}

/// <summary>
/// TableFormer model variant selection.
/// </summary>
public enum TableFormerVariant
{
    /// <summary>
    /// Fast model - optimized for speed (~300ms per table).
    /// </summary>
    Fast,

    /// <summary>
    /// Base model - balanced speed and accuracy (~500ms per table).
    /// </summary>
    Base,

    /// <summary>
    /// Accurate model - highest accuracy (~1s per table).
    /// </summary>
    Accurate
}
