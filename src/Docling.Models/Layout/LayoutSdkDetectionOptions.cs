using System;
using System.IO;
using LayoutSdk;
using LayoutSdk.Configuration;

namespace Docling.Models.Layout;

/// <summary>
/// Configures how the Layout SDK based detection service executes the Docling Heron model.
/// </summary>
public sealed class LayoutSdkDetectionOptions
{
    /// <summary>
    /// Gets or sets the default language hint passed to the SDK. Defaults to English.
    /// </summary>
    public DocumentLanguage Language { get; init; } = DocumentLanguage.English;

    /// <summary>
    /// When <c>true</c> the SDK renders overlay images alongside predictions. Defaults to <c>false</c>.
    /// </summary>
    public bool GenerateOverlay { get; init; }

    /// <summary>
    /// When <c>true</c> the service verifies that all model files exist before first use. Defaults to <c>true</c>.
    /// </summary>
    public bool ValidateModelFiles { get; init; } = true;

    /// <summary>
    /// Optional directory used to materialise temporary PNG files before invoking the SDK.
    /// When not specified the system temporary path is used.
    /// </summary>
    public string? WorkingDirectory { get; init; }

    /// <summary>
    /// Keeps the intermediate PNG artefacts on disk for troubleshooting when set to <c>true</c>.
    /// Defaults to <c>false</c> so files are eagerly deleted.
    /// </summary>
    public bool KeepTemporaryFiles { get; init; }

    /// <summary>
    /// Controls the maximum number of concurrent inference requests executed against the SDK.
    /// Values must be greater than or equal to one. Defaults to <c>1</c>.
    /// </summary>
    public int MaxDegreeOfParallelism { get; init; } = 1;

    /// <summary>
    /// Selects which runtime implementation is used to execute the layout model.
    /// Defaults to the ONNX runtime to mirror the Python pipeline.
    /// </summary>
    public LayoutRuntime Runtime { get; init; } = LayoutRuntime.Onnx;

    /// <summary>
    /// Enables the advanced non-maximum suppression heuristic provided by the Layout SDK.
    /// Defaults to <c>true</c> to preserve parity with the Python pipeline.
    /// </summary>
    public bool EnableAdvancedNonMaxSuppression { get; init; } = true;

    /// <summary>
    /// When <c>true</c> the runner captures a timing breakdown for persistence, inference, and post-processing.
    /// Disabled by default to avoid the overhead outside of performance investigations.
    /// </summary>
    public bool EnableProfiling { get; init; }

    internal LayoutSdkDetectionOptions Clone() => new()
    {
        Language = Language,
        GenerateOverlay = GenerateOverlay,
        ValidateModelFiles = ValidateModelFiles,
        WorkingDirectory = WorkingDirectory,
        KeepTemporaryFiles = KeepTemporaryFiles,
        MaxDegreeOfParallelism = MaxDegreeOfParallelism,
        Runtime = Runtime,
        EnableAdvancedNonMaxSuppression = EnableAdvancedNonMaxSuppression,
        EnableProfiling = EnableProfiling,
    };

    internal void EnsureValid()
    {
        if (MaxDegreeOfParallelism <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxDegreeOfParallelism), "MaxDegreeOfParallelism must be positive.");
        }

        if (!string.IsNullOrWhiteSpace(WorkingDirectory))
        {
            _ = Path.GetFullPath(WorkingDirectory);
        }
    }
}
