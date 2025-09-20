using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Docling.Pipelines.Options;

public enum PictureDescriptionMode
{
    Disabled,
    Api,
    VisionLanguageModel,
}

/// <summary>
/// Options used when generating natural language descriptions for detected pictures.
/// </summary>
public sealed class PictureDescriptionOptions
{
    private ReadOnlyDictionary<string, string> _headers = ReadOnly(new Dictionary<string, string>());
    private ReadOnlyDictionary<string, string> _parameters = ReadOnly(new Dictionary<string, string>());
    private string? _modelRepoId = "docling/smolvlm";
    private double _pictureAreaThreshold = 0.05d;
    private int _batchSize = 8;
    private double _scale = 2d;
    private TimeSpan _apiTimeout = TimeSpan.FromSeconds(20);
    private int _apiConcurrency = 1;

    public PictureDescriptionMode Mode { get; init; } = PictureDescriptionMode.VisionLanguageModel;

    public int BatchSize
    {
        get => _batchSize;
        init => _batchSize = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Batch size must be positive.");
    }

    public double Scale
    {
        get => _scale;
        init => _scale = value > 0d
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Scale must be positive.");
    }

    public double PictureAreaThreshold
    {
        get => _pictureAreaThreshold;
        init
        {
            if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0d || value > 1d)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Threshold must be between 0 (exclusive) and 1 (inclusive).");
            }

            _pictureAreaThreshold = value;
        }
    }

    public Uri? ApiEndpoint { get; init; }

    public IReadOnlyDictionary<string, string> Headers
    {
        get => _headers;
        init => _headers = ReadOnly(value ?? throw new ArgumentNullException(nameof(value)));
    }

    public IReadOnlyDictionary<string, string> Parameters
    {
        get => _parameters;
        init => _parameters = ReadOnly(value ?? throw new ArgumentNullException(nameof(value)));
    }

    public TimeSpan ApiTimeout
    {
        get => _apiTimeout;
        init => _apiTimeout = value > TimeSpan.Zero
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Timeout must be greater than zero.");
    }

    public int ApiConcurrency
    {
        get => _apiConcurrency;
        init => _apiConcurrency = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Concurrency must be positive.");
    }

    public string Prompt { get; init; } = "Describe this image in a few sentences.";

    public string? Provenance { get; init; }

    public string? ModelRepoId
    {
        get => _modelRepoId;
        init => _modelRepoId = value;
    }

    /// <summary>
    /// Validates that the option set is coherent for the selected mode.
    /// </summary>
    public void EnsureValid()
    {
        if (Mode == PictureDescriptionMode.Api && ApiEndpoint is null)
        {
            throw new InvalidOperationException("An API endpoint must be provided when picture description runs in API mode.");
        }

        if (Mode == PictureDescriptionMode.VisionLanguageModel && string.IsNullOrWhiteSpace(ModelRepoId))
        {
            throw new InvalidOperationException("A model repository identifier must be specified for VLM picture description mode.");
        }
    }

    private static ReadOnlyDictionary<string, string> ReadOnly(IEnumerable<KeyValuePair<string, string>> source)
    {
        var copy = new Dictionary<string, string>();
        foreach (var pair in source)
        {
            copy[pair.Key] = pair.Value;
        }

        return new ReadOnlyDictionary<string, string>(copy);
    }
}
