using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TableFormerSdk.Configuration;

/// <summary>
/// Represents the normalization parameters from the TableFormer config.
/// </summary>
public sealed record NormalizationParameters
{
    [JsonPropertyName("mean")]
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Performance", "CA1819:Properties should not return arrays", Justification = "DTO for JSON serialization")]
    public float[] Mean { get; init; } = Array.Empty<float>();

    [JsonPropertyName("std")]
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Performance", "CA1819:Properties should not return arrays", Justification = "DTO for JSON serialization")]
    public float[] Std { get; init; } = Array.Empty<float>();

    [JsonPropertyName("state")]
    public bool Enabled { get; init; } = true;

    public static NormalizationParameters Default => new()
    {
        Mean = [0.94247851f, 0.94254675f, 0.94292611f],
        Std = [0.17910956f, 0.17940403f, 0.17931663f],
        Enabled = true
    };
}

/// <summary>
/// Dataset configuration from TableFormer config JSON.
/// </summary>
public sealed record DatasetConfig
{
    [JsonPropertyName("resized_image")]
    public int ResizedImage { get; init; } = 448;

    [JsonPropertyName("image_normalization")]
    public NormalizationParameters? ImageNormalization { get; init; }
}

/// <summary>
/// Model configuration from TableFormer config JSON.
/// </summary>
public sealed record ModelConfig
{
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    [JsonPropertyName("enc_image_size")]
    public int EncoderImageSize { get; init; } = 28;

    [JsonPropertyName("bbox_classes")]
    public int BboxClasses { get; init; } = 2;

    [JsonPropertyName("hidden_dim")]
    public int HiddenDim { get; init; } = 512;
}

/// <summary>
/// Root configuration object loaded from tableformer_*_config.json.
/// </summary>
public sealed record TableFormerConfig
{
    [JsonPropertyName("dataset")]
    public DatasetConfig? Dataset { get; init; }

    [JsonPropertyName("model")]
    public ModelConfig? Model { get; init; }

    [JsonPropertyName("dataset_wordmap")]
    public JsonElement? DatasetWordmap { get; init; }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        AllowTrailingCommas = true,
        ReadCommentHandling = JsonCommentHandling.Skip
    };

    /// <summary>
    /// Load TableFormer config from JSON file.
    /// </summary>
    public static TableFormerConfig LoadFromFile(string configPath)
    {
        if (string.IsNullOrWhiteSpace(configPath))
        {
            throw new ArgumentException("Config path is empty", nameof(configPath));
        }

        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"Config file not found: {configPath}", configPath);
        }

        var json = File.ReadAllText(configPath);
        var config = JsonSerializer.Deserialize<TableFormerConfig>(json, JsonOptions);

        if (config is null)
        {
            throw new InvalidOperationException($"Failed to parse config from: {configPath}");
        }

        return config;
    }

    /// <summary>
    /// Normalization parameters with fallback to default PubTabNet values.
    /// </summary>
    public NormalizationParameters NormalizationParameters => Dataset?.ImageNormalization ?? NormalizationParameters.Default;

    /// <summary>
    /// Target image size for preprocessing.
    /// </summary>
    public int TargetImageSize => Dataset?.ResizedImage ?? 448;

    /// <summary>
    /// Number of bbox classes (not including background).
    /// </summary>
    public int BboxClasses => Model?.BboxClasses ?? 2;
}
