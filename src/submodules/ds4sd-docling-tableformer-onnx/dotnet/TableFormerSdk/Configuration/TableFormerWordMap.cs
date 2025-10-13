using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TableFormerSdk.Configuration;

/// <summary>
/// Represents the word map (vocabulary) for TableFormer OTSL tags.
/// </summary>
public sealed record TableFormerWordMap
{
    [JsonPropertyName("word_map_tag")]
    public Dictionary<string, int>? WordMapTag { get; init; }

    [JsonPropertyName("word_map_cell")]
    public Dictionary<string, int>? WordMapCell { get; init; }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        AllowTrailingCommas = true,
        ReadCommentHandling = JsonCommentHandling.Skip
    };

    /// <summary>
    /// Load word map from JSON file.
    /// </summary>
    public static TableFormerWordMap LoadFromFile(string wordMapPath)
    {
        if (string.IsNullOrWhiteSpace(wordMapPath))
        {
            throw new ArgumentException("WordMap path is empty", nameof(wordMapPath));
        }

        if (!File.Exists(wordMapPath))
        {
            throw new FileNotFoundException($"WordMap file not found: {wordMapPath}", wordMapPath);
        }

        var json = File.ReadAllText(wordMapPath);
        var wordMap = JsonSerializer.Deserialize<TableFormerWordMap>(json, JsonOptions);

        if (wordMap is null)
        {
            throw new InvalidOperationException($"Failed to parse word map from: {wordMapPath}");
        }

        return wordMap;
    }

    /// <summary>
    /// Get token ID from tag token string.
    /// </summary>
    public int GetTagTokenId(string token)
    {
        if (WordMapTag is null)
        {
            throw new InvalidOperationException("WordMapTag is not loaded");
        }

        if (WordMapTag.TryGetValue(token, out var id))
        {
            return id;
        }

        // Return <unk> token if not found
        return WordMapTag.GetValueOrDefault("<unk>", 1);
    }

    /// <summary>
    /// Get tag token string from ID.
    /// </summary>
    public string GetTagToken(int id)
    {
        if (WordMapTag is null)
        {
            throw new InvalidOperationException("WordMapTag is not loaded");
        }

        // Build reverse mapping
        var reverseMap = WordMapTag.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

        if (reverseMap.TryGetValue(id, out var token))
        {
            return token;
        }

        return "<unk>";
    }

    /// <summary>
    /// Get special token IDs.
    /// </summary>
    public (int start, int end, int pad, int unk) GetSpecialTokens()
    {
        if (WordMapTag is null)
        {
            throw new InvalidOperationException("WordMapTag is not loaded");
        }

        return (
            WordMapTag.GetValueOrDefault("<start>", 2),
            WordMapTag.GetValueOrDefault("<end>", 3),
            WordMapTag.GetValueOrDefault("<pad>", 0),
            WordMapTag.GetValueOrDefault("<unk>", 1)
        );
    }

    /// <summary>
    /// Check if token is a cell token (fcel, ecel, lcel, xcel, ucel).
    /// </summary>
    public static bool IsCellToken(string token)
    {
        return token is "fcel" or "ecel" or "lcel" or "xcel" or "ucel";
    }

    /// <summary>
    /// Get default PubTabNet word map with standard token mappings.
    /// </summary>
    public static TableFormerWordMap CreateDefault()
    {
        return new TableFormerWordMap
        {
            WordMapTag = new Dictionary<string, int>
            {
                ["<pad>"] = 0,
                ["<unk>"] = 1,
                ["<start>"] = 2,
                ["<end>"] = 3,
                ["ecel"] = 4,
                ["fcel"] = 5,
                ["lcel"] = 6,
                ["ucel"] = 7,
                ["xcel"] = 8,
                ["nl"] = 9,
                ["ched"] = 10,
                ["rhed"] = 11,
                ["srow"] = 12
            }
        };
    }
}
