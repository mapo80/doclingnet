using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Docling.Core.Models.Tables;

/// <summary>
/// Implements the autoregressive loop for TableFormer tag generation.
/// Handles OTSL (Ordered Table Structure Language) sequence generation with error correction.
/// </summary>
internal sealed class TableFormerAutoregressive
{
    private readonly TableFormerOnnxComponents _components;

    // OTSL Vocabulary (13 tokens total)
    private readonly Dictionary<string, int> _tokenToId = new()
    {
        ["<start>"] = 0,
        ["<end>"] = 1,
        ["<pad>"] = 2,
        ["fcel"] = 3,
        ["ecel"] = 4,
        ["lcel"] = 5,
        ["xcel"] = 6,
        ["ucel"] = 7,
        ["nl"] = 8,
        ["ched"] = 9,
        ["rhed"] = 10,
        ["srow"] = 11
    };

    private readonly Dictionary<int, string> _idToToken = new();
    private readonly int _maxSteps = 1024;

    public TableFormerAutoregressive(TableFormerOnnxComponents components)
    {
        _components = components ?? throw new ArgumentNullException(nameof(components));

        // Build reverse mapping
        foreach (var (token, id) in _tokenToId)
        {
            _idToToken[id] = token;
        }
    }

    /// <summary>
    /// Generate OTSL tag sequence autoregressively.
    /// Returns the sequence of tag hidden states for bbox prediction.
    /// </summary>
    public List<DenseTensor<float>> GenerateTags(
        DenseTensor<float> memory,
        DenseTensor<bool> encoderMask)
    {
        var tagHiddenStates = new List<DenseTensor<float>>();
        var generatedTokens = new List<long>();
        var currentTags = new DenseTensor<long>(new[] { 1, 1 }); // [1, 1] - start token

        // Start with <start> token (index 0)
        currentTags[0, 0] = 0;
        generatedTokens.Add(0);

        var step = 0;
        var endToken = _tokenToId["<end>"];

        while (step < _maxSteps)
        {
            // Run decoder step
            var (logits, hiddenState) = _components.RunTagTransformerDecoderStep(
                currentTags, memory, encoderMask);

            // Get next token (greedy decoding for now)
            var nextToken = GetNextToken(logits);

            // Early stopping on <end> token
            if (nextToken == endToken)
            {
                break;
            }

            generatedTokens.Add(nextToken);

            // Apply structure error correction
            var correctedToken = ApplyStructureCorrection(generatedTokens, nextToken);

            if (correctedToken != nextToken)
            {
                // If we corrected the token, we need to regenerate the hidden state
                // For simplicity, we'll use the original hidden state
                // In a more sophisticated implementation, we might want to backtrack
            }

            // Update current tags for next step
            var newCurrentTags = new DenseTensor<long>(new[] { 1, generatedTokens.Count });
            for (int i = 0; i < generatedTokens.Count; i++)
            {
                newCurrentTags[0, i] = generatedTokens[i];
            }

            currentTags = newCurrentTags;

            // Collect hidden state for bbox prediction (only for cell tokens)
            if (IsCellToken(_idToToken[(int)correctedToken]))
            {
                tagHiddenStates.Add(hiddenState.ToDenseTensor());
            }

            step++;
        }

        return tagHiddenStates;
    }

    private static long GetNextToken(DenseTensor<float> logits)
    {
        // Simple greedy decoding - take argmax
        var logitsArray = logits.ToArray();
        var maxIndex = 0;
        var maxValue = float.NegativeInfinity;

        for (int i = 0; i < logitsArray.Length; i++)
        {
            if (logitsArray[i] > maxValue)
            {
                maxValue = logitsArray[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private long ApplyStructureCorrection(List<long> generatedTokens, long nextToken)
    {
        var nextTokenStr = _idToToken[(int)nextToken];

        // Rule 1: First line should use lcel instead of xcel
        if (nextTokenStr == "xcel" && IsFirstLine(generatedTokens))
        {
            return _tokenToId["lcel"];
        }

        // Rule 2: After ucel, lcel should become fcel
        if (nextTokenStr == "lcel")
        {
            var lastToken = generatedTokens.LastOrDefault();
            if (lastToken == _tokenToId["ucel"])
            {
                return _tokenToId["fcel"];
            }
        }

        return nextToken;
    }

    private bool IsFirstLine(List<long> tokens)
    {
        // Check if we're still in the first row (no "nl" token yet)
        return !tokens.Contains(_tokenToId["nl"]);
    }

    private static bool IsCellToken(string token)
    {
        return token is "fcel" or "ecel" or "lcel" or "xcel" or "ucel";
    }

    /// <summary>
    /// Convert token sequence to human-readable OTSL string.
    /// </summary>
    public string TokensToOtsl(IEnumerable<long> tokens)
    {
        return string.Join(" ", tokens.Select(id => _idToToken[(int)id]));
    }

    /// <summary>
    /// Convert OTSL string to token sequence.
    /// </summary>
    public int[] OtslToTokens(string otsl)
    {
        return otsl.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                  .Select(token => _tokenToId[token])
                  .ToArray();
    }
}