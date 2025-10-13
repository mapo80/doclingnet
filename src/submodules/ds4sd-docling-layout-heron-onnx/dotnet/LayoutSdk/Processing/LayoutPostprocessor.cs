using System;
using System.Buffers;
using System.Collections.Generic;
using LayoutSdk.Inference;

namespace LayoutSdk.Processing;

/// <summary>
/// Mirrors the HuggingFace <c>RTDetrImageProcessor.post_process_object_detection</c> implementation.
/// </summary>
public sealed class LayoutPostprocessor
{
    private readonly LayoutPostprocessOptions _options;

    public LayoutPostprocessor(LayoutPostprocessOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (_options.Labels == null || _options.Labels.Count == 0)
        {
            throw new ArgumentException("At least one label must be configured.", nameof(options));
        }
    }

    /// <summary>
    /// Converts raw model outputs into absolute bounding boxes aligned with HuggingFace's Python pipeline.
    /// </summary>
    /// <param name="backendResult">Raw logits and box predictions from the inference backend.</param>
    /// <param name="targetHeight">Original image height (in pixels).</param>
    /// <param name="targetWidth">Original image width (in pixels).</param>
    public IReadOnlyList<BoundingBox> Postprocess(LayoutBackendResult backendResult, int targetHeight, int targetWidth)
    {
        ArgumentNullException.ThrowIfNull(backendResult);

        if (targetHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetHeight));
        }

        if (targetWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetWidth));
        }

        var scoresShape = backendResult.ScoresShape.Span;
        var boxesShape = backendResult.BoxesShape.Span;

        if (scoresShape[0] != boxesShape[0])
        {
            throw new InvalidOperationException("Scores and boxes batch dimensions must match.");
        }

        var batchSize = scoresShape[0];
        if (batchSize != 1)
        {
            throw new NotSupportedException("Only batch size 1 is supported in the current pipeline.");
        }

        var numQueries = scoresShape[1];
        var numClasses = scoresShape[2];
        if (boxesShape[1] != numQueries || boxesShape[2] != 4)
        {
            throw new InvalidOperationException("Box tensor shape must be [batch, queries, 4].");
        }

        var scores = backendResult.GetScores().Span;
        var candidates = _options.UseFocalLoss
            ? SelectWithFocalLoss(scores, scoresShape, batchIndex: 0)
            : SelectWithSoftmax(scores, scoresShape, batchIndex: 0);

        if (candidates.Count == 0)
        {
            return Array.Empty<BoundingBox>();
        }

        var results = new List<BoundingBox>(candidates.Count);
        var boxes = backendResult.GetBoxes().Span;
        var batchOffset = 0;
        var stride = boxesShape[2];

        foreach (var candidate in candidates)
        {
            if (candidate.QueryIndex < 0 || candidate.QueryIndex >= numQueries)
            {
                continue;
            }

            if (candidate.ClassIndex < 0)
            {
                continue;
            }

            var label = ResolveLabel(candidate.ClassIndex);
            var threshold = _options.GetThreshold(label);
            if (!(candidate.Score > threshold))
            {
                continue;
            }

            var sourceIndex = batchOffset + candidate.QueryIndex * stride;
            var cx = boxes[sourceIndex + 0];
            var cy = boxes[sourceIndex + 1];
            var width = boxes[sourceIndex + 2];
            var height = boxes[sourceIndex + 3];

            var halfWidth = width / 2f;
            var halfHeight = height / 2f;

            var x0 = Math.Clamp((cx - halfWidth) * targetWidth, 0f, targetWidth);
            var y0 = Math.Clamp((cy - halfHeight) * targetHeight, 0f, targetHeight);
            var x1 = Math.Clamp((cx + halfWidth) * targetWidth, 0f, targetWidth);
            var y1 = Math.Clamp((cy + halfHeight) * targetHeight, 0f, targetHeight);

            var boxWidth = x1 - x0;
            var boxHeight = y1 - y0;
            if (boxWidth <= 0f || boxHeight <= 0f)
            {
                continue;
            }

            results.Add(new BoundingBox(
                (float)x0,
                (float)y0,
                (float)boxWidth,
                (float)boxHeight,
                label,
                candidate.Score));
        }

        return results;
    }

    private IReadOnlyList<DetectionCandidate> SelectWithFocalLoss(
        ReadOnlySpan<float> scores,
        ReadOnlySpan<int> scoresShape,
        int batchIndex)
    {
        var numQueries = scoresShape[1];
        var numClasses = scoresShape[2];
        var totalEntries = numQueries * numClasses;

        var entries = new DetectionCandidate[totalEntries];
        var batchOffset = batchIndex * totalEntries;

        for (var query = 0; query < numQueries; query++)
        {
            for (var cls = 0; cls < numClasses; cls++)
            {
                var flatIndex = query * numClasses + cls;
                var logitIndex = batchOffset + flatIndex;
                var score = Sigmoid(scores[logitIndex]);
                entries[flatIndex] = new DetectionCandidate(score, query, cls, flatIndex);
            }
        }

        Array.Sort(entries, static (a, b) =>
        {
            var byScore = b.Score.CompareTo(a.Score);
            return byScore != 0 ? byScore : a.FlattenedIndex.CompareTo(b.FlattenedIndex);
        });

        var take = Math.Min(numQueries, entries.Length);
        if (take == entries.Length)
        {
            return entries;
        }

        var subset = new DetectionCandidate[take];
        Array.Copy(entries, subset, take);
        return subset;
    }

    private IReadOnlyList<DetectionCandidate> SelectWithSoftmax(
        ReadOnlySpan<float> scores,
        ReadOnlySpan<int> scoresShape,
        int batchIndex)
    {
        var numQueries = scoresShape[1];
        var numClasses = scoresShape[2];
        var entries = new DetectionCandidate[numQueries];
        var batchOffset = batchIndex * numQueries * numClasses;

        float[]? rented = null;
        Span<float> probabilities = numClasses <= 256
            ? stackalloc float[numClasses]
            : (rented = ArrayPool<float>.Shared.Rent(numClasses)).AsSpan(0, numClasses);

        try
        {
            for (var query = 0; query < numQueries; query++)
            {
                var rowOffset = batchOffset + query * numClasses;
                var row = scores.Slice(rowOffset, numClasses);
                ComputeSoftmax(row, probabilities);

                var bestClass = 0;
                var bestScore = probabilities[0];
                // In non-focal mode the last class is the "no-object" class.
                var limit = numClasses - 1;
                for (var cls = 1; cls < limit; cls++)
                {
                    var score = probabilities[cls];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = cls;
                    }
                }

                entries[query] = new DetectionCandidate(
                    bestScore,
                    query,
                    bestClass,
                    query * limit + bestClass);
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        return entries;
    }

    private string ResolveLabel(int classIndex)
    {
        if (classIndex >= 0 && classIndex < _options.Labels.Count)
        {
            return _options.Labels[classIndex];
        }

        return $"Label-{classIndex}";
    }

    private static float Sigmoid(float value)
    {
        var neg = -value;
        return 1f / (1f + MathF.Exp(neg));
    }

    private static void ComputeSoftmax(ReadOnlySpan<float> logits, Span<float> destination)
    {
        var max = float.NegativeInfinity;
        for (var i = 0; i < destination.Length; i++)
        {
            var value = logits[i];
            destination[i] = value;
            if (value > max)
            {
                max = value;
            }
        }

        var sum = 0f;
        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] = MathF.Exp(destination[i] - max);
            sum += destination[i];
        }

        if (sum == 0f)
        {
            return;
        }

        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] /= sum;
        }
    }

    private readonly struct DetectionCandidate
    {
        public DetectionCandidate(float score, int queryIndex, int classIndex, int flattenedIndex)
        {
            Score = score;
            QueryIndex = queryIndex;
            ClassIndex = classIndex;
            FlattenedIndex = flattenedIndex;
        }

        public float Score { get; }

        public int QueryIndex { get; }

        public int ClassIndex { get; }

        public int FlattenedIndex { get; }
    }
}
