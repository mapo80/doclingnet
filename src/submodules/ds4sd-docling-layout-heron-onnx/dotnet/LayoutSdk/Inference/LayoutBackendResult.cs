using System;
using LayoutSdk.Processing;

namespace LayoutSdk.Inference;

public sealed class LayoutBackendResult : IDisposable
{
    private readonly int[] _scoresShape;
    private readonly int[] _boxesShape;
    private bool _disposed;

    internal LayoutBackendResult(
        TensorOwner boxes,
        int[] boxesShape,
        TensorOwner scores,
        int[] scoresShape)
    {
        Boxes = boxes ?? throw new ArgumentNullException(nameof(boxes));
        Scores = scores ?? throw new ArgumentNullException(nameof(scores));
        _boxesShape = boxesShape is { Length: > 0 } ? (int[])boxesShape.Clone()
            : throw new ArgumentException("Boxes shape must be provided.", nameof(boxesShape));
        _scoresShape = scoresShape is { Length: > 0 } ? (int[])scoresShape.Clone()
            : throw new ArgumentException("Scores shape must be provided.", nameof(scoresShape));

        if (_scoresShape.Length != 3)
        {
            throw new ArgumentException("Scores tensor must have rank 3 [batch, queries, classes].", nameof(scoresShape));
        }

        if (_boxesShape.Length != 3 || _boxesShape[^1] != 4)
        {
            throw new ArgumentException("Boxes tensor must have rank 3 [batch, queries, 4].", nameof(boxesShape));
        }

        if (_scoresShape[0] != _boxesShape[0] || _scoresShape[1] != _boxesShape[1])
        {
            throw new ArgumentException("Scores and boxes tensors must share batch and query dimensions.");
        }
    }

    internal TensorOwner Boxes { get; }

    internal TensorOwner Scores { get; }

    public ReadOnlyMemory<int> BoxesShape => _boxesShape;

    public ReadOnlyMemory<int> ScoresShape => _scoresShape;

    public ReadOnlyMemory<float> GetBoxes() => Boxes.AsReadOnlyMemory();

    public ReadOnlyMemory<float> GetScores() => Scores.AsReadOnlyMemory();

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Scores.Dispose();
        Boxes.Dispose();
        _disposed = true;
    }
}
