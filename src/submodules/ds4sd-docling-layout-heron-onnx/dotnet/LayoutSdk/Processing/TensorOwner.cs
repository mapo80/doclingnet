using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LayoutSdk.Processing;

internal sealed class TensorOwner : IDisposable
{
    private readonly NamedOnnxValue _value;
    private readonly IDisposable? _disposable;
    private bool _disposed;

    private TensorOwner(NamedOnnxValue value, Memory<float> buffer)
    {
        _value = value;
        _disposable = value as IDisposable;
        Memory = buffer;
    }

    public static TensorOwner FromNamedValue(NamedOnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value.Value is not DenseTensor<float> tensor)
        {
            throw new InvalidOperationException("Expected a float tensor.");
        }

        return new TensorOwner(value, tensor.Buffer);
    }

    public Memory<float> Memory { get; }

    public ReadOnlyMemory<float> AsReadOnlyMemory() => Memory;

    public ReadOnlySpan<float> AsReadOnlySpan() => Memory.Span;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposable?.Dispose();
        _disposed = true;
    }
}
