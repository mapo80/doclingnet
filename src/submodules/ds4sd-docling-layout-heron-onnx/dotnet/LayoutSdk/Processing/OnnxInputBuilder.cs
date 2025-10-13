using System;
using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LayoutSdk.Processing;

internal sealed class OnnxInputBuilder
{
    private static readonly ArrayPool<float> Pool = ArrayPool<float>.Shared;
    private readonly string _inputName;

    public OnnxInputBuilder(string inputName)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must be provided.", nameof(inputName));
        }

        _inputName = inputName;
    }

    public OnnxValueOwner CreateInput(ReadOnlySpan<float> source, int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);

        if (shape.Length == 0)
        {
            throw new ArgumentException("Input shape must consist of positive dimensions.", nameof(shape));
        }

        var length = 1;
        for (var i = 0; i < shape.Length; i++)
        {
            var dimension = shape[i];
            if (dimension <= 0)
            {
                throw new ArgumentException("Input shape must consist of positive dimensions.", nameof(shape));
            }

            length = checked(length * dimension);
        }

        if (source.Length != length)
        {
            throw new ArgumentException("Source span length does not match provided shape.", nameof(source));
        }

        var buffer = Pool.Rent(length);
        source.CopyTo(buffer.AsSpan(0, length));

        return new OnnxValueOwner(_inputName, buffer, length, shape);
    }

    internal sealed class OnnxValueOwner : IDisposable
    {
        private readonly float[] _buffer;
        private readonly int _length;
        private bool _disposed;

        public OnnxValueOwner(string inputName, float[] buffer, int length, int[] shape)
        {
            _buffer = buffer;
            _length = length;
            Shape = (int[])shape.Clone();
            Value = NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<float>(_buffer.AsMemory(0, length), Shape));
        }

        public int[] Shape { get; }

        public NamedOnnxValue Value { get; }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            Array.Clear(_buffer, 0, _length);
            Pool.Return(_buffer);
            _disposed = true;
        }
    }
}
