using System;
using System.Buffers;

namespace LayoutSdk.Processing;

public sealed class ImageTensor : IDisposable
{
    private static readonly ArrayPool<float> Pool = ArrayPool<float>.Shared;
    private readonly float[] _buffer;
    private readonly int _length;
    private readonly bool _isPooled;
    private bool _disposed;

    private ImageTensor(float[] buffer, int length, bool fromPool, int width, int height, int channels)
    {
        _buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        _length = length;
        _isPooled = fromPool;
        Width = width;
        Height = height;
        Channels = channels;
        Length = length;
    }

    public int Width { get; }

    public int Height { get; }

    public int Channels { get; }

    public int Length { get; }

    public static ImageTensor Rent(int width, int height, int channels) =>
        RentPooled(channels, height, width);

    public static ImageTensor RentPooled(int channels, int height, int width)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (channels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(channels));
        }

        var length = checked(channels * height * width);
        var buffer = Pool.Rent(length);
        return new ImageTensor(buffer, length, fromPool: true, width, height, channels);
    }

    public Span<float> AsSpan() => _buffer.AsSpan(0, Length);

    public float[] Buffer => _buffer;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        if (_isPooled)
        {
            Array.Clear(_buffer, 0, _length);
            Pool.Return(_buffer);
        }

        _disposed = true;
    }
}
