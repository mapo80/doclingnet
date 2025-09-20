using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Docling.Backends;

internal static class StreamUtilities
{
    public static async Task<byte[]> ReadAllBytesAsync(Stream source, CancellationToken cancellationToken)
    {
        if (source is MemoryStream memoryStream && memoryStream.TryGetBuffer(out var buffer))
        {
            return buffer.Array is null
                ? memoryStream.ToArray()
                : buffer.Array.AsSpan(buffer.Offset, buffer.Count).ToArray();
        }

        if (source.CanSeek)
        {
            source.Seek(0, SeekOrigin.Begin);
        }

        using var pooledStream = new MemoryStream();
        await source.CopyToAsync(pooledStream, cancellationToken).ConfigureAwait(false);
        return pooledStream.ToArray();
    }

    public static MemoryStream AsMemoryStream(byte[] buffer)
    {
        return new MemoryStream(buffer, writable: false);
    }
}
