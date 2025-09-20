using System;
using System.Collections.Generic;
using Docling.Backends.Pdf;
using Docling.Core.Primitives;

namespace Docling.Backends.Storage;

/// <summary>
/// Provides an in-memory cache to reuse <see cref="PageImage"/> instances across pipeline stages.
/// </summary>
public sealed class PageImageStore : IDisposable, IAsyncDisposable
{
    private readonly Dictionary<int, PageImage> _images = new();
    private readonly object _gate = new();
    private bool _disposed;

    public void Add(PageImage image, bool overwrite = false)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(image);

        PageImage clone;
        try
        {
            clone = image.Clone();
        }
        finally
        {
            image.Dispose();
        }

        lock (_gate)
        {
            if (_images.TryGetValue(clone.Page.PageNumber, out var existing))
            {
                if (!overwrite)
                {
                    clone.Dispose();
                    throw new InvalidOperationException($"A page image for index {clone.Page.PageNumber} is already present.");
                }

                _images[clone.Page.PageNumber] = clone;
                existing.Dispose();
                return;
            }

            _images.Add(clone.Page.PageNumber, clone);
        }
    }

    public bool TryRent(PageReference page, out PageImage image)
    {
        ThrowIfDisposed();

        lock (_gate)
        {
            if (_images.TryGetValue(page.PageNumber, out var cached))
            {
                image = cached.Clone();
                return true;
            }
        }

        image = null!;
        return false;
    }

    public PageImage Rent(PageReference page)
    {
        if (TryRent(page, out var image))
        {
            return image;
        }

        throw new KeyNotFoundException($"No cached page image for index {page.PageNumber}.");
    }

    public bool Contains(PageReference page)
    {
        ThrowIfDisposed();

        lock (_gate)
        {
            return _images.ContainsKey(page.PageNumber);
        }
    }

    public void Clear()
    {
        ThrowIfDisposed();

        PageImage[] images;
        lock (_gate)
        {
            images = new PageImage[_images.Count];
            var i = 0;
            foreach (var entry in _images)
            {
                images[i++] = entry.Value;
            }

            _images.Clear();
        }

        foreach (var image in images)
        {
            image.Dispose();
        }
    }

    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Clear();
        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
