using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Pdf;
using Docling.Core.Primitives;
using SkiaSharp;

namespace Docling.Tests.Infrastructure;

internal sealed class FakePdfRenderer : IPdfPageRenderer, IAsyncDisposable, IDisposable
{
    private readonly List<PageImage> _images;
    private bool _disposed;

    public FakePdfRenderer(int pageCount)
    {
        _images = new List<PageImage>(pageCount);
        try
        {
            for (var i = 0; i < pageCount; i++)
            {
                SKBitmap? bitmap = null;
                try
                {
                    bitmap = new SKBitmap(2, 2);
                    var metadata = new PageImageMetadata("doc", $"page-{i}", "application/pdf", new Dictionary<string, string>
                    {
                        ["seed"] = i.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    });
                    _images.Add(new PageImage(new PageReference(i, 200), bitmap, metadata));
                    bitmap = null;
                }
                finally
                {
                    bitmap?.Dispose();
                }
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public Task<int> GetPageCountAsync(Stream pdfStream, CancellationToken cancellationToken) => Task.FromResult(_images.Count);

    public async IAsyncEnumerable<PageImage> RenderAsync(Stream pdfStream, IReadOnlyCollection<int>? pages, PdfRenderSettings settings, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var pageIndices = pages ?? CreateDefaultPageOrder();
        foreach (var index in pageIndices)
        {
            yield return _images[index].Clone();
            await Task.Yield();
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

        foreach (var image in _images)
        {
            image.Dispose();
        }

        _images.Clear();
        _disposed = true;
    }

    private int[] CreateDefaultPageOrder()
    {
        var indices = new int[_images.Count];
        for (var i = 0; i < _images.Count; i++)
        {
            indices[i] = i;
        }

        return indices;
    }
}
