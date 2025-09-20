using System;
using Docling.Backends.Pdf;
using Docling.Backends.Storage;
using Docling.Core.Primitives;
using FluentAssertions;
using SkiaSharp;
using Xunit;

namespace Docling.Tests;

public sealed class PageImageStoreTests
{
    [Fact]
    public void AddStoresCloneAndDisposesInput()
    {
        using var store = new PageImageStore();
        using var bitmap = new SKBitmap(4, 4);
        using var image = new PageImage(new PageReference(0, 200), bitmap.Copy() ?? throw new InvalidOperationException());

        store.Add(image);

        store.Contains(new PageReference(0, 200)).Should().BeTrue();
    }

    [Fact]
    public void RentReturnsDistinctClone()
    {
        using var store = new PageImageStore();
        using var bitmap = new SKBitmap(4, 4);
        using var image = new PageImage(new PageReference(1, 200), bitmap.Copy() ?? throw new InvalidOperationException());

        store.Add(image);

        using var rented = store.Rent(new PageReference(1, 200));
        rented.Should().NotBeSameAs(image);
        rented.Metadata.Should().NotBeNull();
    }

    [Fact]
    public void AddWithOverwriteReplacesExisting()
    {
        using var store = new PageImageStore();
        using var originalBitmap = new SKBitmap(2, 2);
        using var original = new PageImage(new PageReference(2, 200), originalBitmap.Copy() ?? throw new InvalidOperationException());

        store.Add(original);

        using var replacementBitmap = new SKBitmap(8, 8);
        using var replacement = new PageImage(new PageReference(2, 200), replacementBitmap.Copy() ?? throw new InvalidOperationException());
        store.Add(replacement, overwrite: true);

        using var rented = store.Rent(new PageReference(2, 200));
        rented.Width.Should().Be(8);
    }

    [Fact]
    public void AddThrowsWhenDuplicateWithoutOverwrite()
    {
        using var store = new PageImageStore();
        using var bitmap = new SKBitmap(2, 2);
        using var first = new PageImage(new PageReference(3, 200), bitmap.Copy() ?? throw new InvalidOperationException());
        using var second = new PageImage(new PageReference(3, 200), bitmap.Copy() ?? throw new InvalidOperationException());

        store.Add(first);
        var action = () => store.Add(second);

        action.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void TryRentReturnsFalseWhenMissing()
    {
        using var store = new PageImageStore();
        store.TryRent(new PageReference(0, 200), out var rented).Should().BeFalse();
        rented.Should().BeNull();
        rented?.Dispose();
    }

    [Fact]
    public void DisposeClearsStore()
    {
        using var bitmap = new SKBitmap(4, 4);
        var store = new PageImageStore();
        try
        {
            using var image = new PageImage(new PageReference(0, 200), bitmap.Copy() ?? throw new InvalidOperationException());
            store.Add(image);
        }
        finally
        {
            store.Dispose();
        }

        Action action = () => store.Contains(new PageReference(0, 200));
        action.Should().Throw<ObjectDisposedException>();
    }
}
