using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Image;
using Docling.Backends.Pdf;
using Docling.Core.Primitives;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using Xunit;

namespace Docling.Tests;

public sealed class ImageBackendTests
{
    [Fact]
    public async Task LoadAsyncYieldsConfiguredImages()
    {
        var descriptors = new List<ImageSourceDescriptor>
        {
            CreateDescriptor(0, 10, 20, "first.png", "image/png"),
            CreateDescriptor(1, 12, 24, "second.png", "image/png", dpi: 200,
                metadata: new Dictionary<string, string> { ["colorSpace"] = "rgb" }),
        };

        var options = new ImageBackendOptions
        {
            Sources = descriptors,
            DefaultDpi = 150,
            DocumentId = "doc-2",
            SourceName = "images",
            Metadata = new Dictionary<string, string> { ["pipeline"] = "unit-test" },
        };

        var backend = new ImageBackend(options, NullLogger<ImageBackend>.Instance);
        var images = new List<PageImage>();

        await foreach (var image in backend.LoadAsync(CancellationToken.None))
        {
            images.Add(image);
        }

        images.Should().HaveCount(2);
        images[0].Page.Should().Be(new PageReference(0, 150));
        images[0].Metadata.SourceId.Should().Be("doc-2");
        images[0].Metadata.Properties.Should().ContainKey("pipeline");
        images[0].Metadata.Properties["fileName"].Should().Be("first.png");
        images[1].Page.Dpi.Should().Be(200);
        images[1].Metadata.Properties.Should().Contain(new KeyValuePair<string, string>("colorSpace", "rgb"));

        foreach (var image in images)
        {
            image.Dispose();
        }
    }

    [Fact]
    public async Task LoadAsyncDisposesStreams()
    {
        await using var stream = new TrackingStream(CreateImageStream(8, 8));
        var descriptor = new ImageSourceDescriptor
        {
            Identifier = "tracked",
            FileName = "tracked.png",
            MediaType = "image/png",
            StreamFactory = _ => Task.FromResult<Stream>(stream),
        };

        var backend = new ImageBackend(new ImageBackendOptions { Sources = new[] { descriptor } }, NullLogger<ImageBackend>.Instance);

        await foreach (var image in backend.LoadAsync(CancellationToken.None))
        {
            image.Dispose();
        }

        stream.DisposeAsyncCalled.Should().BeTrue();
    }

    [Fact]
    public async Task LoadAsyncThrowsWhenBitmapCannotBeDecoded()
    {
        var descriptor = new ImageSourceDescriptor
        {
            Identifier = "broken",
            StreamFactory = _ => Task.FromResult<Stream>(new MemoryStream(new byte[] { 1, 2, 3 })),
        };

        var backend = new ImageBackend(new ImageBackendOptions { Sources = new[] { descriptor } }, NullLogger<ImageBackend>.Instance);

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in backend.LoadAsync(CancellationToken.None))
            {
            }
        });
    }

    private static ImageSourceDescriptor CreateDescriptor(int index, int width, int height, string fileName, string mediaType, int? dpi = null, IReadOnlyDictionary<string, string>? metadata = null)
    {
        return new ImageSourceDescriptor
        {
            Identifier = string.Create(CultureInfo.InvariantCulture, $"img-{index}"),
            FileName = fileName,
            MediaType = mediaType,
            Dpi = dpi,
            Metadata = metadata,
            StreamFactory = _ => Task.FromResult<Stream>(CreateImageStream(width, height)),
        };
    }

    private static MemoryStream CreateImageStream(int width, int height)
    {
        using var bitmap = new SKBitmap(width, height);
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        return new MemoryStream(data.ToArray());
    }

    private sealed class TrackingStream : Stream, IAsyncDisposable
    {
        private readonly MemoryStream _inner;

        public TrackingStream(MemoryStream inner)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        }

        public bool DisposeAsyncCalled { get; private set; }

        public override bool CanRead => _inner.CanRead;

        public override bool CanSeek => _inner.CanSeek;

        public override bool CanWrite => _inner.CanWrite;

        public override long Length => _inner.Length;

        public override long Position
        {
            get => _inner.Position;
            set => _inner.Position = value;
        }

        public override void Flush() => _inner.Flush();

        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);

        public override long Seek(long offset, SeekOrigin origin) => _inner.Seek(offset, origin);

        public override void SetLength(long value) => _inner.SetLength(value);

        public override void Write(byte[] buffer, int offset, int count) => _inner.Write(buffer, offset, count);

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _inner.Dispose();
            }
            base.Dispose(disposing);
        }

        public override async ValueTask DisposeAsync()
        {
            DisposeAsyncCalled = true;
            await _inner.DisposeAsync().ConfigureAwait(false);
            await base.DisposeAsync().ConfigureAwait(false);
        }
    }
}
