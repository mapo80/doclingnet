using System;
using System.Threading;
using System.Threading.Tasks;
using Docling.Models.Ocr;
using Docling.Pipelines.Options;
using Microsoft.Extensions.Logging.Abstractions;

namespace Docling.Tests.Ocr;

public sealed class OcrServiceFactoryTests
{
    [Fact]
    public void CreateThrowsWhenOptionsNull()
    {
        var factory = new OcrServiceFactory();
        Assert.Throws<ArgumentNullException>(() => factory.Create(null!));
    }

    [Fact]
    public void CreateThrowsForUnsupportedEngine()
    {
        var factory = new OcrServiceFactory();
        Assert.Throws<NotSupportedException>(() => factory.Create(new RapidOcrOptions()));
    }

    [Fact]
    public void CreateUsesInjectedEasyOcrBuilder()
    {
        var options = new EasyOcrOptions { Languages = new[] { "en" } };
        using var expected = new FakeService();
        var factory = new OcrServiceFactory(NullLoggerFactory.Instance, _ => expected);

        using var service = factory.Create(options);

        Assert.Same(expected, service);
    }

    private sealed class FakeService : IOcrService
    {
        public void Dispose()
        {
        }

        public IAsyncEnumerable<OcrLine> RecognizeAsync(OcrRequest request, CancellationToken cancellationToken = default)
        {
            return GetEmpty();
        }

        private static async IAsyncEnumerable<OcrLine> GetEmpty()
        {
            await Task.CompletedTask;
            yield break;
        }
    }
}
