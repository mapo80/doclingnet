using System;
using Docling.Pipelines.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Docling.Models.Ocr;

public interface IOcrServiceFactory
{
    IOcrService Create(OcrOptions options);
}

public sealed class OcrServiceFactory : IOcrServiceFactory
{
    private readonly ILoggerFactory _loggerFactory;
    private readonly Func<EasyOcrOptions, IOcrService> _easyOcrFactory;

    public OcrServiceFactory()
        : this(NullLoggerFactory.Instance, null)
    {
    }

    public OcrServiceFactory(ILoggerFactory loggerFactory)
        : this(loggerFactory, null)
    {
    }

    internal OcrServiceFactory(ILoggerFactory loggerFactory, Func<EasyOcrOptions, IOcrService>? easyOcrFactory)
    {
        _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        _easyOcrFactory = easyOcrFactory ?? (options => new EasyOcrService(options, _loggerFactory.CreateLogger<EasyOcrService>()));
    }

    public IOcrService Create(OcrOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        return options.Engine switch
        {
            OcrEngine.EasyOcr when options is EasyOcrOptions easy => _easyOcrFactory(easy),
            _ => throw new NotSupportedException($"OCR engine '{options.Engine}' is not supported."),
        };
    }
}
