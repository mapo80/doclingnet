using System;
using System.Text.Json;

namespace Docling.Models.Layout;

/// <summary>
/// Configures how the HTTP layout detection adapter communicates with the Python service.
/// </summary>
public sealed class LayoutServiceOptions
{
    private JsonSerializerOptions? _serializerOptions;

    public Uri Endpoint
    {
        get => _endpoint ?? throw new InvalidOperationException("Endpoint has not been configured.");
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            _endpoint = value;
        }
    }

    public TimeSpan RequestTimeout
    {
        get => _requestTimeout;
        init
        {
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(value, TimeSpan.Zero);
            _requestTimeout = value;
        }
    }

    public JsonSerializerOptions SerializerOptions
    {
        get => _serializerOptions ??= CreateDefaultSerializerOptions();
        init => _serializerOptions = value is null ? null : new JsonSerializerOptions(value);
    }

    private Uri? _endpoint;
    private TimeSpan _requestTimeout = TimeSpan.FromSeconds(60);

    public LayoutServiceOptions Clone() => new()
    {
        Endpoint = Endpoint,
        RequestTimeout = RequestTimeout,
        SerializerOptions = new JsonSerializerOptions(SerializerOptions),
    };

    public void EnsureValid()
    {
        if (_endpoint is null)
        {
            throw new InvalidOperationException("The layout service endpoint must be configured.");
        }

        if (!_endpoint.IsAbsoluteUri)
        {
            throw new InvalidOperationException("The layout service endpoint must be an absolute URI.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(_requestTimeout, TimeSpan.Zero);
    }

    private static JsonSerializerOptions CreateDefaultSerializerOptions() => new(JsonSerializerDefaults.Web)
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = false,
    };
}

/// <summary>
/// Represents an error reported by the layout detection service.
/// </summary>
public sealed class LayoutServiceException : Exception
{
    public LayoutServiceException()
    {
    }

    public LayoutServiceException(string message)
        : base(message)
    {
    }

    public LayoutServiceException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
