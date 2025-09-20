using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace Docling.Pipelines.Abstractions;

/// <summary>
/// Shared state passed across pipeline stages.
/// </summary>
public sealed class PipelineContext
{
    private readonly ConcurrentDictionary<string, object> _state = new(StringComparer.OrdinalIgnoreCase);

    public PipelineContext(IServiceProvider services)
    {
        Services = services ?? throw new ArgumentNullException(nameof(services));
    }

    public IServiceProvider Services { get; }

    public void Set<T>(string key, T value)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        _state[key] = value!;
    }

    public T GetRequired<T>(string key)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);

        if (TryGet<T>(key, out var value))
        {
            return value;
        }

        throw new KeyNotFoundException($"Pipeline context is missing required entry '{key}'.");
    }

    public bool TryGet<T>(string key, out T value)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);

        if (_state.TryGetValue(key, out var stored) && stored is T typed)
        {
            value = typed;
            return true;
        }

        value = default!;
        return false;
    }
}
