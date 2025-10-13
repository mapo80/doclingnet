using System;
using System.Collections.Concurrent;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Backends;

internal sealed class BackendRegistry : IDisposable
{
    private readonly ConcurrentDictionary<BackendKey, ITableFormerBackend> _backends = new();
    private readonly ITableFormerBackendFactory _backendFactory;

    public BackendRegistry(ITableFormerBackendFactory backendFactory)
    {
        _backendFactory = backendFactory ?? throw new ArgumentNullException(nameof(backendFactory));
    }

    public ITableFormerBackend GetOrCreateBackend(TableFormerRuntime runtime, TableFormerModelVariant variant)
    {
        return _backends.GetOrAdd(new BackendKey(runtime, variant), key => _backendFactory.CreateBackend(key.Runtime, key.Variant));
    }

    public void RegisterBackend(TableFormerRuntime runtime, TableFormerModelVariant variant, ITableFormerBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);

        if (runtime == TableFormerRuntime.Auto)
        {
            throw new ArgumentException("Cannot register a backend for the Auto runtime", nameof(runtime));
        }

        _backends[new BackendKey(runtime, variant)] = backend;
    }

    public void Dispose()
    {
        foreach (var backend in _backends.Values)
        {
            if (backend is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }

        _backends.Clear();
    }
}
