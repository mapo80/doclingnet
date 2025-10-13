using TableFormerSdk.Enums;

namespace TableFormerSdk.Backends;

internal interface ITableFormerBackendFactory
{
    ITableFormerBackend CreateBackend(TableFormerRuntime runtime, TableFormerModelVariant variant);
}
