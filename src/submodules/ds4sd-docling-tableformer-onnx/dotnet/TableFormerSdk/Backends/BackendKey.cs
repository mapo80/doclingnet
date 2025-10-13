using System;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Backends;

internal readonly record struct BackendKey(TableFormerRuntime Runtime, TableFormerModelVariant Variant)
{
    public override string ToString() => $"{Runtime}:{Variant}";
}
