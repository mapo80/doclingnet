using SkiaSharp;
using System.Collections.Generic;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

public interface ITableFormerBackend
{
    IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath);
}
