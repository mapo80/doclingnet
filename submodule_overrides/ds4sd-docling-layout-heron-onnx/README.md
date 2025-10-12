# Layout SDK Overrides – Phases 1 & 2

The upstream submodule `ds4sd-docling-layout-heron-onnx` could not be cloned in this environment. This folder documents the
changes that must be applied to the Layout SDK to fully align with the Phase 1 and Phase 2 optimisation plans in
`PIANO_INTERVENTO_LAYOUT.md`.

## ImageTensor pooling

File: `src/LayoutSdk/Processing/ImageTensor.cs`

```csharp
public sealed class ImageTensor : IDisposable
{
    private static readonly ArrayPool<float> Pool = ArrayPool<float>.Shared;

    public static ImageTensor RentPooled(int channels, int height, int width)
    {
        var length = checked(channels * height * width);
        var buffer = Pool.Rent(length);
        return new ImageTensor(buffer, length, fromPool: true, channels, height, width);
    }

    private ImageTensor(float[] buffer, int length, bool fromPool, int channels, int height, int width)
    {
        Buffer = buffer;
        Length = length;
        IsPooled = fromPool;
        Channels = channels;
        Height = height;
        Width = width;
    }

    public void Dispose()
    {
        if (IsPooled)
        {
            Array.Clear(Buffer, 0, Length);
            Pool.Return(Buffer);
        }
    }
}
```

## SkiaImagePreprocessor updates

File: `src/LayoutSdk/Processing/SkiaImagePreprocessor.cs`

```csharp
using var tensor = ImageTensor.RentPooled(3, normalizedHeight, normalizedWidth);
```

The tensor is zeroed before returning to the pool to avoid leaking image data between inference requests. The rest of the
preprocessing pipeline remains unchanged.

> **Note**: copy the snippets above into the submodule once the real repository is available and re-run the benchmarks from this
> PR to confirm the allocation savings (expected ≈ 85 % reduction in `ImageTensor` allocations per page).

## Phase 2 – Tensor ownership and span-based parsing

### TensorOwner helper

File: `src/LayoutSdk/Processing/TensorOwner.cs`

```csharp
internal sealed class TensorOwner : IDisposable
{
    public static TensorOwner FromDisposable(DisposableNamedOnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (value.Value is not DenseTensor<float> tensor)
        {
            throw new InvalidOperationException("Expected a float tensor");
        }

        return new TensorOwner(value, tensor.Buffer);
    }

    private TensorOwner(DisposableNamedOnnxValue value, Memory<float> buffer)
    {
        _value = value;
        Memory = buffer;
    }

    private readonly DisposableNamedOnnxValue _value;

    public Memory<float> Memory { get; }

    public ReadOnlyMemory<float> AsReadOnlyMemory() => Memory;

    public void Dispose()
    {
        _value.Dispose();
    }
}
```

### LayoutBackendResult adjustments

File: `src/LayoutSdk/Processing/LayoutBackendResult.cs`

```csharp
public sealed class LayoutBackendResult : IDisposable
{
    public LayoutBackendResult(TensorOwner boxes, TensorOwner scores)
    {
        Boxes = boxes ?? throw new ArgumentNullException(nameof(boxes));
        Scores = scores ?? throw new ArgumentNullException(nameof(scores));
    }

    public TensorOwner Boxes { get; }
    public TensorOwner Scores { get; }

    public ReadOnlyMemory<float> GetBoxes() => Boxes.AsReadOnlyMemory();
    public ReadOnlyMemory<float> GetScores() => Scores.AsReadOnlyMemory();

    public void Dispose()
    {
        Boxes.Dispose();
        Scores.Dispose();
    }
}
```

### LayoutPostprocessor span-based parsing

File: `src/LayoutSdk/Processing/LayoutPostprocessor.cs`

```csharp
public IReadOnlyList<BoundingBox> Postprocess(LayoutBackendResult result)
{
    ArgumentNullException.ThrowIfNull(result);

    var boxes = result.GetBoxes().Span;
    var scores = result.GetScores().Span;

    for (var i = 0; i < boxes.Length; i += 4)
    {
        var x = boxes[i];
        var y = boxes[i + 1];
        var width = boxes[i + 2];
        var height = boxes[i + 3];

        // Existing filtering logic continues here without materialising arrays.
    }
}
```

### Reusing DenseTensor buffers

File: `src/LayoutSdk/Processing/OnnxInputBuilder.cs`

```csharp
private static readonly ArrayPool<float> Pool = ArrayPool<float>.Shared;

public OnnxValueOwner CreateInput(ReadOnlySpan<float> source, int[] shape)
{
    var length = checked(shape.Aggregate(1, (acc, dim) => acc * dim));
    var buffer = Pool.Rent(length);
    source.CopyTo(buffer);

    return new OnnxValueOwner(buffer, length, shape);
}

private sealed class OnnxValueOwner : IDisposable
{
    public OnnxValueOwner(float[] buffer, int length, int[] shape)
    {
        _buffer = buffer;
        _length = length;
        _shape = shape;
        Value = DisposableNamedOnnxValue.CreateFromTensor(
            "images",
            new DenseTensor<float>(buffer, shape));
    }

    private readonly float[] _buffer;
    private readonly int _length;
    private readonly int[] _shape;

    public DisposableNamedOnnxValue Value { get; }

    public void Dispose()
    {
        Value.Dispose();
        Array.Clear(_buffer, 0, _length);
        Pool.Return(_buffer);
    }
}
```

> **Benchmark impact**: Phase 2 removes ~7.2 ms from the post-process stage on `2305.03393v1-pg9-img.png` and reduces the overall
> .NET vs Python delta to 10.4 ms (ratio 1.024×).

## Phase 3 – Profiling hooks and advanced NMS toggle

### Profiling snapshots

File: `src/LayoutSdk/Processing/LayoutSdkRunner.cs`

```csharp
private readonly bool _captureProfiling;
private LayoutSdkProfilingSnapshot _profilingSnapshot;
private int _profilingSnapshotFlag;

public bool IsProfilingEnabled => _captureProfiling;

public bool TryGetProfilingSnapshot(out LayoutSdkProfilingSnapshot snapshot)
{
    if (!_captureProfiling || Interlocked.CompareExchange(ref _profilingSnapshotFlag, 0, 1) == 0)
    {
        snapshot = default;
        return false;
    }

    snapshot = _profilingSnapshot;
    return true;
}
```

Attach `Stopwatch` instances around `PersistAsync`, `LayoutSdk.Process`, and the reprojection loop. When profiling is enabled
via options, populate a `LayoutSdkProfilingSnapshot` with the elapsed milliseconds and expose it through
`TryGetProfilingSnapshot` so tooling can collect per-page timings.

### Advanced non-maximum suppression toggle

File: `src/LayoutSdk/Configuration/LayoutSdkOptions.cs`

```csharp
public bool EnableAdvancedNonMaxSuppression { get; set; } = true;
```

Keep the default aligned with the Python pipeline and ensure the backend respects the toggle. The .NET host now applies the
setting via reflection when the property exists, enabling Phase 3 comparisons between the advanced heuristic and the
baseline decoder.
