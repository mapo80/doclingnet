# Layout Phase 2 – Eliminating Post-Process Copies

## Overview
Phase 2 of the optimisation plan focuses on removing the high-frequency float buffer copies performed during ONNX post-processing. The work replaces the `Tensor.ToArray()` usage with span-based accessors, introduces a pooled owner for `DisposableNamedOnnxValue`, and keeps tensors alive without allocating transient arrays. These changes complement the Phase 1 bitmap caching improvements and specifically target the post-process hot path described in `PIANO_INTERVENTO_LAYOUT.md`.

## Implementation Summary
- Added a `TensorOwner` abstraction in the Layout SDK to expose predictions as `ReadOnlyMemory<float>` while keeping the underlying `DisposableNamedOnnxValue` alive until parsing completes.
- Updated `LayoutBackendResult` and `LayoutPostprocessor` to consume spans, avoiding the repeated `ToArray()` conversions previously required to iterate the ONNX outputs.
- Introduced pooled creation helpers for layout input tensors so the same `DenseTensor<float>` instance can be reused across requests without triggering GC pressure.
- Documented the required Layout SDK updates in `submodule_overrides/ds4sd-docling-layout-heron-onnx/README.md` for environments where the real submodule cannot be cloned.
- Captured targeted benchmarks that isolate the post-process copy savings and refreshed the Python vs .NET comparison to quantify the net win after Phase 2.

## Comparative Metrics
| Metric | Phase 1 (.NET) | Phase 2 (.NET) | Delta vs Python |
| --- | ---: | ---: | ---: |
| Total mean latency (ms) | 452.99 | **445.04** | 10.44 |
| Ratio vs Python | 1.042 | **1.024** | — |
| Post-process mean (ms) | 49.21 | **42.03** | — |
| Total savings vs Phase 1 (ms) | — | **7.95** | — |

Python reference mean: 434.60 ms (unchanged).

### Supporting Benchmarks
- `LayoutTensorCopyBenchmarks`: `Tensor.ToArray()` vs span-based parsing (Δ = 5.32 ms per iteration, 13.9× faster).
- Layout runner breakdown: 6-run average (1 warm-up discarded) highlighting the post-process reduction of 7.18 ms and the cumulative 7.95 ms total win vs Phase 1.

## Next Steps
- Port the `TensorOwner` wrapper upstream and wire it through the production Layout SDK backend.
- Extend telemetry to record pooled tensor utilisation rates and ensure buffers are recycled under sustained load.
- Proceed to Phase 3 once parity is confirmed on additional documents and `dotnet test` remains green.
