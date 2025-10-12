# Layout Phase 3 – Fine-Grained Profiling and Final Validation

## Overview
Phase 3 closes the optimisation plan described in `PIANO_INTERVENTO_LAYOUT.md` by focusing on micro profiling and
configurability. The work introduces a first-party performance runner that captures pooled profiling telemetry from the
Layout SDK, exposes a toggle for the advanced non-maximum suppression (NMS) heuristics, and aggregates Python vs .NET
measurements for three representative pages. Combined with the new dotnet-trace guidance, the layout pipeline now
reaches the ≤ 10 ms delta target with deterministic instrumentation support.

## Implementation Summary
- Extended `LayoutSdkDetectionOptions` and `LayoutSdkRunner` with a profiling-aware execution path that records the
  persistence, inference, and post-process timings for each page without impacting production defaults.
- Added an advanced NMS toggle to the CLI parsing layer (`ConvertCommandOptions`) and ensured the Layout SDK runner applies
  the setting even when the upstream property is not present, enabling A/B comparisons from configuration alone.
- Introduced the `LayoutPerfRunner` tool to execute repeated layout inferences with warm-up discards, JSON summaries, and
  optional overrides for the NMS flag.
- Captured updated benchmark artefacts (`python_timings.json`, `dotnet_runs.json`, `dotnet_trace_summary.json`) and a
  consolidated aggregation script (`layout_phase3_metrics.py`) that reports the comparative metrics below.

## Comparative Metrics (3 Images)
| Sample | Python Total (ms) | .NET Total (ms) | Δ (.NET - Python) | Ratio (.NET / Python) | Advanced NMS Gain (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2305.03393v1-pg9-img.png` | 434.60 | **440.60** | 6.00 | 1.014 | 7.20 |
| `amt_handbook_page_0001.png` | 486.30 | **494.12** | 7.82 | 1.016 | 5.57 |
| `amt_handbook_page_0002.png` | 479.54 | **487.88** | 8.34 | 1.017 | 5.22 |
| **Mean** | **466.81** | **474.20** | **7.39** | **1.016** | **6.00** |

Python reference metrics originate from six-run samples (one warm-up) of the existing `infer_onnx.py` script, while the
.NET timings come from the new `LayoutPerfRunner` JSON outputs. The advanced NMS gain column compares `.NET` runs with
and without the heuristic enabled, validating the configurable toggle.

## Profiling Highlights
- `LayoutSdkRunner` now publishes per-page snapshots that `LayoutPerfRunner` and the pipeline can consume for regression
  tracking. The mean Phase 3 post-process cost drops to 46.55–37.44 ms depending on the page, and the persistence stage
  accounts for ≤ 14 ms thanks to the pooled image tensor work from previous phases.【F:results/layout_phase3/dotnet_runs.json†L1-L189】
- A representative `dotnet-trace` session shows that the remaining hotspots reside in `LayoutSdk.Postprocess` and the
  span-based parsing loop, with no residual `TensorOwner` churn. Developers can cross-check future regressions by
  replaying the recorded command.【F:results/layout_phase3/dotnet_trace_summary.json†L1-L18】

## Next Steps
- Integrate the profiling snapshots with the existing telemetry pipeline so production captures the same breakdown under load.
- Promote `LayoutPerfRunner` to CI smoke tests to detect regressions across the sampled pages before shipping.
- Package the advanced NMS toggle inside the layout service configuration so downstream orchestrators can flip the
  heuristic without rebuilding the tooling.
