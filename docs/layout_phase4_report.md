# Layout Phase 4 – Consolidated Benchmark Summary

## Overview
Phase 4 concludes the optimisation plan from `PIANO_INTERVENTO_LAYOUT.md` by consolidating the benchmark
artefacts gathered across the previous milestones. The new aggregation script merges the canonical
single-page samples from Phases 1–2 with the expanded three-image dataset validated in Phase 3. The
resulting roll-up quantifies the cumulative latency savings, highlights the mean performance gap across
multiple documents, and provides a single JSON payload suitable for downstream dashboards.

## Canonical Page Timeline (`2305.03393v1-pg9-img.png`)
| Stage | Python Total (ms) | .NET Total (ms) | Δ (.NET - Python) | Ratio (.NET / Python) | Δ Improvement vs Previous (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (pre Phase 1) | 434.60 | 465.10 | 30.50 | 1.070 | – |
| Phase 1 | 434.60 | 452.79 | 18.19 | 1.042 | 12.31 |
| Phase 2 | 434.60 | 445.04 | 10.44 | 1.024 | 7.75 |
| Phase 3 | 434.60 | **440.60** | **6.00** | **1.014** | **4.44** |

The consolidated view confirms a cumulative 24.5 ms reduction versus the initial .NET baseline while
the Python reference remains unchanged, keeping accuracy parity intact.【F:results/layout_phase4/phase4_metrics.json†L1-L29】

## Multi-Image Averages (Phase 3 Sample Set)
- Mean Python latency: 466.81 ms
- Mean .NET latency: 474.20 ms
- Mean delta: 7.39 ms (ratio 1.016×)
- Advanced NMS mean gain: 6.00 ms

These aggregates align with the profiling-focused sampling introduced in Phase 3 and remain available
for regression tracking through `results/layout_phase3/phase3_metrics.json`. The Phase 4 export simply
makes the data discoverable alongside the canonical timeline for dashboards and documentation.【F:results/layout_phase4/phase4_metrics.json†L29-L39】

## Deliverables
- `eng/tools/layout_phase4_metrics.py`: generates the consolidated roll-up and persists
  `results/layout_phase4/phase4_metrics.json` for reporting pipelines.
- Updated documentation (`docs/layout_phase4_report.md`) providing the comparative tables and
the cumulative savings narrative for stakeholders.
