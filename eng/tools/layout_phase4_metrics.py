#!/usr/bin/env python3
"""Aggregate cross-phase layout optimisation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_timeline_entry(
    label: str,
    python_mean_ms: float,
    dotnet_mean_ms: float,
    delta_ms: float,
    ratio: float,
    previous_delta_ms: Optional[float],
) -> Dict[str, Any]:
    improvement_ms: Optional[float] = None
    if previous_delta_ms is not None:
        improvement_ms = previous_delta_ms - delta_ms

    return {
        "label": label,
        "python_mean_ms": python_mean_ms,
        "dotnet_mean_ms": dotnet_mean_ms,
        "delta_ms": delta_ms,
        "ratio": ratio,
        "incremental_delta_reduction_ms": improvement_ms,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    phase1_metrics = load_json(repo_root / "results" / "layout_phase1" / "phase1_metrics.json")
    phase2_metrics = load_json(repo_root / "results" / "layout_phase2" / "phase2_metrics.json")
    phase3_metrics = load_json(repo_root / "results" / "layout_phase3" / "phase3_metrics.json")

    python_baseline = float(phase1_metrics["python_mean_ms"])
    dotnet_baseline = float(phase1_metrics["dotnet_mean_ms_before"])
    baseline_delta = float(phase1_metrics["delta_ms_before"])
    baseline_ratio = float(phase1_metrics["ratio_before"])

    phase1_dotnet = float(phase1_metrics["dotnet_mean_ms_after"])
    phase1_delta = float(phase1_metrics["delta_ms_after"])
    phase1_ratio = float(phase1_metrics["ratio_after"])

    phase2_dotnet = float(phase2_metrics["dotnet_phase2_mean_ms"])
    phase2_delta = float(phase2_metrics["delta_ms_phase2"])
    phase2_ratio = float(phase2_metrics["ratio_phase2"])

    canonical_id = "2305.03393v1-pg9-img.png"
    canonical_sample = next(
        (sample for sample in phase3_metrics.get("samples", []) if sample.get("id") == canonical_id),
        None,
    )
    if canonical_sample is None:
        raise SystemExit(f"Canonical sample {canonical_id!r} missing from Phase 3 metrics")

    phase3_python = float(canonical_sample["python_total_mean_ms"])
    phase3_dotnet = float(canonical_sample["dotnet_total_mean_ms"])
    phase3_delta = float(canonical_sample["delta_ms"])
    phase3_ratio = float(canonical_sample["ratio"])

    timeline: List[Dict[str, Any]] = []
    timeline.append(
        build_timeline_entry(
            label="Baseline (pre Phase 1)",
            python_mean_ms=python_baseline,
            dotnet_mean_ms=dotnet_baseline,
            delta_ms=baseline_delta,
            ratio=baseline_ratio,
            previous_delta_ms=None,
        )
    )
    timeline.append(
        build_timeline_entry(
            label="Phase 1",
            python_mean_ms=python_baseline,
            dotnet_mean_ms=phase1_dotnet,
            delta_ms=phase1_delta,
            ratio=phase1_ratio,
            previous_delta_ms=baseline_delta,
        )
    )
    timeline.append(
        build_timeline_entry(
            label="Phase 2",
            python_mean_ms=python_baseline,
            dotnet_mean_ms=phase2_dotnet,
            delta_ms=phase2_delta,
            ratio=phase2_ratio,
            previous_delta_ms=phase1_delta,
        )
    )
    timeline.append(
        build_timeline_entry(
            label="Phase 3",
            python_mean_ms=phase3_python,
            dotnet_mean_ms=phase3_dotnet,
            delta_ms=phase3_delta,
            ratio=phase3_ratio,
            previous_delta_ms=phase2_delta,
        )
    )

    total_delta_reduction = baseline_delta - phase3_delta

    payload = {
        "canonical_sample_id": canonical_id,
        "timeline": timeline,
        "total_delta_reduction_ms": total_delta_reduction,
        "phase3_multi_sample_mean": {
            "sample_count": phase3_metrics.get("sample_count", 0),
            "python_mean_ms": phase3_metrics.get("python_mean_ms", 0.0),
            "dotnet_mean_ms": phase3_metrics.get("dotnet_mean_ms", 0.0),
            "delta_mean_ms": phase3_metrics.get("delta_mean_ms", 0.0),
            "ratio_mean": phase3_metrics.get("ratio_mean", 0.0),
            "advanced_nms_mean_gain_ms": phase3_metrics.get("advanced_nms_mean_gain_ms"),
        },
    }

    output_path = repo_root / "results" / "layout_phase4" / "phase4_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
