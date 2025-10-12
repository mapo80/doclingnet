#!/usr/bin/env python3
"""Aggregate layout Phase 3 benchmark artefacts into a comparative summary."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_or_zero(values: Iterable[float]) -> float:
    sequence = list(values)
    return mean(sequence) if sequence else 0.0


def build_sample_summary(
    sample_id: str,
    python_sample: Dict[str, Any],
    dotnet_samples: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    advanced = next((item for item in dotnet_samples if item.get("advanced_nms", True)), None)
    if advanced is None:
        return None

    disabled = next((item for item in dotnet_samples if not item.get("advanced_nms", True)), None)

    python_total = float(python_sample["total_mean_ms"])
    dotnet_total = float(advanced["total_mean_ms"])
    delta = dotnet_total - python_total
    ratio = dotnet_total / python_total if python_total else float("inf")

    gain = None
    if disabled is not None:
        gain = float(disabled["total_mean_ms"]) - dotnet_total

    return {
        "id": sample_id,
        "path": advanced["path"],
        "python_total_mean_ms": python_total,
        "dotnet_total_mean_ms": dotnet_total,
        "delta_ms": delta,
        "ratio": ratio,
        "python_breakdown_ms": {
            "preprocess": float(python_sample["preprocess_mean_ms"]),
            "inference": float(python_sample["inference_mean_ms"]),
            "postprocess": float(python_sample["postprocess_mean_ms"]),
        },
        "dotnet_breakdown_ms": {
            "preprocess": float(advanced["preprocess_mean_ms"]),
            "inference": float(advanced["inference_mean_ms"]),
            "postprocess": float(advanced["postprocess_mean_ms"]),
        },
        "advanced_nms_gain_ms": gain,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    python_metrics = load_json(repo_root / "results" / "layout_phase3" / "python_timings.json")
    dotnet_metrics = load_json(repo_root / "results" / "layout_phase3" / "dotnet_runs.json")

    python_samples = {sample["id"]: sample for sample in python_metrics.get("samples", [])}
    dotnet_samples: Dict[str, List[Dict[str, Any]]] = {}
    for sample in dotnet_metrics.get("samples", []):
        dotnet_samples.setdefault(sample["id"], []).append(sample)

    summaries: List[Dict[str, Any]] = []
    total_python: List[float] = []
    total_dotnet: List[float] = []
    deltas: List[float] = []
    gains: List[float] = []

    for sample_id, python_sample in python_samples.items():
        entries = dotnet_samples.get(sample_id)
        if not entries:
            continue

        summary = build_sample_summary(sample_id, python_sample, entries)
        if summary is None:
            continue

        summaries.append(summary)
        total_python.append(summary["python_total_mean_ms"])
        total_dotnet.append(summary["dotnet_total_mean_ms"])
        deltas.append(summary["delta_ms"])
        gain = summary.get("advanced_nms_gain_ms")
        if gain is not None:
            gains.append(gain)

    python_mean = mean_or_zero(total_python)
    dotnet_mean = mean_or_zero(total_dotnet)
    delta_mean = dotnet_mean - python_mean
    ratio_mean = dotnet_mean / python_mean if python_mean else float("inf")

    summary_payload = {
        "sample_count": len(summaries),
        "python_mean_ms": python_mean,
        "dotnet_mean_ms": dotnet_mean,
        "delta_mean_ms": delta_mean,
        "ratio_mean": ratio_mean,
        "max_delta_ms": max(deltas) if deltas else 0.0,
        "min_delta_ms": min(deltas) if deltas else 0.0,
        "advanced_nms_mean_gain_ms": mean_or_zero(gains),
        "samples": summaries,
    }

    output_path = repo_root / "results" / "layout_phase3" / "phase3_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
