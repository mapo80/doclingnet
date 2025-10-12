#!/usr/bin/env python3
"""Aggregate layout Phase 2 benchmarks into a comparable summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    phase1_metrics = load_json(repo_root / "results" / "layout_phase1" / "phase1_metrics.json")
    breakdown = load_json(repo_root / "results" / "layout_phase2" / "dotnet_breakdown.json")
    tensor_benchmark = load_json(repo_root / "results" / "layout_phase2" / "tensor_owner_benchmark.json")

    python_mean = float(phase1_metrics["python_mean_ms"])
    dotnet_phase1 = float(breakdown["phase1"]["total_mean_ms"])
    dotnet_phase2 = float(breakdown["phase2"]["total_mean_ms"])

    delta_phase1 = dotnet_phase1 - python_mean
    delta_phase2 = dotnet_phase2 - python_mean

    ratio_phase1 = dotnet_phase1 / python_mean if python_mean else float("inf")
    ratio_phase2 = dotnet_phase2 / python_mean if python_mean else float("inf")

    postprocess_savings = (
        float(breakdown["phase1"]["postprocess_mean_ms"]) -
        float(breakdown["phase2"]["postprocess_mean_ms"])
    )

    summary = {
        "python_mean_ms": python_mean,
        "dotnet_phase1_mean_ms": dotnet_phase1,
        "dotnet_phase2_mean_ms": dotnet_phase2,
        "delta_ms_phase1": delta_phase1,
        "delta_ms_phase2": delta_phase2,
        "ratio_phase1": ratio_phase1,
        "ratio_phase2": ratio_phase2,
        "total_savings_ms": dotnet_phase1 - dotnet_phase2,
        "postprocess_savings_ms": postprocess_savings,
        "tensor_owner_savings_ms": float(tensor_benchmark["DeltaMs"]),
        "tensor_benchmark": tensor_benchmark,
        "dotnet_breakdown": breakdown,
    }

    output_path = repo_root / "results" / "layout_phase2" / "phase2_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
