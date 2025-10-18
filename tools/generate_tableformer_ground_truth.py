#!/usr/bin/env python3
"""Generate TableFormer Python ground truth fixtures for parity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

try:
    from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "docling-ibm-models is required. Install with 'pip install docling-ibm-models --no-deps'."
    ) from exc


def _load_model(config_path: Path, weights_path: Path, device: torch.device) -> TableModel04_rs:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config.setdefault("model", {})["save_dir"] = str(weights_path.parent)

    state_dict = load_file(str(weights_path))
    dataset_wordmap = config["dataset_wordmap"]
    init_data = {
        "word_map": {
            "word_map_tag": dataset_wordmap["word_map_tag"],
            "word_map_cell": dataset_wordmap["word_map_cell"],
        }
    }

    model = TableModel04_rs(config, init_data=init_data, device=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _run_inference(model: TableModel04_rs, image_path: Path, max_steps: int) -> dict[str, Any]:
    image = np.load(str(image_path))
    image_tensor = torch.from_numpy(image)

    with torch.no_grad():
        sequence, bbox_classes, bbox_coords = model.predict(
            image_tensor, max_steps=max_steps, k=1, return_attention=False
        )

    return {
        "sequence": [int(idx) for idx in sequence],
        "bbox_classes": bbox_classes.tolist(),
        "bbox_coords": bbox_coords.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to tm_config.json")
    parser.add_argument("weights", type=Path, help="Path to tableformer_fast.safetensors")
    parser.add_argument("image", type=Path, help="Path to preprocessed input .npy file")
    parser.add_argument("output", type=Path, help="Destination JSON file for the fixtures")
    parser.add_argument("--max-steps", type=int, default=1024, help="Maximum decoding steps")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = _load_model(args.config, args.weights, device)
    result = _run_inference(model, args.image, max_steps=args.max_steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Ground truth saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
