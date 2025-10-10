#!/usr/bin/env python3
"""Compare layout detections between Python and .NET runs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Box:
    label: str
    left: float
    top: float
    width: float
    height: float
    confidence: float | None
    source: str
    page_width: float
    page_height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def as_normalised(self) -> Tuple[float, float, float, float]:
        return (
            self.left / self.page_width if self.page_width else 0.0,
            self.top / self.page_height if self.page_height else 0.0,
            self.width / self.page_width if self.page_width else 0.0,
            self.height / self.page_height if self.page_height else 0.0,
        )


@dataclass
class MatchResult:
    python_box: Box
    dotnet_box: Box
    iou: float


def _load_python_boxes(path: Path) -> Tuple[List[Box], Dict[str, float]]:
    data = json.loads(path.read_text())
    metadata = data.get("metadata", {})
    page_width = float(metadata.get("originalWidth", 1.0))
    page_height = float(metadata.get("originalHeight", 1.0))

    boxes: List[Box] = []
    for det in data.get("detections", []):
        page = det.get("page", {})
        boxes.append(
            Box(
                label=det.get("label", "Unknown"),
                left=float(page.get("left", 0.0)),
                top=float(page.get("top", 0.0)),
                width=float(page.get("width", 0.0)),
                height=float(page.get("height", 0.0)),
                confidence=float(det.get("confidence", 0.0)) if det.get("confidence") is not None else None,
                source="python",
                page_width=page_width,
                page_height=page_height,
            )
        )

    return boxes, {"page_width": page_width, "page_height": page_height}


def _load_dotnet_boxes(path: Path) -> Tuple[List[Box], Dict[str, float]]:
    data = json.loads(path.read_text())
    normalisations = data.get("data", {}).get("normalisations", [])
    if normalisations:
        page_width = float(normalisations[0].get("originalWidth", 1.0))
        page_height = float(normalisations[0].get("originalHeight", 1.0))
    else:
        page_width = 1.0
        page_height = 1.0

    boxes: List[Box] = []
    for item in data.get("data", {}).get("items", []):
        bbox = item.get("boundingBox", {})
        boxes.append(
            Box(
                label=item.get("kind", "Unknown"),
                left=float(bbox.get("left", 0.0)),
                top=float(bbox.get("top", 0.0)),
                width=float(bbox.get("width", 0.0)),
                height=float(bbox.get("height", 0.0)),
                confidence=None,
                source="dotnet",
                page_width=page_width,
                page_height=page_height,
            )
        )

    return boxes, {"page_width": page_width, "page_height": page_height}


def _compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_left = max(ax, bx)
    inter_top = max(ay, by)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)

    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _greedy_match(python_boxes: List[Box], dotnet_boxes: List[Box]) -> List[MatchResult]:
    pairs: List[Tuple[int, int, float]] = []
    for i, p in enumerate(python_boxes):
        pn = p.as_normalised()
        for j, d in enumerate(dotnet_boxes):
            if p.label.lower() != d.label.lower():
                continue
            dn = d.as_normalised()
            iou = _compute_iou(pn, dn)
            if iou > 0:
                pairs.append((i, j, iou))

    pairs.sort(key=lambda item: item[2], reverse=True)
    matched_python = set()
    matched_dotnet = set()
    matches: List[MatchResult] = []

    for i, j, iou in pairs:
        if i in matched_python or j in matched_dotnet:
            continue
        matched_python.add(i)
        matched_dotnet.add(j)
        matches.append(MatchResult(python_boxes[i], dotnet_boxes[j], iou))

    return matches


def _summarise_matches(matches: List[MatchResult]) -> Dict[str, float]:
    if not matches:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    values = sorted(match.iou for match in matches)
    count = len(values)
    mean = sum(values) / count
    median = values[count // 2] if count % 2 == 1 else (values[count // 2 - 1] + values[count // 2]) / 2.0
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "min": values[0],
        "max": values[-1],
    }


def _format_table(rows: List[List[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        line = " | ".join(padded)
        lines.append(line)
        if idx == 0:
            lines.append(" | ".join("-" * widths[i] for i in range(len(widths))))
    return "\n".join(lines)


def build_report(
    python_boxes: List[Box],
    dotnet_boxes: List[Box],
    matches: List[MatchResult],
    output_dir: Path,
    python_meta: Dict[str, float],
    dotnet_meta: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _summarise_matches(matches)
    python_counts: Dict[str, int] = {}
    for box in python_boxes:
        python_counts[box.label] = python_counts.get(box.label, 0) + 1

    dotnet_counts: Dict[str, int] = {}
    for box in dotnet_boxes:
        dotnet_counts[box.label] = dotnet_counts.get(box.label, 0) + 1

    markdown_lines = ["# Layout comparison", ""]
    markdown_lines.append("## Summary")
    markdown_lines.append("")
    markdown_lines.append(f"- Python boxes: {len(python_boxes)}")
    markdown_lines.append(f"- .NET boxes: {len(dotnet_boxes)}")
    markdown_lines.append(f"- Matches: {summary['count']} (mean IoU {summary['mean']:.3f}, median {summary['median']:.3f})")
    markdown_lines.append("")

    markdown_lines.append("### Counts per label")
    markdown_lines.append("")
    label_rows = [["Label", "Python", ".NET"]]
    labels = sorted(set(list(python_counts.keys()) + list(dotnet_counts.keys())))
    for label in labels:
        label_rows.append([
            label,
            str(python_counts.get(label, 0)),
            str(dotnet_counts.get(label, 0)),
        ])
    markdown_lines.append(_format_table(label_rows))
    markdown_lines.append("")

    if matches:
        markdown_lines.append("### Matched boxes (normalised deltas)")
        markdown_lines.append("")
        match_rows = [["#", "IoU", "Δx", "Δy", "Δw", "Δh"]]
        for idx, match in enumerate(matches, start=1):
            p_norm = match.python_box.as_normalised()
            d_norm = match.dotnet_box.as_normalised()
            deltas = [abs(p - d) for p, d in zip(p_norm, d_norm)]
            match_rows.append(
                [
                    str(idx),
                    f"{match.iou:.3f}",
                    f"{deltas[0]:.3f}",
                    f"{deltas[1]:.3f}",
                    f"{deltas[2]:.3f}",
                    f"{deltas[3]:.3f}",
                ]
            )
        markdown_lines.append(_format_table(match_rows))
        markdown_lines.append("")

    metadata_section = {
        "python": python_meta,
        "dotnet": dotnet_meta,
        "matches": summary,
        "python_count": len(python_boxes),
        "dotnet_count": len(dotnet_boxes),
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata_section, indent=2))
    (output_dir / "report.md").write_text("\n".join(markdown_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_layout", type=Path, help="Path to the Python layout JSON")
    parser.add_argument("dotnet_layout", type=Path, help="Path to the .NET layout workflow JSON")
    parser.add_argument("output", type=Path, help="Directory where the report should be stored")
    args = parser.parse_args()

    python_boxes, python_meta = _load_python_boxes(args.python_layout)
    dotnet_boxes, dotnet_meta = _load_dotnet_boxes(args.dotnet_layout)

    matches = _greedy_match(python_boxes, dotnet_boxes)
    build_report(python_boxes, dotnet_boxes, matches, args.output, python_meta, dotnet_meta)


if __name__ == "__main__":
    main()
