#!/usr/bin/env python3
"""Generate comparison metrics between Python and .NET Markdown exports."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import List, Tuple


@dataclass
class DocumentStats:
    lines: int
    words: int
    characters: int
    blank_lines: int


@dataclass
class MarkdownDiffResult:
    python_stats: DocumentStats
    dotnet_stats: DocumentStats
    similarity: Tuple[float, float, int, int]
    missing_characters: Counter
    spurious_characters: Counter
    diff_text: str

    def to_markdown(self) -> str:
        ratio, jaccard, shared, total_unique = self.similarity

        md_lines: List[str] = []
        md_lines.append("# Python vs .NET Markdown Comparison")
        md_lines.append("")
        md_lines.append("## Summary Metrics")
        md_lines.append("")
        md_lines.append("| Metric | Python | .NET | Delta (.NET - Python) |")
        md_lines.append("| --- | ---: | ---: | ---: |")
        md_lines.append(
            f"| Lines | {self.python_stats.lines} | {self.dotnet_stats.lines} | {self.dotnet_stats.lines - self.python_stats.lines:+d} |"
        )
        md_lines.append(
            f"| Words | {self.python_stats.words} | {self.dotnet_stats.words} | {self.dotnet_stats.words - self.python_stats.words:+d} |"
        )
        md_lines.append(
            f"| Characters | {self.python_stats.characters} | {self.dotnet_stats.characters} | {self.dotnet_stats.characters - self.python_stats.characters:+d} |"
        )
        md_lines.append(
            f"| Blank Lines | {self.python_stats.blank_lines} | {self.dotnet_stats.blank_lines} | {self.dotnet_stats.blank_lines - self.python_stats.blank_lines:+d} |"
        )
        md_lines.append("")
        md_lines.append("")
        md_lines.append("## Similarity Scores")
        md_lines.append("")
        md_lines.append(f"- SequenceMatcher ratio: **{ratio:.4f}**")
        md_lines.append(
            f"- Line-level Jaccard overlap: **{jaccard:.4f}** ({shared} shared / {total_unique} total unique lines)"
        )
        md_lines.append("")
        md_lines.append("")
        md_lines.append("## Character Discrepancies")
        md_lines.append("")
        md_lines.append("Characters missing from the .NET output compared to Python:")
        md_lines.append("| Character | Count | ASCII |")
        md_lines.append("| --- | ---: | ---: |")
        md_lines.extend(table_from_counter(self.missing_characters))
        md_lines.append("")
        md_lines.append("Spurious characters introduced by the .NET output:")
        md_lines.append("| Character | Count | ASCII |")
        md_lines.append("| --- | ---: | ---: |")
        md_lines.extend(table_from_counter(self.spurious_characters))
        md_lines.append("")
        md_lines.append("")
        md_lines.append("## Line-Level Diff")
        md_lines.append("")
        md_lines.append("```diff")
        md_lines.append(self.diff_text)
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("## Raw JSON Summary")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(self.to_summary_dict(), indent=2, ensure_ascii=False))
        md_lines.append("```")

        return "\n".join(md_lines)

    def to_summary_dict(self) -> dict:
        ratio, jaccard, shared, total_unique = self.similarity
        return {
            "summary": {
                "python": self.python_stats.__dict__,
                "dotnet": self.dotnet_stats.__dict__,
            },
            "similarity": {
                "sequence_matcher_ratio": ratio,
                "line_jaccard": jaccard,
                "shared_unique_lines": shared,
                "total_unique_lines": total_unique,
            },
            "missing_characters": {k: v for k, v in self.missing_characters.items()},
            "spurious_characters": {k: v for k, v in self.spurious_characters.items()},
        }


class MarkdownDiffGenerator:
    def __init__(self, python_label: str, dotnet_label: str) -> None:
        self.python_label = python_label
        self.dotnet_label = dotnet_label

    def generate(self, python_text: str, dotnet_text: str) -> MarkdownDiffResult:
        python_stats = compute_stats(python_text)
        dotnet_stats = compute_stats(dotnet_text)
        similarity = compute_similarity(python_text, dotnet_text)
        missing, spurious = diff_counters(python_text, dotnet_text)
        diff_text = build_diff(
            python_text,
            dotnet_text,
            self.python_label,
            self.dotnet_label,
        )
        return MarkdownDiffResult(
            python_stats=python_stats,
            dotnet_stats=dotnet_stats,
            similarity=similarity,
            missing_characters=missing,
            spurious_characters=spurious,
            diff_text=diff_text,
        )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compute_stats(text: str) -> DocumentStats:
    lines = text.splitlines()
    words = len(text.split())
    blank_lines = sum(1 for line in lines if line.strip() == "")
    return DocumentStats(
        lines=len(lines),
        words=words,
        characters=len(text),
        blank_lines=blank_lines,
    )


def compute_similarity(python_text: str, dotnet_text: str) -> Tuple[float, float, int, int]:
    matcher = SequenceMatcher(None, python_text, dotnet_text)
    ratio = matcher.ratio()

    python_lines = set(python_text.splitlines())
    dotnet_lines = set(dotnet_text.splitlines())
    shared = len(python_lines & dotnet_lines)
    total_unique = len(python_lines | dotnet_lines)
    jaccard = shared / total_unique if total_unique else 1.0
    return ratio, jaccard, shared, total_unique


def diff_counters(python_text: str, dotnet_text: str) -> Tuple[Counter, Counter]:
    python_counter = Counter(python_text)
    dotnet_counter = Counter(dotnet_text)

    missing = Counter()
    for char, count in python_counter.items():
        delta = count - dotnet_counter.get(char, 0)
        if delta > 0:
            missing[char] = delta

    spurious = Counter()
    for char, count in dotnet_counter.items():
        delta = count - python_counter.get(char, 0)
        if delta > 0:
            spurious[char] = delta

    return missing, spurious


def format_char(char: str) -> str:
    if char == "\n":
        return "\\n"
    if char == "\t":
        return "\\t"
    if char == "\r":
        return "\\r"
    if char == " ":
        return "' '"
    return repr(char)[1:-1] if char not in {"'", '"'} else repr(char)


def table_from_counter(counter: Counter, limit: int = 20) -> List[str]:
    rows: List[str] = []
    for char, count in counter.most_common(limit):
        display = format_char(char)
        rows.append(f"| {display} | {count} | {ord(char)} |")
    if len(counter) > limit:
        rows.append("| ... | ... | ... |")
    return rows


def build_diff(python_text: str, dotnet_text: str, python_label: str, dotnet_label: str) -> str:
    python_lines = python_text.splitlines()
    dotnet_lines = dotnet_text.splitlines()
    diff_lines = list(
        unified_diff(
            python_lines,
            dotnet_lines,
            fromfile=python_label,
            tofile=dotnet_label,
            lineterm="",
        )
    )
    return "\n".join(diff_lines)


def write_markdown_report(output_dir: Path, report: MarkdownDiffResult) -> None:
    (output_dir / "report.md").write_text(report.to_markdown(), encoding="utf-8")


def write_summary_json(output_path: Path, report: MarkdownDiffResult) -> None:
    summary = json.dumps(report.to_summary_dict(), indent=2, ensure_ascii=False)
    output_path.write_text(summary + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_markdown", type=Path)
    parser.add_argument("dotnet_markdown", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--python-label", default="python-cli/docling.md")
    parser.add_argument("--dotnet-label", default="dotnet-cli/docling.md")
    args = parser.parse_args()

    python_text = read_text(args.python_markdown)
    dotnet_text = read_text(args.dotnet_markdown)

    generator = MarkdownDiffGenerator(args.python_label, args.dotnet_label)
    report = generator.generate(python_text, dotnet_text)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    write_markdown_report(output_dir, report)
    write_summary_json(output_dir / "summary.json", report)


if __name__ == "__main__":
    main()
