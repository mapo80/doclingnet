"""Extract PNG images from the FinTabNet parquet sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def extract_images(parquet_path: Path, output_dir: Path) -> list[Path]:
    df = pd.read_parquet(parquet_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for _, row in df.iterrows():
        image_info = row["image"]
        image_bytes = image_info["bytes"] if isinstance(image_info, dict) else image_info
        filename = row["filename"]
        output_path = output_dir / filename
        output_path.write_bytes(image_bytes)
        written.append(output_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FinTabNet PNG samples from parquet")
    parser.add_argument("parquet", type=Path, help="Path to FinTabNet parquet file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where PNG files are created")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.parquet.with_suffix("") / "images"
    written = extract_images(args.parquet, output_dir)
    print(f"Extracted {len(written)} images to {output_dir}")


if __name__ == "__main__":
    main()
