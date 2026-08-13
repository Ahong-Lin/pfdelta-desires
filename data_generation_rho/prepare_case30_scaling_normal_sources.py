#!/usr/bin/env python3
"""Build a canonical normal-PF source directory for endpoint scaling."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def hard_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(f"Destination exists with another source: {destination}")
    os.link(source, destination)


def link_range(source_dir: Path, target_dir: Path, first: int, last: int, source_offset: int) -> int:
    linked = 0
    for load_id in range(first, last + 1):
        source_id = load_id + source_offset
        source = source_dir / f"sample_{source_id}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing normal PF source: {source}")
        hard_link(source, target_dir / f"sample_{load_id}.json")
        linked += 1
    return linked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-range",
        action="append",
        required=True,
        metavar="DIR:TARGET_FIRST:TARGET_LAST:SOURCE_FIRST",
        help="Map a locally numbered source directory onto a canonical load-ID range.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    seen_loads: set[int] = set()
    for text in args.source_range:
        try:
            source_dir_text, first_text, last_text, source_first_text = text.rsplit(":", 3)
            first, last, source_first = int(first_text), int(last_text), int(source_first_text)
        except ValueError as error:
            raise ValueError(f"Invalid --source-range: {text}") from error
        if first > last:
            raise ValueError(f"Invalid target range in {text}")
        target_loads = set(range(first, last + 1))
        if seen_loads & target_loads:
            raise ValueError(f"Overlapping target load range in {text}")
        seen_loads.update(target_loads)
        linked += link_range(
            Path(source_dir_text).expanduser().resolve(),
            output_dir,
            first,
            last,
            source_first - first,
        )
    print(f"Canonical normal source ready: {output_dir}; linked={linked}, loads={len(seen_loads)}")


if __name__ == "__main__":
    main()
