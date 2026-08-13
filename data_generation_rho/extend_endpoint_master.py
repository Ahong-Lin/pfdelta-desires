#!/usr/bin/env python3
"""Append a paired endpoint block to an existing endpoint master with hard links."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def read_records(root: Path) -> list[dict]:
    with (root / "splits" / "train.json").open(encoding="utf-8") as handle:
        return json.load(handle)["records"]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, separators=(",", ":"))
            handle.write("\n")


def validate_pairs(records: list[dict], label: str) -> set[int]:
    by_load: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_load[int(record["load_setting_id"])].append(record)
    for load_id, rows in by_load.items():
        rhos = sorted(float(row["rho"]) for row in rows)
        branches = {str(row["branch_id"]) for row in rows}
        if rhos != [0.0, 1.0] or len(branches) != 1:
            raise ValueError(f"{label}: invalid paired endpoints for load {load_id}")
    return set(by_load)


def load_rows(records: list[dict]) -> list[dict]:
    by_load: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_load[int(record["load_setting_id"])].append(record)
    return [
        {
            "split": "train",
            "load_setting_id": load_id,
            "base_sample_id": load_id,
            "sample_ids": [row["sample_id"] for row in rows],
            "selected_branch_ids": str(rows[0]["branch_id"]),
            "rho_values": "0,1",
        }
        for load_id, rows in sorted(by_load.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-master", required=True)
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    base_root = Path(args.base_master).resolve()
    extension_root = Path(args.extension_root).resolve()
    output_root = Path(args.output_root).resolve()
    base_records = read_records(base_root)
    extension_records = read_records(extension_root)
    base_loads = validate_pairs(base_records, "base")
    extension_loads = validate_pairs(extension_records, "extension")
    overlap = base_loads & extension_loads
    if overlap:
        raise ValueError(f"Base/extension load overlap: {sorted(overlap)[:5]}")

    if output_root.exists():
        if not args.clean:
            raise FileExistsError(f"Output exists: {output_root}")
        shutil.rmtree(output_root)
    for directory in ["raw", "splits", "manifests", "analysis"]:
        (output_root / directory).mkdir(parents=True)

    records: list[dict] = []
    for source_root, source_records, role in [
        (base_root, base_records, "base_master"),
        (extension_root, extension_records, "extension"),
    ]:
        for source in source_records:
            record = dict(source)
            record["sample_id"] = len(records) + 1
            record["file"] = f"raw/sample_{record['sample_id']}.json"
            record["split"] = "train"
            record["source_dataset_root"] = str(source_root)
            record["source_sample_id"] = source["sample_id"]
            record["source_sample_file"] = source["file"]
            record["source_role"] = role
            os.link(source_root / source["file"], output_root / record["file"])
            records.append(record)

    rows = load_rows(records)
    write_json(output_root / "splits" / "train.json", {"split": "train", "records": records, "load_settings": rows, "failures": []})
    for split in ["val", "test"]:
        write_json(output_root / "splits" / f"{split}.json", {"split": split, "records": [], "load_settings": [], "failures": []})
    write_jsonl(output_root / "manifests" / "samples.jsonl", records)
    write_jsonl(output_root / "manifests" / "load_settings.jsonl", rows)
    (output_root / "manifests" / "failures.jsonl").touch()
    write_json(
        output_root / "meta.json",
        {
            "design": "Ordered extension of a paired endpoint master.",
            "records": len(records),
            "loads": len(rows),
            "base_master": str(base_root),
            "extension_root": str(extension_root),
        },
    )
    print(f"Built {output_root}: records={len(records)}, loads={len(rows)}")


if __name__ == "__main__":
    main()
