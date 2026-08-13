#!/usr/bin/env python3
"""Build a PFDelta-style unpaired endpoint baseline.

The output dataset mixes two unpaired pools:

* rho=1 records copied from an existing endpoint/control dataset.
* rho=0 records copied from a random-load/random-branch PF-rho dataset.

This is intended to mimic PFDelta task 1.2's unpaired N / N-1 mixture while
staying in the branch-rho setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, separators=(",", ":"))
            handle.write("\n")


def copy_sample(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_split_records(root: Path, split: str, rho: float | None) -> list[dict]:
    payload = read_json(root / "splits" / f"{split}.json")
    records = payload.get("records", [])
    if rho is None:
        return records
    return [record for record in records if abs(float(record["rho"]) - rho) < 1e-10]


def parse_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        split, count = item.split(":", 1)
        counts[split.strip()] = int(count)
    return counts


def make_record(
    source_root: Path,
    source_record: dict,
    sample_id: int,
    split: str,
    output_rel: str,
    role: str,
    task_name: str,
) -> dict:
    record = dict(source_record)
    record["sample_id"] = sample_id
    record["file"] = output_rel
    record["split"] = split
    record["task_name"] = task_name
    record["source_dataset_root"] = str(source_root)
    record["source_sample_id"] = source_record.get("sample_id")
    record["source_sample_file"] = source_record.get("file")
    record["source_role"] = role
    record["experiment_name"] = task_name
    return record


def update_sample_meta(path: Path, sample_id: int, split: str, task_name: str, source_root: Path, source_record: dict, role: str) -> None:
    sample = read_json(path)
    meta = sample.setdefault("rho_meta", {})
    meta["sample_id"] = sample_id
    meta["split"] = split
    meta["task_name"] = task_name
    meta["source_dataset_root"] = str(source_root)
    meta["source_sample_id"] = source_record.get("sample_id")
    meta["source_sample_file"] = source_record.get("file")
    meta["source_role"] = role
    write_json(path, sample)


def write_summary(output_dir: Path, records_by_split: dict[str, list[dict]]) -> None:
    fieldnames = [
        "split",
        "samples",
        "load_settings",
        "branches",
        "branch_ids",
        "rho_values",
        "rho_counts",
        "source_roles",
    ]
    rows = []
    for split in ["train", "val", "test"]:
        records = records_by_split.get(split, [])
        loads = {int(record["load_setting_id"]) for record in records if "load_setting_id" in record}
        branches = Counter(str(record["branch_id"]) for record in records if "branch_id" in record)
        rhos = Counter(f"{float(record['rho']):.10g}" for record in records if "rho" in record)
        roles = Counter(str(record.get("source_role", "")) for record in records)
        rows.append(
            {
                "split": split,
                "samples": len(records),
                "load_settings": len(loads),
                "branches": len(branches),
                "branch_ids": ",".join(sorted(branches, key=int)) if branches else "",
                "rho_values": ",".join(sorted(rhos, key=float)) if rhos else "",
                "rho_counts": ";".join(f"{key}:{rhos[key]}" for key in sorted(rhos, key=float)),
                "source_roles": ";".join(f"{key}:{roles[key]}" for key in sorted(roles)),
            }
        )
    with (output_dir / "analysis" / "split_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho1-root", required=True)
    parser.add_argument("--rho0-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--counts", default="train:41976,val:6012")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rho1_root = Path(args.rho1_root).resolve()
    rho0_root = Path(args.rho0_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    counts = parse_counts(args.counts)
    rng = random.Random(args.seed)

    if output_dir.exists():
        if not args.clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    records_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    load_rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    all_records: list[dict] = []
    all_load_rows: list[dict] = []
    source_map: list[dict] = []
    next_sample_id = 1

    for split in ["train", "val"]:
        total = counts.get(split, 0)
        if total <= 0:
            continue
        if total % 2 != 0:
            raise ValueError(f"Split count must be even for 50/50 mix: {split}={total}")
        each = total // 2
        rho1_records = load_split_records(rho1_root, split, 1.0)
        rho0_records = load_split_records(rho0_root, split, 0.0)
        if len(rho1_records) < each:
            raise RuntimeError(f"Need {each} rho=1 {split} records, found {len(rho1_records)}")
        if len(rho0_records) < each:
            raise RuntimeError(f"Need {each} rho=0 {split} records, found {len(rho0_records)}")
        selected = [
            ("rho1_normal_pool", rho1_root, record)
            for record in rng.sample(rho1_records, each)
        ] + [
            ("rho0_random_edge_pool", rho0_root, record)
            for record in rng.sample(rho0_records, each)
        ]
        rng.shuffle(selected)

        for role, source_root, source_record in selected:
            out_rel = f"raw/sample_{next_sample_id}.json"
            out_path = output_dir / out_rel
            copy_sample(source_root / source_record["file"], out_path)
            update_sample_meta(out_path, next_sample_id, split, args.task_name, source_root, source_record, role)
            out_record = make_record(source_root, source_record, next_sample_id, split, out_rel, role, args.task_name)
            records_by_split[split].append(out_record)
            all_records.append(out_record)
            source_map.append(
                {
                    "split": split,
                    "sample_id": next_sample_id,
                    "source_role": role,
                    "load_setting_id": source_record.get("load_setting_id"),
                    "branch_id": source_record.get("branch_id"),
                    "rho": source_record.get("rho"),
                    "source_dataset_root": str(source_root),
                    "source_sample_id": source_record.get("sample_id"),
                    "source_file": source_record.get("file"),
                }
            )
            load_row = {
                "split": split,
                "load_setting_id": int(source_record["load_setting_id"]),
                "base_sample_id": int(source_record.get("base_sample_id", source_record["load_setting_id"])),
                "source_file": source_record.get("source_file"),
                "sample_ids": [next_sample_id],
                "selected_branch_ids": str(source_record.get("branch_id", "")),
                "rho_values": f"{float(source_record['rho']):.10g}",
                "source_role": role,
            }
            load_rows_by_split[split].append(load_row)
            all_load_rows.append(load_row)
            next_sample_id += 1

    for split in ["train", "val", "test"]:
        write_json(
            output_dir / "splits" / f"{split}.json",
            {
                "split": split,
                "records": records_by_split[split],
                "load_settings": load_rows_by_split[split],
                "failures": [],
            },
        )

    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", all_load_rows)
    (output_dir / "manifests" / "failures.jsonl").touch()
    write_summary(output_dir, records_by_split)
    write_json(
        output_dir / "meta.json",
        {
            "task_name": args.task_name,
            "created_from": "pfdelta-style unpaired endpoint baseline",
            "rho1_root": str(rho1_root),
            "rho0_root": str(rho0_root),
            "counts": counts,
            "seed": args.seed,
            "note": "rho=1 and rho=0 pools are sampled independently, mimicking PFDelta task 1.2's unpaired N/N-1 mixture in a branch-rho setting.",
        },
    )
    with (output_dir / "analysis" / "source_map.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "sample_id", "source_role", "load_setting_id", "branch_id", "rho", "source_dataset_root", "source_sample_id", "source_file"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_map)
    print(json.dumps({"output_dir": str(output_dir), "samples": len(all_records)}, indent=2))


if __name__ == "__main__":
    main()
