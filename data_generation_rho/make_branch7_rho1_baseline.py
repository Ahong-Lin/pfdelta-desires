#!/usr/bin/env python3
"""Build a branch-7 rho=1-only baseline with the fixed rho=0 test split.

The baseline trains only on normal-topology branch-7 samples (rho=1) and
reuses the fixed branch-7 rho=0 test split from the PF-rho benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path


def rho_key(value: float | str) -> str:
    return f"{round(float(value), 10):.10g}"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, separators=(",", ":"))
            handle.write("\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(src: Path, dst: Path, copy_raw: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_raw:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_rho1_records(source_roots: list[Path], branch_id: str) -> tuple[dict[int, dict], dict[int, dict]]:
    sample_index: dict[int, dict] = {}
    load_records: dict[int, dict] = {}

    for source_root in source_roots:
        for rec in read_jsonl(source_root / "manifests" / "load_settings.jsonl"):
            load_id = int(rec["load_setting_id"])
            rec = dict(rec)
            rec["_source_root"] = str(source_root)
            load_records[load_id] = rec

        for rec in read_jsonl(source_root / "manifests" / "samples.jsonl"):
            if str(rec.get("branch_id")) != str(branch_id):
                continue
            if rho_key(rec.get("rho")) != "1":
                continue
            load_id = int(rec["load_setting_id"])
            if load_id in sample_index:
                raise RuntimeError(f"Duplicate rho=1 sample for load_setting_id={load_id}")
            rec = dict(rec)
            rec["_source_root"] = str(source_root)
            sample_index[load_id] = rec

    missing_loads = sorted(set(sample_index) - set(load_records))
    if missing_loads:
        raise RuntimeError(f"Missing load records for {len(missing_loads)} rho=1 samples")

    return load_records, sample_index


def fixed_split_loads(fixed_split_root: Path, split: str) -> list[int]:
    split_path = fixed_split_root / "splits" / f"{split}.json"
    data = json.load(split_path.open("r", encoding="utf-8"))
    loads = sorted({int(rec["load_setting_id"]) for rec in data["records"]})
    return loads


def fixed_test_records(fixed_split_root: Path) -> list[dict]:
    split_path = fixed_split_root / "splits" / "test.json"
    data = json.load(split_path.open("r", encoding="utf-8"))
    records = [dict(rec) for rec in data["records"]]
    for rec in records:
        if rho_key(rec["rho"]) != "0":
            raise RuntimeError("Fixed test split contains nonzero rho")
    return records


def make_source_record(
    source_rec: dict,
    out_sample_id: int,
    out_rel: str,
    split: str,
    experiment_name: str,
) -> dict:
    out_rec = dict(source_rec)
    source_root = out_rec.pop("_source_root")
    out_rec["sample_id"] = out_sample_id
    out_rec["file"] = out_rel
    out_rec["split"] = split
    out_rec["source_dataset_root"] = source_root
    out_rec["source_sample_id"] = source_rec["sample_id"]
    out_rec["source_sample_file"] = source_rec["file"]
    out_rec["source_load_setting_id"] = source_rec["load_setting_id"]
    out_rec["selected_branch_id"] = source_rec["branch_id"]
    out_rec["experiment_name"] = experiment_name
    return out_rec


def make_fixed_test_record(
    fixed_rec: dict,
    fixed_split_root: Path,
    out_sample_id: int,
    out_rel: str,
    experiment_name: str,
) -> dict:
    out_rec = dict(fixed_rec)
    out_rec["sample_id"] = out_sample_id
    out_rec["file"] = out_rel
    out_rec["split"] = "test"
    out_rec["fixed_test_root"] = str(fixed_split_root)
    out_rec["fixed_test_sample_id"] = fixed_rec["sample_id"]
    out_rec["fixed_test_file"] = fixed_rec["file"]
    out_rec["selected_branch_id"] = fixed_rec["branch_id"]
    out_rec["experiment_name"] = experiment_name
    return out_rec


def build_split(
    output_dir: Path,
    split: str,
    records: list[dict],
    load_records: dict[int, dict],
    source_map_rows: list[dict],
) -> None:
    split_loads = []
    by_load: dict[int, list[int]] = {}
    for rec in records:
        by_load.setdefault(int(rec["load_setting_id"]), []).append(int(rec["sample_id"]))

    for load_id in sorted(by_load):
        load_rec = dict(load_records.get(load_id, {}))
        load_rec.pop("_source_root", None)
        load_rec["split"] = split
        load_rec["load_setting_id"] = load_id
        load_rec["sample_ids"] = by_load[load_id]
        load_rec["num_selected_rhos"] = len(by_load[load_id])
        split_loads.append(load_rec)
        source_map_rows.append(
            {
                "split": split,
                "load_setting_id": load_id,
                "selected_branch_id": records[0].get("selected_branch_id", "7") if records else "7",
                "rho_values": ",".join(rho_key(rec["rho"]) for rec in records if int(rec["load_setting_id"]) == load_id),
                "sample_ids": ",".join(str(sample_id) for sample_id in by_load[load_id]),
            }
        )

    write_json(
        output_dir / "splits" / f"{split}.json",
        {
            "split": split,
            "records": records,
            "load_settings": split_loads,
            "failures": [],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", action="append", required=True)
    parser.add_argument("--fixed-split-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--train-samples", type=int, default=41976)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--prefer-low-load-ids",
        action="store_true",
        help=(
            "Select training load settings by ascending load_setting_id instead "
            "of random shuffle. This keeps the existing first-10000 rho=1 "
            "samples before filling from later normal samples."
        ),
    )
    parser.add_argument("--copy-raw", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source_roots = [Path(path).resolve() for path in args.source_bank]
    fixed_split_root = Path(args.fixed_split_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    experiment_name = output_dir.name

    if output_dir.exists():
        if not args.clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "manifests", "splits", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    load_records, sample_index = load_rho1_records(source_roots, args.branch_id)
    test_loads = fixed_split_loads(fixed_split_root, "test")
    val_loads = fixed_split_loads(fixed_split_root, "val")
    test_set = set(test_loads)
    val_set = set(val_loads)
    if test_set & val_set:
        raise RuntimeError("Fixed val/test load settings overlap")

    missing_val = sorted(val_set - set(sample_index))
    if missing_val:
        raise RuntimeError(f"Missing rho=1 val samples for {len(missing_val)} loads")

    candidate_train_loads = sorted(set(sample_index) - test_set - val_set)
    if not args.prefer_low_load_ids:
        rng = random.Random(args.seed)
        rng.shuffle(candidate_train_loads)
    train_loads = candidate_train_loads[: args.train_samples]
    if len(train_loads) < args.train_samples:
        raise RuntimeError(
            f"Need {args.train_samples} rho=1 training loads, found {len(train_loads)}"
        )

    all_records = []
    source_map_rows = []
    split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    next_sample_id = 1

    for split, loads in [("train", train_loads), ("val", val_loads)]:
        for load_id in loads:
            source_rec = sample_index[load_id]
            source_root = Path(source_rec["_source_root"])
            src_file = source_root / source_rec["file"]
            out_rel = f"raw/sample_{next_sample_id}.json"
            link_or_copy(src_file, output_dir / out_rel, copy_raw=args.copy_raw)
            out_rec = make_source_record(
                source_rec,
                next_sample_id,
                out_rel,
                split,
                experiment_name,
            )
            split_records[split].append(out_rec)
            all_records.append(out_rec)
            next_sample_id += 1

    test_records = fixed_test_records(fixed_split_root)
    fixed_test_load_records = {}
    fixed_test_split = json.load((fixed_split_root / "splits" / "test.json").open("r", encoding="utf-8"))
    for rec in fixed_test_split.get("load_settings", []):
        fixed_test_load_records[int(rec["load_setting_id"])] = rec

    for fixed_rec in test_records:
        src_file = fixed_split_root / fixed_rec["file"]
        out_rel = f"raw/sample_{next_sample_id}.json"
        link_or_copy(src_file, output_dir / out_rel, copy_raw=args.copy_raw)
        out_rec = make_fixed_test_record(
            fixed_rec,
            fixed_split_root,
            next_sample_id,
            out_rel,
            experiment_name,
        )
        split_records["test"].append(out_rec)
        all_records.append(out_rec)
        next_sample_id += 1

    load_records_for_split = dict(load_records)
    load_records_for_split.update(fixed_test_load_records)
    for split in ["train", "val", "test"]:
        build_split(
            output_dir,
            split,
            split_records[split],
            load_records_for_split,
            source_map_rows,
        )

    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(
        output_dir / "manifests" / "load_settings.jsonl",
        [dict(load_records_for_split[load_id]) for load_id in sorted(set(train_loads) | val_set | test_set)],
    )
    (output_dir / "manifests" / "failures.jsonl").touch()

    summary_rows = []
    for split in ["train", "val", "test"]:
        records = split_records[split]
        loads = sorted({int(rec["load_setting_id"]) for rec in records})
        rhos = sorted({rho_key(rec["rho"]) for rec in records}, key=float)
        summary_rows.append(
            {
                "split": split,
                "load_settings": len(loads),
                "rho_values_per_load": len(records) // max(len(loads), 1),
                "samples": len(records),
                "rho_values": ",".join(rhos),
            }
        )
    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        summary_rows,
        ["split", "load_settings", "rho_values_per_load", "samples", "rho_values"],
    )
    write_csv(
        output_dir / "analysis" / "load_source_map.csv",
        source_map_rows,
        ["split", "load_setting_id", "selected_branch_id", "rho_values", "sample_ids"],
    )

    train_set = {int(rec["load_setting_id"]) for rec in split_records["train"]}
    val_set_actual = {int(rec["load_setting_id"]) for rec in split_records["val"]}
    test_set_actual = {int(rec["load_setting_id"]) for rec in split_records["test"]}
    overlaps = {
        "train_val": len(train_set & val_set_actual),
        "train_test": len(train_set & test_set_actual),
        "val_test": len(val_set_actual & test_set_actual),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Split load settings overlap: {overlaps}")
    if {rho_key(rec["rho"]) for rec in split_records["train"]} != {"1"}:
        raise RuntimeError("Train split is not rho=1 only")
    if {rho_key(rec["rho"]) for rec in split_records["val"]} != {"1"}:
        raise RuntimeError("Val split is not rho=1 only")
    if {rho_key(rec["rho"]) for rec in split_records["test"]} != {"0"}:
        raise RuntimeError("Test split is not rho=0 only")

    meta = {
        "experiment_name": experiment_name,
        "source_banks": [str(path) for path in source_roots],
        "fixed_split_root": str(fixed_split_root),
        "branch_id": args.branch_id,
        "seed": args.seed,
        "train_selection_strategy": (
            "ascending_load_setting_id" if args.prefer_low_load_ids else "random_shuffle"
        ),
        "selection": summary_rows,
        "split_load_overlaps": overlaps,
        "target": "PF_solution(rho)",
        "notes": (
            "Baseline train/val use branch-7 rho=1 normal PF samples only. "
            "Test reuses the fixed branch-7 rho=0 split from the B benchmark."
        ),
    }
    write_json(output_dir / "meta.json", meta)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
