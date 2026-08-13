#!/usr/bin/env python3
"""Build a branch-7 endpoint baseline from rho=0 and rho=1 PF banks.

The training split contains the same number of branch-7 rho=0 and rho=1
samples, using load settings that are disjoint from the fixed branch-7 rho=0
test split used by the rho6/rho11 experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def rho_key(value: float | str) -> str:
    return f"{round(float(value), 10):.10g}"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def load_branch_records(source_root: Path, branch_id: str, rho: float) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for rec in read_jsonl(source_root / "manifests" / "samples.jsonl"):
        if str(rec.get("branch_id")) != str(branch_id):
            continue
        if rho_key(rec.get("rho")) != rho_key(rho):
            continue
        load_id = int(rec["load_setting_id"])
        if load_id in records:
            raise RuntimeError(
                f"Duplicate branch={branch_id}, rho={rho_key(rho)} sample for "
                f"load_setting_id={load_id} in {source_root}"
            )
        rec = dict(rec)
        rec["_source_root"] = str(source_root)
        records[load_id] = rec
    return records


def fixed_test_records(fixed_test_root: Path, branch_id: str) -> list[dict]:
    split = read_json(fixed_test_root / "splits" / "test.json")
    records = []
    for rec in split.get("records", []):
        if str(rec.get("branch_id")) != str(branch_id):
            raise RuntimeError(f"Fixed test has branch {rec.get('branch_id')}, expected {branch_id}")
        if rho_key(rec.get("rho")) != "0":
            raise RuntimeError("Fixed test split contains nonzero rho")
        rec = dict(rec)
        rec["_source_root"] = str(fixed_test_root)
        records.append(rec)
    if not records:
        raise RuntimeError(f"No fixed test records found in {fixed_test_root}")
    return records


def make_output_record(
    source_rec: dict,
    sample_id: int,
    out_rel: str,
    split: str,
    source_root: Path,
    experiment_name: str,
) -> dict:
    out_rec = dict(source_rec)
    out_rec.pop("_source_root", None)
    out_rec["sample_id"] = sample_id
    out_rec["file"] = out_rel
    out_rec["split"] = split
    out_rec["source_dataset_root"] = str(source_root)
    out_rec["source_sample_id"] = source_rec["sample_id"]
    out_rec["source_sample_file"] = source_rec["file"]
    out_rec["source_load_setting_id"] = source_rec["load_setting_id"]
    out_rec["selected_branch_id"] = source_rec["branch_id"]
    out_rec["experiment_name"] = experiment_name
    return out_rec


def split_load_settings(records: list[dict], split: str) -> list[dict]:
    by_load: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_load[int(rec["load_setting_id"])].append(rec)

    load_settings = []
    for load_id in sorted(by_load):
        rows = by_load[load_id]
        sample_ids = [int(row["sample_id"]) for row in rows]
        branches = sorted({str(row["branch_id"]) for row in rows}, key=int)
        rhos = sorted({rho_key(row["rho"]) for row in rows}, key=float)
        load_settings.append(
            {
                "split": split,
                "load_setting_id": load_id,
                "sample_ids": sample_ids,
                "selected_branch_ids": ",".join(branches),
                "rho_values": ",".join(rhos),
                "num_samples": len(rows),
            }
        )
    return load_settings


def write_root(
    output_dir: Path,
    split_sources: dict[str, list[dict]],
    meta: dict,
    clean: bool,
    copy_raw: bool,
) -> None:
    if output_dir.exists():
        if not clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "manifests", "splits", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    all_records = []
    load_rows = []
    source_map_rows = []
    split_records: dict[str, list[dict]] = {}
    sample_id = 1
    experiment_name = output_dir.name

    for split in ["train", "val", "test"]:
        out_records = []
        for source_rec in split_sources[split]:
            source_root = Path(source_rec["_source_root"])
            out_rel = f"raw/sample_{sample_id}.json"
            link_or_copy(source_root / source_rec["file"], output_dir / out_rel, copy_raw)
            out_rec = make_output_record(
                source_rec=source_rec,
                sample_id=sample_id,
                out_rel=out_rel,
                split=split,
                source_root=source_root,
                experiment_name=experiment_name,
            )
            out_records.append(out_rec)
            all_records.append(out_rec)
            source_map_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "load_setting_id": out_rec["load_setting_id"],
                    "branch_id": out_rec["branch_id"],
                    "rho": rho_key(out_rec["rho"]),
                    "source_dataset_root": out_rec["source_dataset_root"],
                    "source_sample_id": out_rec["source_sample_id"],
                    "source_load_setting_id": out_rec["source_load_setting_id"],
                    "source_file": out_rec["source_sample_file"],
                }
            )
            sample_id += 1
        split_records[split] = out_records
        split_loads = split_load_settings(out_records, split)
        load_rows.extend(split_loads)
        write_json(
            output_dir / "splits" / f"{split}.json",
            {
                "split": split,
                "records": out_records,
                "load_settings": split_loads,
                "failures": [],
            },
        )

    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", load_rows)
    (output_dir / "manifests" / "failures.jsonl").touch()

    summary_rows = []
    for split, records in split_records.items():
        loads = {int(rec["load_setting_id"]) for rec in records}
        branches = sorted({str(rec["branch_id"]) for rec in records}, key=int)
        rhos = sorted({rho_key(rec["rho"]) for rec in records}, key=float)
        rho_counts = Counter(rho_key(rec["rho"]) for rec in records)
        summary_rows.append(
            {
                "split": split,
                "samples": len(records),
                "load_settings": len(loads),
                "branches": len(branches),
                "branch_ids": ",".join(branches),
                "rho_values": ",".join(rhos),
                "rho_counts": ";".join(f"{rho}:{rho_counts[rho]}" for rho in sorted(rho_counts, key=float)),
            }
        )

    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        summary_rows,
        ["split", "samples", "load_settings", "branches", "branch_ids", "rho_values", "rho_counts"],
    )
    write_csv(
        output_dir / "analysis" / "source_map.csv",
        source_map_rows,
        [
            "split",
            "sample_id",
            "load_setting_id",
            "branch_id",
            "rho",
            "source_dataset_root",
            "source_sample_id",
            "source_load_setting_id",
            "source_file",
        ],
    )

    train_loads = {int(rec["load_setting_id"]) for rec in split_records["train"]}
    val_loads = {int(rec["load_setting_id"]) for rec in split_records["val"]}
    test_loads = {int(rec["load_setting_id"]) for rec in split_records["test"]}
    overlaps = {
        "train_val": sorted(train_loads & val_loads),
        "train_test": sorted(train_loads & test_loads),
        "val_test": sorted(val_loads & test_loads),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Split load settings overlap: {overlaps}")

    train_counts = Counter(rho_key(rec["rho"]) for rec in split_records["train"])
    if train_counts.get("0", 0) != train_counts.get("1", 0):
        raise RuntimeError(f"Train rho counts are not balanced: {dict(train_counts)}")
    if set(train_counts) != {"0", "1"}:
        raise RuntimeError(f"Train split is not endpoint-only: {dict(train_counts)}")
    if {rho_key(rec["rho"]) for rec in split_records["test"]} != {"0"}:
        raise RuntimeError("Test split is not rho=0 only")

    meta = dict(meta)
    meta["selection"] = summary_rows
    meta["split_load_overlaps"] = overlaps
    write_json(output_dir / "meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho0-bank", required=True)
    parser.add_argument("--rho1-bank", required=True)
    parser.add_argument("--fixed-test-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--train-loads", type=int, default=20988)
    parser.add_argument("--val-loads", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    rho0_bank = Path(args.rho0_bank).resolve()
    rho1_bank = Path(args.rho1_bank).resolve()
    fixed_test_root = Path(args.fixed_test_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    rho0_records = load_branch_records(rho0_bank, args.branch_id, 0.0)
    rho1_records = load_branch_records(rho1_bank, args.branch_id, 1.0)
    test_records = fixed_test_records(fixed_test_root, args.branch_id)
    fixed_test_loads = {int(rec["load_setting_id"]) for rec in test_records}

    common_loads = sorted(set(rho0_records) & set(rho1_records) - fixed_test_loads)
    required = args.train_loads + args.val_loads
    if len(common_loads) < required:
        raise RuntimeError(
            f"Need {required} common rho0/rho1 non-test loads, found {len(common_loads)}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(common_loads)
    val_loads = sorted(common_loads[: args.val_loads])
    train_loads = sorted(common_loads[args.val_loads: required])

    train_sources = []
    val_sources = []
    for load_id in train_loads:
        train_sources.extend([rho0_records[load_id], rho1_records[load_id]])
    for load_id in val_loads:
        val_sources.extend([rho0_records[load_id], rho1_records[load_id]])

    write_root(
        output_dir=output_dir,
        split_sources={
            "train": train_sources,
            "val": val_sources,
            "test": test_records,
        },
        meta={
            "experiment_name": output_dir.name,
            "branch_id": str(args.branch_id),
            "rho0_bank": str(rho0_bank),
            "rho1_bank": str(rho1_bank),
            "fixed_test_root": str(fixed_test_root),
            "seed": args.seed,
            "train_loads": len(train_loads),
            "val_loads": len(val_loads),
            "target": "PF_solution(rho)",
            "train_design": "endpoint mix: each selected load_setting contributes branch7 rho=0 and rho=1",
            "leakage_rule": "train/val use only common rho0/rho1 loads outside the fixed test load_setting_id set",
        },
        clean=args.clean,
        copy_raw=args.copy_raw,
    )

    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
