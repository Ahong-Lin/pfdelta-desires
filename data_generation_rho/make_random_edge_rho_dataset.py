#!/usr/bin/env python3
"""Build a random-edge/random-rho dataset from existing PF-rho banks.

This script does not solve new PF instances.  It hard-links successful samples
from one or more source banks and selects either:

* one random successful (branch, rho) sample per unique load_setting, or
* random successful samples with load reuse enabled.
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


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def parse_csv_set(text: str | None) -> set[str] | None:
    if not text:
        return None
    values = {item.strip() for item in text.split(",") if item.strip()}
    return values if values else None


def split_load_ids(root: Path, split: str) -> set[int]:
    path = root / "splits" / f"{split}.json"
    if not path.exists():
        return set()
    payload = read_json(path)
    return {int(record["load_setting_id"]) for record in payload.get("records", [])}


def excluded_load_ids(roots: list[str], splits: list[str]) -> set[int]:
    excluded: set[int] = set()
    for root_text in roots:
        root = Path(root_text).resolve()
        for split in splits:
            excluded |= split_load_ids(root, split)
    return excluded


def source_records_from_bank(
    source_bank: Path,
    allowed_branches: set[str] | None,
    allowed_rhos: set[str] | None,
    excluded_loads: set[int],
) -> tuple[list[dict], dict[int, dict]]:
    samples_path = source_bank / "manifests" / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {samples_path}")

    records = []
    for rec in read_jsonl(samples_path):
        load_id = int(rec["load_setting_id"])
        if load_id in excluded_loads:
            continue
        branch_id = str(rec["branch_id"])
        rho = rho_key(rec["rho"])
        if allowed_branches is not None and branch_id not in allowed_branches:
            continue
        if allowed_rhos is not None and rho not in allowed_rhos:
            continue
        rec["_source_bank"] = str(source_bank)
        records.append(rec)

    load_records = {}
    load_path = source_bank / "manifests" / "load_settings.jsonl"
    if load_path.exists():
        for rec in read_jsonl(load_path):
            load_id = int(rec["load_setting_id"])
            if load_id not in excluded_loads:
                load_records[load_id] = rec

    return records, load_records


def source_records(
    source_banks: list[Path],
    allowed_branches: set[str] | None,
    allowed_rhos: set[str] | None,
    excluded_loads: set[int],
) -> tuple[list[dict], dict[int, dict]]:
    records: list[dict] = []
    load_records: dict[int, dict] = {}
    for source_bank in source_banks:
        bank_records, bank_load_records = source_records_from_bank(
            source_bank,
            allowed_branches,
            allowed_rhos,
            excluded_loads,
        )
        records.extend(bank_records)
        load_records.update(bank_load_records)
    return records, load_records


def choose_unique_loads(records: list[dict], count: int, rng: random.Random) -> list[dict]:
    by_load: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_load[int(rec["load_setting_id"])].append(rec)

    load_ids = sorted(by_load)
    if len(load_ids) < count:
        raise RuntimeError(
            f"Need {count} unique load_setting values, found {len(load_ids)}. "
            "Use --allow-load-reuse for sample-count training, or generate more PF-rho samples."
        )

    selected_loads = rng.sample(load_ids, count)
    return [rng.choice(by_load[load_id]) for load_id in selected_loads]


def choose_records(records: list[dict], count: int, rng: random.Random) -> list[dict]:
    if len(records) < count:
        raise RuntimeError(f"Need {count} records, found {len(records)}.")
    return rng.sample(records, count)


def make_out_record(source_rec: dict, sample_id: int, out_rel: str, split: str) -> dict:
    rec = dict(source_rec)
    source_bank = Path(rec.pop("_source_bank"))
    rec["sample_id"] = sample_id
    rec["file"] = out_rel
    rec["split"] = split
    rec["source_dataset_root"] = str(source_bank)
    rec["source_sample_id"] = source_rec["sample_id"]
    rec["source_sample_file"] = source_rec["file"]
    rec["source_load_setting_id"] = source_rec["load_setting_id"]
    rec["selected_branch_id"] = source_rec["branch_id"]
    rec["experiment_name"] = "random_edge_random_rho"
    return rec


def prepare_output(output_dir: Path, clean: bool) -> None:
    if output_dir.exists():
        if not clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def write_dataset(
    output_dir: Path,
    split_sources: dict[str, list[dict]],
    load_records: dict[int, dict],
    clean: bool,
    copy_raw: bool,
    meta: dict,
) -> None:
    prepare_output(output_dir, clean)

    all_records = []
    source_map = []
    next_sample_id = 1

    for split in ["train", "val", "test"]:
        out_records = []
        load_rows = []
        for source_rec in split_sources.get(split, []):
            source_bank = Path(source_rec["_source_bank"])
            out_rel = f"raw/sample_{next_sample_id}.json"
            link_or_copy(source_bank / source_rec["file"], output_dir / out_rel, copy_raw)
            out_rec = make_out_record(source_rec, next_sample_id, out_rel, split)
            out_records.append(out_rec)
            all_records.append(out_rec)
            source_map.append(
                {
                    "split": split,
                    "sample_id": next_sample_id,
                    "load_setting_id": int(source_rec["load_setting_id"]),
                    "branch_id": str(source_rec["branch_id"]),
                    "rho": rho_key(source_rec["rho"]),
                    "source_dataset_root": str(source_bank),
                    "source_sample_id": source_rec["sample_id"],
                    "source_file": source_rec["file"],
                }
            )
            next_sample_id += 1

        for rec in out_records:
            load_id = int(rec["load_setting_id"])
            load_row = dict(load_records.get(load_id, {"load_setting_id": load_id}))
            load_row["split"] = split
            load_row["load_setting_id"] = load_id
            load_row["selected_branch_ids"] = str(rec["branch_id"])
            load_row["rho_values"] = rho_key(rec["rho"])
            load_row["sample_ids"] = [rec["sample_id"]]
            load_rows.append(load_row)

        write_json(
            output_dir / "splits" / f"{split}.json",
            {"split": split, "records": out_records, "load_settings": load_rows, "failures": []},
        )

    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", [
        row for split in ["train", "val", "test"]
        for row in read_json(output_dir / "splits" / f"{split}.json").get("load_settings", [])
    ])
    (output_dir / "manifests" / "failures.jsonl").touch()

    summary = []
    branch_counts = Counter()
    rho_counts = Counter()
    for split, rows in split_sources.items():
        loads = {int(row["load_setting_id"]) for row in rows}
        branches = Counter(str(row["branch_id"]) for row in rows)
        rhos = Counter(rho_key(row["rho"]) for row in rows)
        summary.append(
            {
                "split": split,
                "samples": len(rows),
                "unique_load_settings": len(loads),
                "branches": len(branches),
                "rho_values": len(rhos),
                "branch_counts": ";".join(f"{k}:{branches[k]}" for k in sorted(branches, key=int)),
                "rho_counts": ";".join(f"{k}:{rhos[k]}" for k in sorted(rhos, key=float)),
            }
        )
        branch_counts.update(branches)
        rho_counts.update(rhos)

    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        summary,
        ["split", "samples", "unique_load_settings", "branches", "rho_values", "branch_counts", "rho_counts"],
    )
    write_csv(
        output_dir / "analysis" / "source_map.csv",
        source_map,
        ["split", "sample_id", "load_setting_id", "branch_id", "rho", "source_dataset_root", "source_sample_id", "source_file"],
    )
    write_json(output_dir / "meta.json", {**meta, "summary": summary})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=14000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--branches", help="Optional comma-separated branch ids.")
    parser.add_argument("--rhos", help="Optional comma-separated rho values.")
    parser.add_argument("--exclude-root", action="append", default=[])
    parser.add_argument("--exclude-splits", default="test")
    parser.add_argument("--allow-load-reuse", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    source_banks = [Path(path).resolve() for path in args.source_bank]
    output_dir = Path(args.output_dir).resolve()
    rng = random.Random(args.seed)
    allowed_branches = parse_csv_set(args.branches)
    allowed_rhos = parse_csv_set(args.rhos)
    if allowed_rhos is not None:
        allowed_rhos = {rho_key(rho) for rho in allowed_rhos}
    exclude_splits = [item.strip() for item in args.exclude_splits.split(",") if item.strip()]
    excluded = excluded_load_ids(args.exclude_root, exclude_splits)

    records, load_records = source_records(source_banks, allowed_branches, allowed_rhos, excluded)
    if not records:
        raise RuntimeError("No eligible source records found.")

    split_sources = {}
    remaining = list(records)
    for split, count in [("train", args.train_count), ("val", args.val_count), ("test", args.test_count)]:
        if count <= 0:
            split_sources[split] = []
            continue
        if args.allow_load_reuse:
            selected = choose_records(remaining, count, rng)
            selected_keys = {id(row) for row in selected}
            remaining = [row for row in remaining if id(row) not in selected_keys]
        else:
            selected = choose_unique_loads(remaining, count, rng)
            selected_loads = {int(row["load_setting_id"]) for row in selected}
            remaining = [row for row in remaining if int(row["load_setting_id"]) not in selected_loads]
        split_sources[split] = selected

    meta = {
        "source_banks": [str(path) for path in source_banks],
        "selection": "one random successful branch/rho sample per load_setting"
        if not args.allow_load_reuse
        else "random successful branch/rho samples, load reuse allowed",
        "seed": args.seed,
        "allowed_branches": sorted(allowed_branches, key=int) if allowed_branches else "all",
        "allowed_rhos": sorted(allowed_rhos, key=float) if allowed_rhos else "all",
        "excluded_load_settings": len(excluded),
        "allow_load_reuse": args.allow_load_reuse,
    }
    write_dataset(output_dir, split_sources, load_records, args.clean, args.copy_raw, meta)
    print(json.dumps({"output_dir": str(output_dir), "meta": meta}, indent=2))


if __name__ == "__main__":
    main()
