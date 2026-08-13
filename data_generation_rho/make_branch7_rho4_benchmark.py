#!/usr/bin/env python3
"""Build branch-7 rho4 benchmark from existing PF-rho banks.

The rho4 experiment uses rho={0,0.3,0.6,1}.  It keeps the fixed branch-7
rho=0 test split from the existing rho6 benchmark and avoids using those
loads for train/val.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path


RHOS_4 = [0.0, 0.3, 0.6, 1.0]


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


def load_fixed_test_loads(fixed_test_root: Path) -> list[int]:
    split = read_json(fixed_test_root / "splits" / "test.json")
    loads = sorted({int(record["load_setting_id"]) for record in split["records"]})
    if not loads:
        raise RuntimeError(f"No test loads in {fixed_test_root}")
    return loads


def add_bank_records(
    bank_root: Path,
    branch_id: str,
    wanted_rhos: set[str],
    sample_index: dict[tuple[int, str], dict],
    load_records: dict[int, dict],
) -> None:
    load_path = bank_root / "manifests" / "load_settings.jsonl"
    if load_path.exists():
        for rec in read_jsonl(load_path):
            load_id = int(rec["load_setting_id"])
            load_records.setdefault(load_id, {**rec, "_source_load_bank": str(bank_root)})

    for rec in read_jsonl(bank_root / "manifests" / "samples.jsonl"):
        if str(rec.get("branch_id")) != str(branch_id):
            continue
        rho = rho_key(rec["rho"])
        if rho not in wanted_rhos:
            continue
        load_id = int(rec["load_setting_id"])
        key = (load_id, rho)
        if key in sample_index:
            continue
        sample_index[key] = {**rec, "_source_bank": str(bank_root)}


def complete_loads(sample_index: dict[tuple[int, str], dict], rhos: list[float]) -> list[int]:
    required = {rho_key(rho) for rho in rhos}
    loads = {}
    for load_id, rho in sample_index:
        loads.setdefault(load_id, set()).add(rho)
    return sorted(load_id for load_id, seen in loads.items() if required <= seen)


def ensure_required_samples(load_ids: list[int], sample_index: dict[tuple[int, str], dict], rhos: list[float]) -> None:
    missing = []
    for load_id in load_ids:
        for rho in rhos:
            key = (load_id, rho_key(rho))
            if key not in sample_index:
                missing.append(key)
    if missing:
        preview = ", ".join(f"{load}:{rho}" for load, rho in missing[:20])
        raise RuntimeError(f"Missing {len(missing)} required samples. First: {preview}")


def make_out_record(
    source_rec: dict,
    sample_id: int,
    out_rel: str,
    split: str,
    experiment_name: str,
) -> dict:
    rec = dict(source_rec)
    source_bank = rec.pop("_source_bank")
    rec["sample_id"] = sample_id
    rec["file"] = out_rel
    rec["split"] = split
    rec["source_dataset_root"] = source_bank
    rec["source_sample_id"] = source_rec["sample_id"]
    rec["source_sample_file"] = source_rec["file"]
    rec["source_load_setting_id"] = source_rec["load_setting_id"]
    rec["selected_branch_id"] = source_rec["branch_id"]
    rec["experiment_name"] = experiment_name
    return rec


def build_split(
    output_dir: Path,
    split: str,
    load_ids: list[int],
    rhos: list[float],
    sample_index: dict[tuple[int, str], dict],
    load_records: dict[int, dict],
    next_sample_id: int,
    branch_id: str,
    experiment_name: str,
    copy_raw: bool,
) -> tuple[list[dict], list[dict], list[dict], int]:
    records = []
    split_load_records = []
    source_map_rows = []

    for load_id in load_ids:
        sample_ids = []
        for rho in rhos:
            source_rec = sample_index[(load_id, rho_key(rho))]
            source_bank = Path(source_rec["_source_bank"])
            out_rel = f"raw/sample_{next_sample_id}.json"
            link_or_copy(source_bank / source_rec["file"], output_dir / out_rel, copy_raw)
            out_rec = make_out_record(
                source_rec=source_rec,
                sample_id=next_sample_id,
                out_rel=out_rel,
                split=split,
                experiment_name=experiment_name,
            )
            records.append(out_rec)
            sample_ids.append(next_sample_id)
            next_sample_id += 1

        load_rec = dict(load_records.get(load_id, {"load_setting_id": load_id}))
        load_rec.pop("_source_load_bank", None)
        load_rec["split"] = split
        load_rec["load_setting_id"] = load_id
        load_rec["selected_branch_id"] = branch_id
        load_rec["source_load_setting_id"] = load_id
        load_rec["sample_ids"] = sample_ids
        load_rec["num_selected_rhos"] = len(rhos)
        split_load_records.append(load_rec)
        source_map_rows.append(
            {
                "split": split,
                "load_setting_id": load_id,
                "selected_branch_id": branch_id,
                "rho_values": ",".join(rho_key(rho) for rho in rhos),
                "sample_ids": ",".join(str(sample_id) for sample_id in sample_ids),
            }
        )

    return records, split_load_records, source_map_rows, next_sample_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", action="append", required=True)
    parser.add_argument("--fixed-test-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--train-loads", type=int, default=10494)
    parser.add_argument("--val-loads", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "manifests", "splits", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    wanted_rhos = {rho_key(rho) for rho in RHOS_4}
    sample_index: dict[tuple[int, str], dict] = {}
    load_records: dict[int, dict] = {}
    for bank in args.bank:
        add_bank_records(
            bank_root=Path(bank).resolve(),
            branch_id=args.branch_id,
            wanted_rhos=wanted_rhos,
            sample_index=sample_index,
            load_records=load_records,
        )

    fixed_test_root = Path(args.fixed_test_root).resolve()
    test_loads = load_fixed_test_loads(fixed_test_root)
    test_set = set(test_loads)
    complete = complete_loads(sample_index, RHOS_4)
    eligible = [load_id for load_id in complete if load_id not in test_set]

    needed = args.train_loads + args.val_loads
    if len(eligible) < needed:
        raise RuntimeError(
            f"Need {needed} complete non-test load settings, found {len(eligible)}. "
            f"Complete={len(complete)}, fixed_test={len(test_set)}"
        )

    first_bank_eligible = [load_id for load_id in eligible if load_id <= 10000]
    extra_eligible = [load_id for load_id in eligible if load_id > 10000]
    rng = random.Random(args.seed)
    rng.shuffle(first_bank_eligible)
    rng.shuffle(extra_eligible)

    # Reuse as many pre-10000 non-test loads as possible, then fill with extras.
    selected_pool = first_bank_eligible + extra_eligible
    train_loads = selected_pool[: args.train_loads]
    val_loads = selected_pool[args.train_loads: args.train_loads + args.val_loads]
    selected = train_loads + val_loads + test_loads
    ensure_required_samples(train_loads + val_loads, sample_index, RHOS_4)
    ensure_required_samples(test_loads, sample_index, [0.0])

    all_records = []
    all_load_rows = []
    source_map_rows = []
    summary_rows = []
    next_sample_id = 1
    experiment_name = output_dir.name

    split_specs = {
        "train": (train_loads, RHOS_4),
        "val": (val_loads, RHOS_4),
        "test": (test_loads, [0.0]),
    }
    for split in ["train", "val", "test"]:
        load_ids, rhos = split_specs[split]
        records, load_rows, map_rows, next_sample_id = build_split(
            output_dir=output_dir,
            split=split,
            load_ids=load_ids,
            rhos=rhos,
            sample_index=sample_index,
            load_records=load_records,
            next_sample_id=next_sample_id,
            branch_id=args.branch_id,
            experiment_name=experiment_name,
            copy_raw=args.copy_raw,
        )
        write_json(
            output_dir / "splits" / f"{split}.json",
            {"split": split, "records": records, "load_settings": load_rows, "failures": []},
        )
        all_records.extend(records)
        all_load_rows.extend(load_rows)
        source_map_rows.extend(map_rows)
        summary_rows.append(
            {
                "split": split,
                "load_settings": len(load_ids),
                "rho_values_per_load": len(rhos),
                "samples": len(records),
                "rho_values": ",".join(rho_key(rho) for rho in rhos),
                "max_load_id": max(load_ids),
                "min_load_id": min(load_ids),
            }
        )

    train_set = set(train_loads)
    val_set = set(val_loads)
    overlaps = {
        "train_val": len(train_set & val_set),
        "train_test": len(train_set & test_set),
        "val_test": len(val_set & test_set),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Split load overlap detected: {overlaps}")

    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", all_load_rows)
    (output_dir / "manifests" / "failures.jsonl").touch()
    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        summary_rows,
        ["split", "load_settings", "rho_values_per_load", "samples", "rho_values", "min_load_id", "max_load_id"],
    )
    write_csv(
        output_dir / "analysis" / "load_source_map.csv",
        source_map_rows,
        ["split", "load_setting_id", "selected_branch_id", "rho_values", "sample_ids"],
    )
    write_json(
        output_dir / "meta.json",
        {
            "experiment_name": experiment_name,
            "source_banks": [str(Path(bank).resolve()) for bank in args.bank],
            "fixed_test_root": str(fixed_test_root),
            "branch_id": args.branch_id,
            "train_loads": len(train_loads),
            "val_loads": len(val_loads),
            "test_loads": len(test_loads),
            "rhos": RHOS_4,
            "split_load_overlaps": overlaps,
            "complete_loads_total": len(complete),
            "complete_non_test_loads": len(eligible),
            "selected_train_extra_loads_gt10000": sum(load_id > 10000 for load_id in train_loads),
            "selected_val_extra_loads_gt10000": sum(load_id > 10000 for load_id in val_loads),
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": summary_rows,
                "overlaps": overlaps,
                "selected_train_extra_loads_gt10000": sum(load_id > 10000 for load_id in train_loads),
                "selected_val_extra_loads_gt10000": sum(load_id > 10000 for load_id in val_loads),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
