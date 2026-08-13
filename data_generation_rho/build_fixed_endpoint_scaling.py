#!/usr/bin/env python3
"""Freeze and assemble the case30 endpoint-scaling data protocol.

Protocol v1 uses disjoint contiguous load-ID ranges:
  val:   1..2000
  test:  2001..8800
  train: 8801 onward

Every load contributes one full-topology rho=1 record and one single-line
rho=0 record.  Branch schedules are globally balanced and deterministic.
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
from typing import Iterable


NORMAL_LOOP_BRANCHES = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "14", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
    "29", "30", "31", "32", "33", "35", "37", "38", "39", "40", "41",
]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, separators=(",", ":"))
            handle.write("\n")


def hard_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError as error:
        raise RuntimeError(f"Hard-link failed: {src} -> {dst}: {error}") from error


def bank_availability(bank_root: Path, load_ids: list[int]) -> dict[int, set[str]]:
    """Return only load/branch pairs with a successful rho=0 PF record."""
    wanted_loads = set(load_ids)
    available = {load_id: set() for load_id in load_ids}
    manifest = bank_root / "manifests" / "samples.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            load_id = int(record["load_setting_id"])
            if load_id in wanted_loads and rho_key(record["rho"]) == "0":
                available[load_id].add(str(record["branch_id"]))
    missing = [load_id for load_id, branches in available.items() if not branches]
    if missing:
        raise RuntimeError(f"No successful rho=0 branch in bank for loads: {missing[:10]}")
    return available


def balanced_schedule(load_ids: list[int], seed: int, available: dict[int, set[str]] | None = None) -> list[dict]:
    """Assign branches exactly evenly while respecting known rho=0 feasibility."""
    target = Counter({
        branch: len(load_ids) // len(NORMAL_LOOP_BRANCHES) + (index < len(load_ids) % len(NORMAL_LOOP_BRANCHES))
        for index, branch in enumerate(NORMAL_LOOP_BRANCHES)
    })
    for attempt in range(100):
        rng = random.Random(seed + attempt)
        remaining = Counter(target)
        assignments: dict[int, str] = {}
        # The rare loads with fewer converged outages must be assigned first.
        ordered_loads = sorted(
            load_ids,
            key=lambda load_id: (len(available[load_id]) if available else len(NORMAL_LOOP_BRANCHES), rng.random()),
        )
        for load_id in ordered_loads:
            candidates = available[load_id] if available else set(NORMAL_LOOP_BRANCHES)
            candidates = [branch for branch in candidates if remaining[branch] > 0]
            if not candidates:
                break
            max_remaining = max(remaining[branch] for branch in candidates)
            best = [branch for branch in candidates if remaining[branch] == max_remaining]
            branch = rng.choice(best)
            assignments[load_id] = branch
            remaining[branch] -= 1
        if len(assignments) == len(load_ids) and not any(remaining.values()):
            return [
                {"load_setting_id": load_id, "branch_id": assignments[load_id]}
                for load_id in load_ids
            ]
    raise RuntimeError("Could not make an exact branch-balanced feasible schedule after 100 attempts")


def plan_paths(protocol_dir: Path) -> dict[str, Path]:
    plans = protocol_dir / "plans"
    return {
        "val": plans / "val_2000.jsonl",
        "test": plans / "test_6800.jsonl",
        "train": plans / "train_master_40000.jsonl",
    }


def freeze_protocol(args: argparse.Namespace) -> None:
    protocol_dir = Path(args.protocol_dir).resolve()
    if protocol_dir.exists():
        if not args.clean:
            raise FileExistsError(f"Protocol directory exists: {protocol_dir}. Use --clean.")
        shutil.rmtree(protocol_dir)
    protocol_dir.mkdir(parents=True)

    val_ids = list(range(1, 2001))
    test_ids = list(range(2001, 8801))
    train_ids = list(range(8801, 48801))
    bank_root = Path(args.bank_root).resolve()
    paths = plan_paths(protocol_dir)
    write_jsonl(paths["val"], balanced_schedule(val_ids, args.seed + 1, bank_availability(bank_root, val_ids)))
    write_jsonl(paths["test"], balanced_schedule(test_ids, args.seed + 2, bank_availability(bank_root, test_ids)))
    # Only the first 1,200 training loads already exist in the full-rho bank.
    # Select those exact outage pairs from successful bank records; later loads
    # are solved by the planned endpoint generator with a recorded fallback.
    bank_train_ids = [load_id for load_id in train_ids if load_id <= 10000]
    new_train_ids = [load_id for load_id in train_ids if load_id > 10000]
    train_plan = balanced_schedule(bank_train_ids, args.seed + 3, bank_availability(bank_root, bank_train_ids))
    train_plan += balanced_schedule(new_train_ids, args.seed + 4)
    write_jsonl(paths["train"], train_plan)

    write_json(
        protocol_dir / "protocol.json",
        {
            "name": "case30_endpoint_scaling_v1",
            "seed": args.seed,
            "bank_root": str(bank_root),
            "branch_ids": NORMAL_LOOP_BRANCHES,
            "val_load_ids": [1, 2000],
            "test_load_ids": [2001, 8800],
            "train_master_load_ids": [8801, 48800],
            "sizes": {"val_loads": 2000, "test_loads": 6800, "train_master_loads": 40000},
            "design": "Each load has one rho=1 normal record and one branch-balanced rho=0 outage record.",
        },
    )
    print(f"Froze protocol in {protocol_dir}")


def read_plan(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def rho_key(value: float | str) -> str:
    return f"{float(value):.10g}"


def selected_bank_records(bank_root: Path, wanted: dict[tuple[int, str, str], str]) -> dict[str, dict]:
    manifest = bank_root / "manifests" / "samples.jsonl"
    found: dict[str, dict] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (int(record["load_setting_id"]), str(record["branch_id"]), rho_key(record["rho"]))
            role = wanted.get(key)
            if role is not None:
                if role in found:
                    raise RuntimeError(f"Duplicate bank record for {key}")
                found[role] = record
    missing = sorted(set(wanted.values()) - set(found))
    if missing:
        raise RuntimeError(f"Bank is missing {len(missing)} planned samples; first: {missing[:5]}")
    return found


def output_record(source_root: Path, source_record: dict, sample_id: int, split: str, file_name: str, role: str) -> dict:
    record = dict(source_record)
    record.update(
        {
            "sample_id": sample_id,
            "file": file_name,
            "split": split,
            "task_name": "case30_endpoint_scaling_v1",
            "experiment_name": "case30_endpoint_scaling_v1",
            "source_dataset_root": str(source_root),
            "source_sample_id": source_record["sample_id"],
            "source_sample_file": source_record["file"],
            "source_role": role,
        }
    )
    return record


def load_rows(records: list[dict], split: str) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["load_setting_id"])].append(record)
    return [
        {
            "split": split,
            "load_setting_id": load_id,
            "base_sample_id": load_id,
            "sample_ids": [record["sample_id"] for record in samples],
            "selected_branch_ids": ",".join(sorted({str(record["branch_id"]) for record in samples}, key=int)),
            "rho_values": ",".join(sorted({rho_key(record["rho"]) for record in samples}, key=float)),
        }
        for load_id, samples in sorted(grouped.items())
    ]


def write_dataset(output_dir: Path, split_records: dict[str, list[tuple[str, dict]]], source_root: Path, clean: bool) -> None:
    if output_dir.exists():
        if not clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean.")
        shutil.rmtree(output_dir)
    for directory in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / directory).mkdir(parents=True)

    next_id = 1
    all_records: list[dict] = []
    split_outputs: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split in ["train", "val", "test"]:
        for role, source_record in split_records.get(split, []):
            file_name = f"raw/sample_{next_id}.json"
            hard_link(source_root / source_record["file"], output_dir / file_name)
            record = output_record(source_root, source_record, next_id, split, file_name, role)
            split_outputs[split].append(record)
            all_records.append(record)
            next_id += 1

    for split, records in split_outputs.items():
        write_json(output_dir / "splits" / f"{split}.json", {"split": split, "records": records, "load_settings": load_rows(records, split), "failures": []})
    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", [row for split in ["train", "val", "test"] for row in load_rows(split_outputs[split], split)])
    (output_dir / "manifests" / "failures.jsonl").touch()

    with (output_dir / "analysis" / "split_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "samples", "unique_loads", "rho0", "rho1", "branches"])
        writer.writeheader()
        for split, records in split_outputs.items():
            writer.writerow(
                {
                    "split": split,
                    "samples": len(records),
                    "unique_loads": len({int(record["load_setting_id"]) for record in records}),
                    "rho0": sum(rho_key(record["rho"]) == "0" for record in records),
                    "rho1": sum(rho_key(record["rho"]) == "1" for record in records),
                    "branches": ";".join(f"{branch}:{count}" for branch, count in sorted(Counter(str(record["branch_id"]) for record in records).items(), key=lambda item: int(item[0]))),
                }
            )


def build_evaluations(args: argparse.Namespace) -> None:
    protocol_dir = Path(args.protocol_dir).resolve()
    bank_root = Path(args.bank_root).resolve()
    plans = plan_paths(protocol_dir)
    val_plan = read_plan(plans["val"])
    test_plan = read_plan(plans["test"])
    wanted: dict[tuple[int, str, str], str] = {}
    for label, plan, rhos in [("val", val_plan, ["0", "1"]), ("test", test_plan, ["0", "1"])]:
        for row in plan:
            for rho in rhos:
                wanted[(int(row["load_setting_id"]), str(row["branch_id"]), rho)] = f"{label}:{row['load_setting_id']}:{rho}"
    selected = selected_bank_records(bank_root, wanted)

    val_records: list[tuple[str, dict]] = []
    for row in val_plan:
        load_id = int(row["load_setting_id"])
        val_records.append(("normal_rho1", selected[f"val:{load_id}:1"]))
        val_records.append(("outage_rho0", selected[f"val:{load_id}:0"]))
    test0_records = [("outage_rho0", selected[f"test:{int(row['load_setting_id'])}:0"]) for row in test_plan]
    test1_records = [("normal_rho1", selected[f"test:{int(row['load_setting_id'])}:1"]) for row in test_plan]

    output_base = Path(args.output_base).resolve()
    write_dataset(output_base / "rho_pf_scaling_val_endpoint_4000", {"val": val_records}, bank_root, args.clean)
    write_dataset(output_base / "rho_pf_scaling_T34_final_rho0_6800", {"test": test0_records}, bank_root, args.clean)
    write_dataset(output_base / "rho_pf_scaling_T34_final_rho1_6800", {"test": test1_records}, bank_root, args.clean)
    print(f"Built fixed val/test roots under {output_base}")


def records_from_root(root: Path, split: str) -> list[dict]:
    return read_json(root / "splits" / f"{split}.json")["records"]


def assemble_training(args: argparse.Namespace) -> None:
    master_root = Path(args.master_root).resolve()
    val_root = Path(args.val_root).resolve()
    output_base = Path(args.output_base).resolve()
    master_records = records_from_root(master_root, "train")
    val_records = records_from_root(val_root, "val")
    if len(master_records) < max(args.sizes):
        raise RuntimeError(f"Master has {len(master_records)} records; need {max(args.sizes)}")
    if len(val_records) != 4000:
        raise RuntimeError(f"Expected 4000 fixed val records, found {len(val_records)}")

    val_loads = {int(record["load_setting_id"]) for record in val_records}
    for size in args.sizes:
        if size % 2:
            raise ValueError(f"Training size must be even: {size}")
        train_records = master_records[:size]
        train_loads = {int(record["load_setting_id"]) for record in train_records}
        if len(train_loads) != size // 2:
            raise RuntimeError(f"Expected {size // 2} paired train loads, found {len(train_loads)}")
        if train_loads & val_loads:
            raise RuntimeError("Train/val load leakage")
        root = output_base / f"rho_pf_scaling_endpoint_{size // 1000}k"
        if root.exists():
            if not args.clean:
                raise FileExistsError(f"Output exists: {root}. Use --clean.")
            shutil.rmtree(root)
        for directory in ["raw", "splits", "manifests", "analysis"]:
            (root / directory).mkdir(parents=True)

        outputs: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        next_id = 1
        for split, records, source_root, role in [
            ("train", train_records, master_root, "nested_master_train"),
            ("val", val_records, val_root, "fixed_endpoint_val"),
        ]:
            for source_record in records:
                file_name = f"raw/sample_{next_id}.json"
                hard_link(source_root / source_record["file"], root / file_name)
                outputs[split].append(output_record(source_root, source_record, next_id, split, file_name, role))
                next_id += 1
        for split in ["train", "val", "test"]:
            write_json(root / "splits" / f"{split}.json", {"split": split, "records": outputs[split], "load_settings": load_rows(outputs[split], split), "failures": []})
        write_jsonl(root / "manifests" / "samples.jsonl", outputs["train"] + outputs["val"])
        write_jsonl(root / "manifests" / "load_settings.jsonl", load_rows(outputs["train"], "train") + load_rows(outputs["val"], "val"))
        (root / "manifests" / "failures.jsonl").touch()
        write_json(
            root / "meta.json",
            {
                "protocol": "case30_endpoint_scaling_v1",
                "train_samples": size,
                "train_loads": size // 2,
                "val_samples": len(val_records),
                "val_loads": len(val_loads),
                "design": "Nested prefix of the fixed 80k endpoint master; rho=0/rho=1 are paired by load.",
            },
        )
        print(f"Built {root}")


def assemble_master(args: argparse.Namespace) -> None:
    """Combine bank-backed and newly solved paired endpoints into one 80k master."""
    protocol_dir = Path(args.protocol_dir).resolve()
    bank_root = Path(args.bank_root).resolve()
    solved_root = Path(args.solved_root).resolve()
    output_root = Path(args.output_root).resolve()
    plan = read_plan(plan_paths(protocol_dir)["train"])
    bank_plan = {
        int(row["load_setting_id"]): str(row["branch_id"])
        for row in plan
        if int(row["load_setting_id"]) <= 10000
    }
    bank_records: list[dict] = []
    with (bank_root / "manifests" / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            load_id = int(record["load_setting_id"])
            if (
                load_id in bank_plan
                and str(record["branch_id"]) == bank_plan[load_id]
                and rho_key(record["rho"]) in {"0", "1"}
            ):
                bank_records.append(record)
    solved_records = records_from_root(solved_root, "train")
    by_source: dict[tuple[int, str, str], tuple[Path, dict]] = {}
    for source_root, records in [(bank_root, bank_records), (solved_root, solved_records)]:
        for record in records:
            key = (int(record["load_setting_id"]), str(record["branch_id"]), rho_key(record["rho"]))
            if key in by_source:
                raise RuntimeError(f"Duplicate source record for {key}")
            by_source[key] = (source_root, record)

    solved_outage_branch: dict[int, str] = {}
    for record in solved_records:
        if rho_key(record["rho"]) != "0":
            continue
        load_id = int(record["load_setting_id"])
        branch_id = str(record["branch_id"])
        if load_id in solved_outage_branch and solved_outage_branch[load_id] != branch_id:
            raise RuntimeError(f"Multiple solved outage branches for load={load_id}")
        solved_outage_branch[load_id] = branch_id

    if output_root.exists():
        if not args.clean:
            raise FileExistsError(f"Output exists: {output_root}. Use --clean.")
        shutil.rmtree(output_root)
    for directory in ["raw", "splits", "manifests", "analysis"]:
        (output_root / directory).mkdir(parents=True)

    records: list[dict] = []
    next_id = 1
    fallbacks: list[dict] = []
    for row in plan:
        load_id = int(row["load_setting_id"])
        planned_branch = str(row["branch_id"])
        actual_branch = solved_outage_branch.get(load_id, planned_branch)
        for rho, role in [("1", "normal_rho1"), ("0", "outage_rho0")]:
            key = (load_id, actual_branch, rho)
            try:
                source_root, source_record = by_source[key]
            except KeyError as error:
                raise RuntimeError(f"Missing {rho=} source for load={load_id}, branch={actual_branch}") from error
            file_name = f"raw/sample_{next_id}.json"
            hard_link(source_root / source_record["file"], output_root / file_name)
            records.append(output_record(source_root, source_record, next_id, "train", file_name, role))
            next_id += 1
        if actual_branch != planned_branch:
            fallbacks.append({"load_setting_id": load_id, "planned_branch_id": planned_branch, "actual_branch_id": actual_branch})

    write_json(output_root / "splits" / "train.json", {"split": "train", "records": records, "load_settings": load_rows(records, "train"), "failures": []})
    for split in ["val", "test"]:
        write_json(output_root / "splits" / f"{split}.json", {"split": split, "records": [], "load_settings": [], "failures": []})
    write_jsonl(output_root / "manifests" / "samples.jsonl", records)
    write_jsonl(output_root / "manifests" / "load_settings.jsonl", load_rows(records, "train"))
    (output_root / "manifests" / "failures.jsonl").touch()
    with (output_root / "analysis" / "branch_fallbacks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["load_setting_id", "planned_branch_id", "actual_branch_id"])
        writer.writeheader()
        writer.writerows(fallbacks)
    write_json(
        output_root / "meta.json",
        {
            "protocol": "case30_endpoint_scaling_v1",
            "samples": len(records),
            "loads": len(records) // 2,
            "source_bank": str(bank_root),
            "source_solved": str(solved_root),
            "fallback_count": len(fallbacks),
            "design": "Ordered paired rho=1/rho=0 endpoint master. Dataset sizes are even-length prefixes.",
        },
    )
    print(f"Built master {output_root}: {len(records)} records, fallbacks={len(fallbacks)}")


def parse_sizes(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def make_planned_schedule(args: argparse.Namespace) -> None:
    """Create a deterministic balanced one-outage-per-load schedule."""
    if args.start_load_id > args.end_load_id:
        raise ValueError("start-load-id must not exceed end-load-id")
    load_ids = list(range(args.start_load_id, args.end_load_id + 1))
    schedule = balanced_schedule(load_ids, args.seed)
    output = Path(args.output).resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Schedule exists: {output}. Use --force to replace it.")
    write_jsonl(output, schedule)
    print(f"Wrote {len(schedule)} branch-balanced planned loads to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--protocol-dir", required=True)
    freeze.add_argument("--bank-root", required=True, help="Existing 10k full-rho bank used to choose feasible val/test outages.")
    freeze.add_argument("--seed", type=int, default=20260731)
    freeze.add_argument("--clean", action="store_true")
    freeze.set_defaults(func=freeze_protocol)

    evaluate = subparsers.add_parser("build-evaluations")
    evaluate.add_argument("--protocol-dir", required=True)
    evaluate.add_argument("--bank-root", required=True)
    evaluate.add_argument("--output-base", required=True)
    evaluate.add_argument("--clean", action="store_true")
    evaluate.set_defaults(func=build_evaluations)

    train = subparsers.add_parser("assemble-training")
    train.add_argument("--master-root", required=True)
    train.add_argument("--val-root", required=True)
    train.add_argument("--output-base", required=True)
    train.add_argument("--sizes", type=parse_sizes, default=[5000, 10000, 20000, 40000, 80000])
    train.add_argument("--clean", action="store_true")
    train.set_defaults(func=assemble_training)

    master = subparsers.add_parser("assemble-master")
    master.add_argument("--protocol-dir", required=True)
    master.add_argument("--bank-root", required=True)
    master.add_argument("--solved-root", required=True)
    master.add_argument("--output-root", required=True)
    master.add_argument("--clean", action="store_true")
    master.set_defaults(func=assemble_master)

    schedule = subparsers.add_parser("make-schedule")
    schedule.add_argument("--start-load-id", type=int, required=True)
    schedule.add_argument("--end-load-id", type=int, required=True)
    schedule.add_argument("--seed", type=int, default=20260801)
    schedule.add_argument("--output", required=True)
    schedule.add_argument("--force", action="store_true")
    schedule.set_defaults(func=make_planned_schedule)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
