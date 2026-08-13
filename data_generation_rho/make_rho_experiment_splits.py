#!/usr/bin/env python3
"""Build CANOS-ready PFDelta-rho experiment datasets from fine rho banks.

The script copies selected raw samples into new standalone PFDelta-rho
directories with:

    raw/sample_*.json
    manifests/{samples,load_settings,failures}.jsonl
    splits/{train,val,test}.json

It does not modify source banks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


SOURCE_BANKS = [
    ("old30", "case30/rho_bank_normal_loop_edges_30_0.01"),
    ("extra50", "case30/rho_bank_normal_loop_edges_extra50_0.01"),
    ("new50", "case30/rho_bank_normal_loop_edges_new50_0.01"),
]

EXPERIMENTS = [
    ("60L80R", 60, 80),
    ("80L60R", 80, 60),
    ("100L48R", 100, 48),
]


def rho_key(value: float) -> str:
    return f"{round(float(value), 10):.10g}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            json.dump(row, f, separators=(",", ":"))
            f.write("\n")


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")


def evenly_spaced_rhos(count: int) -> list[float]:
    if count < 2:
        raise ValueError("rho count must be at least 2 so rho=0 and rho=1 are included")

    grid = [round(i / 100, 10) for i in range(101)]
    chosen = []
    used = set()
    for i in range(count):
        idx = round(i * 100 / (count - 1))
        idx = max(0, min(100, idx))
        if idx not in used:
            chosen.append(grid[idx])
            used.add(idx)

    if len(chosen) < count:
        for idx in range(101):
            if idx not in used:
                chosen.append(grid[idx])
                used.add(idx)
                if len(chosen) == count:
                    break

    return sorted(chosen)


def load_complete_curves(source_root: Path, source_banks: list[tuple[str, str]], branch_id: str):
    complete_loads = []
    sample_index = {}
    load_records = {}

    expected_rhos = {rho_key(i / 100) for i in range(101)}

    for source_tag, rel_dir in source_banks:
        bank_dir = source_root / rel_dir
        full_curves_path = bank_dir / "analysis" / "fine_full_rho_success_curves.csv"
        if not full_curves_path.exists():
            raise FileNotFoundError(f"Missing full-curve CSV: {full_curves_path}")

        complete_load_ids = set()
        with full_curves_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row["branch_id"]) == branch_id:
                    complete_load_ids.add(int(row["load_setting_id"]))

        for rec in read_jsonl(bank_dir / "manifests" / "load_settings.jsonl"):
            load_id = int(rec["load_setting_id"])
            if load_id in complete_load_ids:
                load_key = (source_tag, load_id)
                load_records[load_key] = rec

        curve_rhos = {}
        for rec in read_jsonl(bank_dir / "manifests" / "samples.jsonl"):
            load_id = int(rec["load_setting_id"])
            if load_id not in complete_load_ids:
                continue
            if str(rec["branch_id"]) != branch_id:
                continue
            key = (source_tag, load_id, rho_key(rec["rho"]))
            sample_index[key] = rec
            curve_key = (source_tag, load_id)
            curve_rhos.setdefault(curve_key, set()).add(rho_key(rec["rho"]))

        for load_id in sorted(complete_load_ids):
            load_key = (source_tag, load_id)
            if curve_rhos.get(load_key, set()) >= expected_rhos:
                complete_loads.append(load_key)

    return complete_loads, load_records, sample_index


def copy_sample(
    source_root: Path,
    source_banks: dict[str, str],
    output_dir: Path,
    source_tag: str,
    source_rec: dict,
    new_sample_id: int,
    new_load_setting_id: int,
    split: str,
    experiment_name: str,
) -> dict:
    source_file = source_root / source_banks[source_tag] / source_rec["file"]
    with source_file.open() as f:
        sample = json.load(f)

    rho_meta = sample.setdefault("rho_meta", {})
    rho_meta["sample_id"] = new_sample_id
    rho_meta["load_setting_id"] = new_load_setting_id
    rho_meta["split"] = split
    rho_meta["task_name"] = experiment_name
    rho_meta["source_bank"] = source_tag
    rho_meta["source_file"] = source_rec["file"]
    rho_meta["source_sample_id"] = source_rec["sample_id"]
    rho_meta["source_load_setting_id"] = source_rec["load_setting_id"]

    out_rel = f"raw/sample_{new_sample_id}.json"
    out_path = output_dir / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(sample, f, separators=(",", ":"))

    rec = dict(source_rec)
    rec["sample_id"] = new_sample_id
    rec["file"] = out_rel
    rec["split"] = split
    rec["load_setting_id"] = new_load_setting_id
    rec["source_bank"] = source_tag
    rec["source_file"] = source_rec["file"]
    rec["source_sample_id"] = source_rec["sample_id"]
    rec["source_load_setting_id"] = source_rec["load_setting_id"]
    return rec


def build_experiment(
    source_root: Path,
    output_root: Path,
    source_banks: list[tuple[str, str]],
    complete_loads: list[tuple[str, int]],
    load_records: dict,
    sample_index: dict,
    branch_id: str,
    experiment_tag: str,
    train_load_count: int,
    train_rho_count: int,
    clean: bool,
) -> None:
    output_dir = output_root / f"rho_exp_branch{branch_id}_{experiment_tag}"
    if output_dir.exists():
        if not clean:
            raise FileExistsError(
                f"Output dir already exists: {output_dir}. "
                "Pass --clean if you want to overwrite it."
            )
        shutil.rmtree(output_dir)
    (output_dir / "raw").mkdir(parents=True)
    (output_dir / "manifests").mkdir(parents=True)
    (output_dir / "splits").mkdir(parents=True)
    (output_dir / "analysis").mkdir(parents=True)
    (output_dir / "manifests" / "failures.jsonl").touch()

    source_bank_map = dict(source_banks)
    train_loads = complete_loads[:train_load_count]
    val_loads = complete_loads[100:110]
    test_loads = complete_loads[110:123]
    train_rhos = evenly_spaced_rhos(train_rho_count)
    val_rhos = [round(i / 100, 10) for i in range(101)]
    test_rhos = [0.0]

    split_specs = {
        "train": (train_loads, train_rhos),
        "val": (val_loads, val_rhos),
        "test": (test_loads, test_rhos),
    }

    load_id_map = {load_key: idx + 1 for idx, load_key in enumerate(complete_loads)}
    source_map_rows = []
    split_summary_rows = []
    sample_id = 1

    for split, (loads, rhos) in split_specs.items():
        split_records = []
        split_load_records = []

        for load_key in loads:
            source_tag, source_load_id = load_key
            new_load_id = load_id_map[load_key]
            source_load_rec = load_records[load_key]
            out_load_rec = dict(source_load_rec)
            out_load_rec["load_setting_id"] = new_load_id
            out_load_rec["split"] = split
            out_load_rec["source_bank"] = source_tag
            out_load_rec["source_load_setting_id"] = source_load_id
            out_load_rec["selected_branch_id"] = branch_id
            append_jsonl(output_dir / "manifests" / "load_settings.jsonl", out_load_rec)
            split_load_records.append(out_load_rec)

            source_map_rows.append({
                "new_load_setting_id": new_load_id,
                "split": split,
                "source_bank": source_tag,
                "source_load_setting_id": source_load_id,
                "selected_branch_id": branch_id,
            })

            sample_ids = []
            for rho in rhos:
                source_sample = sample_index[(source_tag, source_load_id, rho_key(rho))]
                rec = copy_sample(
                    source_root=source_root,
                    source_banks=source_bank_map,
                    output_dir=output_dir,
                    source_tag=source_tag,
                    source_rec=source_sample,
                    new_sample_id=sample_id,
                    new_load_setting_id=new_load_id,
                    split=split,
                    experiment_name=f"rho_exp_branch{branch_id}_{experiment_tag}",
                )
                append_jsonl(output_dir / "manifests" / "samples.jsonl", rec)
                split_records.append(rec)
                sample_ids.append(sample_id)
                sample_id += 1

            split_load_records[-1]["sample_ids"] = sample_ids

        write_json(output_dir / "splits" / f"{split}.json", {
            "split": split,
            "records": split_records,
            "load_settings": split_load_records,
            "failures": [],
        })
        split_summary_rows.append({
            "split": split,
            "load_settings": len(loads),
            "rho_values_per_load": len(rhos),
            "samples": len(split_records),
        })

    with (output_dir / "analysis" / "load_source_map.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_load_setting_id",
                "split",
                "source_bank",
                "source_load_setting_id",
                "selected_branch_id",
            ],
        )
        writer.writeheader()
        writer.writerows(source_map_rows)

    with (output_dir / "analysis" / "split_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "load_settings",
                "rho_values_per_load",
                "samples",
            ],
        )
        writer.writeheader()
        writer.writerows(split_summary_rows)

    write_jsonl(
        output_dir / "analysis" / "train_rhos.jsonl",
        [{"rho_index_in_selection": i, "rho": rho} for i, rho in enumerate(train_rhos)],
    )

    write_json(output_dir / "meta.json", {
        "experiment": experiment_tag,
        "branch_id": branch_id,
        "train_loads": train_load_count,
        "train_rhos_per_load": train_rho_count,
        "train_samples": train_load_count * train_rho_count,
        "val_loads": len(val_loads),
        "val_rhos_per_load": len(val_rhos),
        "val_samples": len(val_loads) * len(val_rhos),
        "test_loads": len(test_loads),
        "test_rhos": test_rhos,
        "test_samples": len(test_loads) * len(test_rhos),
        "rho_selection": {
            "train_rhos": train_rhos,
            "val_rhos": val_rhos,
            "test_rhos": test_rhos,
        },
        "source_banks": dict(source_banks),
    })

    print(
        output_dir,
        "train",
        train_load_count * train_rho_count,
        "val",
        len(val_loads) * len(val_rhos),
        "test",
        len(test_loads) * len(test_rhos),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--output-root", default="case30")
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Overwrite existing rho_exp_branch*_ directories.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = (source_root / args.output_root).resolve()

    complete_loads, load_records, sample_index = load_complete_curves(
        source_root,
        SOURCE_BANKS,
        branch_id=args.branch_id,
    )
    if len(complete_loads) < 123:
        raise RuntimeError(
            f"Need at least 123 complete load settings for branch {args.branch_id}, "
            f"found {len(complete_loads)}"
        )

    random.Random(args.seed).shuffle(complete_loads)
    complete_loads = complete_loads[:123]

    for tag, train_load_count, train_rho_count in EXPERIMENTS:
        build_experiment(
            source_root=source_root,
            output_root=output_root,
            source_banks=SOURCE_BANKS,
            complete_loads=complete_loads,
            load_records=load_records,
            sample_index=sample_index,
            branch_id=args.branch_id,
            experiment_tag=tag,
            train_load_count=train_load_count,
            train_rho_count=train_rho_count,
            clean=args.clean,
        )


if __name__ == "__main__":
    main()
