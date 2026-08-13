#!/usr/bin/env python3
"""Build fixed branch-7 PF-rho CANOS benchmark splits.

The source bank is the coarse PF-rho bank:

    case30/rho_pf_bank_normal_loop_edges_10000_0p1

This script creates two standalone PFDelta-rho roots for the equal-train-sample
comparison:

    rho_pf_branch7_B_rho11
    rho_pf_branch7_B_rho6

Both experiments share the same held-out test load settings and test only
rho=0. The train/val/test partition is by load_setting_id to avoid leakage.
Raw files are hard-linked by default to avoid duplicating the large JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path


EXPECTED_RHOS_11 = [round(i / 10, 10) for i in range(11)]
RHOS_6 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

EXPERIMENTS = {
    "rho11": {
        "train_loads": 3816,
        "train_rhos": EXPECTED_RHOS_11,
        "val_rhos": EXPECTED_RHOS_11,
    },
    "rho6": {
        "train_loads": 6996,
        "train_rhos": RHOS_6,
        "val_rhos": RHOS_6,
    },
}


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


def parse_rho_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {rho_key(part) for part in str(value).split(",") if str(part).strip()}


def load_complete_branch_curves(source_bank: Path, branch_id: str) -> list[int]:
    expected = {rho_key(rho) for rho in EXPECTED_RHOS_11}
    complete = []
    curves_path = source_bank / "manifests" / "curves.jsonl"
    for rec in read_jsonl(curves_path):
        if str(rec.get("branch_id")) != str(branch_id):
            continue
        success_rhos = parse_rho_list(rec.get("success_rhos"))
        if success_rhos >= expected:
            complete.append(int(rec["load_setting_id"]))
    return sorted(set(complete))


def collect_source_records(
    source_bank: Path,
    branch_id: str,
    load_ids: set[int],
    required_rhos: set[str],
) -> tuple[dict[int, dict], dict[tuple[int, str], dict]]:
    load_records = {}
    for rec in read_jsonl(source_bank / "manifests" / "load_settings.jsonl"):
        load_id = int(rec["load_setting_id"])
        if load_id in load_ids:
            load_records[load_id] = rec

    sample_index = {}
    for rec in read_jsonl(source_bank / "manifests" / "samples.jsonl"):
        if str(rec.get("branch_id")) != str(branch_id):
            continue
        load_id = int(rec["load_setting_id"])
        if load_id not in load_ids:
            continue
        rho = rho_key(rec["rho"])
        if rho not in required_rhos:
            continue
        sample_index[(load_id, rho)] = rec

    missing_loads = sorted(load_ids - set(load_records))
    if missing_loads:
        raise RuntimeError(f"Missing load records for {len(missing_loads)} load settings")

    missing_samples = [
        (load_id, rho)
        for load_id in sorted(load_ids)
        for rho in sorted(required_rhos, key=float)
        if (load_id, rho) not in sample_index
    ]
    if missing_samples:
        preview = ", ".join(f"{load}:{rho}" for load, rho in missing_samples[:10])
        raise RuntimeError(f"Missing {len(missing_samples)} source samples; first: {preview}")

    return load_records, sample_index


def link_or_copy(src: Path, dst: Path, copy_raw: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_raw:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_records_for_experiment(
    source_bank: Path,
    output_dir: Path,
    sample_index: dict[tuple[int, str], dict],
    load_records: dict[int, dict],
    split_loads: dict[str, list[int]],
    split_rhos: dict[str, list[float]],
    branch_id: str,
    experiment_name: str,
    copy_raw: bool,
) -> tuple[list[dict], list[dict]]:
    source_map_rows = []
    summary_rows = []
    all_sample_rows = []
    all_load_rows = []
    next_sample_id = 1

    for split in ["train", "val", "test"]:
        records = []
        split_load_records = []
        loads = split_loads[split]
        rhos = split_rhos[split]

        for load_id in loads:
            source_load_rec = dict(load_records[load_id])
            source_load_rec["split"] = split
            source_load_rec["selected_branch_id"] = branch_id
            source_load_rec["source_load_setting_id"] = load_id
            source_load_rec["num_selected_rhos"] = len(rhos)
            sample_ids = []

            for rho in rhos:
                key = (load_id, rho_key(rho))
                source_rec = sample_index[key]
                source_file = source_bank / source_rec["file"]
                out_rel = f"raw/sample_{next_sample_id}.json"
                out_file = output_dir / out_rel
                link_or_copy(source_file, out_file, copy_raw=copy_raw)

                out_rec = dict(source_rec)
                out_rec["sample_id"] = next_sample_id
                out_rec["file"] = out_rel
                out_rec["split"] = split
                out_rec["source_sample_id"] = source_rec["sample_id"]
                out_rec["source_file"] = source_rec["file"]
                out_rec["source_load_setting_id"] = load_id
                out_rec["selected_branch_id"] = branch_id
                out_rec["experiment_name"] = experiment_name
                records.append(out_rec)
                all_sample_rows.append(out_rec)
                sample_ids.append(next_sample_id)
                next_sample_id += 1

            source_load_rec["sample_ids"] = sample_ids
            split_load_records.append(source_load_rec)
            all_load_rows.append(source_load_rec)
            source_map_rows.append(
                {
                    "split": split,
                    "load_setting_id": load_id,
                    "selected_branch_id": branch_id,
                    "rho_values": ",".join(rho_key(rho) for rho in rhos),
                    "sample_ids": ",".join(str(sid) for sid in sample_ids),
                }
            )

        write_json(
            output_dir / "splits" / f"{split}.json",
            {
                "split": split,
                "records": records,
                "load_settings": split_load_records,
                "failures": [],
            },
        )
        summary_rows.append(
            {
                "split": split,
                "load_settings": len(loads),
                "rho_values_per_load": len(rhos),
                "samples": len(records),
                "rho_values": ",".join(rho_key(rho) for rho in rhos),
            }
        )

    return all_sample_rows, all_load_rows, summary_rows, source_map_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_experiment(
    source_bank: Path,
    output_root: Path,
    experiment_tag: str,
    branch_id: str,
    train_loads: list[int],
    val_loads: list[int],
    test_loads: list[int],
    load_records: dict[int, dict],
    sample_index: dict[tuple[int, str], dict],
    copy_raw: bool,
    clean: bool,
    seed: int,
) -> Path:
    spec = EXPERIMENTS[experiment_tag]
    experiment_name = f"rho_pf_branch{branch_id}_B_{experiment_tag}"
    output_dir = output_root / experiment_name

    if output_dir.exists():
        if not clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "manifests", "splits", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    selected_train_loads = train_loads[: spec["train_loads"]]
    split_loads = {
        "train": selected_train_loads,
        "val": val_loads,
        "test": test_loads,
    }
    split_rhos = {
        "train": spec["train_rhos"],
        "val": spec["val_rhos"],
        "test": [0.0],
    }

    sample_rows, load_rows, summary_rows, source_map_rows = split_records_for_experiment(
        source_bank=source_bank,
        output_dir=output_dir,
        sample_index=sample_index,
        load_records=load_records,
        split_loads=split_loads,
        split_rhos=split_rhos,
        branch_id=branch_id,
        experiment_name=experiment_name,
        copy_raw=copy_raw,
    )

    write_jsonl(output_dir / "manifests" / "samples.jsonl", sample_rows)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", load_rows)
    (output_dir / "manifests" / "failures.jsonl").touch()

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

    write_json(
        output_dir / "meta.json",
        {
            "experiment_name": experiment_name,
            "source_bank": str(source_bank),
            "branch_id": branch_id,
            "split_seed": seed,
            "selection": {
                "train_loads": len(selected_train_loads),
                "train_rhos": split_rhos["train"],
                "train_samples": len(selected_train_loads) * len(split_rhos["train"]),
                "val_loads": len(val_loads),
                "val_rhos": split_rhos["val"],
                "val_samples": len(val_loads) * len(split_rhos["val"]),
                "test_loads": len(test_loads),
                "test_rhos": split_rhos["test"],
                "test_samples": len(test_loads) * len(split_rhos["test"]),
            },
            "raw_storage": "copy" if copy_raw else "hardlink_or_copy_fallback",
            "target": "PF_solution(rho)",
            "notes": (
                "Split is by load_setting_id. rho11 and rho6 share the same "
                "held-out test load settings; test uses rho=0 only."
            ),
        },
    )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--test-loads", type=int, default=2000)
    parser.add_argument("--val-loads", type=int, default=1000)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true", help="Copy raw JSON instead of hard-linking")
    args = parser.parse_args()

    source_bank = Path(args.source_bank).resolve()
    output_root = Path(args.output_root).resolve()
    complete_loads = load_complete_branch_curves(source_bank, args.branch_id)
    random.Random(args.seed).shuffle(complete_loads)

    needed_train_pool = max(spec["train_loads"] for spec in EXPERIMENTS.values())
    needed_total = args.test_loads + args.val_loads + needed_train_pool
    if len(complete_loads) < needed_total:
        raise RuntimeError(
            f"Need {needed_total} complete branch-{args.branch_id} load settings, "
            f"found {len(complete_loads)}"
        )

    test_loads = complete_loads[: args.test_loads]
    val_loads = complete_loads[args.test_loads: args.test_loads + args.val_loads]
    train_pool = complete_loads[
        args.test_loads + args.val_loads:
        args.test_loads + args.val_loads + needed_train_pool
    ]

    required_loads = set(test_loads + val_loads + train_pool)
    required_rhos = {rho_key(rho) for rho in EXPECTED_RHOS_11}
    load_records, sample_index = collect_source_records(
        source_bank=source_bank,
        branch_id=args.branch_id,
        load_ids=required_loads,
        required_rhos=required_rhos,
    )

    output_dirs = []
    for experiment_tag in ["rho11", "rho6"]:
        output_dirs.append(
            build_experiment(
                source_bank=source_bank,
                output_root=output_root,
                experiment_tag=experiment_tag,
                branch_id=args.branch_id,
                train_loads=train_pool,
                val_loads=val_loads,
                test_loads=test_loads,
                load_records=load_records,
                sample_index=sample_index,
                copy_raw=args.copy_raw,
                clean=args.clean,
                seed=args.seed,
            )
        )

    print(json.dumps({
        "source_bank": str(source_bank),
        "complete_branch_loads": len(complete_loads),
        "seed": args.seed,
        "test_loads": len(test_loads),
        "val_loads": len(val_loads),
        "train_pool_loads": len(train_pool),
        "output_dirs": [str(path) for path in output_dirs],
    }, indent=2))


if __name__ == "__main__":
    main()
