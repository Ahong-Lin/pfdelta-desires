#!/usr/bin/env python3
"""Build a rho-aware normal-operation baseline for CANOS.

This script converts original PFDelta case30 normal-operation samples into the
PFDelta-rho directory layout.  All branches receive an edge-level rho=1.0, and
one selected branch is marked as rho_controlled=true.  The test split is copied
from an existing rho experiment so the baseline is evaluated on the same
rho=0 outage samples as the rho-continuation models.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(row, f, separators=(",", ":"))
        f.write("\n")


def load_vector(sample: dict[str, Any]) -> list[float]:
    loads = sorted(sample["network"]["load"].items(), key=lambda x: int(x[0]))
    pd = [float(load["pd"]) for _, load in loads]
    qd = [float(load["qd"]) for _, load in loads]
    return pd + qd


def rounded_vector_key(values: list[float], digits: int = 10) -> tuple[float, ...]:
    return tuple(round(float(x), digits) for x in values)


def is_close_vector(a: list[float], b: list[float], tol: float) -> bool:
    return len(a) == len(b) and max(abs(x - y) for x, y in zip(a, b)) <= tol


def ordered_normal_files(pfdelta_root: Path, case_name: str) -> list[Path]:
    raw_dir = pfdelta_root / case_name / "n" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing normal raw directory: {raw_dir}")

    shuffle_path = pfdelta_root / "shuffle_files" / "n" / "raw_shuffle.json"
    if not shuffle_path.exists():
        return sorted(raw_dir.glob("sample_*.json"), key=lambda p: int(p.stem.split("_")[-1]))

    shuffle_raw = read_json(shuffle_path)
    shuffle_map = {int(k): int(v) for k, v in shuffle_raw.items()}
    shuffle_keys = sorted(shuffle_map)
    raw_fnames = [raw_dir / f"sample_{i + 1}.json" for i in shuffle_keys]
    ordered = [raw_fnames[shuffle_map[i]] for i in shuffle_keys]
    return [path for path in ordered if path.exists()]


def patch_rho_one(
    sample: dict[str, Any],
    branch_id: str,
    sample_id: int,
    load_setting_id: int,
    split: str,
    task_name: str,
    source_file: str,
    source_kind: str,
) -> dict[str, Any]:
    sample = copy.deepcopy(sample)
    network = sample["network"]
    branch_key = str(branch_id)
    if branch_key not in network["branch"]:
        raise KeyError(f"Branch {branch_key} is not in sample network")

    for key, branch in network["branch"].items():
        if int(branch.get("br_status", 1)) != 1:
            raise ValueError(f"Source normal sample has inactive branch {key}")

        for field in ["br_r", "br_x", "g_fr", "b_fr", "g_to", "b_to", "br_status"]:
            base_field = f"{field}_base"
            if base_field not in branch:
                branch[base_field] = branch[field]

        rho = 1.0
        branch["rho"] = rho
        branch["rho_controlled"] = str(key) == branch_key
        branch["br_status"] = 1
        branch["br_r"] = branch["br_r_base"] / rho
        branch["br_x"] = branch["br_x_base"] / rho
        branch["g_fr"] = branch["g_fr_base"] * rho
        branch["b_fr"] = branch["b_fr_base"] * rho
        branch["g_to"] = branch["g_to_base"] * rho
        branch["b_to"] = branch["b_to_base"] * rho

    controlled = network["branch"][branch_key]
    sample["rho_meta"] = {
        "sample_id": sample_id,
        "load_setting_id": load_setting_id,
        "branch_id": branch_key,
        "rho": 1.0,
        "rho_index": 0,
        "split": split,
        "case_name": "case30",
        "task_name": task_name,
        "br_status": 1,
        "f_bus": controlled["f_bus"],
        "t_bus": controlled["t_bus"],
        "source_kind": source_kind,
        "source_file": source_file,
    }
    return sample


def make_success_record(
    sample: dict[str, Any],
    sample_id: int,
    load_setting_id: int,
    split: str,
    branch_id: str,
    out_rel: str,
    source_file: str,
    source_kind: str,
) -> dict[str, Any]:
    branch = sample["network"]["branch"][str(branch_id)]
    solution = sample.get("solution", {})
    return {
        "sample_id": sample_id,
        "file": out_rel,
        "split": split,
        "case_name": "case30",
        "load_setting_id": load_setting_id,
        "branch_id": str(branch_id),
        "f_bus": branch["f_bus"],
        "t_bus": branch["t_bus"],
        "rho": 1.0,
        "rho_index": 0,
        "rho_grid": "1.0",
        "br_status": 1,
        "termination_status": solution.get("termination_status", ""),
        "objective": solution.get("objective"),
        "source_kind": source_kind,
        "source_file": source_file,
    }


def write_sample(
    output_dir: Path,
    sample: dict[str, Any],
    sample_id: int,
) -> str:
    out_rel = f"raw/sample_{sample_id}.json"
    out_path = output_dir / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(sample, f, separators=(",", ":"))
    return out_rel


def prepare_output(output_dir: Path, clean: bool) -> None:
    if output_dir.exists():
        if not clean:
            raise FileExistsError(
                f"Output dir already exists: {output_dir}. "
                "Pass --clean if you want to overwrite it."
            )
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests" / "failures.jsonl").touch()


def copy_rho_test_split(
    output_dir: Path,
    rho_test_root: Path,
    branch_id: str,
    task_name: str,
    next_sample_id: int,
    next_load_setting_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    split_path = rho_test_root / "splits" / "test.json"
    split_data = read_json(split_path)
    records = split_data.get("records", [])
    if not records:
        raise ValueError(f"No test records found in {split_path}")

    out_records = []
    out_load_records = []
    source_rows = []

    for source_rec in records:
        source_file = rho_test_root / source_rec["file"]
        sample = read_json(source_file)
        sample = copy.deepcopy(sample)
        sample_id = next_sample_id
        load_setting_id = next_load_setting_id

        rho_meta = sample.setdefault("rho_meta", {})
        rho_meta["sample_id"] = sample_id
        rho_meta["load_setting_id"] = load_setting_id
        rho_meta["split"] = "test"
        rho_meta["task_name"] = task_name
        rho_meta["source_kind"] = "copied_rho0_test"
        rho_meta["source_test_root"] = str(rho_test_root)
        rho_meta["source_file"] = source_rec["file"]
        rho_meta["source_sample_id"] = source_rec.get("sample_id")
        rho_meta["source_load_setting_id"] = source_rec.get("load_setting_id")

        out_rel = write_sample(output_dir, sample, sample_id)
        out_rec = dict(source_rec)
        out_rec.update({
            "sample_id": sample_id,
            "file": out_rel,
            "split": "test",
            "load_setting_id": load_setting_id,
            "source_kind": "copied_rho0_test",
            "source_test_root": str(rho_test_root),
            "source_file": source_rec["file"],
            "source_sample_id": source_rec.get("sample_id"),
            "source_load_setting_id": source_rec.get("load_setting_id"),
        })
        out_records.append(out_rec)
        append_jsonl(output_dir / "manifests" / "samples.jsonl", out_rec)

        out_load = {
            "load_setting_id": load_setting_id,
            "split": "test",
            "load_vector": load_vector(sample),
            "sample_ids": [sample_id],
            "selected_branch_id": str(branch_id),
            "source_kind": "copied_rho0_test",
            "source_test_root": str(rho_test_root),
            "source_load_setting_id": source_rec.get("load_setting_id"),
        }
        out_load_records.append(out_load)
        append_jsonl(output_dir / "manifests" / "load_settings.jsonl", out_load)

        source_rows.append({
            "split": "test",
            "new_load_setting_id": load_setting_id,
            "new_sample_id": sample_id,
            "source_kind": "copied_rho0_test",
            "source_file": source_rec["file"],
            "source_load_setting_id": source_rec.get("load_setting_id", ""),
        })

        next_sample_id += 1
        next_load_setting_id += 1

    return out_records, out_load_records, source_rows, next_sample_id, next_load_setting_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfdelta-root", default="data/pfdelta_data")
    parser.add_argument("--rho-test-root", default="data_generation_rho/case30/rho_exp_branch7_60L80R")
    parser.add_argument("--output-dir", default="data_generation_rho/case30/rho_exp_branch7_rho1_baseline_4800")
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--train-samples", type=int, default=4800)
    parser.add_argument("--val-samples", type=int, default=1010)
    parser.add_argument("--exclude-load-tol", type=float, default=1e-9)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    pfdelta_root = Path(args.pfdelta_root).resolve()
    rho_test_root = Path(args.rho_test_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_name = output_dir.name

    prepare_output(output_dir, args.clean)

    test_vectors = []
    test_split = read_json(rho_test_root / "splits" / "test.json")
    for rec in test_split.get("records", []):
        test_vectors.append(load_vector(read_json(rho_test_root / rec["file"])))

    target_counts = {"train": args.train_samples, "val": args.val_samples}
    split_records = {"train": [], "val": []}
    split_loads = {"train": [], "val": []}
    source_rows = []
    selected_keys = set()
    sample_id = 1
    load_setting_id = 1

    normal_files = ordered_normal_files(pfdelta_root, "case30")
    normal_iter = iter(normal_files)

    for split in ["train", "val"]:
        while len(split_records[split]) < target_counts[split]:
            try:
                source_path = next(normal_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Not enough normal case30 samples. Need {target_counts}, "
                    f"got train={len(split_records['train'])}, val={len(split_records['val'])}."
                ) from exc

            source_sample = read_json(source_path)
            vector = load_vector(source_sample)
            if any(is_close_vector(vector, test_vec, args.exclude_load_tol) for test_vec in test_vectors):
                continue
            key = rounded_vector_key(vector)
            if key in selected_keys:
                continue

            try:
                sample = patch_rho_one(
                    source_sample,
                    branch_id=str(args.branch_id),
                    sample_id=sample_id,
                    load_setting_id=load_setting_id,
                    split=split,
                    task_name=task_name,
                    source_file=str(source_path.relative_to(pfdelta_root)),
                    source_kind="normal_rho1",
                )
            except ValueError:
                continue

            out_rel = write_sample(output_dir, sample, sample_id)
            rec = make_success_record(
                sample=sample,
                sample_id=sample_id,
                load_setting_id=load_setting_id,
                split=split,
                branch_id=str(args.branch_id),
                out_rel=out_rel,
                source_file=str(source_path.relative_to(pfdelta_root)),
                source_kind="normal_rho1",
            )
            split_records[split].append(rec)
            append_jsonl(output_dir / "manifests" / "samples.jsonl", rec)

            load_rec = {
                "load_setting_id": load_setting_id,
                "split": split,
                "load_vector": vector,
                "sample_ids": [sample_id],
                "selected_branch_id": str(args.branch_id),
                "source_kind": "normal_rho1",
                "source_file": str(source_path.relative_to(pfdelta_root)),
            }
            split_loads[split].append(load_rec)
            append_jsonl(output_dir / "manifests" / "load_settings.jsonl", load_rec)

            source_rows.append({
                "split": split,
                "new_load_setting_id": load_setting_id,
                "new_sample_id": sample_id,
                "source_kind": "normal_rho1",
                "source_file": str(source_path.relative_to(pfdelta_root)),
                "source_load_setting_id": "",
            })

            selected_keys.add(key)
            sample_id += 1
            load_setting_id += 1

    test_records, test_loads, test_rows, sample_id, load_setting_id = copy_rho_test_split(
        output_dir=output_dir,
        rho_test_root=rho_test_root,
        branch_id=str(args.branch_id),
        task_name=task_name,
        next_sample_id=sample_id,
        next_load_setting_id=load_setting_id,
    )
    source_rows.extend(test_rows)

    write_json(output_dir / "splits" / "train.json", {
        "split": "train",
        "records": split_records["train"],
        "load_settings": split_loads["train"],
        "failures": [],
    })
    write_json(output_dir / "splits" / "val.json", {
        "split": "val",
        "records": split_records["val"],
        "load_settings": split_loads["val"],
        "failures": [],
    })
    write_json(output_dir / "splits" / "test.json", {
        "split": "test",
        "records": test_records,
        "load_settings": test_loads,
        "failures": [],
    })

    with (output_dir / "analysis" / "source_map.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "new_load_setting_id",
                "new_sample_id",
                "source_kind",
                "source_file",
                "source_load_setting_id",
            ],
        )
        writer.writeheader()
        writer.writerows(source_rows)

    split_summary = [
        {"split": "train", "load_settings": len(split_loads["train"]), "samples": len(split_records["train"]), "rho": 1.0},
        {"split": "val", "load_settings": len(split_loads["val"]), "samples": len(split_records["val"]), "rho": 1.0},
        {"split": "test", "load_settings": len(test_loads), "samples": len(test_records), "rho": 0.0},
    ]
    with (output_dir / "analysis" / "split_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "load_settings", "samples", "rho"])
        writer.writeheader()
        writer.writerows(split_summary)

    write_json(output_dir / "meta.json", {
        "task_name": task_name,
        "case_name": "case30",
        "branch_id": str(args.branch_id),
        "rho_definition": (
            "Baseline train/val samples use rho=1 on every branch. The selected "
            "branch is marked rho_controlled=true, br_status=1, and branch "
            "parameters follow br_r=br_r_base/rho, br_x=br_x_base/rho, "
            "g/b shunts = base*rho. Test is copied from the rho=0 branch7 split."
        ),
        "train_samples": len(split_records["train"]),
        "val_samples": len(split_records["val"]),
        "test_samples": len(test_records),
        "pfdelta_root": str(pfdelta_root),
        "rho_test_root": str(rho_test_root),
    })

    print(f"Output: {output_dir}")
    for row in split_summary:
        print(f"{row['split']}: samples={row['samples']}, loads={row['load_settings']}, rho={row['rho']}")


if __name__ == "__main__":
    main()
