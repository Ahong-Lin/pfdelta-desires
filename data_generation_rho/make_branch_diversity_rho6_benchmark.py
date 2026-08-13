#!/usr/bin/env python3
"""Build branch-diversity rho6 benchmarks from an existing PF-rho bank.

The source bank is expected to contain successful PF samples indexed by
``manifests/samples.jsonl`` and per-load/branch curves in
``manifests/curves.jsonl``.

Outputs:

* ``rho_pf_branchset6_B_disjoint_rho6_41976``
* ``rho_pf_branchset6_B_shared_rho6_41976``
* ``rho_pf_allbranch_B_balanced_rho6_41976``
* ``rho_pf_test_branchset6_B_rho0``
* ``rho_pf_test_allbranch_B_rho0``

All raw JSON files are hard-linked by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


RHO6 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BRANCHSET6 = ["4", "5", "7", "20", "21", "23"]


def rho_key(value: float | str) -> str:
    return f"{round(float(value), 10):.10g}"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def parse_rhos(value: str | None) -> set[str]:
    if not value:
        return set()
    return {rho_key(part) for part in str(value).split(",") if str(part).strip()}


def fixed_loads(fixed_split_root: Path, split: str) -> list[int]:
    data = read_json(fixed_split_root / "splits" / f"{split}.json")
    return sorted({int(record["load_setting_id"]) for record in data["records"]})


def load_curve_index(source_bank: Path) -> tuple[dict[tuple[int, str], set[str]], list[str]]:
    curve_rhos: dict[tuple[int, str], set[str]] = {}
    branches = set()
    for rec in read_jsonl(source_bank / "manifests" / "curves.jsonl"):
        load_id = int(rec["load_setting_id"])
        branch_id = str(rec["branch_id"])
        branches.add(branch_id)
        curve_rhos[(load_id, branch_id)] = parse_rhos(rec.get("success_rhos"))
    return curve_rhos, sorted(branches, key=int)


def has_rhos(curve_rhos: dict[tuple[int, str], set[str]], load_id: int, branch_id: str, rhos: list[float]) -> bool:
    available = curve_rhos.get((int(load_id), str(branch_id)), set())
    return {rho_key(rho) for rho in rhos}.issubset(available)


def select_disjoint_pairs(
    curve_rhos: dict[tuple[int, str], set[str]],
    branch_ids: list[str],
    candidate_loads: list[int],
    loads_per_branch: int,
) -> list[tuple[int, str]]:
    used_loads: set[int] = set()
    pairs: list[tuple[int, str]] = []
    for branch_id in branch_ids:
        selected = []
        for load_id in candidate_loads:
            if load_id in used_loads:
                continue
            if not has_rhos(curve_rhos, load_id, branch_id, RHO6):
                continue
            selected.append((load_id, branch_id))
            used_loads.add(load_id)
            if len(selected) == loads_per_branch:
                break
        if len(selected) < loads_per_branch:
            raise RuntimeError(
                f"Need {loads_per_branch} disjoint loads for branch {branch_id}, "
                f"found {len(selected)}"
            )
        pairs.extend(selected)
    return pairs


def select_shared_pairs(
    curve_rhos: dict[tuple[int, str], set[str]],
    branch_ids: list[str],
    candidate_loads: list[int],
    num_loads: int,
) -> list[tuple[int, str]]:
    selected_loads = []
    for load_id in candidate_loads:
        if all(has_rhos(curve_rhos, load_id, branch_id, RHO6) for branch_id in branch_ids):
            selected_loads.append(load_id)
        if len(selected_loads) == num_loads:
            break
    if len(selected_loads) < num_loads:
        raise RuntimeError(f"Need {num_loads} shared loads, found {len(selected_loads)}")
    return [(load_id, branch_id) for load_id in selected_loads for branch_id in branch_ids]


def select_balanced_pairs(
    curve_rhos: dict[tuple[int, str], set[str]],
    branch_ids: list[str],
    candidate_loads: list[int],
    total_pairs: int,
    prefer_unique_loads: bool = True,
) -> list[tuple[int, str]]:
    base = total_pairs // len(branch_ids)
    extra = total_pairs % len(branch_ids)
    targets = {
        branch_id: base + (1 if idx < extra else 0)
        for idx, branch_id in enumerate(branch_ids)
    }
    used_loads: set[int] = set()
    pairs: list[tuple[int, str]] = []

    for branch_id in branch_ids:
        selected = []
        for load_id in candidate_loads:
            if prefer_unique_loads and load_id in used_loads:
                continue
            if not has_rhos(curve_rhos, load_id, branch_id, RHO6):
                continue
            selected.append((load_id, branch_id))
            used_loads.add(load_id)
            if len(selected) == targets[branch_id]:
                break

        if len(selected) < targets[branch_id] and prefer_unique_loads:
            for load_id in candidate_loads:
                if not has_rhos(curve_rhos, load_id, branch_id, RHO6):
                    continue
                pair = (load_id, branch_id)
                if pair in selected:
                    continue
                selected.append(pair)
                if len(selected) == targets[branch_id]:
                    break

        if len(selected) < targets[branch_id]:
            raise RuntimeError(
                f"Need {targets[branch_id]} balanced pairs for branch {branch_id}, "
                f"found {len(selected)}"
            )
        pairs.extend(selected)
    return pairs


def collect_needed_samples(source_bank: Path, needed_keys: set[tuple[int, str, str]]) -> dict[tuple[int, str, str], dict]:
    sample_index = {}
    for rec in read_jsonl(source_bank / "manifests" / "samples.jsonl"):
        key = (int(rec["load_setting_id"]), str(rec["branch_id"]), rho_key(rec["rho"]))
        if key in needed_keys:
            sample_index[key] = rec
    missing = sorted(needed_keys - set(sample_index), key=lambda item: (item[0], int(item[1]), float(item[2])))
    if missing:
        preview = ", ".join(f"{load}:{branch}:{rho}" for load, branch, rho in missing[:10])
        raise RuntimeError(f"Missing {len(missing)} selected source samples; first: {preview}")
    return sample_index


def records_for_pairs(
    pairs: list[tuple[int, str]],
    rhos: list[float],
    sample_index: dict[tuple[int, str, str], dict],
) -> list[dict]:
    records = []
    for load_id, branch_id in pairs:
        for rho in rhos:
            records.append(sample_index[(int(load_id), str(branch_id), rho_key(rho))])
    return records


def make_output_record(source_rec: dict, sample_id: int, out_rel: str, split: str, source_root: Path, experiment_name: str) -> dict:
    rec = dict(source_rec)
    rec["sample_id"] = sample_id
    rec["file"] = out_rel
    rec["split"] = split
    rec["source_dataset_root"] = str(source_root)
    rec["source_sample_id"] = source_rec["sample_id"]
    rec["source_sample_file"] = source_rec["file"]
    rec["source_load_setting_id"] = source_rec["load_setting_id"]
    rec["selected_branch_id"] = source_rec["branch_id"]
    rec["experiment_name"] = experiment_name
    return rec


def write_root(
    output_dir: Path,
    source_bank: Path,
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
    source_map_rows = []
    next_sample_id = 1
    split_output_records: dict[str, list[dict]] = {}

    for split in ["train", "val", "test"]:
        out_records = []
        for source_rec in split_sources[split]:
            out_rel = f"raw/sample_{next_sample_id}.json"
            link_or_copy(source_bank / source_rec["file"], output_dir / out_rel, copy_raw)
            out_rec = make_output_record(
                source_rec=source_rec,
                sample_id=next_sample_id,
                out_rel=out_rel,
                split=split,
                source_root=source_bank,
                experiment_name=output_dir.name,
            )
            out_records.append(out_rec)
            all_records.append(out_rec)
            next_sample_id += 1
        split_output_records[split] = out_records

    load_manifest_rows = []
    for split, out_records in split_output_records.items():
        by_load: dict[int, list[dict]] = defaultdict(list)
        for rec in out_records:
            by_load[int(rec["load_setting_id"])].append(rec)

        split_loads = []
        for load_id in sorted(by_load):
            rows = by_load[load_id]
            sample_ids = [int(row["sample_id"]) for row in rows]
            branches = sorted({str(row["branch_id"]) for row in rows}, key=int)
            rhos = sorted({rho_key(row["rho"]) for row in rows}, key=float)
            load_rec = {
                "load_setting_id": load_id,
                "split": split,
                "sample_ids": sample_ids,
                "selected_branch_ids": ",".join(branches),
                "rho_values": ",".join(rhos),
                "num_samples": len(rows),
            }
            split_loads.append(load_rec)
            load_manifest_rows.append(load_rec)
            source_map_rows.append(load_rec)

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
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", load_manifest_rows)
    (output_dir / "manifests" / "failures.jsonl").touch()

    summary_rows = []
    branch_rows = []
    for split, out_records in split_output_records.items():
        loads = sorted({int(rec["load_setting_id"]) for rec in out_records})
        branches = sorted({str(rec["branch_id"]) for rec in out_records}, key=int)
        rhos = sorted({rho_key(rec["rho"]) for rec in out_records}, key=float)
        summary_rows.append(
            {
                "split": split,
                "samples": len(out_records),
                "load_settings": len(loads),
                "branches": len(branches),
                "branch_ids": ",".join(branches),
                "rho_values": ",".join(rhos),
            }
        )
        counts = defaultdict(int)
        for rec in out_records:
            counts[str(rec["branch_id"])] += 1
        for branch_id in branches:
            branch_rows.append(
                {
                    "split": split,
                    "branch_id": branch_id,
                    "samples": counts[branch_id],
                    "sample_groups_at_rho6": counts[branch_id] // len(rhos) if rhos else 0,
                }
            )

    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        summary_rows,
        ["split", "samples", "load_settings", "branches", "branch_ids", "rho_values"],
    )
    write_csv(
        output_dir / "analysis" / "branch_counts.csv",
        branch_rows,
        ["split", "branch_id", "samples", "sample_groups_at_rho6"],
    )
    write_csv(
        output_dir / "analysis" / "load_source_map.csv",
        source_map_rows,
        ["split", "load_setting_id", "sample_ids", "selected_branch_ids", "rho_values", "num_samples"],
    )

    meta = dict(meta)
    meta["selection"] = summary_rows
    write_json(output_dir / "meta.json", meta)


def placeholder_splits(test_records: list[dict]) -> dict[str, list[dict]]:
    if not test_records:
        raise RuntimeError("Cannot build test root without test records")
    return {
        "train": [test_records[0]],
        "val": [test_records[0]],
        "test": test_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--fixed-split-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    source_bank = Path(args.source_bank).resolve()
    fixed_split_root = Path(args.fixed_split_root).resolve()
    output_root = Path(args.output_root).resolve()

    test_loads = fixed_loads(fixed_split_root, "test")
    val_loads = fixed_loads(fixed_split_root, "val")
    blocked_loads = set(test_loads) | set(val_loads)

    curve_rhos, all_branches = load_curve_index(source_bank)
    candidate_train_loads = [load_id for load_id in range(1, 10001) if load_id not in blocked_loads]

    train_disjoint_pairs = select_disjoint_pairs(curve_rhos, BRANCHSET6, candidate_train_loads, 1166)
    val_disjoint_pairs = select_balanced_pairs(curve_rhos, BRANCHSET6, val_loads, 1000, prefer_unique_loads=True)

    train_shared_pairs = select_shared_pairs(curve_rhos, BRANCHSET6, candidate_train_loads, 1166)
    val_shared_pairs = select_shared_pairs(curve_rhos, BRANCHSET6, val_loads, 167)

    train_all_pairs = select_balanced_pairs(curve_rhos, all_branches, candidate_train_loads, 6996, prefer_unique_loads=True)
    val_all_pairs = select_balanced_pairs(curve_rhos, all_branches, val_loads, 1000, prefer_unique_loads=True)

    t7_pairs = [(load_id, "7") for load_id in test_loads if has_rhos(curve_rhos, load_id, "7", [0.0])]
    t6_pairs = [
        (load_id, branch_id)
        for load_id in test_loads
        for branch_id in BRANCHSET6
        if has_rhos(curve_rhos, load_id, branch_id, [0.0])
    ]
    t34_pairs = [
        (load_id, branch_id)
        for load_id in test_loads
        for branch_id in all_branches
        if has_rhos(curve_rhos, load_id, branch_id, [0.0])
    ]

    needed_keys = set()
    for pairs, rhos in [
        (train_disjoint_pairs, RHO6),
        (val_disjoint_pairs, RHO6),
        (train_shared_pairs, RHO6),
        (val_shared_pairs, RHO6),
        (train_all_pairs, RHO6),
        (val_all_pairs, RHO6),
        (t7_pairs, [0.0]),
        (t6_pairs, [0.0]),
        (t34_pairs, [0.0]),
    ]:
        for load_id, branch_id in pairs:
            for rho in rhos:
                needed_keys.add((int(load_id), str(branch_id), rho_key(rho)))

    sample_index = collect_needed_samples(source_bank, needed_keys)

    t7_records = records_for_pairs(t7_pairs, [0.0], sample_index)
    t6_records = records_for_pairs(t6_pairs, [0.0], sample_index)
    t34_records = records_for_pairs(t34_pairs, [0.0], sample_index)

    common_meta = {
        "source_bank": str(source_bank),
        "fixed_split_root": str(fixed_split_root),
        "rho_values_train": RHO6,
        "branchset6": BRANCHSET6,
        "all_branches": all_branches,
        "blocked_test_loads": len(test_loads),
        "blocked_val_loads": len(val_loads),
        "target": "PF_solution(rho)",
        "leakage_rule": "train and val exclude the fixed test load_setting_id set",
    }

    write_root(
        output_root / "rho_pf_branchset6_B_disjoint_rho6_41976",
        source_bank,
        {
            "train": records_for_pairs(train_disjoint_pairs, RHO6, sample_index),
            "val": records_for_pairs(val_disjoint_pairs, RHO6, sample_index),
            "test": t7_records,
        },
        {
            **common_meta,
            "experiment_name": "rho_pf_branchset6_B_disjoint_rho6_41976",
            "train_design": "6996 unique loads, each assigned to one of six branches; rho6 per load-branch pair",
        },
        args.clean,
        args.copy_raw,
    )

    write_root(
        output_root / "rho_pf_branchset6_B_shared_rho6_41976",
        source_bank,
        {
            "train": records_for_pairs(train_shared_pairs, RHO6, sample_index),
            "val": records_for_pairs(val_shared_pairs, RHO6, sample_index),
            "test": t7_records,
        },
        {
            **common_meta,
            "experiment_name": "rho_pf_branchset6_B_shared_rho6_41976",
            "train_design": "1166 shared loads, each covering six branches; rho6 per load-branch pair",
        },
        args.clean,
        args.copy_raw,
    )

    write_root(
        output_root / "rho_pf_allbranch_B_balanced_rho6_41976",
        source_bank,
        {
            "train": records_for_pairs(train_all_pairs, RHO6, sample_index),
            "val": records_for_pairs(val_all_pairs, RHO6, sample_index),
            "test": t7_records,
        },
        {
            **common_meta,
            "experiment_name": "rho_pf_allbranch_B_balanced_rho6_41976",
            "train_design": "6996 load-branch pairs balanced across all normal-loop branches; rho6 per pair",
        },
        args.clean,
        args.copy_raw,
    )

    write_root(
        output_root / "rho_pf_test_branchset6_B_rho0",
        source_bank,
        placeholder_splits(t6_records),
        {
            **common_meta,
            "experiment_name": "rho_pf_test_branchset6_B_rho0",
            "test_design": "fixed test loads x branchset6 x rho=0, skipping failed PF samples",
            "expected_attempts": len(test_loads) * len(BRANCHSET6),
            "actual_test_samples": len(t6_records),
        },
        args.clean,
        args.copy_raw,
    )

    write_root(
        output_root / "rho_pf_test_allbranch_B_rho0",
        source_bank,
        placeholder_splits(t34_records),
        {
            **common_meta,
            "experiment_name": "rho_pf_test_allbranch_B_rho0",
            "test_design": "fixed test loads x all normal-loop branches x rho=0, skipping failed PF samples",
            "expected_attempts": len(test_loads) * len(all_branches),
            "actual_test_samples": len(t34_records),
        },
        args.clean,
        args.copy_raw,
    )

    outputs = [
        output_root / "rho_pf_branchset6_B_disjoint_rho6_41976",
        output_root / "rho_pf_branchset6_B_shared_rho6_41976",
        output_root / "rho_pf_allbranch_B_balanced_rho6_41976",
        output_root / "rho_pf_test_branchset6_B_rho0",
        output_root / "rho_pf_test_allbranch_B_rho0",
    ]
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
