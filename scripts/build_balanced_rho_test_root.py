#!/usr/bin/env python3
"""Build a balanced fixed-size PFDelta-rho test root from another test root."""

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


def target_counts(branches: list[str], total_samples: int) -> dict[str, int]:
    base = total_samples // len(branches)
    extra = total_samples % len(branches)
    return {
        branch_id: base + (1 if idx < extra else 0)
        for idx, branch_id in enumerate(branches)
    }


def select_records(records: list[dict], total_samples: int, seed: int) -> list[dict]:
    by_branch: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_branch[str(rec["branch_id"])].append(rec)

    branches = sorted(by_branch, key=int)
    targets = target_counts(branches, total_samples)
    rng = random.Random(seed)
    for rows in by_branch.values():
        rng.shuffle(rows)

    used_loads: set[int] = set()
    selected: list[dict] = []
    selected_keys: set[tuple[int, str]] = set()

    for branch_id in branches:
        branch_selected = []
        for rec in by_branch[branch_id]:
            load_id = int(rec["load_setting_id"])
            if load_id in used_loads:
                continue
            branch_selected.append(rec)
            used_loads.add(load_id)
            selected_keys.add((load_id, branch_id))
            if len(branch_selected) == targets[branch_id]:
                break

        if len(branch_selected) < targets[branch_id]:
            for rec in by_branch[branch_id]:
                load_id = int(rec["load_setting_id"])
                key = (load_id, branch_id)
                if key in selected_keys:
                    continue
                branch_selected.append(rec)
                selected_keys.add(key)
                if len(branch_selected) == targets[branch_id]:
                    break

        if len(branch_selected) < targets[branch_id]:
            raise RuntimeError(
                f"Need {targets[branch_id]} records for branch {branch_id}, "
                f"found {len(branch_selected)}"
            )
        selected.extend(branch_selected)

    rng.shuffle(selected)
    if len(selected) != total_samples:
        raise RuntimeError(f"Selected {len(selected)} records, expected {total_samples}")
    return selected


def make_output_record(source_rec: dict, sample_id: int, out_rel: str, source_root: Path, output_name: str) -> dict:
    rec = dict(source_rec)
    rec["sample_id"] = sample_id
    rec["file"] = out_rel
    rec["split"] = "test"
    rec["source_dataset_root"] = str(source_root)
    rec["source_sample_id"] = source_rec["sample_id"]
    rec["source_sample_file"] = source_rec["file"]
    rec["source_load_setting_id"] = source_rec["load_setting_id"]
    rec["experiment_name"] = output_name
    return rec


def build_load_settings(records: list[dict], split: str) -> list[dict]:
    by_load: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_load[int(rec["load_setting_id"])].append(rec)

    rows = []
    for load_id in sorted(by_load):
        recs = by_load[load_id]
        rows.append(
            {
                "split": split,
                "load_setting_id": load_id,
                "sample_ids": [int(rec["sample_id"]) for rec in recs],
                "selected_branch_ids": ",".join(sorted({str(rec["branch_id"]) for rec in recs}, key=int)),
                "rho_values": ",".join(sorted({rho_key(rec["rho"]) for rec in recs}, key=float)),
                "num_samples": len(recs),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    source_split = read_json(source_root / "splits" / "test.json")
    selected_sources = select_records(source_split["records"], args.total_samples, args.seed)

    out_records = []
    source_map_rows = []
    for sample_id, source_rec in enumerate(selected_sources, start=1):
        out_rel = f"raw/sample_{sample_id}.json"
        link_or_copy(source_root / source_rec["file"], output_dir / out_rel, args.copy_raw)
        out_rec = make_output_record(source_rec, sample_id, out_rel, source_root, output_dir.name)
        out_records.append(out_rec)
        source_map_rows.append(
            {
                "sample_id": sample_id,
                "load_setting_id": out_rec["load_setting_id"],
                "branch_id": out_rec["branch_id"],
                "rho": rho_key(out_rec["rho"]),
                "source_sample_id": out_rec["source_sample_id"],
                "source_load_setting_id": out_rec["source_load_setting_id"],
                "source_file": out_rec["source_sample_file"],
            }
        )

    placeholder = [out_records[0]]
    split_payloads = {
        "train": placeholder,
        "val": placeholder,
        "test": out_records,
    }
    all_load_settings = []
    for split, records in split_payloads.items():
        load_settings = build_load_settings(records, split)
        all_load_settings.extend(load_settings)
        write_json(
            output_dir / "splits" / f"{split}.json",
            {
                "split": split,
                "records": records,
                "load_settings": load_settings,
                "failures": [],
            },
        )

    write_jsonl(output_dir / "manifests" / "samples.jsonl", out_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", all_load_settings)
    (output_dir / "manifests" / "failures.jsonl").touch()

    branch_counts = Counter(str(rec["branch_id"]) for rec in out_records)
    load_counts = Counter(int(rec["load_setting_id"]) for rec in out_records)
    rhos = Counter(rho_key(rec["rho"]) for rec in out_records)
    summary = [
        {
            "split": "test",
            "samples": len(out_records),
            "load_settings": len(load_counts),
            "branches": len(branch_counts),
            "branch_ids": ",".join(sorted(branch_counts, key=int)),
            "rho_values": ",".join(sorted(rhos, key=float)),
            "max_samples_per_load": max(load_counts.values()),
        }
    ]
    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        summary,
        ["split", "samples", "load_settings", "branches", "branch_ids", "rho_values", "max_samples_per_load"],
    )
    write_csv(
        output_dir / "analysis" / "branch_counts.csv",
        [
            {"branch_id": branch_id, "samples": branch_counts[branch_id]}
            for branch_id in sorted(branch_counts, key=int)
        ],
        ["branch_id", "samples"],
    )
    write_csv(
        output_dir / "analysis" / "source_map.csv",
        source_map_rows,
        ["sample_id", "load_setting_id", "branch_id", "rho", "source_sample_id", "source_load_setting_id", "source_file"],
    )
    write_json(
        output_dir / "meta.json",
        {
            "source_root": str(source_root),
            "output_dir": str(output_dir),
            "total_samples": args.total_samples,
            "seed": args.seed,
            "selection": summary,
            "branch_counts": dict(sorted(branch_counts.items(), key=lambda item: int(item[0]))),
            "rho_counts": dict(rhos),
            "note": "Evaluation root only. train/val contain one placeholder sample because PFDelta-rho processing expects all split files.",
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary[0]}, indent=2))


if __name__ == "__main__":
    main()
