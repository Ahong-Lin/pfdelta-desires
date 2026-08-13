#!/usr/bin/env python3
"""Build endpoint baseline plus nested uniform-rho extra datasets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_split_records(root: Path, split: str) -> list[dict]:
    return list(read_json(root / "splits" / f"{split}.json").get("records", []))


def sample_source_path(root: Path, record: dict) -> Path:
    return root / record["file"]


def copy_and_update_sample(
    source_root: Path,
    source_record: dict,
    output_dir: Path,
    output_rel: str,
    sample_id: int,
    split: str,
    task_name: str,
    source_role: str,
) -> None:
    dst = output_dir / output_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_source_path(source_root, source_record), dst)

    sample = read_json(dst)
    meta = sample.setdefault("rho_meta", {})
    meta["sample_id"] = sample_id
    meta["split"] = split
    meta["task_name"] = task_name
    meta["source_dataset_root"] = str(source_root)
    meta["source_sample_id"] = source_record.get("sample_id")
    meta["source_sample_file"] = source_record.get("file")
    meta["source_role"] = source_role
    write_json(dst, sample)


def make_output_record(
    source_root: Path,
    source_record: dict,
    output_rel: str,
    sample_id: int,
    split: str,
    task_name: str,
    source_role: str,
) -> dict:
    record = dict(source_record)
    record["sample_id"] = sample_id
    record["file"] = output_rel
    record["split"] = split
    record["task_name"] = task_name
    record["experiment_name"] = task_name
    record["source_dataset_root"] = str(source_root)
    record["source_sample_id"] = source_record.get("sample_id")
    record["source_sample_file"] = source_record.get("file")
    record["source_role"] = source_role
    return record


def make_load_row(record: dict, split: str, sample_id: int, source_role: str) -> dict:
    return {
        "split": split,
        "load_setting_id": int(record["load_setting_id"]),
        "base_sample_id": int(record.get("base_sample_id", record["load_setting_id"])),
        "source_file": record.get("source_file"),
        "sample_ids": [sample_id],
        "selected_branch_ids": str(record.get("branch_id", "")),
        "rho_values": f"{float(record['rho']):.12g}",
        "source_role": source_role,
    }


def add_records(
    output_dir: Path,
    task_name: str,
    source_root: Path,
    source_records: list[dict],
    split: str,
    source_role: str,
    next_sample_id: int,
    out_records: list[dict],
    out_loads: list[dict],
    source_map: list[dict],
) -> int:
    for source_record in source_records:
        out_rel = f"raw/sample_{next_sample_id}.json"
        copy_and_update_sample(
            source_root,
            source_record,
            output_dir,
            out_rel,
            next_sample_id,
            split,
            task_name,
            source_role,
        )
        out_record = make_output_record(
            source_root,
            source_record,
            out_rel,
            next_sample_id,
            split,
            task_name,
            source_role,
        )
        out_records.append(out_record)
        out_loads.append(make_load_row(source_record, split, next_sample_id, source_role))
        source_map.append(
            {
                "split": split,
                "sample_id": next_sample_id,
                "source_role": source_role,
                "load_setting_id": source_record.get("load_setting_id"),
                "branch_id": source_record.get("branch_id"),
                "rho": source_record.get("rho"),
                "source_dataset_root": str(source_root),
                "source_sample_id": source_record.get("sample_id"),
                "source_file": source_record.get("file"),
            }
        )
        next_sample_id += 1
    return next_sample_id


def write_summary(output_dir: Path, records_by_split: dict[str, list[dict]]) -> None:
    rows = []
    for split in ["train", "val", "test"]:
        records = records_by_split.get(split, [])
        loads = {int(row["load_setting_id"]) for row in records if "load_setting_id" in row}
        branches = Counter(str(row["branch_id"]) for row in records if "branch_id" in row)
        roles = Counter(str(row.get("source_role", "")) for row in records)
        endpoint = sum(1 for row in records if float(row.get("rho", -1.0)) in (0.0, 1.0))
        mid = len(records) - endpoint
        rows.append(
            {
                "split": split,
                "samples": len(records),
                "load_settings": len(loads),
                "branches": len(branches),
                "branch_ids": ",".join(sorted(branches, key=int)) if branches else "",
                "endpoint_samples": endpoint,
                "midrho_samples": mid,
                "source_roles": ";".join(f"{key}:{roles[key]}" for key in sorted(roles)),
            }
        )
    with (output_dir / "analysis" / "split_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "samples",
            "load_settings",
            "branches",
            "branch_ids",
            "endpoint_samples",
            "midrho_samples",
            "source_roles",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_ids(records: list[dict]) -> set[int]:
    return {int(record["load_setting_id"]) for record in records if "load_setting_id" in record}


def build_dataset(
    base_root: Path,
    extra_root: Path,
    output_dir: Path,
    task_name: str,
    extra_count: int,
    clean: bool,
    require_extra_loads_in_base_train: bool,
) -> None:
    if output_dir.exists():
        if not clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    base_train = load_split_records(base_root, "train")
    base_val = load_split_records(base_root, "val")
    extra_train = load_split_records(extra_root, "train")
    if len(extra_train) < extra_count:
        raise RuntimeError(f"Need {extra_count} extra train records, found {len(extra_train)}")

    selected_extra = extra_train[:extra_count]
    if require_extra_loads_in_base_train:
        base_train_loads = load_ids(base_train)
        selected_extra_loads = load_ids(selected_extra)
        missing_loads = selected_extra_loads - base_train_loads
        if missing_loads:
            preview = ", ".join(str(value) for value in sorted(missing_loads)[:10])
            raise RuntimeError(
                "Extra samples use load settings outside the endpoint baseline train split: "
                f"{preview}"
            )
        if len(selected_extra_loads) != len(selected_extra):
            raise RuntimeError(
                "Same-load rho augmentation must contain one extra sample per load setting; "
                f"got {len(selected_extra)} samples from {len(selected_extra_loads)} load settings"
            )

    records_by_split = {"train": [], "val": [], "test": []}
    loads_by_split = {"train": [], "val": [], "test": []}
    source_map: list[dict] = []
    next_sample_id = 1

    next_sample_id = add_records(
        output_dir,
        task_name,
        base_root,
        base_train,
        "train",
        "endpoint_baseline_train",
        next_sample_id,
        records_by_split["train"],
        loads_by_split["train"],
        source_map,
    )
    next_sample_id = add_records(
        output_dir,
        task_name,
        extra_root,
        selected_extra,
        "train",
        f"uniformrho_extra_{extra_count}",
        next_sample_id,
        records_by_split["train"],
        loads_by_split["train"],
        source_map,
    )
    next_sample_id = add_records(
        output_dir,
        task_name,
        base_root,
        base_val,
        "val",
        "endpoint_baseline_val",
        next_sample_id,
        records_by_split["val"],
        loads_by_split["val"],
        source_map,
    )

    for split in ["train", "val", "test"]:
        write_json(
            output_dir / "splits" / f"{split}.json",
            {
                "split": split,
                "records": records_by_split[split],
                "load_settings": loads_by_split[split],
                "failures": [],
            },
        )

    all_records = records_by_split["train"] + records_by_split["val"]
    all_loads = loads_by_split["train"] + loads_by_split["val"]
    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", all_loads)
    (output_dir / "manifests" / "failures.jsonl").touch()
    write_summary(output_dir, records_by_split)

    with (output_dir / "analysis" / "source_map.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "sample_id",
            "source_role",
            "load_setting_id",
            "branch_id",
            "rho",
            "source_dataset_root",
            "source_sample_id",
            "source_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_map)

    write_json(
        output_dir / "meta.json",
        {
            "task_name": task_name,
            "base_root": str(base_root),
            "extra_root": str(extra_root),
            "extra_count": extra_count,
            "train_samples": len(records_by_split["train"]),
            "val_samples": len(records_by_split["val"]),
            "test_samples": 0,
            "require_extra_loads_in_base_train": require_extra_loads_in_base_train,
            "note": "Train is endpoint baseline train plus nested uniform-rho extra samples. Val is copied exactly from the endpoint baseline.",
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "task_name": task_name,
                "train_samples": len(records_by_split["train"]),
                "val_samples": len(records_by_split["val"]),
                "extra_count": extra_count,
            },
            indent=2,
        )
    )


def parse_counts(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--extra-root", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--counts", default="5000,10000,20000")
    parser.add_argument("--name-prefix", default="rho_pf_endpoint_pfdelta_style_randomedge_unpaired_41976_plus_uniformrho_extra")
    parser.add_argument(
        "--name-template",
        default="{prefix}{extra_count}_{train_total}",
        help="Format fields: prefix, extra_count, train_total.",
    )
    parser.add_argument(
        "--require-extra-loads-in-base-train",
        action="store_true",
        help="Fail unless every extra sample uses a distinct load setting in the base train split.",
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    base_root = Path(args.base_root).resolve()
    extra_root = Path(args.extra_root).resolve()
    output_parent = Path(args.output_parent).resolve()
    base_train_count = len(load_split_records(base_root, "train"))
    for extra_count in parse_counts(args.counts):
        task_name = args.name_template.format(
            prefix=args.name_prefix,
            extra_count=extra_count,
            train_total=base_train_count + extra_count,
        )
        build_dataset(
            base_root,
            extra_root,
            output_parent / task_name,
            task_name,
            extra_count,
            args.clean,
            args.require_extra_loads_in_base_train,
        )


if __name__ == "__main__":
    main()
