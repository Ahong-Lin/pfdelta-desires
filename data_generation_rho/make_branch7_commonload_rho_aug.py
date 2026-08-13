#!/usr/bin/env python3
"""Build common-load branch-7 rho augmentation roots.

The train/val load settings are copied from an endpoint baseline root, while
the selected rho grid is looked up in one or more existing PF-rho banks.  This
keeps load diversity fixed and changes only the amount of mid-rho supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def rho_key(value: float | str) -> str:
    return f"{round(float(value), 10):.10g}"


def parse_rhos(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("At least one rho value is required.")
    return values


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


def split_load_ids(root: Path, split: str, branch_id: str) -> list[int]:
    payload = read_json(root / "splits" / f"{split}.json")
    load_ids = sorted(
        {
            int(record["load_setting_id"])
            for record in payload.get("records", [])
            if str(record.get("branch_id")) == branch_id
        }
    )
    if not load_ids:
        raise RuntimeError(f"No branch {branch_id} load settings in {root} split={split}")
    return load_ids


def add_bank_records(
    bank_root: Path,
    branch_id: str,
    wanted_rhos: set[str],
    sample_index: dict[tuple[int, str], dict],
    load_records: dict[int, dict],
) -> int:
    samples_path = bank_root / "manifests" / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing bank manifest: {samples_path}")

    load_path = bank_root / "manifests" / "load_settings.jsonl"
    if load_path.exists():
        for rec in read_jsonl(load_path):
            value = rec.get("load_setting_id")
            if value is None:
                continue
            load_id = int(value)
            load_records.setdefault(load_id, {**rec, "_source_load_bank": str(bank_root)})

    count = 0
    for rec in read_jsonl(samples_path):
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
        count += 1
    return count


def ensure_required(load_ids: list[int], rhos: list[float], sample_index: dict[tuple[int, str], dict]) -> None:
    missing = []
    for load_id in load_ids:
        for rho in rhos:
            key = (load_id, rho_key(rho))
            if key not in sample_index:
                missing.append(key)
    if missing:
        preview = ", ".join(f"{load}:{rho}" for load, rho in missing[:20])
        raise RuntimeError(f"Missing {len(missing)} samples. First missing: {preview}")


def make_record(source_rec: dict, sample_id: int, out_rel: str, split: str, output_name: str) -> dict:
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
    rec["experiment_name"] = output_name
    return rec


def build_split(
    output_dir: Path,
    split: str,
    load_ids: list[int],
    rhos: list[float],
    sample_index: dict[tuple[int, str], dict],
    load_records: dict[int, dict],
    sample_id_start: int,
    branch_id: str,
    copy_raw: bool,
) -> tuple[list[dict], list[dict], list[dict], int]:
    records = []
    load_rows = []
    source_map_rows = []
    sample_id = sample_id_start

    for load_id in load_ids:
        split_sample_ids = []
        for rho in rhos:
            key = (load_id, rho_key(rho))
            source_rec = sample_index[key]
            source_bank = Path(source_rec["_source_bank"])
            out_rel = f"raw/sample_{sample_id}.json"
            link_or_copy(source_bank / source_rec["file"], output_dir / out_rel, copy_raw)
            out_record = make_record(source_rec, sample_id, out_rel, split, output_dir.name)
            records.append(out_record)
            split_sample_ids.append(sample_id)
            source_map_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "load_setting_id": load_id,
                    "branch_id": branch_id,
                    "rho": rho_key(rho),
                    "source_dataset_root": source_bank,
                    "source_sample_id": source_rec["sample_id"],
                    "source_file": source_rec["file"],
                }
            )
            sample_id += 1

        load_row = dict(load_records.get(load_id, {"load_setting_id": load_id}))
        load_row.pop("_source_load_bank", None)
        load_row["split"] = split
        load_row["load_setting_id"] = load_id
        load_row["selected_branch_id"] = branch_id
        load_row["source_load_setting_id"] = load_id
        load_row["sample_ids"] = split_sample_ids
        load_row["rho_values"] = ",".join(rho_key(rho) for rho in rhos)
        load_row["num_selected_rhos"] = len(rhos)
        load_rows.append(load_row)

    return records, load_rows, source_map_rows, sample_id


def summarize_split(split: str, records: list[dict]) -> dict:
    load_ids = sorted({int(record["load_setting_id"]) for record in records})
    rho_counts = Counter(rho_key(record["rho"]) for record in records)
    return {
        "split": split,
        "load_settings": len(load_ids),
        "samples": len(records),
        "branches": len({str(record["branch_id"]) for record in records}),
        "branch_ids": ",".join(sorted({str(record["branch_id"]) for record in records}, key=int)),
        "rho_values": ",".join(sorted(rho_counts, key=float)),
        "rho_values_per_load": len(rho_counts),
        "rho_counts": ";".join(f"{rho}:{rho_counts[rho]}" for rho in sorted(rho_counts, key=float)),
        "min_load_id": min(load_ids),
        "max_load_id": max(load_ids),
    }


def prepare_output(output_dir: Path, clean: bool) -> None:
    if output_dir.exists():
        if not clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bank", action="append", required=True)
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--rhos", required=True)
    parser.add_argument("--test-rhos", default="0")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    endpoint_root = Path(args.endpoint_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    train_rhos = parse_rhos(args.rhos)
    val_rhos = list(train_rhos)
    test_rhos = parse_rhos(args.test_rhos)
    wanted_rhos = {rho_key(rho) for rho in train_rhos + val_rhos + test_rhos}

    train_loads = split_load_ids(endpoint_root, "train", args.branch_id)
    val_loads = split_load_ids(endpoint_root, "val", args.branch_id)
    test_loads = split_load_ids(endpoint_root, "test", args.branch_id)

    overlaps = {
        "train_val": len(set(train_loads) & set(val_loads)),
        "train_test": len(set(train_loads) & set(test_loads)),
        "val_test": len(set(val_loads) & set(test_loads)),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Endpoint split load overlap detected: {overlaps}")

    sample_index: dict[tuple[int, str], dict] = {}
    load_records: dict[int, dict] = {}
    bank_counts = []
    for bank in args.bank:
        bank_root = Path(bank).resolve()
        count = add_bank_records(bank_root, args.branch_id, wanted_rhos, sample_index, load_records)
        bank_counts.append({"bank": str(bank_root), "indexed_records": count})

    ensure_required(train_loads, train_rhos, sample_index)
    ensure_required(val_loads, val_rhos, sample_index)
    ensure_required(test_loads, test_rhos, sample_index)

    prepare_output(output_dir, args.clean)

    all_records = []
    all_load_rows = []
    source_map_rows = []
    summary_rows = []
    sample_id = 1
    specs = {
        "train": (train_loads, train_rhos),
        "val": (val_loads, val_rhos),
        "test": (test_loads, test_rhos),
    }
    for split in ["train", "val", "test"]:
        load_ids, rhos = specs[split]
        records, load_rows, map_rows, sample_id = build_split(
            output_dir,
            split,
            load_ids,
            rhos,
            sample_index,
            load_records,
            sample_id,
            args.branch_id,
            args.copy_raw,
        )
        write_json(
            output_dir / "splits" / f"{split}.json",
            {"split": split, "records": records, "load_settings": load_rows, "failures": []},
        )
        all_records.extend(records)
        all_load_rows.extend(load_rows)
        source_map_rows.extend(map_rows)
        summary_rows.append(summarize_split(split, records))

    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", all_load_rows)
    (output_dir / "manifests" / "failures.jsonl").touch()
    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        summary_rows,
        [
            "split",
            "load_settings",
            "samples",
            "branches",
            "branch_ids",
            "rho_values",
            "rho_values_per_load",
            "rho_counts",
            "min_load_id",
            "max_load_id",
        ],
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
            "source_file",
        ],
    )
    write_json(
        output_dir / "meta.json",
        {
            "experiment_name": output_dir.name,
            "endpoint_root": str(endpoint_root),
            "output_dir": str(output_dir),
            "branch_id": args.branch_id,
            "train_rhos": [rho_key(rho) for rho in train_rhos],
            "val_rhos": [rho_key(rho) for rho in val_rhos],
            "test_rhos": [rho_key(rho) for rho in test_rhos],
            "split_load_overlaps": overlaps,
            "bank_counts": bank_counts,
            "design": (
                "Common-load rho augmentation: train/val loads come from the "
                "endpoint baseline, with only the rho grid changed."
            ),
            "summary": summary_rows,
        },
    )

    print(json.dumps({"output_dir": str(output_dir), "summary": summary_rows, "overlaps": overlaps}, indent=2))


if __name__ == "__main__":
    main()
