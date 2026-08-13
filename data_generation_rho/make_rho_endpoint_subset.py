#!/usr/bin/env python3
"""Create an endpoint-only subset from an existing PFDelta-rho root."""

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


def load_settings_for_records(records: list[dict], split: str) -> list[dict]:
    by_load: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_load[int(record["load_setting_id"])].append(record)

    rows = []
    for load_id in sorted(by_load):
        load_records = by_load[load_id]
        rows.append(
            {
                "split": split,
                "load_setting_id": load_id,
                "sample_ids": [int(record["sample_id"]) for record in load_records],
                "selected_branch_ids": ",".join(
                    sorted({str(record["branch_id"]) for record in load_records}, key=int)
                ),
                "rho_values": ",".join(
                    sorted({rho_key(record["rho"]) for record in load_records}, key=float)
                ),
                "num_samples": len(load_records),
            }
        )
    return rows


def make_record(
    source_record: dict,
    sample_id: int,
    out_rel: str,
    split: str,
    source_root: Path,
    output_name: str,
) -> dict:
    record = dict(source_record)
    record["sample_id"] = sample_id
    record["file"] = out_rel
    record["split"] = split
    record["source_dataset_root"] = str(source_root)
    record["source_sample_id"] = source_record["sample_id"]
    record["source_sample_file"] = source_record["file"]
    record["source_load_setting_id"] = source_record["load_setting_id"]
    record["experiment_name"] = output_name
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rhos", default="0,1")
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    selected_rhos = {rho_key(value) for value in args.rhos.split(",") if value.strip()}
    branch_id = str(args.branch_id)

    if output_dir.exists():
        if not args.clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    all_records = []
    load_rows = []
    source_map_rows = []
    summary_rows = []
    sample_id = 1

    for split in ["train", "val", "test"]:
        source_split = read_json(source_root / "splits" / f"{split}.json")
        filtered_sources = [
            record
            for record in source_split.get("records", [])
            if str(record.get("branch_id")) == branch_id
            and rho_key(record.get("rho")) in selected_rhos
        ]
        if not filtered_sources:
            raise RuntimeError(f"No records selected for split={split}")

        out_records = []
        for source_record in filtered_sources:
            out_rel = f"raw/sample_{sample_id}.json"
            link_or_copy(source_root / source_record["file"], output_dir / out_rel, args.copy_raw)
            record = make_record(
                source_record=source_record,
                sample_id=sample_id,
                out_rel=out_rel,
                split=split,
                source_root=source_root,
                output_name=output_dir.name,
            )
            out_records.append(record)
            all_records.append(record)
            source_map_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "load_setting_id": record["load_setting_id"],
                    "branch_id": record["branch_id"],
                    "rho": rho_key(record["rho"]),
                    "source_sample_id": record["source_sample_id"],
                    "source_load_setting_id": record["source_load_setting_id"],
                    "source_file": record["source_sample_file"],
                }
            )
            sample_id += 1

        split_loads = load_settings_for_records(out_records, split)
        load_rows.extend(split_loads)
        write_json(
            output_dir / "splits" / f"{split}.json",
            {
                "split": split,
                "records": out_records,
                "load_settings": split_loads,
                "failures": [],
            },
        )

        loads = {int(record["load_setting_id"]) for record in out_records}
        rho_counts = Counter(rho_key(record["rho"]) for record in out_records)
        summary_rows.append(
            {
                "split": split,
                "samples": len(out_records),
                "load_settings": len(loads),
                "branches": len({str(record["branch_id"]) for record in out_records}),
                "branch_ids": ",".join(sorted({str(record["branch_id"]) for record in out_records}, key=int)),
                "rho_values": ",".join(sorted(rho_counts, key=float)),
                "rho_counts": ";".join(
                    f"{rho}:{rho_counts[rho]}" for rho in sorted(rho_counts, key=float)
                ),
            }
        )

    write_jsonl(output_dir / "manifests" / "samples.jsonl", all_records)
    write_jsonl(output_dir / "manifests" / "load_settings.jsonl", load_rows)
    (output_dir / "manifests" / "failures.jsonl").touch()
    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        summary_rows,
        ["split", "samples", "load_settings", "branches", "branch_ids", "rho_values", "rho_counts"],
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
            "source_sample_id",
            "source_load_setting_id",
            "source_file",
        ],
    )

    train_loads = {
        int(record["load_setting_id"])
        for record in read_json(output_dir / "splits" / "train.json")["records"]
    }
    val_loads = {
        int(record["load_setting_id"])
        for record in read_json(output_dir / "splits" / "val.json")["records"]
    }
    test_loads = {
        int(record["load_setting_id"])
        for record in read_json(output_dir / "splits" / "test.json")["records"]
    }
    overlaps = {
        "train_val": sorted(train_loads & val_loads),
        "train_test": sorted(train_loads & test_loads),
        "val_test": sorted(val_loads & test_loads),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Split load settings overlap: {overlaps}")

    write_json(
        output_dir / "meta.json",
        {
            "source_root": str(source_root),
            "output_dir": str(output_dir),
            "branch_id": branch_id,
            "selected_rhos": sorted(selected_rhos, key=float),
            "selection": summary_rows,
            "split_load_overlaps": overlaps,
            "design": "Endpoint-only subset from an existing rho6 branch7 benchmark.",
        },
    )

    print(json.dumps({"output_dir": str(output_dir), "selection": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
