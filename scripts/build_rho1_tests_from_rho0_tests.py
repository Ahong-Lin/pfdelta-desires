#!/usr/bin/env python3
"""Build rho=1 test roots matching existing rho=0 test roots.

The script takes one or more existing PFDelta-rho rho=0 test roots, looks up
the same (load_setting_id, branch_id) pairs at rho=1 in a source rho bank, and
writes evaluation-only PFDelta-rho roots.  It also checks that the selected
load settings do not appear in train/val splits of existing experiment roots.
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


def collect_record_load_ids(record: dict) -> set[int]:
    load_ids = set()
    for key in ["load_setting_id", "source_load_setting_id", "source_base_sample_id"]:
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            load_ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return load_ids


def collect_excluded_loads(case_dir: Path, output_roots: set[Path]) -> tuple[set[int], list[dict]]:
    excluded: set[int] = set()
    rows: list[dict] = []

    for split_path in sorted(case_dir.glob("*/splits/train.json")):
        root = split_path.parents[1].resolve()
        if root in output_roots:
            continue
        # Test-only roots should not count as model training datasets.
        if root.name.startswith("rho_pf_test_"):
            continue

        for split in ["train", "val"]:
            path = root / "splits" / f"{split}.json"
            if not path.exists():
                continue
            data = read_json(path)
            before = len(excluded)
            record_count = 0
            for record in data.get("records", []):
                excluded.update(collect_record_load_ids(record))
                record_count += 1
            rows.append(
                {
                    "root": str(root),
                    "split": split,
                    "records": record_count,
                    "new_excluded_loads": len(excluded) - before,
                    "total_excluded_loads": len(excluded),
                }
            )
    return excluded, rows


def source_test_records(source_root: Path) -> list[dict]:
    split_path = source_root / "splits" / "test.json"
    data = read_json(split_path)
    records = data.get("records", [])
    if not records:
        raise RuntimeError(f"No test records found in {split_path}")
    for record in records:
        if rho_key(record.get("rho")) != "0":
            raise RuntimeError(f"Source test is not rho=0 only: {split_path}")
    return records


def collect_needed_keys(jobs: list[tuple[Path, Path]]) -> dict[tuple[int, str], list[dict]]:
    needed: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for source_root, _ in jobs:
        for record in source_test_records(source_root):
            load_id = int(record["load_setting_id"])
            branch_id = str(record["branch_id"])
            needed[(load_id, branch_id)].append(record)
    return needed


def load_rho1_index(bank_root: Path, needed_keys: set[tuple[int, str]]) -> dict[tuple[int, str], dict]:
    samples_path = bank_root / "manifests" / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing bank manifest: {samples_path}")

    found: dict[tuple[int, str], dict] = {}
    for record in read_jsonl(samples_path):
        if rho_key(record.get("rho")) != "1":
            continue
        key = (int(record["load_setting_id"]), str(record["branch_id"]))
        if key not in needed_keys or key in found:
            continue
        found[key] = record
        if len(found) == len(needed_keys):
            break

    missing = sorted(needed_keys - set(found), key=lambda item: (item[0], int(item[1])))
    if missing:
        preview = ", ".join(f"(load={load}, branch={branch})" for load, branch in missing[:10])
        raise RuntimeError(f"Missing {len(missing)} rho=1 bank samples. First missing: {preview}")
    return found


def make_output_record(source_rec: dict, sample_id: int, out_rel: str, bank_root: Path, output_name: str) -> dict:
    record = dict(source_rec)
    record["sample_id"] = sample_id
    record["file"] = out_rel
    record["split"] = "test"
    record["source_dataset_root"] = str(bank_root)
    record["source_sample_id"] = source_rec["sample_id"]
    record["source_sample_file"] = source_rec["file"]
    record["source_load_setting_id"] = source_rec["load_setting_id"]
    record["experiment_name"] = output_name
    return record


def build_load_settings(records: list[dict], split: str) -> list[dict]:
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


def prepare_output(output_dir: Path, clean: bool) -> None:
    if output_dir.exists():
        if not clean:
            raise FileExistsError(f"Output exists: {output_dir}. Use --clean to overwrite.")
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def build_one_root(
    source_root: Path,
    output_dir: Path,
    bank_root: Path,
    rho1_index: dict[tuple[int, str], dict],
    excluded_loads: set[int],
    clean: bool,
    copy_raw: bool,
) -> dict:
    source_records = source_test_records(source_root)
    source_loads = {int(record["load_setting_id"]) for record in source_records}
    leaked_loads = sorted(source_loads & excluded_loads)
    if leaked_loads:
        preview = ",".join(str(load_id) for load_id in leaked_loads[:20])
        raise RuntimeError(
            f"{source_root} has {len(leaked_loads)} test loads present in existing train/val. "
            f"First leaked load ids: {preview}"
        )

    prepare_output(output_dir, clean)

    out_records = []
    source_map_rows = []
    for sample_id, source_record in enumerate(source_records, start=1):
        key = (int(source_record["load_setting_id"]), str(source_record["branch_id"]))
        bank_record = rho1_index[key]
        out_rel = f"raw/sample_{sample_id}.json"
        link_or_copy(bank_root / bank_record["file"], output_dir / out_rel, copy_raw)
        out_record = make_output_record(bank_record, sample_id, out_rel, bank_root, output_dir.name)
        out_records.append(out_record)
        source_map_rows.append(
            {
                "sample_id": sample_id,
                "load_setting_id": out_record["load_setting_id"],
                "branch_id": out_record["branch_id"],
                "rho": rho_key(out_record["rho"]),
                "rho0_source_root": str(source_root),
                "rho0_source_sample_id": source_record["sample_id"],
                "rho1_source_sample_id": bank_record["sample_id"],
                "rho1_source_file": bank_record["file"],
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

    branch_counts = Counter(str(record["branch_id"]) for record in out_records)
    load_counts = Counter(int(record["load_setting_id"]) for record in out_records)
    rho_counts = Counter(rho_key(record["rho"]) for record in out_records)
    summary = {
        "split": "test",
        "samples": len(out_records),
        "load_settings": len(load_counts),
        "branches": len(branch_counts),
        "branch_ids": ",".join(sorted(branch_counts, key=int)),
        "rho_values": ",".join(sorted(rho_counts, key=float)),
        "max_samples_per_load": max(load_counts.values()),
        "excluded_train_val_overlap": 0,
    }
    write_csv(
        output_dir / "analysis" / "split_summary.csv",
        [summary],
        [
            "split",
            "samples",
            "load_settings",
            "branches",
            "branch_ids",
            "rho_values",
            "max_samples_per_load",
            "excluded_train_val_overlap",
        ],
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
        [
            "sample_id",
            "load_setting_id",
            "branch_id",
            "rho",
            "rho0_source_root",
            "rho0_source_sample_id",
            "rho1_source_sample_id",
            "rho1_source_file",
        ],
    )
    write_json(
        output_dir / "meta.json",
        {
            "source_rho0_root": str(source_root),
            "rho1_bank_root": str(bank_root),
            "output_dir": str(output_dir),
            "selection": summary,
            "note": (
                "Evaluation root only. train/val contain one placeholder sample "
                "because PFDelta-rho processing expects all split files."
            ),
        },
    )
    return summary


def parse_job(raw_job: str) -> tuple[Path, Path]:
    if "=" not in raw_job:
        raise ValueError("--job must be formatted as SOURCE_ROOT=OUTPUT_DIR")
    source, output = raw_job.split("=", 1)
    return Path(source).resolve(), Path(output).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--case-dir", default="data_generation_rho/case30")
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        help="SOURCE_RHO0_TEST_ROOT=OUTPUT_RHO1_TEST_ROOT. Can be repeated.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    bank_root = Path(args.bank_root).resolve()
    case_dir = Path(args.case_dir).resolve()
    jobs = [parse_job(job) for job in args.job]
    output_roots = {output for _, output in jobs}

    excluded_loads, excluded_rows = collect_excluded_loads(case_dir, output_roots)
    needed = collect_needed_keys(jobs)
    leaked_needed = sorted({load_id for load_id, _ in needed} & excluded_loads)
    if leaked_needed:
        preview = ",".join(str(load_id) for load_id in leaked_needed[:20])
        raise RuntimeError(
            f"Requested tests include {len(leaked_needed)} loads found in existing train/val. "
            f"First leaked load ids: {preview}"
        )

    rho1_index = load_rho1_index(bank_root, set(needed))
    summaries = []
    for source_root, output_dir in jobs:
        summary = build_one_root(
            source_root=source_root,
            output_dir=output_dir,
            bank_root=bank_root,
            rho1_index=rho1_index,
            excluded_loads=excluded_loads,
            clean=args.clean,
            copy_raw=args.copy_raw,
        )
        summaries.append({"output_dir": str(output_dir), **summary})

    for _, output_dir in jobs:
        write_csv(
            output_dir / "analysis" / "excluded_train_val_roots.csv",
            excluded_rows,
            ["root", "split", "records", "new_excluded_loads", "total_excluded_loads"],
        )

    print(json.dumps({"excluded_loads": len(excluded_loads), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
