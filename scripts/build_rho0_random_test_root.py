#!/usr/bin/env python3
"""Build a small random rho=0 PFDelta-rho test root.

The output uses the PFDelta-rho reader layout and is intended for evaluation
only.  The same records are written to train/val/test split files so the
current PFDelta-rho processor, which materializes all three split files, can
build its processed cache.  Evaluation should use ``--split test``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from pathlib import Path
from typing import Any


SOURCE_BANKS = [
    ("old30", "case30/rho_bank_normal_loop_edges_30_0.01"),
    ("extra50", "case30/rho_bank_normal_loop_edges_extra50_0.01"),
    ("new50", "case30/rho_bank_normal_loop_edges_new50_0.01"),
]


def rho_key(value: Any) -> str:
    return f"{round(float(value), 10):.10g}"


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
    loads = sorted(sample["network"]["load"].items(), key=lambda item: int(item[0]))
    return [float(load["pd"]) for _, load in loads] + [
        float(load["qd"]) for _, load in loads
    ]


def collect_candidates(source_root: Path, branch_id: str) -> list[tuple[str, str, dict[str, Any]]]:
    candidates = []
    seen_loads = set()
    for source_tag, rel_dir in SOURCE_BANKS:
        bank_dir = source_root / rel_dir
        for rec in read_jsonl(bank_dir / "manifests" / "samples.jsonl"):
            if str(rec.get("branch_id")) != branch_id:
                continue
            if rho_key(rec.get("rho")) != "0":
                continue
            load_key = (source_tag, int(rec["load_setting_id"]))
            if load_key in seen_loads:
                continue
            seen_loads.add(load_key)
            candidates.append((source_tag, rel_dir, rec))
    return candidates


def prepare_output(output_dir: Path, clean: bool) -> None:
    if output_dir.exists():
        if not clean:
            raise FileExistsError(
                f"Output dir already exists: {output_dir}. "
                "Use --clean if you want to overwrite it."
            )
        shutil.rmtree(output_dir)
    for subdir in ["raw", "splits", "manifests", "analysis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests" / "failures.jsonl").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data_generation_rho")
    parser.add_argument(
        "--output-dir",
        default="data_generation_rho/case30/rho_exp_branch7_random13_rho0_seed20260714",
    )
    parser.add_argument("--branch-id", default="7")
    parser.add_argument("--num-samples", type=int, default=13)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    branch_id = str(args.branch_id)

    prepare_output(output_dir, args.clean)

    candidates = collect_candidates(source_root, branch_id)
    if len(candidates) < args.num_samples:
        raise RuntimeError(
            f"Need {args.num_samples} branch={branch_id}, rho=0 samples, "
            f"found {len(candidates)}."
        )

    rng = random.Random(args.seed)
    selected = candidates[:]
    rng.shuffle(selected)
    selected = selected[: args.num_samples]

    records = []
    load_settings = []
    source_rows = []
    task_name = output_dir.name

    for sample_id, (source_tag, rel_dir, source_rec) in enumerate(selected, start=1):
        source_file = source_root / rel_dir / source_rec["file"]
        sample = copy.deepcopy(read_json(source_file))
        rho_meta = sample.setdefault("rho_meta", {})
        source_load_setting_id = source_rec.get("load_setting_id")
        source_sample_id = source_rec.get("sample_id")
        load_setting_id = sample_id

        rho_meta.update(
            {
                "sample_id": sample_id,
                "load_setting_id": load_setting_id,
                "split": "test",
                "task_name": task_name,
                "source_kind": "random_rho0_test",
                "source_bank": source_tag,
                "source_file": source_rec["file"],
                "source_sample_id": source_sample_id,
                "source_load_setting_id": source_load_setting_id,
            }
        )

        out_rel = f"raw/sample_{sample_id}.json"
        with (output_dir / out_rel).open("w") as f:
            json.dump(sample, f, separators=(",", ":"))

        rec = dict(source_rec)
        rec.update(
            {
                "sample_id": sample_id,
                "file": out_rel,
                "split": "test",
                "load_setting_id": load_setting_id,
                "source_kind": "random_rho0_test",
                "source_bank": source_tag,
                "source_file": source_rec["file"],
                "source_sample_id": source_sample_id,
                "source_load_setting_id": source_load_setting_id,
            }
        )
        records.append(rec)
        append_jsonl(output_dir / "manifests" / "samples.jsonl", rec)

        load_rec = {
            "load_setting_id": load_setting_id,
            "split": "test",
            "load_vector": load_vector(sample),
            "sample_ids": [sample_id],
            "selected_branch_id": branch_id,
            "source_kind": "random_rho0_test",
            "source_bank": source_tag,
            "source_load_setting_id": source_load_setting_id,
        }
        load_settings.append(load_rec)
        append_jsonl(output_dir / "manifests" / "load_settings.jsonl", load_rec)

        source_rows.append(
            {
                "sample_id": sample_id,
                "load_setting_id": load_setting_id,
                "source_bank": source_tag,
                "source_sample_id": source_sample_id,
                "source_load_setting_id": source_load_setting_id,
                "source_file": source_rec["file"],
                "rho": rec["rho"],
                "branch_id": rec["branch_id"],
            }
        )

    split_payload = {
        "records": records,
        "load_settings": load_settings,
        "failures": [],
    }
    for split in ["train", "val", "test"]:
        payload = {"split": split, **split_payload}
        write_json(output_dir / "splits" / f"{split}.json", payload)

    with (output_dir / "analysis" / "source_map.csv").open("w") as f:
        columns = list(source_rows[0].keys())
        f.write(",".join(columns) + "\n")
        for row in source_rows:
            f.write(",".join(str(row[col]) for col in columns) + "\n")

    write_json(
        output_dir / "meta.json",
        {
            "task_name": task_name,
            "case_name": "case30",
            "branch_id": branch_id,
            "rho": 0.0,
            "seed": args.seed,
            "num_samples": len(records),
            "source_banks": dict(SOURCE_BANKS),
            "note": (
                "Evaluation root only. train/val/test split files intentionally "
                "contain the same records because PFDelta-rho processing builds "
                "all split caches before loading the requested split."
            ),
        },
    )

    print(f"Output: {output_dir}")
    print(f"num_samples={len(records)}")
    print("source_map:")
    for row in source_rows:
        print(row)


if __name__ == "__main__":
    main()
