#!/usr/bin/env python3
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys


def read_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_rhos(values):
    return ",".join(f"{x:g}" for x in sorted(values))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_rho_bank.py <rho_bank_dir>")

    base = Path(sys.argv[1])
    out = base / "analysis"
    out.mkdir(exist_ok=True)

    samples = read_jsonl(base / "manifests" / "samples.jsonl")
    failures = read_jsonl(base / "manifests" / "failures.jsonl")
    loads = read_jsonl(base / "manifests" / "load_settings.jsonl")

    success_by_lb = defaultdict(list)
    failure_by_lb = defaultdict(list)
    branch_info = {}

    for rec in samples:
        key = (rec["split"], int(rec["load_setting_id"]), str(rec["branch_id"]))
        success_by_lb[key].append(float(rec["rho"]))
        branch_info[str(rec["branch_id"])] = (rec.get("f_bus"), rec.get("t_bus"))

    for rec in failures:
        key = (rec["split"], int(rec["load_setting_id"]), str(rec["branch_id"]))
        failure_by_lb[key].append(float(rec["rho"]))
        branch_info[str(rec["branch_id"])] = (rec.get("f_bus"), rec.get("t_bus"))

    split_order = {"train": 0, "val": 1, "test": 2}
    all_keys = sorted(
        set(success_by_lb) | set(failure_by_lb),
        key=lambda x: (split_order.get(x[0], 9), x[1], int(x[2])),
    )

    with (out / "success_by_load_branch.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "load_setting_id",
            "branch_id",
            "f_bus",
            "t_bus",
            "success_rho_count",
            "success_rhos",
            "failed_rho_count",
            "failed_rhos",
        ])
        for split, load_setting_id, branch_id in all_keys:
            success_rhos = success_by_lb.get((split, load_setting_id, branch_id), [])
            failed_rhos = failure_by_lb.get((split, load_setting_id, branch_id), [])
            f_bus, t_bus = branch_info.get(branch_id, ("", ""))
            writer.writerow([
                split,
                load_setting_id,
                branch_id,
                f_bus,
                t_bus,
                len(success_rhos),
                fmt_rhos(success_rhos),
                len(failed_rhos),
                fmt_rhos(failed_rhos),
            ])

    load_split = {int(rec["load_setting_id"]): rec["split"] for rec in loads}
    load_success = Counter(int(rec["load_setting_id"]) for rec in samples)
    load_failure = Counter(int(rec["load_setting_id"]) for rec in failures)

    with (out / "success_by_load_setting.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "load_setting_id", "success_samples", "failure_samples", "attempted", "success_rate"])
        for load_setting_id in sorted(set(load_split) | set(load_success) | set(load_failure)):
            success = load_success[load_setting_id]
            failure = load_failure[load_setting_id]
            total = success + failure
            writer.writerow([
                load_split.get(load_setting_id, ""),
                load_setting_id,
                success,
                failure,
                total,
                round(success / total, 6) if total else "",
            ])

    rho_success = Counter(float(rec["rho"]) for rec in samples)
    rho_failure = Counter(float(rec["rho"]) for rec in failures)

    with (out / "success_by_rho.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rho", "success_samples", "failure_samples", "attempted", "success_rate"])
        for rho in sorted(set(rho_success) | set(rho_failure)):
            success = rho_success[rho]
            failure = rho_failure[rho]
            total = success + failure
            writer.writerow([f"{rho:g}", success, failure, total, round(success / total, 6) if total else ""])

    branch_success = Counter(str(rec["branch_id"]) for rec in samples)
    branch_failure = Counter(str(rec["branch_id"]) for rec in failures)

    with (out / "success_by_branch.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["branch_id", "f_bus", "t_bus", "success_samples", "failure_samples", "attempted", "success_rate"])
        for branch_id in sorted(set(branch_success) | set(branch_failure), key=int):
            success = branch_success[branch_id]
            failure = branch_failure[branch_id]
            total = success + failure
            f_bus, t_bus = branch_info.get(branch_id, ("", ""))
            writer.writerow([branch_id, f_bus, t_bus, success, failure, total, round(success / total, 6) if total else ""])

    split_success = Counter(rec["split"] for rec in samples)
    split_failure = Counter(rec["split"] for rec in failures)

    print("OUTPUT_DIR", base)
    print("ANALYSIS_DIR", out)
    print()
    print("SPLIT SUMMARY")
    for split in ["train", "val", "test"]:
        success = split_success[split]
        failure = split_failure[split]
        total = success + failure
        print(f"{split:5s} success={success:5d} failure={failure:5d} attempted={total:5d} success_rate={success / total:.3f}")

    print()
    print("LOAD SETTING SUMMARY")
    for load_setting_id in sorted(load_split):
        success = load_success[load_setting_id]
        failure = load_failure[load_setting_id]
        total = success + failure
        print(
            f"{load_split[load_setting_id]:5s} load_setting_id={load_setting_id:2d} "
            f"success={success:3d}/374 failure={failure:3d} rate={success / total:.3f}"
        )

    print()
    print("RHO SUMMARY")
    for rho in sorted(set(rho_success) | set(rho_failure)):
        success = rho_success[rho]
        failure = rho_failure[rho]
        total = success + failure
        print(f"rho={rho:>3g} success={success:4d}/{total:4d} rate={success / total:.3f}")

    print()
    print("BEST/WORST BRANCHES BY SUCCESS RATE")
    branch_rows = []
    for branch_id in sorted(set(branch_success) | set(branch_failure), key=int):
        success = branch_success[branch_id]
        failure = branch_failure[branch_id]
        total = success + failure
        f_bus, t_bus = branch_info.get(branch_id, ("", ""))
        branch_rows.append((success / total if total else 0.0, branch_id, f_bus, t_bus, success, total))

    for rate, branch_id, f_bus, t_bus, success, total in sorted(branch_rows)[:8]:
        print(f"low  branch={branch_id:>2s} {f_bus}-{t_bus:<2} success={success:3d}/{total:3d} rate={rate:.3f}")
    for rate, branch_id, f_bus, t_bus, success, total in sorted(branch_rows, reverse=True)[:8]:
        print(f"high branch={branch_id:>2s} {f_bus}-{t_bus:<2} success={success:3d}/{total:3d} rate={rate:.3f}")

    print()
    print("CSV FILES")
    for name in [
        "success_by_load_branch.csv",
        "success_by_load_setting.csv",
        "success_by_rho.csv",
        "success_by_branch.csv",
    ]:
        print(out / name)


if __name__ == "__main__":
    main()
