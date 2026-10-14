#!/usr/bin/env python3
"""Small, locked launcher. Output data never depends on a server-specific path."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from contextlib import contextmanager

BUNDLE = Path(__file__).resolve().parents[1]


def environment():
    env = os.environ.copy()
    env.setdefault("JULIA_DEPOT_PATH", str(BUNDLE / ".julia_depot"))
    env.setdefault("JULIA_NUM_THREADS", "1")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def julia_executable():
    local = BUNDLE / ".runtime/julia-1.9.4/bin/julia"
    value = os.environ.get("JULIA_BIN", str(local) if local.is_file() else "julia")
    found = shutil.which(value)
    if not found:
        raise RuntimeError("Julia is unavailable. Run setup_env.sh; set JULIA_BIN if needed.")
    return found


def invoke(command, env):
    print("RUN:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=BUNDLE, env=env, check=True)


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".pipeline.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another pipeline is already using this output directory.") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield


def storage_report(root, allow_tmpfs=False):
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    print(f"Output: {root}\nAvailable disk: {usage.free / 2**30:.1f} GiB", flush=True)
    if sys.platform == "linux" and shutil.which("findmnt"):
        query = subprocess.run(
            ["findmnt", "-n", "-o", "FSTYPE,TARGET", "-T", str(root)],
            text=True, capture_output=True, check=False,
        )
        print("Filesystem:", query.stdout.strip() or "unknown", flush=True)
        if query.stdout.split()[:1] == ["tmpfs"] and not allow_tmpfs:
            raise RuntimeError("Output is on tmpfs. Choose persistent storage (or explicitly --allow-tmpfs for smoke only).")


def status(root):
    for name in ("state.json", "splits/audit.json", "splits/split_audit.json"):
        path = root / name
        if path.exists():
            print(f"\n{name}")
            content = json.loads(path.read_text())
            if name == "state.json":
                content = {k: v for k, v in content.items() if not k.startswith("branch_")}
            else:
                content = {
                    "parent_overlap_train_val_test": content.get("parent_overlap_train_val_test"),
                    "raw_metadata_verified": content.get("raw_metadata_verified"),
                    "splits": {key: {field: value.get(field) for field in (
                        "parents", "endpoint_records", "branch_count_min_max",
                        "cluster_total_variation_vs_pool")}
                        for key, value in content.get("splits", {}).items()},
                }
            print(json.dumps(content, ensure_ascii=False, indent=2))
    planpath = root / "plan.json"
    if planpath.exists():
        plan = json.loads(planpath.read_text())
        print(f"Planned endpoints: {plan['target_endpoint_records']:,}")
    else:
        print("No generation plan found yet.")


def estimate(root, config):
    """Use completed smoke raw bytes, not an invented per-record storage figure."""
    plan = json.loads((root / "plan.json").read_text())
    manifest = root / "manifests/parents.jsonl"
    sizes = []
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            sizes.extend((root / file).stat().st_size for file in row["files"])
    if not sizes:
        return
    # Supplied case has 502 eligible branches. Read the full value from plan
    # when available; smoke's eligible count is deliberately smaller.
    eligible = int(plan.get("full_eligible_branch_count", 502))
    total = int(config["train_endpoint_records"]) + 2 * eligible * (
        int(config["val_parents_per_branch"]) + int(config["test_parents_per_branch"])
    )
    projected = sum(sizes) / len(sizes) * total / 2**30
    report = {
        "measured_endpoint_files": len(sizes),
        "measured_mean_raw_bytes": sum(sizes) / len(sizes),
        "full_target_endpoint_records": total,
        "projected_raw_GiB": projected,
        "note": "Excludes manifests, sampler state, input memmap, future graph caches and filesystem overhead; use additional headroom. Four smoke branches do not characterize acceptance rate or production runtime.",
    }
    (root / "storage_estimate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Measured raw storage projection: {projected:.1f} GiB for {total:,} endpoints; reserve extra space for indexes/caches.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "smoke", "full", "split", "verify", "status"])
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--config", type=Path, default=BUNDLE / "config.json")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--allow-tmpfs", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    env = environment()
    if args.action == "plan":
        invoke([julia_executable(), "--startup-file=no", f"--project={BUNDLE / 'data_generation'}",
                BUNDLE / "data_generation/generate_case500.jl", "--config", config_path,
                "--mode", "full", "--plan"], env)
        return
    if args.output is None:
        parser.error("Supply an absolute output path on persistent storage.")
    if not args.output.is_absolute():
        parser.error("output must be an absolute path")
    root = args.output.expanduser().resolve()
    if args.action == "status":
        status(root)
        return
    if args.action == "full" and args.allow_tmpfs:
        parser.error("--allow-tmpfs is only intended for smoke runs")
    with output_lock(root):
        storage_report(root, args.allow_tmpfs)
        if args.action in ("smoke", "full"):
            cmd = [julia_executable(), "--startup-file=no", f"--project={BUNDLE / 'data_generation'}",
                   BUNDLE / "data_generation/generate_case500.jl", "--config", config_path,
                   "--mode", args.action, "--output", root]
            if args.workers is not None:
                cmd.extend(["--workers", args.workers])
            invoke(cmd, env)
        invoke([sys.executable, BUNDLE / "scripts/split_dataset.py", "--root", root,
                "--config", config_path, "--verify-raw"], env)
        if args.action == "smoke":
            estimate(root, config)
        print("Complete. Frozen data indexes are under:", root / "splits", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
