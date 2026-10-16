#!/usr/bin/env python3
"""Resume a frozen experiment with explicit GPUs, three independent runs per GPU.

No automatic repeated retry: failed runs are skipped unless --retry-failed is
specified, and each run is launched at most once per invocation. A successful
run is skipped only after validating its completion/config/data identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from configure_runs import fingerprint, read_json, file_sha256, valid_sha256

ROOT = Path(__file__).resolve().parents[1]
STOP = False


def stop_requested(_signum, _frame):
    global STOP
    STOP = True


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def parse_gpus(value):
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item or not (item.isdigit() or item.startswith("GPU-")) for item in items):
        raise ValueError("Use explicit GPU ordinals or UUIDs, e.g. --gpus 0,1,2")
    if len(items) != len(set(items)):
        raise ValueError("GPU IDs must not repeat")
    return items


def validate_gpu_allocation(gpus, environment=None):
    """GPU arguments use the literal identifiers in any inherited allocation."""
    environment = os.environ if environment is None else environment
    inherited = environment.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return
    allowed = {item.strip() for item in inherited.split(",") if item.strip()}
    if not set(gpus) <= allowed:
        raise ValueError(
            f"Requested GPU IDs {gpus} conflict with inherited CUDA_VISIBLE_DEVICES={inherited!r}. "
            "Use identifiers from that allocation, not renumbered visible ordinals; this launcher will not expand it."
        )


def verify_bound_files(entries):
    """Stream each unique frozen data file once before starting any child."""
    bindings = {}
    fields = ["data_index", "parent_manifest"] + [
        f"{split}_{kind}" for split in ("train", "val", "test") for kind in ("cache", "parent_ids")]
    for entry in entries:
        config = read_json(entry["config"])
        if fingerprint(config) != entry["config_hash"]:
            raise ValueError(f"Frozen configuration was changed: {entry['config']}")
        for field in fields:
            path = str(Path(config[field]).resolve())
            digest = config.get(field + "_sha256")
            if not valid_sha256(digest):
                raise ValueError(f"Missing or invalid SHA256 binding for {field}: {entry['config']}")
            if path in bindings and bindings[path] != digest:
                raise ValueError(f"Conflicting frozen hashes for the same data file: {path}")
            bindings[path] = digest
    for number, (path, expected) in enumerate(bindings.items(), 1):
        size = Path(path).stat().st_size
        print(f"Verifying frozen file {number}/{len(bindings)} ({size / 2**30:.3f} GiB): {path}", flush=True)
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen data content changed: {path}; expected {expected}, got {actual}")
    print(f"Verified {len(bindings)} unique frozen data files before launch.", flush=True)
    return len(bindings)


def locked_run(path):
    path = Path(path) / ".run.lock"
    if not path.exists():
        return False
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def classify(entry, prior=None):
    config = read_json(entry["config"])
    if fingerprint(config) != entry["config_hash"]:
        raise ValueError(f"Frozen configuration was changed: {entry['config']}")
    if config["run_dir"] != entry["run_dir"]:
        raise ValueError("Run directory differs from the frozen manifest")
    run_dir = Path(entry["run_dir"])
    completion = run_dir / "Evaluation/completion.json"
    if completion.exists():
        value = read_json(completion)
        if value.get("finished") and not value.get("smoke_only", False):
            if value.get("config_hash") != entry["config_hash"] or value.get("data_hash") != config["data_hash"]:
                raise ValueError(f"Completed run has different provenance: {run_dir}")
            return "completed"
    if locked_run(run_dir):
        return "running"
    progress = run_dir / "progress.json"
    if progress.exists() and read_json(progress).get("status") == "failed":
        return "failed"
    if prior and (prior.get("status") == "failed" or (prior.get("status") != "interrupted" and prior.get("exit_code") not in (None, 0, 130, 143))):
        return "failed"
    return "pending"


def inspect_runs(manifest_path, gpus, jobs_per_gpu, retry_failed=False):
    if jobs_per_gpu < 1:
        raise ValueError("jobs-per-gpu must be positive")
    manifest = read_json(manifest_path)
    if not manifest.get("runs"):
        raise ValueError("Frozen manifest contains no runs")
    state_path = Path(manifest_path).parent / "queue_state.json"
    previous = read_json(state_path).get("runs", {}) if state_path.exists() else {}
    states = {entry["config"]: classify(entry, previous.get(entry["config"])) for entry in manifest["runs"]}
    pending = [entry for entry in manifest["runs"]
               if states[entry["config"]] == "pending" or (retry_failed and states[entry["config"]] == "failed")]
    slots = [gpu for gpu in gpus for _ in range(jobs_per_gpu)]
    return manifest, states, pending, slots, previous


def gpu_preflight(gpus):
    """Check the explicitly selected cards once; do not launch 70 doomed jobs."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    script = """import json, sys, torch
expected = int(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
    raise SystemExit(f"CUDA preflight failed: requested {expected} devices, visible {torch.cuda.device_count()}")
print(json.dumps({"cuda_runtime":torch.version.cuda,"devices":[torch.cuda.get_device_name(i) for i in range(expected)]}))
"""
    subprocess.run([sys.executable, "-c", script, str(len(gpus))],
                   env=environment, check=True, timeout=90)


def run_queue(manifest_path, gpus, jobs_per_gpu=3, retry_failed=False, dry_run=False, poll_seconds=10):
    validate_gpu_allocation(gpus)
    manifest_path = Path(manifest_path).resolve()
    manifest, states, pending, slots, previous = inspect_runs(manifest_path, gpus, jobs_per_gpu, retry_failed)
    counts = dict(Counter(states.values()))
    print(json.dumps({"status": counts, "gpus": gpus, "jobs_per_gpu": jobs_per_gpu,
                      "total_slots": len(slots), "eligible_this_invocation": len(pending),
                      "dry_run": dry_run}, indent=2), flush=True)
    if dry_run:
        for slot, entry in zip(slots, pending):
            print(f"GPU {slot}: {entry['scale']} seed{entry['seed']} -> {entry['run_dir']}")
        return 0
    if any(status == "running" for status in states.values()):
        raise RuntimeError("A selected run still holds its run lock. An old queue or manually launched process may still be running; wait for it to stop before restarting this queue.")
    control = manifest_path.parent
    lock = (control / ".queue.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Another queue owns this experiment; use --dry-run to inspect it") from exc
    # Re-inspect under the queue lock; never use a stale pre-lock task list.
    manifest, states, pending, slots, previous = inspect_runs(manifest_path, gpus, jobs_per_gpu, retry_failed)
    if any(status == "running" for status in states.values()):
        lock.close()
        raise RuntimeError("A run started during queue initialization; refusing duplicate scheduling")
    state_path = control / "queue_state.json"
    records = dict(previous)
    active = {}
    failures = []

    def save_state():
        atomic_json(state_path, {"queue_pid": os.getpid(), "gpus": gpus,
                                "jobs_per_gpu": jobs_per_gpu, "updated_at": time.time(),
                                "runs": records, "stopping": STOP})

    def poll_children():
        for slot, item in list(active.items()):
            process, entry, output = item
            code = process.poll()
            if code is None:
                continue
            output.close()
            key = entry["config"]
            # Zero exit without a completion record is a failed full training run.
            completed = classify(entry) == "completed"
            if not STOP and (code != 0 or not completed):
                failures.append(key)
                status = "failed"
            elif completed:
                status = "completed"
            else:
                status = "interrupted"
            records[key].update(status=status, exit_code=code,
                                finished_at=time.time(), completion_verified=completed)
            print(f"{status}: {entry['scale']} seed{entry['seed']} exit={code}", flush=True)
            del active[slot]
            save_state()

    try:
        if pending:
            if any(read_json(entry["config"]).get("device") != "cuda:0" for entry in pending):
                raise ValueError("GPU queue requires device=cuda:0; run CPU smoke checks directly")
            gpu_preflight(gpus)
            verify_bound_files(pending)
        save_state()
        while pending or active:
            poll_children()
            if STOP:
                break
            for slot, gpu in enumerate(slots):
                if slot in active or not pending:
                    continue
                entry = pending.pop(0)
                status = classify(entry)
                if status == "completed":
                    continue
                if status == "running":
                    raise RuntimeError(f"Run was launched elsewhere: {entry['run_dir']}")
                config = read_json(entry["config"])
                log_dir = Path(entry["run_dir"]) / "Logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                output = (log_dir / "console.log").open("a", buffering=1)
                environment = os.environ.copy()
                # Use the explicitly requested identifier, validated against any
                # inherited CUDA allocation. The child sees it as cuda:0.
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["PYTHONUNBUFFERED"] = "1"
                environment["OMP_NUM_THREADS"] = str(config["torch_threads"])
                environment["MKL_NUM_THREADS"] = str(config["torch_threads"])
                command = [sys.executable, str(ROOT / "scripts/endpoint_training_v2/train_endpoint_run.py"),
                           "--config", entry["config"]]
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=output, stderr=subprocess.STDOUT)
                active[slot] = (process, entry, output)
                records[entry["config"]] = {"status": "running", "pid": process.pid,
                                            "gpu": gpu, "slot": slot, "started_at": time.time(),
                                            "exit_code": None}
                save_state()
                print(f"Started GPU {gpu}: {entry['scale']} seed{entry['seed']} pid={process.pid}", flush=True)
            if active:
                time.sleep(max(0.1, poll_seconds))
        if STOP:
            print("Stopping children; committed epoch checkpoints remain resumable.", flush=True)
            for process, _, _ in active.values():
                if process.poll() is None:
                    process.terminate()
            deadline = time.monotonic() + 60
            while active and time.monotonic() < deadline:
                poll_children()
                if active:
                    time.sleep(0.5)
            for process, _, _ in active.values():
                if process.poll() is None:
                    process.kill()
            for process, _, _ in active.values():
                process.wait()
            poll_children()
            return 130
        print(f"Queue finished; failed this invocation: {len(failures)}. Failed runs require --retry-failed.", flush=True)
        return 1 if failures or any(status == "failed" for status in states.values()) and not retry_failed else 0
    finally:
        # Also stop our children on unexpected launcher errors.
        for process, _, output in active.values():
            if process.poll() is None:
                process.terminate()
            output.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Results/0-1labels/_control/runs.json")
    parser.add_argument("--gpus", required=True, help="Explicit GPU IDs/UUIDs, e.g. 0,1,2; must belong to inherited CUDA_VISIBLE_DEVICES if set")
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true", help="Retry each previously failed run once in this invocation")
    parser.add_argument("--dry-run", action="store_true", help="Print states/initial slot assignment without launching or writing")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    return run_queue(args.manifest, parse_gpus(args.gpus), args.jobs_per_gpu,
                     args.retry_failed, args.dry_run, args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
