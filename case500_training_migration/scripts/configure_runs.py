#!/usr/bin/env python3
"""Freeze portable CANOS-PF runs from prepared case500 caches (no training)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_KEYS = ("epochs", "batch_size", "val_batch_size", "val_every", "lr", "hidden_dim",
            "k_steps", "num_workers", "torch_threads", "device")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require_ready(item, split):
    if item.get("ready") is not True:
        raise ValueError(f"{split} cache is not ready; finish preparing this cache before configuring runs")
    if not item.get("cache") or not valid_sha256(item.get("sha256")):
        raise ValueError(f"{split} cache has no valid SHA256 identity")
    if not valid_sha256(item.get("parent_ids_sha256")):
        raise ValueError(f"{split} parent IDs have no valid SHA256 identity")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing immutable configuration differs: {path}; use a new results root")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def existing_path(base, item):
    value = Path(item)
    path = (base / value).resolve() if not value.is_absolute() else value.resolve()
    if not path.is_file():
        raise ValueError(f"Missing prepared data file: {path}")
    return str(path)


def build_configs(data_root, results_root, experiment, scales=None, seeds=None, repo_root=ROOT):
    data_root, results_root = Path(data_root).resolve(), Path(results_root).resolve()
    index_path = data_root / "index.json"
    index = read_json(index_path)
    if index["case_name"] != experiment["case_name"]:
        raise ValueError("Prepared data case_name does not match experiment configuration")
    selected_scales = list(scales if scales is not None else experiment["scales"])
    selected_seeds = list(seeds if seeds is not None else experiment["seeds"])
    if not selected_scales or len(selected_scales) != len(set(selected_scales)):
        raise ValueError("Select at least one distinct scale")
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("Select at least one distinct seed")
    if any(scale not in experiment["scales"] for scale in selected_scales):
        raise ValueError("Requested scale is not listed in experiment.json")
    if experiment.get("num_workers") != 0 or experiment.get("device") not in ("cuda:0", "cpu"):
        raise ValueError("Preserved protocol uses num_workers=0 and device cuda:0 (or cpu for smoke checks)")
    if not index.get("dataset_hash"):
        raise ValueError("Prepared index is missing dataset_hash")
    shared = {
        "format_version": 1,
        "repo_root": str(Path(repo_root).resolve()),
        "case_name": experiment["case_name"],
        "experiment_id": experiment["experiment_id"],
        "data_hash": index["dataset_hash"],
        "data_index": str(index_path),
        "data_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "experiment_protocol_sha256": fingerprint(experiment),
        "parent_manifest": existing_path(data_root, index["parent_manifest"]),
        **{key: experiment[key] for key in RUN_KEYS},
    }
    shared["parent_manifest_sha256"] = file_sha256(shared["parent_manifest"])
    for split in ("val", "test"):
        item = index[split]
        require_ready(item, split)
        shared[f"{split}_cache"] = existing_path(data_root, item["cache"])
        shared[f"{split}_parent_ids"] = existing_path(data_root, item["parent_ids"])
        shared[f"expected_{split}_count"] = int(item["graphs"])
        shared[f"{split}_cache_sha256"] = item["sha256"]
        shared[f"{split}_parent_ids_sha256"] = item["parent_ids_sha256"]
        if file_sha256(shared[f"{split}_parent_ids"]) != item["parent_ids_sha256"]:
            raise ValueError(f"{split} parent IDs changed since cache preparation")
        if int(item["graphs"]) <= 0 or int(item["graphs"]) % 2:
            raise ValueError(f"{split} must contain complete endpoint pairs")
    prepared = []
    for scale in selected_scales:
        if scale not in index["train"]:
            raise ValueError(f"Scale {scale} has not been prepared; prepare the cache first")
        item = index["train"][scale]
        require_ready(item, scale)
        expected = int(scale[:-1]) * 1000 if scale.endswith("k") else int(scale)
        if int(item["graphs"]) != expected or expected % 2:
            raise ValueError(f"Scale {scale} must count endpoint graphs, not parent settings")
        train_cache = existing_path(data_root, item["cache"])
        train_parents = existing_path(data_root, item["parent_ids"])
        if file_sha256(train_parents) != item["parent_ids_sha256"]:
            raise ValueError(f"{scale} parent IDs changed since cache preparation")
        for seed in selected_seeds:
            prepared.append({
                **shared, "scale": scale, "seed": int(seed),
                "train_cache": train_cache, "train_parent_ids": train_parents,
                "expected_train_count": expected,
                "train_cache_sha256": item["sha256"],
                "train_parent_ids_sha256": item["parent_ids_sha256"],
                "run_dir": str(results_root / scale / f"seed{seed}"),
            })
    return prepared


def freeze_runs(data_root, results_root, experiment, scales=None, seeds=None, repo_root=ROOT):
    configs = build_configs(data_root, results_root, experiment, scales, seeds, repo_root)
    control = Path(results_root).resolve() / "_control"
    runs = []
    for config in configs:
        path = control / "configs" / f"{config['scale']}_seed{config['seed']}.json"
        # Validate all pre-existing files before writing any new configuration.
        if path.exists() and read_json(path) != config:
            raise ValueError(f"Existing immutable configuration differs: {path}; use a new results root")
        runs.append({"scale": config["scale"], "seed": config["seed"],
                     "config": str(path), "config_hash": fingerprint(config),
                     "run_dir": config["run_dir"]})
    manifest = {"format_version": 1, "case_name": experiment["case_name"],
                "experiment_id": experiment["experiment_id"],
                "data_hash": configs[0]["data_hash"], "runs": runs}
    manifest_path = control / "runs.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Frozen run selection differs; use a new results root for a different selection")
    experiment_path = control / "experiment.json"
    if experiment_path.exists() and read_json(experiment_path) != experiment:
        raise ValueError("Existing immutable experiment differs; use a new results root")
    for entry, config in zip(runs, configs):
        immutable_json(entry["config"], config)
    immutable_json(control / "experiment.json", experiment)
    immutable_json(manifest_path, manifest)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Prepared cache directory containing index.json")
    parser.add_argument("--results-root", required=True, help="E.g. ./Results/0-1labels")
    parser.add_argument("--experiment", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--scales", help="Optional comma-separated subset, e.g. 10k,20k,40k")
    parser.add_argument("--seeds", help="Optional comma-separated subset, e.g. 11,22")
    args = parser.parse_args()
    path = freeze_runs(args.data_root, args.results_root, read_json(args.experiment),
                       args.scales.split(",") if args.scales else None,
                       [int(s) for s in args.seeds.split(",")] if args.seeds else None)
    print(f"Frozen {len(read_json(path)['runs'])} runs: {path}")
    print("No training was started. Inspect the configurations, then use manage_runs.py --dry-run.")


if __name__ == "__main__":
    main()
