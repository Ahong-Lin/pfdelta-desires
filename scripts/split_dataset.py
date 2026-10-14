#!/usr/bin/env python3
"""Freeze portable, parent-disjoint endpoint splits from generated PF samples.

Only known PF inputs enter clustering. This program does not inspect label
extremeness, model errors, or engineering limits to choose held-out samples.
All requested training scales are prefixes of ONE stratified parent ordering.
The JSON split schema follows the case118 reader; this is not a training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

# Avoid excessive BLAS threads on a shared generation server.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np


SCHEMA_VERSION = 1


def size_label(records: int) -> str:
    return f"{records // 1000}k" if records % 1000 == 0 else str(records)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def effective_protocol(plan: dict, config: dict) -> dict:
    """Generation plan is authoritative for smoke/full counts, config for split seed."""
    mode = plan.get("mode", "full")
    if mode not in ("full", "smoke"):
        raise ValueError(f"Unknown plan mode {mode!r}")
    raw_branches = plan.get("eligible_branch_ids", plan.get("eligible_branches"))
    if not raw_branches or isinstance(raw_branches, (int, str)):
        raise ValueError("plan.json must contain a list of eligible branch IDs")
    branches = sorted(int(branch) for branch in raw_branches)
    if len(set(branches)) != len(branches):
        raise ValueError("Duplicate eligible branch IDs in plan")
    nlines = len(branches)
    train_parents = int(plan.get("train_parents", int(config["train_endpoint_records"]) // 2))
    if train_parents <= 0:
        raise ValueError("train_parents must be positive")
    train_records = 2 * train_parents
    scales = [int(x) for x in plan.get("train_endpoint_scales", config["train_scales"])]
    if not scales or scales != sorted(set(scales)) or any(x <= 0 or x % 2 for x in scales):
        raise ValueError("Training endpoint scales must be unique increasing positive even integers")
    if scales[-1] != train_records:
        raise ValueError("Largest scale must equal the generation plan's full train size")
    val = int(plan.get("val_parents_per_branch", config["val_parents_per_branch"]))
    test = int(plan.get("test_parents_per_branch", config["test_parents_per_branch"]))
    if val < 1 or test < 1:
        raise ValueError("Validation and test need at least one parent per branch")
    full_quota = equal_branch_quota(train_parents, nlines)
    for field, extra in (("full_train_quota", 0), ("branch_parent_targets", val + test)):
        if field in plan:
            supplied = [int(plan[field][str(branch)]) for branch in branches]
            expected = (full_quota + extra).tolist()
            if supplied != expected:
                raise ValueError(f"{field} disagrees with sorted-branch remainder convention")
    expected_total = train_parents + nlines * (val + test)
    planned_total = plan.get("total_parents", plan.get("target_parents", expected_total))
    if int(planned_total) != expected_total:
        raise ValueError("plan total_parents does not match split quotas")
    if mode == "full" and train_records != int(config["train_endpoint_records"]):
        raise ValueError("Full plan and config disagree on training size")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode, "case_name": str(plan.get("case_name", config.get("case_name", "case500"))),
        "eligible_branches": branches, "train_endpoint_scales": scales,
        "train_parents": train_parents, "val_parents_per_branch": val,
        "test_parents_per_branch": test, "total_parents": expected_total,
        "full_train_quota": full_quota.tolist(),
        "split_seed": int(config.get("split_seed", 20261014)),
        "requested_clusters": int(config.get("clusters", 64)),
        "cluster_fit_max_parents": int(config.get("cluster_fit_max_parents", 50000)),
        "feature_dimension": int(plan.get("feature_dimension", config.get("feature_dimension", 3000))),
        "branch_remainder_rule": "ascending branch ID, first remainder branches receive +1",
        "stratification": "branch quotas, proportional known-input clusters, random mixing within cells",
    }


def equal_branch_quota(total: int, nbranches: int) -> np.ndarray:
    quotient, remainder = divmod(int(total), int(nbranches))
    quota = np.full(nbranches, quotient, dtype=np.int64)
    quota[:remainder] += 1
    return quota


def systematic_quota(total: int, capacities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact proportional allocation, randomized rounding, never beyond capacity."""
    capacities = np.asarray(capacities, dtype=np.int64)
    if total < 0 or total > int(capacities.sum()) or np.any(capacities < 0):
        raise ValueError("Invalid proportional quota/capacity")
    if total == 0:
        return np.zeros_like(capacities)
    ideal = total * (capacities / capacities.sum())
    order = rng.permutation(len(capacities))
    cumulative = np.cumsum(ideal[order])
    cumulative[-1] = total
    rounded = np.diff(np.floor(np.r_[0.0, cumulative] + rng.random())).astype(np.int64)
    quota = np.empty_like(rounded)
    quota[order] = rounded
    if quota.sum() != total or np.any(quota < 0) or np.any(quota > capacities):
        raise ValueError("Numerical failure in proportional allocation")
    return quota


def safe_relative_path(root: Path, value: str) -> tuple[str, Path]:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Non-portable sample path: {value}")
    absolute = (root / candidate).resolve()
    if not absolute.is_relative_to(root.resolve()):
        raise ValueError(f"Sample path escapes dataset root: {value}")
    if not absolute.is_file():
        raise ValueError(f"Missing endpoint file: {value}")
    return candidate.as_posix(), absolute


def read_catalogue(root: Path, protocol: dict, features_path: Path | None,
                   verify_raw: bool = False) -> dict:
    """One JSON row at a time; only the current 3000-value list lives in Python."""
    count, dimension = protocol["total_parents"], protocol["feature_dimension"]
    features = (np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float32,
                                         shape=(count, dimension)) if features_path else None)
    parents = np.empty(count, dtype=np.int64)
    branches = np.empty(count, dtype=np.int64)
    batches = np.empty(count, dtype=np.int32)
    fingerprints = np.empty(count, dtype="S64")
    input_hashes = np.empty(count, dtype="S64")
    batch_codes: dict[str, int] = {}
    files: list[tuple[str, str]] = []
    seen_parent, seen_input, seen_binary_input, seen_file = set(), set(), set(), set()
    branch_set = set(protocol["eligible_branches"])
    manifest_digest = hashlib.sha256()
    actual = 0
    with (root / "manifests/parents.jsonl").open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            manifest_digest.update(line)
            if not line.strip():
                continue
            if actual >= count:
                raise ValueError("Manifest contains more parents than the generation plan")
            row = json.loads(line)
            parent, branch = int(row["parent_id"]), int(row["branch_id"])
            if parent < 1 or parent in seen_parent:
                raise ValueError(f"Bad/duplicate parent_id at manifest line {line_number}: {parent}")
            if branch not in branch_set:
                raise ValueError(f"Unexpected branch ID {branch}")
            fingerprint = str(row["input_fingerprint"])
            if not fingerprint or len(fingerprint) > 64 or not fingerprint.isascii():
                raise ValueError(f"Invalid input_fingerprint for parent {parent}")
            if fingerprint in seen_input:
                raise ValueError(f"Repeated fixed PF inputs at parent {parent}; regenerate this parent")
            vector64 = np.asarray(row["input_features"], dtype="<f8")
            if vector64.shape != (dimension,) or not np.isfinite(vector64).all():
                raise ValueError(f"Invalid known-input feature vector for parent {parent}")
            vector64[vector64 == 0] = 0.0  # Canonicalize signed zero.
            binary_hash = hashlib.sha256(vector64.tobytes()).hexdigest()
            if binary_hash in seen_binary_input:
                raise ValueError(f"Duplicate known-input vector at parent {parent}, even with a different fingerprint")
            pair = row["files"]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Parent {parent} needs [rho1_file, rho0_file]")
            normalized = []
            for rho, value in zip((1.0, 0.0), pair):
                relative, absolute = safe_relative_path(root, str(value))
                if relative in seen_file:
                    raise ValueError(f"Endpoint file reused by multiple records: {relative}")
                seen_file.add(relative)
                if verify_raw:
                    validate_raw_sample(absolute, parent, branch, rho, fingerprint)
                normalized.append(relative)
            if features is not None:
                features[actual] = vector64
            batch_name = str(row.get("source_batch", "unknown"))
            if batch_name not in batch_codes:
                batch_codes[batch_name] = len(batch_codes)
            parents[actual], branches[actual], batches[actual] = parent, branch, batch_codes[batch_name]
            fingerprints[actual], input_hashes[actual] = fingerprint.encode(), binary_hash.encode()
            files.append(tuple(normalized))
            seen_parent.add(parent)
            seen_input.add(fingerprint)
            seen_binary_input.add(binary_hash)
            actual += 1
            if actual % 10000 == 0:
                print(f"Validated {actual:,}/{count:,} parents", flush=True)
    if actual != count:
        raise ValueError(f"Dataset generation incomplete: {actual:,} parents, expected {count:,}")
    actual_counts = np.array([np.count_nonzero(branches == b) for b in protocol["eligible_branches"]])
    expected_counts = (np.asarray(protocol["full_train_quota"])
                       + protocol["val_parents_per_branch"] + protocol["test_parents_per_branch"])
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError("Manifest per-branch counts do not match generation/split quotas")
    if features is not None:
        features.flush()
    return {
        "parent_ids": parents, "branch_ids": branches, "batch_codes": batches,
        "batch_names": list(batch_codes), "input_fingerprints": fingerprints,
        "binary_input_hashes": input_hashes, "files": files, "features": features,
        "manifest_sha256": manifest_digest.hexdigest(),
    }


def validate_raw_sample(path: Path, parent: int, branch: int, rho: float,
                        fingerprint: str) -> None:
    """Check stored generation metadata; do not recompute or solve the PF equations."""
    with path.open(encoding="utf-8") as handle:
        sample = json.load(handle)
    sources = [sample, sample.get("rho_meta", {}), sample.get("metadata", {}),
               sample.get("meta", {}), sample.get("solution", {})]

    def get(*names: str) -> Any:
        for name in names:
            for source in sources:
                if name in source:
                    return source[name]
        return None

    def all_values(*names: str) -> list:
        return [source[name] for source in sources for name in names if name in source]

    values = (get("parent_id", "load_setting_id"), get("branch_id", "selected_branch_id"), get("rho"))
    if any(value is None for value in values):
        raise ValueError(f"Missing endpoint identity metadata: {path}")
    if ((int(values[0]), int(values[1]), float(values[2])) != (parent, branch, rho)
            or any(int(value) != parent for value in all_values("parent_id", "load_setting_id"))
            or any(int(value) != branch for value in all_values("branch_id", "selected_branch_id"))
            or any(float(value) != rho for value in all_values("rho"))):
        raise ValueError(f"Endpoint identity mismatch: {path}")
    stated_hash = get("input_fingerprint")
    if stated_hash is None or any(str(value) != fingerprint for value in all_values("input_fingerprint")):
        raise ValueError(f"Endpoint fixed-input fingerprint mismatch: {path}")
    status = get("termination_status", "status")
    if str(status) != "PF_CONVERGED":
        raise ValueError(f"Unconverged endpoint metadata: {path}: {status}")
    method = get("solver_method", "method")
    if method is None or any(str(value).lower() != "newton" for value in all_values("solver_method", "method")):
        raise ValueError(f"Endpoint is not marked as a Newton-family solve: {path}: {method}")
    fallback = get("solver_fallback")
    if fallback is None or any(str(value).lower() != "none" for value in all_values("solver_fallback")):
        raise ValueError(f"Endpoint records a fallback solver: {path}: {fallback}")
    if get("solver_type") is None or any(str(value) != "AC_PF" for value in all_values("solver_type")):
        raise ValueError(f"Endpoint solver_type is not AC_PF: {path}")
    validation = get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"Missing stored numerical validation metadata: {path}")
    tolerances = {"max_balance_residual": 1e-7, "fixed_bus_pg_error": 1e-8,
                  "fixed_vm_error": 1e-8, "fixed_slack_va_error": 1e-8}
    for name, tolerance in tolerances.items():
        value = float(validation.get(name, float("nan")))
        if not np.isfinite(value) or not 0 <= value <= tolerance:
            raise ValueError(f"Endpoint stored {name} is missing or exceeds {tolerance}: {path}: {value}")
    minimum, maximum = float(validation.get("min_vm", float("nan"))), float(validation.get("max_vm", float("nan")))
    if not np.isfinite([minimum, maximum]).all() or not 0 < minimum <= maximum:
        raise ValueError(f"Endpoint stored voltage bounds are missing or invalid: {path}")
    endpoint_branch = sample.get("network", {}).get("branch", {}).get(str(branch), {})
    if endpoint_branch.get("br_status") != int(rho):
        raise ValueError(f"Endpoint selected branch status disagrees with rho: {path}")


def cluster_inputs(features: np.ndarray, protocol: dict, stage: Path) -> np.ndarray:
    """Fit at most 50k parents; transform/predict the full memory-mapped pool in chunks."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    n = len(features)
    seed = protocol["split_seed"]
    fit_count = min(n, protocol["cluster_fit_max_parents"])
    if fit_count < 1 or protocol["requested_clusters"] < 1:
        raise ValueError("cluster_fit_max_parents and clusters must be positive")
    rng = np.random.default_rng(seed)
    fit_rows = np.sort(rng.choice(n, size=fit_count, replace=False))
    fit_data = np.asarray(features[fit_rows], dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(fit_data)
    active = scaler.var_ > 1e-12
    # Features constant in the random fit sample cannot separate its clusters.
    # Tiny smoke fixtures should not create one stratum per individual sample.
    clusters_count = min(protocol["requested_clusters"], fit_count, max(1, n // 4))
    if not np.any(active):
        raise ValueError("All known PF inputs are constant; cannot build meaningful split strata")
    fit_data = fit_data[:, active]
    mean, scale = scaler.mean_[active].astype(np.float32), scaler.scale_[active].astype(np.float32)
    fit_data -= mean
    fit_data /= scale
    print(f"Clustering {n:,} parents: fit={fit_count:,}, active inputs={int(active.sum())}, "
          f"clusters={clusters_count}", flush=True)
    model = MiniBatchKMeans(n_clusters=clusters_count, random_state=seed, batch_size=2048,
                            n_init=3, max_iter=100, reassignment_ratio=0.01)
    with threadpool_limits(limits=4):
        model.fit(fit_data)
        del fit_data
        labels = np.empty(n, dtype=np.int32)
        for start in range(0, n, 4096):
            block = np.asarray(features[start:start + 4096, active], dtype=np.float32)
            block -= mean
            block /= scale
            labels[start:start + len(block)] = model.predict(block)
    np.savez_compressed(stage / "stratification_model.npz", active_columns=np.flatnonzero(active),
                        mean=mean, scale=scale, centers=model.cluster_centers_, fit_rows=fit_rows,
                        labels=labels)
    return labels


def split_indices(branches: np.ndarray, clusters: np.ndarray, protocol: dict) -> dict[str, np.ndarray]:
    """Shuffle within branch×input cluster, then fill held-outs and nested train layers."""
    rng = np.random.default_rng(protocol["split_seed"])
    values = protocol["eligible_branches"]
    scales = protocol["train_endpoint_scales"]
    nclusters = int(clusters.max()) + 1
    quotas = [equal_branch_quota(records // 2, len(values)) for records in scales]
    for earlier, later in zip(quotas, quotas[1:]):
        if np.any(earlier > later):
            raise ValueError("Branch quotas are not nested")
    validation: list[int] = []
    test: list[int] = []
    layers: list[list[int]] = [[] for _ in scales]
    for index, branch in enumerate(values):
        rows = np.flatnonzero(branches == branch)
        pools = [rng.permutation(rows[clusters[rows] == cluster]) for cluster in range(nclusters)]
        available = np.array([len(pool) for pool in pools], dtype=np.int64)
        positions = np.zeros(nclusters, dtype=np.int64)
        targets = [protocol["val_parents_per_branch"], protocol["test_parents_per_branch"]]
        previous = 0
        for quota in quotas:
            targets.append(int(quota[index] - previous))
            previous = int(quota[index])
        for phase, target in enumerate(targets):
            take = systematic_quota(target, available - positions, rng)
            chosen = np.concatenate([pool[positions[k]:positions[k] + take[k]]
                                     for k, pool in enumerate(pools)])
            positions += take
            destination = validation if phase == 0 else test if phase == 1 else layers[phase - 2]
            destination.extend(chosen.tolist())
        if not np.array_equal(positions, available):
            raise ValueError(f"Unallocated parents for branch {branch}")
    train = np.concatenate([rng.permutation(np.asarray(layer, dtype=np.int64)) for layer in layers])
    result = {"train": train,
              "val": rng.permutation(np.asarray(validation, dtype=np.int64)),
              "test": rng.permutation(np.asarray(test, dtype=np.int64))}
    assigned = np.concatenate(list(result.values()))
    if len(assigned) != len(branches) or len(np.unique(assigned)) != len(branches):
        raise ValueError("Parent leakage or missing rows in split assignment")
    return result


def count_dict(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(key): int(count) for key, count in zip(unique, counts)}


def summarize(rows: np.ndarray, data: dict, clusters: np.ndarray, reference: dict) -> dict:
    branch_counts = count_dict(data["branch_ids"][rows])
    cluster_counts = np.bincount(clusters[rows], minlength=len(reference["cluster_proportions"]))
    source_counts = np.bincount(data["batch_codes"][rows], minlength=len(data["batch_names"]))
    n = len(rows)
    cluster_proportions = cluster_counts / n
    source_proportions = source_counts / n
    return {
        "parents": n, "endpoint_records": 2 * n, "branch_counts": branch_counts,
        "branch_count_min_max": [min(branch_counts.values()), max(branch_counts.values())],
        "cluster_counts": cluster_counts.tolist(), "cluster_proportions": cluster_proportions.tolist(),
        "missing_clusters": np.flatnonzero(cluster_counts == 0).tolist(),
        "cluster_total_variation_vs_pool": float(np.abs(cluster_proportions - reference["cluster_proportions"]).sum() / 2),
        "source_batch_counts": {name: int(value) for name, value in zip(data["batch_names"], source_counts)},
        "source_batch_proportions": {name: float(value) for name, value in zip(data["batch_names"], source_proportions)},
        "source_batch_total_variation_vs_pool": float(np.abs(source_proportions - reference["source_proportions"]).sum() / 2),
    }


def write_split(path: Path, name: str, rows: np.ndarray, data: dict, protocol: dict) -> None:
    """The root-relative `file` key is consumed by PFDeltaRhoCANOS-style loaders."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write('{"split":' + json.dumps(name) + ',"records":[')
        first = True
        for index in rows:
            parent = int(data["parent_ids"][index])
            branch = str(data["branch_ids"][index])
            for rho, relative in zip((1.0, 0.0), data["files"][index]):
                record = {
                    "sample_id": 2 * parent - int(rho), "file": relative,
                    "split": name, "case_name": protocol["case_name"],
                    "load_setting_id": parent, "branch_id": branch, "rho": rho,
                    "rho_mode": "endpoint-paired", "solver_type": "AC_PF",
                }
                if not first:
                    handle.write(",")
                json.dump(record, handle, separators=(",", ":"))
                first = False
        handle.write('],"load_settings":[')
        for position, index in enumerate(rows):
            if position:
                handle.write(",")
            parent = int(data["parent_ids"][index])
            json.dump({"load_setting_id": parent, "sample_ids": [2 * parent - 1, 2 * parent],
                       "selected_branch_id": str(data["branch_ids"][index]), "split": name},
                      handle, separators=(",", ":"))
        handle.write('],"failures":[]}\n')


def validate_frozen(target: Path, signature: dict) -> dict:
    frozen = json.loads((target / "frozen.json").read_text())
    if frozen["signature"] != signature:
        raise ValueError("Frozen split inputs/protocol changed. Use a NEW dataset root; do not resplit this version.")
    for relative, expected in frozen["artifacts"].items():
        path = target / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Frozen split artifact missing or changed: {relative}")
    return frozen


def run(root: Path, config_path: Path, verify_raw: bool = False, dry_run: bool = False) -> dict:
    started = time.monotonic()
    root = root.resolve()
    config = json.loads(config_path.read_text())
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    protocol = effective_protocol(plan, config)
    target = root / "splits"
    manifest_path = root / "manifests/parents.jsonl"
    if target.exists():
        if not (target / "frozen.json").is_file():
            raise ValueError("Existing splits/ has no frozen.json; refusing to replace unknown files")
        signature = {"manifest_sha256": file_hash(manifest_path), "plan_sha256": file_hash(plan_path),
                     "protocol_sha256": object_hash(protocol), "schema_version": SCHEMA_VERSION}
        frozen = validate_frozen(target, signature)
        if verify_raw:
            read_catalogue(root, protocol, None, verify_raw=True)
        print(f"Frozen splits verified; no resampling. Dataset hash {frozen['dataset_hash']}", flush=True)
        return frozen
    if dry_run:
        data = read_catalogue(root, protocol, None, verify_raw=verify_raw)
        result = {"dry_run": True, "protocol": protocol, "manifest_sha256": data["manifest_sha256"]}
        print(json.dumps(result, indent=2), flush=True)
        return result
    root.mkdir(parents=True, exist_ok=True)
    # The final directory is installed only after all checks succeed. A crash
    # leaves a hidden staging directory, never a partially frozen public split.
    with tempfile.TemporaryDirectory(prefix=".split-build-", dir=root) as temporary:
        stage = Path(temporary) / "splits"
        stage.mkdir()
        data = read_catalogue(root, protocol, stage / "parent_features.npy", verify_raw=verify_raw)
        clusters = cluster_inputs(data["features"], protocol, stage)
        assignment = split_indices(data["branch_ids"], clusters, protocol)
        cluster_reference = np.bincount(clusters) / len(clusters)
        source_reference = np.bincount(data["batch_codes"]) / len(clusters)
        reference = {"cluster_proportions": cluster_reference, "source_proportions": source_reference}
        summaries = {}
        for name in ("val", "test"):
            rows = assignment[name]
            write_split(stage / f"{name}.json", name, rows, data, protocol)
            np.save(stage / f"{name}_parent_ids.npy", data["parent_ids"][rows])
            summaries[name] = summarize(rows, data, clusters, reference)
        train_rows = assignment["train"]
        np.save(stage / "train_parent_ids.npy", data["parent_ids"][train_rows])
        np.save(stage / "train_manifest_rows.npy", train_rows)
        for records in protocol["train_endpoint_scales"]:
            label = size_label(records)
            rows = train_rows[:records // 2]
            actual = np.array([np.count_nonzero(data["branch_ids"][rows] == branch)
                               for branch in protocol["eligible_branches"]])
            if not np.array_equal(actual, equal_branch_quota(records // 2, len(actual))):
                raise ValueError(f"Training scale {records} failed exact branch balance")
            write_split(stage / f"train_{label}.json", "train", rows, data, protocol)
            np.save(stage / f"train_{label}_parent_ids.npy", data["parent_ids"][rows])
            summaries[f"train_{label}"] = summarize(rows, data, clusters, reference)
        final_label = size_label(protocol["train_endpoint_scales"][-1])
        # A relative symlink remains valid when moving the complete dataset root.
        (stage / "train.json").symlink_to(f"train_{final_label}.json")
        np.savez_compressed(stage / "parent_metadata.npz", parent_ids=data["parent_ids"],
                            branch_ids=data["branch_ids"], source_batch_codes=data["batch_codes"],
                            source_batch_names=np.asarray(data["batch_names"], dtype=str),
                            input_fingerprints=data["input_fingerprints"],
                            binary_input_hashes=data["binary_input_hashes"], cluster_ids=clusters)
        write_json(stage / "protocol.json", protocol)
        write_json(stage / "audit.json", {
            "protocol": protocol, "splits": summaries,
            "parent_overlap_train_val_test": 0, "input_fingerprint_overlap_train_val_test": 0,
            "duplicate_input_vectors": 0, "raw_metadata_verified": verify_raw,
            "reference_cluster_proportions": cluster_reference.tolist(),
            "known_input_feature_file": "parent_features.npy",
            "feature_rows": "generation manifest row order, NOT training order",
            "notes": ["Only PF known inputs used for stratification; no output voltage/error screening.",
                      "Comparable input strata do not prove identical high-dimensional distributions.",
                      "Training scale counts are endpoint graphs; each parent contributes two graphs.",
                      "Root-relative file schema matches the case118 reader; no old training run is claimed."],
        })
        del data["features"]
        signature = {"manifest_sha256": data["manifest_sha256"], "plan_sha256": file_hash(plan_path),
                     "protocol_sha256": object_hash(protocol), "schema_version": SCHEMA_VERSION}
        artifacts = {path.name: file_hash(path) for path in sorted(stage.iterdir()) if path.is_file()}
        frozen = {"signature": signature, "dataset_hash": object_hash({"signature": signature, "artifacts": artifacts}),
                  "artifacts": artifacts, "completed": True,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(stage / "frozen.json", frozen)
        stage.rename(target)
    print(f"Frozen {len(protocol['train_endpoint_scales'])} nested endpoint scales. "
          f"Dataset hash {frozen['dataset_hash']}", flush=True)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Generated dataset root containing plan.json")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verify-raw", action="store_true", help="Also open endpoint JSONs to check identity/solver metadata")
    parser.add_argument("--dry-run", action="store_true", help="Validate complete manifest without writing new artifacts")
    args = parser.parse_args()
    run(args.root, args.config, verify_raw=args.verify_raw, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
