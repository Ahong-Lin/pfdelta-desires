#!/usr/bin/env python3
"""Freeze balanced nested endpoint subsets, then materialize shared CANOS caches.

Original data, labels and held-out memberships are never modified. Run in the
pfdelta311 environment. --scales selects caches only; all 14 index sets are
always frozen together. No train.json is created: the existing PFDelta reader
will fail explicitly if a requested cache is absent, rather than rebuild an
incorrect full-size dataset. No training is launched by this script.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import numpy as np

SCHEMA = "case118_endpoint_training_v2"
SCALES = (10000, 20000, 40000, 45000, 50000, 55000, 60000, 80000,
          90000, 100000, 110000, 120000, 160000, 320000)
DEFAULT_REPO = Path("/data1/liuy/linhj/pfdelta-case118-acpf/pfdelta-case118")
SOURCE_NAME = "case118_endpoint_id_1280k_v1_20261012"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp, path)


def frozen_array(path, value):
    path = Path(path)
    if path.exists():
        require(np.array_equal(np.load(path), value), f"Existing frozen index differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as f:
        np.save(f, value)
    os.replace(temp, path)


def link_exact(target, link):
    target, link = Path(target).resolve(), Path(link)
    require(target.is_file(), f"Missing shared source: {target}")
    if link.exists() or link.is_symlink():
        require(link.resolve() == target, f"Refusing to replace shared link: {link}")
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))


def parse_scales(text):
    if text == "all":
        return list(SCALES)
    result = []
    for word in text.split(","):
        word = word.strip().lower()
        result.append(int(word[:-1]) * 1000 if word.endswith("k") else int(word))
    require(result and set(result).issubset(SCALES), f"Allowed cache scales: {SCALES}")
    return sorted(set(result))


def systematic_quota(total, capacity, rng):
    """Exact-sum randomized rounding of proportional remaining capacities."""
    capacity = np.asarray(capacity, dtype=np.int64)
    require(0 <= total <= capacity.sum(), "Cluster capacity exceeded")
    if total == 0:
        return np.zeros_like(capacity)
    expected = total * capacity / capacity.sum()
    order = rng.permutation(len(capacity))
    cumulative = np.cumsum(expected[order])
    cumulative[-1] = total
    rounded = np.diff(np.floor(np.r_[0, cumulative] + rng.random())).astype(np.int64)
    quota = np.empty_like(rounded)
    quota[order] = rounded
    require(quota.sum() == total and np.all(quota >= 0) and np.all(quota <= capacity), "Invalid cluster allocation")
    return quota


def balanced_order(lines, clusters, scales, seed):
    """Exact equal-line quotas at every anchor; proportional input-cluster mix.

    Each added layer uses randomized systematic rounding of each line's
    remaining cluster proportions. Randomized rounding avoids systematically
    eliminating small clusters across all lines, while preserving exact line
    totals, capacity constraints and nesting without consulting any labels.
    """
    rng = np.random.default_rng(seed)
    branches = np.unique(lines)
    counts = np.array([(lines == b).sum() for b in branches])
    priority = rng.permutation(len(branches))
    quotas = []
    for records in scales:
        total = records // 2
        q = np.full(len(branches), total // len(branches), dtype=np.int64)
        q[priority[:total % len(branches)]] += 1
        require(np.all(q <= counts), f"Insufficient branch capacity at {records} records")
        quotas.append(q)
    require(all(np.all(a <= b) for a, b in zip(quotas, quotas[1:])), "Non-nested quotas")
    layers = [[] for _ in scales]
    nclusters = int(clusters.max()) + 1
    for bi, branch in enumerate(branches):
        rows = np.flatnonzero(lines == branch)
        pools = [rng.permutation(rows[clusters[rows] == c]) for c in range(nclusters)]
        capacity = np.array([len(p) for p in pools], dtype=np.int64)
        used = np.zeros(nclusters, dtype=np.int64)
        previous = 0
        for si, q in enumerate(quotas):
            total = int(q[bi])
            take = systematic_quota(total - previous, capacity - used, rng)
            selected = np.concatenate([pool[used[c]:used[c] + take[c]] for c, pool in enumerate(pools)])
            layers[si].extend(selected)
            used += take
            previous = total
    ordered = np.concatenate([rng.permutation(np.asarray(layer, dtype=np.int64)) for layer in layers])
    require(len(ordered) == scales[-1] // 2 and len(np.unique(ordered)) == len(ordered), "Bad nested order")
    return ordered


def summaries(rows, features, reference):
    branches, counts = np.unique(features["branch_ids"][rows], return_counts=True)
    parts, pc = np.unique(features["source_part"][rows], return_counts=True)
    cc = np.bincount(features["clusters"][rows], minlength=len(reference))
    return {"parents": int(len(rows)), "records": int(2 * len(rows)),
            "branch_counts": {str(b): int(c) for b, c in zip(branches, counts)},
            "branch_count_min_max": [int(counts.min()), int(counts.max())],
            "cluster_counts": cc.tolist(), "missing_clusters": np.flatnonzero(cc == 0).tolist(),
            "cluster_total_variation_vs_equal_line_training_pool": float(np.abs(cc / len(rows) - reference).sum() / 2),
            "source_part_counts": {str(p): int(c) for p, c in zip(parts, pc)}}


def make_indices(source, root, seed):
    source_index = json.loads((source / "processed/canos_rho_scales/index.json").read_text())
    source_meta = json.loads((source / "meta.json").read_text())
    require(source_meta["status"] == "COMPLETE", "Mother dataset is not complete")
    ids = {s: np.load(source / f"splits/{s}_parent_ids.npy").astype(np.int64) for s in ("train", "val", "test")}
    require([len(ids[s]) for s in ids] == [640000, 8400, 8400], "Unexpected source split sizes")
    joined = np.concatenate(list(ids.values()))
    require(len(np.unique(joined)) == len(joined), "Source parent overlap")
    with np.load(source / "manifests/parent_features.npz") as z:
        features = {k: z[k] for k in z.files}
    all_ids = features["parent_ids"]
    require(np.array_equal(np.sort(joined), np.sort(all_ids)), "Source feature IDs do not cover the splits")
    require(len(np.unique(features["input_fingerprints"])) == len(all_ids), "Duplicate known-input fingerprints in source pool")
    sorter = np.argsort(all_ids)
    rows = {s: sorter[np.searchsorted(all_ids[sorter], p)] for s, p in ids.items()}
    require(all(np.array_equal(all_ids[rows[s]], ids[s]) for s in ids), "Parent-to-feature mapping failed")
    train_lines = features["branch_ids"][rows["train"]]
    branches = np.unique(train_lines)
    require(len(branches) == 168, "Expected 168 eligible branches")
    nclusters = int(features["clusters"].max()) + 1
    require(nclusters == 64, "Expected existing 64 input-only clusters")
    for split in ("val", "test"):
        hb, hc = np.unique(features["branch_ids"][rows[split]], return_counts=True)
        require(np.array_equal(hb, branches) and np.all(hc == 50), f"Bad {split} per-line counts")
    reference = np.mean([np.bincount(features["clusters"][rows["train"]][train_lines == b], minlength=nclusters)
                         / np.sum(train_lines == b) for b in branches], axis=0)
    binding = {"schema": SCHEMA, "sampling_seed": seed, "all_endpoint_scales": list(SCALES),
               "source_features_sha256": sha256(source / "manifests/parent_features.npz"),
               "source_split_hashes": {s: array_hash(p) for s, p in ids.items()},
               "source_cache_sha256": source_index["scales"]["1280k"]["file_sha256"],
               "source_holdout_cache_sha256": {s: source_index["shared"][s]["file_sha256"] for s in ("val", "test")},
               "algorithm": "equal_line_fixed_priority__within_line_systematic_remaining_cluster_rounding_v1"}
    version_hash = object_hash(binding)
    protocol = root / "protocol"
    binding_path = protocol / "binding.json"
    if binding_path.exists():
        require(json.loads(binding_path.read_text()) == binding, "Existing output belongs to a different frozen data version")
    else:
        atomic_json(binding_path, binding)
    ordered = balanced_order(train_lines, features["clusters"][rows["train"]], SCALES, seed)
    report = {"schema": SCHEMA, "dataset_version_hash": version_hash, "binding": binding,
              "source_root": str(source), "target_root": str(root),
              "counts": {"source_training_parents": 640000, "largest_training_parents": 160000,
                         "validation_parents": 8400, "test_parents": 8400},
              "checks": {"split_parent_overlap": 0, "known_input_duplicate_parents": 0,
                         "heldout_membership_unchanged": True, "nested_training": True,
                         "all_scales_cover_64_input_clusters": True,
                         "per_line_holdout_parents": 50, "rho_values": [1.0, 0.0],
                         "new_label_solves": 0, "additional_filters": False},
              "sampling_notes": "Equal line counts (difference <=1); input clusters follow mother-training-pool proportions, not equal cluster counts. Clusters and exact fingerprints cannot guarantee identical full joint distributions.",
              "scales": {}, "holdouts": {}}
    for records in SCALES:
        label = f"{records // 1000}k"
        selected = ordered[:records // 2]
        parents = ids["train"][selected]
        graph_indices = np.column_stack((2 * selected, 2 * selected + 1)).reshape(-1)
        folder = root / "train" / label
        frozen_array(folder / "parent_ids.npy", parents)
        frozen_array(folder / "source_graph_indices.npy", graph_indices)
        info = summaries(rows["train"][selected], features, reference)
        require(info["branch_count_min_max"][1] - info["branch_count_min_max"][0] <= 1, f"Unequal lines: {label}")
        require(not info["missing_clusters"], f"Missing coarse input clusters at {label}: {info['missing_clusters']}")
        info.update(dataset_version_hash=version_hash, parent_ids_sha256=array_hash(parents),
                    source_graph_indices_sha256=array_hash(graph_indices), processed_subdir=f"processed/{label}")
        atomic_json(folder / "meta.json", info)
        report["scales"][label] = info
    for split in ("val", "test"):
        frozen_array(root / "shared" / f"{split}_parent_ids.npy", ids[split])
        link_exact(source / f"processed/canos_rho_scales/shared/{split}.pt", root / f"shared/{split}.pt")
        info = summaries(rows[split], features, reference)
        info.update(parent_ids_sha256=array_hash(ids[split]), file_sha256=source_index["shared"][split]["file_sha256"],
                    bytes=source_index["shared"][split]["bytes"])
        report["holdouts"][split] = info
    # Existing PFDelta reader can load these caches; a missing cache must fail.
    require(not (root / "splits/train.json").exists(), "Unsafe fallback train.json found under new dataset root")
    atomic_json(root / "audit.json", report)
    old_meta = json.loads((root / "meta.json").read_text()) if (root / "meta.json").exists() else {}
    require(not old_meta or old_meta.get("dataset_version_hash") == version_hash, "Dataset metadata version mismatch")
    atomic_json(root / "meta.json", {**old_meta, "schema": SCHEMA, "dataset_version_hash": version_hash,
                "status": old_meta.get("status", "INDICES_FROZEN"), "records_per_parent": 2,
                "training_scales": list(SCALES), "validation_records": 16800, "test_records": 16800,
                "processed_train": "processed/<scale>/train.pt", "processed_holdouts": "shared/<split>.pt"})
    return report, ids


def torch_load(path):
    import torch
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    require(isinstance(payload, tuple) and len(payload) == 2, f"Unexpected PyG payload: {path}")
    return payload


def fields(data, slices):
    import torch
    for store in data.stores:
        key = store._key
        cut_store = slices if key is None else slices[key]
        for attr, value in store.items():
            require(torch.is_tensor(value) and attr in cut_store, f"Unsupported attribute: {key}/{attr}")
            dim = data.__cat_dim__(attr, value, store)
            require(isinstance(dim, int), f"Unsupported concatenation: {key}/{attr}")
            yield key, attr, value, cut_store[attr], dim % value.ndim


def store_at(data, key):
    return data._global_store if key is None else data[key]


def check_pairs(data, parents):
    require(np.array_equal(data.load_setting_id.numpy().reshape(-1), np.repeat(parents, 2)), "Cache parent pairing mismatch")
    require(np.array_equal(data.rho.numpy().reshape(-1), np.tile([1., 0.], len(parents))), "Cache rho ordering mismatch")
    samples = np.column_stack((2 * parents - 1, 2 * parents)).reshape(-1)
    require(np.array_equal(data.rho_sample_id.numpy().reshape(-1), samples), "Cache sample-ID mismatch")


def gather_payload(payload, graph_indices, chunk=2048):
    """Copy every tensor field, keeping local edge indices and PyG slices exact."""
    import torch
    data, slices = payload
    result = copy.copy(data)
    result_slices = {}
    nsource = data.load_setting_id.numel()
    n = len(graph_indices)
    indices = torch.as_tensor(graph_indices, dtype=torch.long)
    for key, attr, value, cuts, dim in fields(data, slices):
        require(cuts.numel() == nsource + 1 and int(cuts[0]) == 0, f"Bad slices: {key}/{attr}")
        width = int(cuts[1])
        require(width > 0 and bool(torch.all(cuts[1:] - cuts[:-1] == width)), f"Variable-size field: {key}/{attr}")
        require(int(cuts[-1]) == value.shape[dim], f"Slice extent mismatch: {key}/{attr}")
        leading = value.movedim(dim, 0)
        trailing = tuple(leading.shape[1:])
        source = leading.reshape(nsource, width, *trailing)
        dest = torch.empty((n, width, *trailing), dtype=value.dtype)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            dest[start:stop] = source[indices[start:stop]]
        store_at(result, key)[attr] = dest.reshape(n * width, *trailing).movedim(0, dim).contiguous()
        outcuts = result_slices if key is None else result_slices.setdefault(key, {})
        outcuts[attr] = torch.arange(n + 1, dtype=torch.int64) * width
    return result, result_slices


def verify_mapping(source_payload, selected_payload, graph_indices):
    """Check every attribute in a deterministic cross-section, not labels alone."""
    import torch
    picks = np.unique(np.linspace(0, len(graph_indices) - 1, min(32, len(graph_indices)), dtype=int))
    for key, attr, value, cuts, dim in fields(*selected_payload):
        source = store_at(source_payload[0], key)[attr]
        scuts = source_payload[1] if key is None else source_payload[1][key]
        for i in picks:
            si = int(graph_indices[i])
            begin, end = int(cuts[i]), int(cuts[i + 1])
            sb, se = int(scuts[attr][si]), int(scuts[attr][si + 1])
            require(torch.equal(value.narrow(dim, begin, end - begin), source.narrow(dim, sb, se - sb)),
                    f"Tensor mapping mismatch: {key}/{attr}, graph {i}")


def cache_binding(root, report, records):
    label = f"{records // 1000}k"
    info = report["scales"][label]
    return {"schema": SCHEMA, "dataset_version_hash": report["dataset_version_hash"], "records": records,
            "parent_ids_sha256": info["parent_ids_sha256"],
            "source_graph_indices_sha256": info["source_graph_indices_sha256"],
            "source_cache_sha256": report["binding"]["source_cache_sha256"]}


def verify_cache(root, report, records, source_payload=None, deep_hash=False):
    label = f"{records // 1000}k"
    path = root / f"processed/{label}/train.pt"
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    binding = cache_binding(root, report, records)
    require(all(metadata.get(k) == v for k, v in binding.items()), f"Stale/incompatible cache: {path}")
    require(path.stat().st_size == metadata["bytes"], f"Cache size mismatch: {path}")
    if deep_hash:
        require(sha256(path) == metadata["file_sha256"], f"Cache checksum mismatch: {path}")
    parents = np.load(root / f"train/{label}/parent_ids.npy")
    indices = np.load(root / f"train/{label}/source_graph_indices.npy")
    payload = torch_load(path)
    check_pairs(payload[0], parents)
    require(payload[0].load_setting_id.numel() == records, f"Wrong cache length: {path}")
    if source_payload is not None:
        verify_mapping(source_payload, payload, indices)
    del payload
    gc.collect()
    return metadata


def build_caches(repo, source, root, report, source_ids, scales, verify_only, deep_hash):
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(repo))
    path = source / "processed/canos_rho_scales/1280k/train.pt"
    index = json.loads((source / "processed/canos_rho_scales/index.json").read_text())
    require(path.stat().st_size == index["scales"]["1280k"]["bytes"], "Source cache byte size differs from frozen metadata")
    missing = [s for s in scales if not (root / f"processed/{s // 1000}k/train.pt").exists()]
    if verify_only:
        require(not missing, f"Missing requested caches: {missing}")
    else:
        estimate = path.stat().st_size * sum(missing) / 1280000
        require(shutil.disk_usage(root).free >= estimate + 20 * 1024**3,
                f"Insufficient disk: need approximately {estimate / 1024**3:.1f} GiB plus 20 GiB reserve")
    if deep_hash:
        require(sha256(path) == report["binding"]["source_cache_sha256"], "Source cache checksum mismatch")
    payload = torch_load(path)
    check_pairs(payload[0], source_ids["train"])
    done = {}
    for records in scales:
        label = f"{records // 1000}k"
        destination = root / f"processed/{label}/train.pt"
        meta_path = destination.with_suffix(".meta.json")
        if destination.exists():
            require(meta_path.exists(), f"Cache without sidecar: {destination}; inspect before manual removal")
        else:
            require(not verify_only, f"Missing cache: {destination}")
            parent_ids = np.load(root / f"train/{label}/parent_ids.npy")
            graph_indices = np.load(root / f"train/{label}/source_graph_indices.npy")
            print(f"Building {label}: {records:,} graphs", flush=True)
            selected = gather_payload(payload, graph_indices)
            check_pairs(selected[0], parent_ids)
            verify_mapping(payload, selected, graph_indices)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(f".train.pt.{os.getpid()}.tmp")
            torch.save(selected, temp)
            metadata = dict(cache_binding(root, report, records), bytes=temp.stat().st_size, file_sha256=sha256(temp))
            # Sidecar first: a crash cannot leave an accepted cache lacking its binding.
            atomic_json(meta_path, metadata)
            os.replace(temp, destination)
            del selected
            gc.collect()
        done[label] = verify_cache(root, report, records, payload, deep_hash)
        print(f"Verified {label}: {done[label]['bytes'] / 1024**3:.3f} GiB", flush=True)
    for split in ("val", "test"):
        shared = root / f"shared/{split}.pt"
        require(shared.stat().st_size == report["holdouts"][split]["bytes"], f"Bad {split} cache size")
        if deep_hash:
            require(sha256(shared) == report["holdouts"][split]["file_sha256"], f"Bad {split} checksum")
        held = torch_load(shared)
        check_pairs(held[0], source_ids[split])
        del held
    del payload
    gc.collect()
    # Exercise the actual registry dataset, including a shuffled PyG batch.
    from core.datasets.pfdelta_rho_dataset import PFDeltaRhoCANOS
    from torch_geometric.loader import DataLoader
    smoke = {}
    label = f"{min(scales) // 1000}k"
    for split, subdir, expected in (("train", f"processed/{label}", min(scales)),
                                    ("val", "shared", 16800), ("test", "shared", 16800)):
        ds = PFDeltaRhoCANOS(root_dir=str(root), case_name="case118", split=split,
                            task=SCHEMA, add_bus_type=True, processed_subdir=subdir)
        require(len(ds) == expected, f"Reader count mismatch: {split}")
        batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)))
        require(batch.num_graphs == 4 and tuple(ds[0]["bus"].x.shape) == (118, 2), f"Reader batch mismatch: {split}")
        smoke[split] = {"records": len(ds), "first_parent": int(ds[0].load_setting_id)}
        del ds, batch
        gc.collect()
    atomic_json(root / "protocol/loader_smoke.json", {"dataset_version_hash": report["dataset_version_hash"], "splits": smoke})
    prior_path = root / "protocol/cache_index.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {"dataset_version_hash": report["dataset_version_hash"], "scales": {}}
    require(prior["dataset_version_hash"] == report["dataset_version_hash"], "Cache index version mismatch")
    prior["scales"].update(done)
    atomic_json(prior_path, prior)
    meta = json.loads((root / "meta.json").read_text())
    meta.update(status="READY" if len(prior["scales"]) == len(SCALES) else "PARTIAL_CACHES", materialized_scales=sorted(prior["scales"]))
    atomic_json(root / "meta.json", meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261013, help="Data split seed, distinct from model seeds")
    parser.add_argument("--scales", default="all", help="Cache subset only, e.g. 10k or 10k,20k; all indices are always frozen")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--indices-only", action="store_true")
    mode.add_argument("--build-caches", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--deep-hash", action="store_true", help="Also rehash complete source/output tensors during verification")
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = (args.source or repo / "data" / SOURCE_NAME).resolve()
    root = (args.output or repo / "data" / SCHEMA).resolve()
    require(root != source and source not in root.parents and root not in source.parents, "Source and target must be separate sibling datasets")
    scales = parse_scales(args.scales)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (root / ".prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another data preparation process is active; refusing concurrent writes")
        report, ids = make_indices(source, root, args.seed)
        print(f"Frozen 14 nested scales, dataset hash {report['dataset_version_hash']}", flush=True)
        if not args.indices_only:
            build_caches(repo, source, root, report, ids, scales, args.verify, args.deep_hash)
        print(json.dumps({"output": str(root), "dataset_version_hash": report["dataset_version_hash"],
                          "mode": "indices" if args.indices_only else "verify" if args.verify else "build",
                          "seconds": round(time.time() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
