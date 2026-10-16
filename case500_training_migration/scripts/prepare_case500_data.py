#!/usr/bin/env python3
"""Materialize CANOS caches from already frozen case500 generator splits.

Never re-split, solve, filter, normalize or modify raw data. The original liuy
PFDeltaRhoCANOS converter is used unchanged. Tensor buffers are disk-backed;
Python holds one endpoint pair at a time, rather than hundreds of thousands of
graph objects. --scales controls cache creation, not frozen memberships.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
SCHEMA = "case500_canos_endpoint_cache_v1"
DEFAULT_SCALES = [10000, 20000, 40000, 45000, 50000, 55000, 60000, 80000,
                  90000, 100000, 110000, 120000, 160000, 320000]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def label(records):
    return f"{records // 1000}k" if records % 1000 == 0 else str(records)


def safe_file(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Non-portable source file: {relative}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"Missing/unsafe source: {path}")
    return path


def vector_hash(values):
    vector = np.asarray(values, dtype="<f8")
    require(vector.ndim == 1 and np.isfinite(vector).all(), "Invalid known-input feature vector")
    vector[vector == 0] = 0.0
    return hashlib.sha256(vector.tobytes()).hexdigest()


def known_inputs(network):
    """Same bus-major features as generator input_description; no output labels."""
    buses = sorted(network["bus"], key=int)
    demand = {bus: [0.0, 0.0] for bus in buses}
    pg = dict.fromkeys(buses, 0.0)
    for load in network["load"].values():
        if int(load.get("status", 1)) == 1:
            target = demand[str(load["load_bus"])]
            target[0] += float(load["pd"])
            target[1] += float(load["qd"])
    for generator in network["gen"].values():
        if int(generator.get("gen_status", 1)) == 1:
            pg[str(generator["gen_bus"])] += float(generator["pg"])
    result = []
    for bus_id in buses:
        bus = network["bus"][bus_id]
        kind = int(bus["bus_type"])
        result.extend([*demand[bus_id], pg[bus_id] if kind == 2 else 0.0,
                       float(bus["vm"]) if kind in (2, 3) else 0.0,
                       float(bus["va"]) if kind == 3 else 0.0, float(kind)])
    return result


def audit_source(source):
    frozen_path = source / "splits/frozen.json"
    frozen = json.loads(frozen_path.read_text())
    require(frozen.get("completed") is True, "Generator splits are not complete/frozen")
    for relative, expected in frozen["artifacts"].items():
        path = safe_file(source / "splits", relative)
        require(file_hash(path) == expected, f"Changed frozen split artifact: {relative}")
    signature = frozen["signature"]
    require(file_hash(source / "plan.json") == signature["plan_sha256"], "Source plan changed")
    require(file_hash(source / "manifests/parents.jsonl") == signature["manifest_sha256"],
            "Source manifest changed")
    protocol = json.loads((source / "splits/protocol.json").read_text())
    require(object_hash(protocol) == signature["protocol_sha256"], "Source protocol changed")
    require(frozen["dataset_hash"] == object_hash({"signature": signature, "artifacts": frozen["artifacts"]}),
            "Source dataset hash mismatch")
    require(protocol["case_name"] == "case500", "Expected case500 frozen generator output")
    if protocol["mode"] == "full":
        require(protocol["train_endpoint_scales"] == DEFAULT_SCALES, "Full case500 scale protocol changed")
        require(protocol["val_parents_per_branch"] == protocol["test_parents_per_branch"] == 50,
                "Full protocol requires 50 held-out parents per eligible branch")
    else:
        require(protocol["mode"] == "smoke", "Unknown generator mode")
    catalogue = {}
    fingerprints, feature_hashes, paths = set(), set(), set()
    with (source / "manifests/parents.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parent = int(row["parent_id"])
            fingerprint = str(row["input_fingerprint"])
            features = row["input_features"]
            binary = vector_hash(features)
            require(len(features) == protocol["feature_dimension"], "Wrong input feature dimension")
            require(parent > 0 and parent not in catalogue, "Duplicate/invalid parent ID")
            require(fingerprint and fingerprint not in fingerprints, "Duplicate known-input fingerprint")
            require(binary not in feature_hashes, "Duplicate known-input vector")
            require(len(row["files"]) == 2, "Parent needs exactly two endpoint files")
            for relative in row["files"]:
                require(relative not in paths, "Endpoint file reused by two parent records")
                safe_file(source, relative)
                paths.add(relative)
            catalogue[parent] = {"parent_id": parent, "branch_id": int(row["branch_id"]),
                                  "input_fingerprint": fingerprint, "input_vector_sha256": binary,
                                  "files": row["files"], "source_batch": row.get("source_batch", "unknown")}
            fingerprints.add(fingerprint)
            feature_hashes.add(binary)
    require(len(catalogue) == protocol["total_parents"], "Incomplete parent catalogue")
    branches = sorted(map(int, protocol["eligible_branches"]))
    memberships = {}
    all_names = ["val", "test"] + [f"train_{label(n)}" for n in protocol["train_endpoint_scales"]]
    for name in all_names:
        ids = np.load(source / f"splits/{name}_parent_ids.npy", allow_pickle=False).astype(np.int64)
        require(len(ids) > 0 and len(np.unique(ids)) == len(ids), f"Duplicate/empty {name} membership")
        require(all(int(parent) in catalogue for parent in ids), f"Unknown {name} parent")
        split = json.loads((source / f"splits/{name}.json").read_text())
        records = split["records"]
        require(len(records) == 2 * len(ids), f"Wrong endpoint count in {name}")
        for i, parent in enumerate(ids.tolist()):
            row = catalogue[parent]
            for j, rho in enumerate((1.0, 0.0)):
                record = records[2 * i + j]
                require(int(record["load_setting_id"]) == parent and float(record["rho"]) == rho
                        and int(record["branch_id"]) == row["branch_id"]
                        and int(record["sample_id"]) == 2 * parent - int(rho)
                        and record["file"] == row["files"][j], f"Pair/order mismatch in {name}")
        counts = Counter(catalogue[int(p)]["branch_id"] for p in ids)
        if name in ("val", "test"):
            target = protocol[f"{name}_parents_per_branch"]
            require(counts == Counter({b: target for b in branches}), f"Uneven held-out lines: {name}")
        else:
            total = len(ids)
            q, r = divmod(total, len(branches))
            expected = {b: q + int(i < r) for i, b in enumerate(branches)}
            require(all(counts[b] == count for b, count in expected.items()) and set(counts) <= set(branches),
                    f"Training line quotas differ from frozen generation protocol: {name}")
        memberships[name] = ids
        del records, split
    train_names = all_names[2:]
    largest = memberships[train_names[-1]]
    for name in train_names:
        require(np.array_equal(memberships[name], largest[:len(memberships[name])]),
                "Training scales are not exact nested prefixes")
    joined = np.concatenate([largest, memberships["val"], memberships["test"]])
    require(len(joined) == len(catalogue) and len(np.unique(joined)) == len(catalogue),
            "Parent overlap or omitted parents across train/val/test")
    return frozen, protocol, catalogue, memberships


def validate_pair(samples, row):
    """Validate stored solver provenance and actual fixed inputs; no new PF solve."""
    for rho, sample in zip((1.0, 0.0), samples):
        meta, result, network = sample["rho_meta"], sample["solution"], sample["network"]
        require(int(meta["parent_id"]) == int(meta["load_setting_id"]) == row["parent_id"]
                and int(meta["branch_id"]) == row["branch_id"] and float(meta["rho"]) == rho
                and int(meta["sample_id"]) == 2 * row["parent_id"] - int(rho), "Raw endpoint identity mismatch")
        require(meta["input_fingerprint"] == row["input_fingerprint"], "Raw fingerprint mismatch")
        require(result["termination_status"] == "PF_CONVERGED", "Endpoint PF did not converge")
        for block in (meta, result):
            require(block.get("solver_type") == "AC_PF" and block.get("solver_method") == "newton"
                    and block.get("solver_fallback") == "none", "Endpoint is not the required Newton AC-PF label")
        qa = result["validation"]
        for key, tolerance in {"max_balance_residual": 1e-7, "fixed_bus_pg_error": 1e-8,
                               "fixed_vm_error": 1e-8, "fixed_slack_va_error": 1e-8}.items():
            value = float(qa[key])
            require(np.isfinite(value) and 0 <= value <= tolerance, f"Invalid stored validation: {key}")
        require(0 < float(qa["min_vm"]) <= float(qa["max_vm"]) < float("inf"), "Invalid voltage metadata")
        # Generator dictionaries retain their ACOPF initial guesses; only the
        # prescribed components enter known_inputs, never unknown solved Pg/Qg.
        require(vector_hash(known_inputs(network)) == row["input_vector_sha256"],
                "Actual fixed inputs disagree with manifest features")
        controlled = [int(key) for key, b in network["branch"].items() if b.get("rho_controlled", False)]
        require(controlled == [row["branch_id"]], "Wrong controlled branch")
        branch = network["branch"][str(row["branch_id"])]
        require(float(branch["rho"]) == rho and int(branch["br_status"]) == int(rho), "Wrong endpoint admittance state")
        for edge in network["branch"].values():
            if int(edge.get("br_status", 1)) == 1 or edge.get("rho_controlled", False):
                require(edge.get("rho_controlled", False) or float(edge.get("rho", 1.0)) == 1.0,
                        "Uncontrolled branch unexpectedly changes rho")
                require(np.isfinite(float(edge["tap"])) and float(edge["tap"]) > 0,
                        "CANOS requires finite positive transformer tap")
                require(abs(float(edge.get("g_fr", 0.0))) <= 1e-14
                        and abs(float(edge.get("g_to", 0.0))) <= 1e-14
                        and abs(float(edge.get("b_fr", 0.0)) - float(edge.get("b_to", 0.0))) <= 1e-14,
                        "Current CANOS/PBL requires zero branch conductance shunts and symmetric susceptance")
                require(np.isfinite([edge["br_r"], edge["br_x"]]).all()
                        and abs(complex(edge["br_r"], edge["br_x"])) > 0, "Invalid finite endpoint branch impedance")
        bus_ids = sorted(map(int, network["bus"]))
        require(bus_ids == list(range(1, len(bus_ids) + 1)), "Converter needs contiguous 1-based bus IDs")
        # The legacy reader assumes these entries are active. Fail rather than
        # silently including inactive load/shunt injections or changing labels.
        require(all(int(x.get("status", 1)) == 1 for group in ("load", "shunt")
                    for x in network[group].values()), "Inactive load/shunt needs explicit converter adaptation")
    n1, n0 = samples[0]["network"], samples[1]["network"]
    for group in ("bus", "gen", "load", "shunt"):
        require(n1[group] == n0[group], f"Paired endpoints changed fixed network/control dictionaries: {group}")
    require(n1["branch"].keys() == n0["branch"].keys(), "Paired branch sets changed")
    for key in n1["branch"]:
        left, right = dict(n1["branch"][key]), dict(n0["branch"][key])
        if int(key) == row["branch_id"]:
            for attr in ("rho", "br_status"):
                left.pop(attr, None)
                right.pop(attr, None)
        require(left == right, f"Unexpected paired branch parameter change: {key}")


def make_reader():
    sys.path.insert(0, str(PACKAGE))
    from core.datasets.pfdelta_rho_dataset import PFDeltaRhoCANOS
    # Invoke conversion only, bypassing InMemoryDataset processing/download.
    reader = PFDeltaRhoCANOS.__new__(PFDeltaRhoCANOS)
    reader.add_bus_type = True
    reader.include_branch_rho = True
    reader.pre_transform = None
    reader.pre_filter = None
    return reader


def graphs_from_raw(source, parents, catalogue, digest):
    import torch
    reader = make_reader()
    for index, parent in enumerate(parents):
        row = catalogue[int(parent)]
        samples = []
        for relative in row["files"]:
            raw = safe_file(source, relative).read_bytes()
            digest.update(relative.encode() + b"\0" + hashlib.sha256(raw).digest())
            samples.append(json.loads(raw))
        validate_pair(samples, row)
        pair = []
        for rho, sample in zip((1.0, 0.0), samples):
            reader._ensure_branch_solution(sample)
            graph = reader.build_heterodata(sample, is_cpf_sample=False)
            expected_types = torch.tensor([int(sample["network"]["bus"][key]["bus_type"])
                                           for key in sorted(sample["network"]["bus"], key=int)])
            require(torch.equal(graph["bus"].bus_type, expected_types),
                    "Reader changed PV/PQ type (for example zero-generation PV); refusing inconsistent fixed bus types")
            require(graph[("bus", "branch", "bus")].edge_attr.shape[1] == 9, "Expected ninth rho edge feature")
            graph.rho = torch.tensor([rho])
            graph.load_setting_id = torch.tensor([int(parent)], dtype=torch.long)
            graph.rho_sample_id = torch.tensor([2 * int(parent) - int(rho)], dtype=torch.long)
            pair.append(graph)
        for kind in ("bus", "PV", "PQ", "slack"):
            require(torch.equal(pair[0][kind].x, pair[1][kind].x), f"Paired graph known inputs differ: {parent}/{kind}")
        require(torch.equal(pair[0]["bus"].bus_type, pair[1]["bus"].bus_type), "Paired bus types differ")
        yield from pair
        if (index + 1) % 1000 == 0:
            print(f"Converted {index + 1:,}/{len(parents):,} paired parents", flush=True)


def tensor_fields(data):
    import torch
    for store in data.stores:
        for attr, value in store.items():
            require(torch.is_tensor(value) and value.ndim >= 1, f"Unsupported graph field {store._key}/{attr}")
            dim = data.__cat_dim__(attr, value, store)
            require(isinstance(dim, int), f"Unsupported concatenation for {attr}")
            yield store._key, attr, value, dim % value.ndim


def store_at(data, key):
    return data._global_store if key is None else data[key]


def save_graph_stream(graphs, count, destination):
    """PyG InMemoryDataset collation, with exact local indices and disk buffers."""
    import torch
    destination.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(graphs)
    first = next(iterator)
    template = copy.copy(first)
    specifications = list(tensor_fields(first))
    need = sum(value.numel() * value.element_size() for _, _, value, _ in specifications) * count
    require(shutil.disk_usage(destination.parent).free > 2.1 * need + 64 * 1024**2,
            f"Insufficient disk: need about {2.1 * need / 1024**3:.1f} GiB for cache + temporary buffers")
    with tempfile.TemporaryDirectory(prefix=".tensor-build-", dir=destination.parent) as temp:
        buffers, slices = {}, {}
        for i, (key, attr, value, dim) in enumerate(specifications):
            shape = list(value.shape)
            shape[dim] *= count
            buffer = torch.from_file(str(Path(temp) / f"tensor_{i}.bin"), shared=True,
                                     size=value.numel() * count, dtype=value.dtype).reshape(shape)
            store_at(template, key)[attr] = buffer
            buffers[(key, attr)] = (buffer, tuple(value.shape), dim)
            cut_store = slices if key is None else slices.setdefault(key, {})
            cut_store[attr] = torch.arange(count + 1, dtype=torch.long) * value.shape[dim]
        def append(graph, i):
            current = list(tensor_fields(graph))
            require([(k, a) for k, a, _, _ in current] == list(buffers), "Graph schema changed between parents")
            for key, attr, value, dim in current:
                buffer, shape, expected_dim = buffers[(key, attr)]
                require(tuple(value.shape) == shape and value.dtype == buffer.dtype and dim == expected_dim,
                        f"Graph field shape/type changed: {key}/{attr}. Check bus types and branch retention.")
                require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Nonfinite graph tensor")
                buffer.narrow(dim, i * shape[dim], shape[dim]).copy_(value)
        append(first, 0)
        actual = 1
        for graph in iterator:
            require(actual < count, "Too many streamed graphs")
            append(graph, actual)
            actual += 1
        require(actual == count, "Too few streamed graphs")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        torch.save((template, slices), temporary)
        temporary.replace(destination)
        del buffers, template
        gc.collect()


def save_prefix(source_cache, records, destination):
    """Copy contiguous tensor prefixes in chunks without rebuilding graph objects."""
    import torch
    data, slices = torch.load(source_cache, map_location="cpu", mmap=True, weights_only=False)
    require(records <= data.load_setting_id.numel(), "Prefix beyond source cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    need = 0
    for key, attr, value, dim in tensor_fields(data):
        cut = (slices if key is None else slices[key])[attr]
        need += int(cut[records]) * (value.numel() // value.shape[dim]) * value.element_size()
    require(shutil.disk_usage(destination.parent).free > 2.1 * need + 64 * 1024**2,
            f"Insufficient disk for prefix cache and buffers: {destination}")
    with tempfile.TemporaryDirectory(prefix=".prefix-build-", dir=destination.parent) as temp:
        result = copy.copy(data)
        output_slices = {}
        for i, (key, attr, value, dim) in enumerate(tensor_fields(data)):
            cut = (slices if key is None else slices[key])[attr]
            extent = int(cut[records])
            shape = list(value.shape)
            shape[dim] = extent
            numel = int(np.prod(shape))
            buffer = torch.from_file(str(Path(temp) / f"tensor_{i}.bin"), shared=True,
                                     size=numel, dtype=value.dtype).reshape(shape)
            # Bound each copy to about 32 MiB; the operating system can evict
            # clean file-backed pages when processing the large 320k cache.
            row_bytes = max(1, value.numel() // value.shape[dim] * value.element_size())
            chunk = max(1, (32 * 1024**2) // row_bytes)
            for begin in range(0, extent, chunk):
                length = min(chunk, extent - begin)
                buffer.narrow(dim, begin, length).copy_(value.narrow(dim, begin, length))
            store_at(result, key)[attr] = buffer
            target_cuts = output_slices if key is None else output_slices.setdefault(key, {})
            target_cuts[attr] = cut[:records + 1].clone()
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        torch.save((result, output_slices), temporary)
        temporary.replace(destination)
        del result
        gc.collect()


def verify_cache(path, parents):
    import torch
    data, slices = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    require(np.array_equal(data.load_setting_id.numpy().reshape(-1), np.repeat(parents, 2)), "Cache parent mapping failed")
    require(np.array_equal(data.rho.numpy().reshape(-1), np.tile([1., 0.], len(parents))), "Cache rho pairing failed")
    expected = np.column_stack((2 * parents - 1, 2 * parents)).reshape(-1)
    require(np.array_equal(data.rho_sample_id.numpy().reshape(-1), expected), "Cache sample mapping failed")
    for key, attr, value, dim in tensor_fields(data):
        cut = (slices if key is None else slices[key])[attr]
        require(len(cut) == 2 * len(parents) + 1 and int(cut[0]) == 0 and int(cut[-1]) == value.shape[dim],
                f"Cache slices invalid: {key}/{attr}")
    return {"graphs": 2 * len(parents), "parent_count": len(parents),
            "sha256": file_hash(path), "bytes": path.stat().st_size,
            "bus_count": int(data["bus"].x.shape[0] // (2 * len(parents)))}


def prepare(source, output, scales="10k", indices_only=False):
    source, output = source.resolve(), output.resolve()
    require(source != output and source not in output.parents and output not in source.parents,
            "Source and cache roots must be separate, non-nested directories")
    frozen, protocol, catalogue, memberships = audit_source(source)
    allowed = protocol["train_endpoint_scales"]
    selected = allowed if scales == "all" else sorted(set(int(s[:-1]) * 1000 if s.endswith("k") else int(s)
                                                        for s in scales.lower().split(",")))
    require(bool(selected) and set(selected) <= set(allowed), f"Available scales: {allowed}")
    output.mkdir(parents=True, exist_ok=True)
    conversion_files = ["core/datasets/pfdelta_dataset.py", "core/datasets/pfdelta_variants.py",
                        "core/datasets/pfdelta_rho_dataset.py", "scripts/prepare_case500_data.py"]
    binding = {"schema": SCHEMA, "source_dataset_hash": frozen["dataset_hash"],
               "converter_sha256": {name: file_hash(PACKAGE / name) for name in conversion_files},
               "include_branch_rho": True, "normalization": None, "rho_order": [1.0, 0.0]}
    dataset_hash = object_hash(binding)
    index_path = output / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        require(index["dataset_hash"] == dataset_hash, "Existing caches belong to different data/converter version")
    else:
        index = {"schema": SCHEMA, "case_name": "case500", "dataset_hash": dataset_hash,
                 "source_dataset_hash": frozen["dataset_hash"], "source_mode": protocol["mode"],
                 "eligible_branch_count": len(protocol["eligible_branches"]),
                 "parent_manifest": "manifests/parents.jsonl", "train": {},
                 "all_endpoint_scales": allowed, "binding": binding}
    (output / "manifests").mkdir(exist_ok=True)
    manifest_path = output / index["parent_manifest"]
    content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                      for row in catalogue.values())
    if manifest_path.exists():
        require(manifest_path.read_text() == content, "Existing copied identity manifest changed")
    else:
        manifest_path.write_text(content)
    del content
    for name, parents in memberships.items():
        train = name.startswith("train_")
        relative = f"train/{name[6:]}/parent_ids.npy" if train else f"shared/{name}_parent_ids.npy"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(np.array_equal(np.load(path), parents), "Previously frozen cache parent IDs changed")
        else:
            np.save(path, parents)
        entry = {"cache": f"train/{name[6:]}/train.pt" if train else f"shared/{name}.pt",
                 "graphs": 2 * len(parents), "parent_count": len(parents), "parent_ids": relative,
                 "parent_ids_sha256": file_hash(path), "ready": False}
        if train:
            index["train"].setdefault(name[6:], entry)
        else:
            index.setdefault(name, entry)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(index_path, index)
    if indices_only:
        return index
    import torch
    torch.set_num_threads(2)
    # Largest requested cache supplies all smaller exact prefixes. Existing
    # smaller caches can be reused, but never rebuilt using a different split.
    names = [f"train_{label(n)}" for n in sorted(selected, reverse=True)] + ["val", "test"]
    prefix_source = None
    prefix_source_hash = None
    for name in names:
        entry = index["train"][name[6:]] if name.startswith("train_") else index[name]
        path = output / entry["cache"]
        if path.exists():
            require(entry.get("ready") and entry.get("sha256"), f"Uncommitted cache exists; inspect before removing: {path}")
            check = verify_cache(path, memberships[name])
            require(check["sha256"] == entry["sha256"], f"Existing cache changed: {path}")
        else:
            print(f"Building {name}: {entry['graphs']:,} endpoint graphs", flush=True)
            raw_digest = hashlib.sha256()
            if name.startswith("train_") and prefix_source is not None:
                save_prefix(prefix_source, entry["graphs"], path)
                provenance = {"derived_prefix_cache_sha256": prefix_source_hash}
            else:
                save_graph_stream(graphs_from_raw(source, memberships[name], catalogue, raw_digest), entry["graphs"], path)
                provenance = {"raw_endpoint_content_sha256": raw_digest.hexdigest()}
            check = verify_cache(path, memberships[name])
            entry.update(check, ready=True, **provenance)
            atomic_json(index_path, index)
        if name.startswith("train_") and prefix_source is None:
            prefix_source = path
            prefix_source_hash = entry["sha256"]
        print(f"Verified {name}: {path.stat().st_size / 1024**3:.3f} GiB", flush=True)
    atomic_json(output / "audit.json", {"dataset_hash": dataset_hash, "parent_overlap": 0,
                "duplicate_known_input_fingerprints": 0, "duplicate_known_input_vectors": 0,
                "preserved_generator_splits": True, "nested_training_prefixes": True,
                "relabelled_samples": 0, "additional_filters": False,
                "raw_validation": "Newton/PF provenance and stored numerical checks plus actual paired fixed inputs; no independent new PF solve",
                "notes": ["Line counts differ by at most one at each scale; input-cluster mixture is inherited unchanged.",
                          "Comparable sampling does not prove identical full joint distributions or dynamic stability."]})
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scales", default="10k", help="Materialize 10k initially, or comma-separated scales / all")
    parser.add_argument("--indices-only", action="store_true")
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    require(source != output and source not in output.parents and output not in source.parents,
            "Source and cache roots must be separate, non-nested directories")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = prepare(args.source_root, args.output_root, args.scales, args.indices_only)
    print(json.dumps({"index": str(args.output_root.resolve() / "index.json"),
                      "dataset_hash": index["dataset_hash"], "mode": "indices" if args.indices_only else "caches"}))


if __name__ == "__main__":
    main()
