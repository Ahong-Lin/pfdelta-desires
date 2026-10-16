"""Small synthetic format fixtures; these are not numerical PF solver tests."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_case500_data", ROOT / "scripts/prepare_case500_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def raw_fixture(parent, branch, rho):
    buses = {str(i): {"bus_type": kind, "vm": 1.0, "va": 0.0, "vmin": .9, "vmax": 1.1}
             for i, kind in ((1, 3), (2, 2), (3, 1))}
    gens = {str(i): {"gen_bus": i, "gen_status": 1, "pg": .2, "qg": .1,
                     "pmin": 0, "pmax": 2, "qmin": -2, "qmax": 2} for i in (1, 2)}
    edges = {}
    for i, (f, t) in enumerate(((1, 2), (2, 3), (3, 1)), 1):
        edge = {"f_bus": f, "t_bus": t, "br_status": 1, "br_r": .01, "br_x": .1,
                "g_fr": 0., "g_to": 0., "b_fr": .01, "b_to": .01, "tap": 1., "shift": 0.,
                "rate_a": 2., "rho": 1., "rho_controlled": i == branch}
        if i == branch:
            edge.update(rho=rho, br_status=int(rho))
        edges[str(i)] = edge
    network = {"bus": buses, "gen": gens, "branch": edges, "shunt": {},
               "load": {"1": {"load_bus": 3, "status": 1, "pd": parent * .1, "qd": parent * .01}}}
    fingerprint = hashlib.sha256(f"parent-{parent}".encode()).hexdigest()
    qa = {"max_balance_residual": 1e-10, "fixed_bus_pg_error": 0., "fixed_vm_error": 0.,
          "fixed_slack_va_error": 0., "min_vm": 1., "max_vm": 1.}
    solved = {"bus": {key: {"vm": 1., "va": 0.} for key in buses},
              "gen": {key: {"pg": .2, "qg": .1} for key in gens}}
    metadata = {"parent_id": parent, "load_setting_id": parent, "branch_id": branch, "rho": rho,
                "sample_id": 2 * parent - int(rho), "input_fingerprint": fingerprint,
                "solver_type": "AC_PF", "solver_method": "newton", "solver_fallback": "none"}
    result = {"solution": solved, "termination_status": "PF_CONVERGED", "validation": qa,
              "solver_type": "AC_PF", "solver_method": "newton", "solver_fallback": "none"}
    return {"network": network, "solution": result, "rho_meta": metadata}


def make_source(root):
    (root / "raw").mkdir(parents=True)
    (root / "splits").mkdir()
    (root / "manifests").mkdir()
    protocol = {"mode": "smoke", "case_name": "case500", "feature_dimension": 18,
                "eligible_branches": [1, 2], "train_endpoint_scales": [4, 8], "total_parents": 8,
                "val_parents_per_branch": 1, "test_parents_per_branch": 1}
    prep.atomic_json(root / "plan.json", {"mode": "smoke", "test_fixture": True})
    prep.atomic_json(root / "splits/protocol.json", protocol)
    rows = {}
    with (root / "manifests/parents.jsonl").open("w") as out:
        for parent in range(1, 9):
            branch = 1 + (parent + 1) % 2
            samples = [raw_fixture(parent, branch, rho) for rho in (1., 0.)]
            files = [f"raw/sample_{2*parent-1}.json", f"raw/sample_{2*parent}.json"]
            for path, sample in zip(files, samples):
                prep.atomic_json(root / path, sample)
            row = {"parent_id": parent, "branch_id": branch, "files": files,
                   "input_fingerprint": samples[0]["rho_meta"]["input_fingerprint"],
                   "input_features": prep.known_inputs(samples[0]["network"])}
            out.write(json.dumps(row) + "\n")
            rows[parent] = row
    memberships = {"train_4": [1, 2], "train_8": [1, 2, 3, 4], "val": [5, 6], "test": [7, 8]}
    for name, parents in memberships.items():
        records = []
        for parent in parents:
            row = rows[parent]
            for i, rho in enumerate((1., 0.)):
                records.append({"load_setting_id": parent, "branch_id": row["branch_id"], "rho": rho,
                                "sample_id": 2 * parent - int(rho), "file": row["files"][i]})
        prep.atomic_json(root / f"splits/{name}.json", {"records": records})
        np.save(root / f"splits/{name}_parent_ids.npy", np.asarray(parents, dtype=np.int64))
    freeze(root)
    return rows


def freeze(root):
    protocol = json.loads((root / "splits/protocol.json").read_text())
    signature = {"plan_sha256": prep.file_hash(root / "plan.json"),
                 "manifest_sha256": prep.file_hash(root / "manifests/parents.jsonl"),
                 "protocol_sha256": prep.object_hash(protocol)}
    artifacts = {path.name: prep.file_hash(path) for path in (root / "splits").iterdir()
                 if path.name != "frozen.json"}
    frozen = {"signature": signature, "artifacts": artifacts, "completed": True}
    frozen["dataset_hash"] = prep.object_hash({"signature": signature, "artifacts": artifacts})
    prep.atomic_json(root / "splits/frozen.json", frozen)


class Case500DataTests(unittest.TestCase):
    def test_pair_rejects_changed_control_and_solver(self):
        pair = [raw_fixture(1, 1, rho) for rho in (1., 0.)]
        row = {"parent_id": 1, "branch_id": 1, "input_fingerprint": pair[0]["rho_meta"]["input_fingerprint"],
               "input_vector_sha256": prep.vector_hash(prep.known_inputs(pair[0]["network"]))}
        prep.validate_pair(pair, row)
        bad = copy.deepcopy(pair)
        bad[1]["network"]["load"]["1"]["pd"] += 1.
        with self.assertRaisesRegex(ValueError, "fixed inputs"):
            prep.validate_pair(bad, row)
        bad = copy.deepcopy(pair)
        bad[1]["solution"]["solver_type"] = "AC_OPF"
        with self.assertRaisesRegex(ValueError, "Newton"):
            prep.validate_pair(bad, row)
        bad = copy.deepcopy(pair)
        bad[0]["network"]["branch"]["1"]["b_fr"] = .02
        with self.assertRaisesRegex(ValueError, "symmetric"):
            prep.validate_pair(bad, row)

    def test_source_rejects_overlap_even_with_updated_frozen_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            make_source(source)
            val = json.loads((source / "splits/val.json").read_text())
            test = json.loads((source / "splits/test.json").read_text())
            test["records"][:2] = val["records"][:2]
            prep.atomic_json(source / "splits/test.json", test)
            np.save(source / "splits/test_parent_ids.npy", np.array([5, 8]))
            freeze(source)
            with self.assertRaisesRegex(ValueError, "overlap"):
                prep.audit_source(source)

    def test_source_rejects_changed_raw_pair_and_frozen_split(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            make_source(source)
            with (source / "splits/test.json").open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "Changed frozen"):
                prep.audit_source(source)

    def test_prepare_exact_prefix_caches_and_original_converter(self):
        import torch
        from torch_geometric.data import InMemoryDataset
        from torch_geometric.data.separate import separate
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "cache"
            make_source(source)
            index = prep.prepare(source, output, scales="all")
            self.assertEqual(index["eligible_branch_count"], 2)
            self.assertTrue(index["train"]["8"]["ready"])
            large = torch.load(output / index["train"]["8"]["cache"], weights_only=False)
            small = torch.load(output / index["train"]["4"]["cache"], weights_only=False)
            frozen, protocol, catalogue, ids = prep.audit_source(source)
            graphs = list(prep.graphs_from_raw(source, ids["train_8"], catalogue, hashlib.sha256()))
            original = InMemoryDataset.collate(graphs)
            for key, attr, value, dim in prep.tensor_fields(large[0]):
                self.assertTrue(torch.equal(value, prep.store_at(original[0], key)[attr]), (key, attr))
                cuts = original[1] if key is None else original[1][key]
                self.assertTrue(torch.equal((large[1] if key is None else large[1][key])[attr], cuts[attr]))
            for i in range(4):
                a = separate(cls=large[0].__class__, batch=large[0], idx=i, slice_dict=large[1], decrement=False)
                b = separate(cls=small[0].__class__, batch=small[0], idx=i, slice_dict=small[1], decrement=False)
                for key, attr, value, dim in prep.tensor_fields(a):
                    self.assertTrue(torch.equal(value, prep.store_at(b, key)[attr]), (key, attr))
            self.assertEqual(large[0][("bus", "branch", "bus")].edge_attr.shape[1], 9)
            again = prep.prepare(source, output, scales="4")
            self.assertEqual(index["dataset_hash"], again["dataset_hash"])
            self.assertEqual(index["train"]["4"]["sha256"], again["train"]["4"]["sha256"])


if __name__ == "__main__":
    unittest.main()
