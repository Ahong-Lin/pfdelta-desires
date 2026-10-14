"""Small pipeline tests: disjoint input groups, exact quotas, nesting, and frozen reruns."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/split_dataset.py"
SPEC = importlib.util.spec_from_file_location("split_dataset", SCRIPT)
splitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splitter)


class SplitPipelineTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        (root / "manifests").mkdir()
        (root / "raw").mkdir()
        config = {
            "train_endpoint_records": 32, "train_scales": [8, 16, 32],
            "val_parents_per_branch": 1, "test_parents_per_branch": 1,
            "split_seed": 123, "clusters": 3, "feature_dimension": 12,
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        plan = {"mode": "full", "eligible_branches": [2, 7, 11, 20], "train_parents": 16,
                "total_parents": 24, "train_endpoint_scales": [8, 16, 32],
                "full_train_quota": {str(b): 4 for b in [2, 7, 11, 20]},
                "branch_parent_targets": {str(b): 6 for b in [2, 7, 11, 20]}}
        (root / "plan.json").write_text(json.dumps(plan))
        with (root / "manifests/parents.jsonl").open("w") as handle:
            for index in range(24):
                parent = index + 1
                branch = [2, 7, 11, 20][index // 6]
                fingerprint = hashlib.sha256(f"inputs-{parent}".encode()).hexdigest()
                paths = []
                for rho in (1, 0):
                    relative = f"raw/{parent}_{rho}.json"
                    (root / relative).write_text(json.dumps({"parent_id": parent, "branch_id": branch,
                        "rho": rho, "input_fingerprint": fingerprint,
                        "termination_status": "PF_CONVERGED", "solver_method": "newton",
                        "solver_type": "AC_PF", "solver_fallback": "none",
                        "validation": {"max_balance_residual": 2e-9, "fixed_bus_pg_error": 0.0,
                                       "fixed_vm_error": 0.0, "fixed_slack_va_error": 0.0,
                                       "min_vm": 0.9, "max_vm": 1.1},
                        "network": {"branch": {str(branch): {"br_status": rho}}}}))
                    paths.append(relative)
                row = {"parent_id": parent, "branch_id": branch, "source_batch": index % 3,
                       "files": paths, "input_fingerprint": fingerprint,
                       "input_features": [float(parent * (j + 1) % 31) for j in range(12)]}
                handle.write(json.dumps(row) + "\n")
        return config_path

    def test_full_freeze_nesting_pairing_and_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_fixture(root)
            first = splitter.run(root, config, verify_raw=True)
            second = splitter.run(root, config)
            self.assertEqual(first["dataset_hash"], second["dataset_hash"])
            groups = {name: np.load(root / "splits" / f"{name}_parent_ids.npy")
                      for name in ("train", "val", "test")}
            self.assertEqual(len(set(np.concatenate(list(groups.values())))), 24)
            for count in (8, 16, 32):
                subset = np.load(root / "splits" / f"train_{count}_parent_ids.npy")
                np.testing.assert_array_equal(subset, groups["train"][:count // 2])
                manifest = json.loads((root / "splits" / f"train_{count}.json").read_text())
                self.assertEqual(len(manifest["records"]), count)
                records = manifest["records"]
                for j in range(0, count, 2):
                    self.assertEqual(records[j]["load_setting_id"], records[j + 1]["load_setting_id"])
                    self.assertEqual([records[j]["rho"], records[j + 1]["rho"]], [1.0, 0.0])
                    self.assertFalse(Path(records[j]["file"]).is_absolute())
                _, counts = np.unique([r["branch_id"] for r in records], return_counts=True)
                self.assertLessEqual(int(counts.max() - counts.min()), 2)
            audit = json.loads((root / "splits/audit.json").read_text())
            self.assertEqual(audit["input_fingerprint_overlap_train_val_test"], 0)
            (root / "splits/val.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
                splitter.run(root, config)

    def test_duplicate_input_rejected_without_public_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_fixture(root)
            path = root / "manifests/parents.jsonl"
            rows = [json.loads(row) for row in path.read_text().splitlines()]
            rows[1]["input_features"] = rows[0]["input_features"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Duplicate known-input vector"):
                splitter.run(root, config)
            self.assertFalse((root / "splits").exists())

    def test_smoke_plan_overrides_large_config_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.make_fixture(root)
            config = json.loads(config_path.read_text())
            config.update(train_endpoint_records=320000, train_scales=[10000, 320000],
                          val_parents_per_branch=50, test_parents_per_branch=50)
            config_path.write_text(json.dumps(config))
            plan_path = root / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan.update(mode="smoke", val_parents_per_branch=1, test_parents_per_branch=1)
            plan_path.write_text(json.dumps(plan))
            result = splitter.run(root, config_path, dry_run=True)
            self.assertEqual(result["protocol"]["total_parents"], 24)
            self.assertFalse((root / "splits").exists())

    def test_nested_branch_quotas_including_remainders(self):
        previous = np.zeros(7, dtype=np.int64)
        for total in range(100):
            quota = splitter.equal_branch_quota(total, 7)
            self.assertEqual(int(quota.sum()), total)
            self.assertLessEqual(int(quota.max() - quota.min()), 1)
            self.assertTrue(np.all(quota >= previous))
            previous = quota

    def test_generator_raw_schema_and_pair_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "endpoint.json"
            sample = {"network": {"branch": {"7": {"br_status": 0}}}, "rho_meta": {"rho": 0.0, "parent_id": 12,
                "load_setting_id": 12, "branch_id": "7", "input_fingerprint": "fixture",
                "solver_method": "newton", "solver_fallback": "none"},
                "solution": {"termination_status": "PF_CONVERGED", "solver_method": "newton",
                             "solver_type": "AC_PF", "solver_fallback": "none",
                             "validation": {"max_balance_residual": 2e-9, "fixed_bus_pg_error": 0.0,
                                            "fixed_vm_error": 0.0, "fixed_slack_va_error": 0.0,
                                            "min_vm": 0.9, "max_vm": 1.1}}}
            path.write_text(json.dumps(sample))
            splitter.validate_raw_sample(path, 12, 7, 0.0, "fixture")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                splitter.validate_raw_sample(path, 12, 7, 1.0, "fixture")
            sample["solution"]["validation"]["max_balance_residual"] = 2e-7
            path.write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, "max_balance_residual"):
                splitter.validate_raw_sample(path, 12, 7, 0.0, "fixture")
            del sample["solution"]["validation"]
            path.write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, "Missing stored numerical"):
                splitter.validate_raw_sample(path, 12, 7, 0.0, "fixture")


if __name__ == "__main__":
    unittest.main()
