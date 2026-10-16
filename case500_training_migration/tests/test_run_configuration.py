"""Portable configuration and scheduling checks; no torch, GPU or training."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import fcntl
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import configure_runs as configure
import manage_runs as manager


class RunConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / "data with spaces"
        self.data.mkdir()
        self.results = self.base / "Results/0-1labels"
        self.experiment = configure.read_json(ROOT / "configs/experiment.json")
        index = {"case_name": "case500", "dataset_hash": "frozen-data-identity",
                 "parent_manifest": "parents.jsonl", "train": {}}
        (self.data / "parents.jsonl").write_text('{"parent_id":1,"branch_id":2}\n')
        for split, graphs in (("val", 100), ("test", 100)):
            index[split] = self.cache(split, graphs)
        for scale in self.experiment["scales"]:
            index["train"][scale] = self.cache(scale, int(scale[:-1]) * 1000)
        (self.data / "index.json").write_text(json.dumps(index))

    def cache(self, stem, graphs):
        (self.data / f"{stem}.pt").write_bytes(b"fixture-cache-not-torch")
        (self.data / f"{stem}.npy").write_bytes(b"fixture-parents-not-numpy")
        return {"cache": f"{stem}.pt", "graphs": graphs,
                "parent_ids": f"{stem}.npy", "parent_count": graphs // 2, "ready": True,
                "sha256": configure.file_sha256(self.data / f"{stem}.pt"),
                "parent_ids_sha256": configure.file_sha256(self.data / f"{stem}.npy")}

    def test_full_protocol_and_portable_paths(self):
        configs = configure.build_configs(self.data, self.results, self.experiment)
        self.assertEqual(len(configs), 70)
        self.assertEqual(configs[0]["scale"], "10k")
        self.assertEqual(configs[-1]["scale"], "320k")
        self.assertEqual(configs[-1]["seed"], 55)
        first = configs[0]
        for key, value in {"case_name": "case500", "epochs": 300, "batch_size": 64,
                           "val_batch_size": 50, "val_every": 2, "lr": 0.0005,
                           "hidden_dim": 128, "k_steps": 20, "num_workers": 0,
                           "torch_threads": 4, "device": "cuda:0"}.items():
            self.assertEqual(first[key], value)
        self.assertEqual(first["train_cache"], str((self.data / "10k.pt").resolve()))
        self.assertEqual(first["expected_train_count"], 10000)
        self.assertEqual(first["data_hash"], "frozen-data-identity")

    def test_immutable_config_prevents_accidental_retarget(self):
        path = configure.freeze_runs(self.data, self.results, self.experiment, ["10k"], [11])
        self.assertEqual(path, configure.freeze_runs(self.data, self.results, self.experiment, ["10k"], [11]))
        changed = dict(self.experiment, batch_size=128)
        with self.assertRaisesRegex(ValueError, "differs"):
            configure.freeze_runs(self.data, self.results, changed, ["10k"], [11])

    def test_reject_case_or_scale_mismatch(self):
        with self.assertRaisesRegex(ValueError, "case_name"):
            configure.build_configs(self.data, self.results, dict(self.experiment, case_name="case118"))
        with self.assertRaisesRegex(ValueError, "not listed"):
            configure.build_configs(self.data, self.results, self.experiment, ["640k"])

    def test_dry_run_assigns_three_per_gpu_without_creating_run_folders(self):
        path = configure.freeze_runs(self.data, self.results, self.experiment, ["10k", "20k"])
        output = io.StringIO()
        with redirect_stdout(output):
            result = manager.run_queue(path, ["2", "5"], dry_run=True)
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue().count("GPU 2:"), 3)
        self.assertEqual(output.getvalue().count("GPU 5:"), 3)
        self.assertFalse((self.results / "10k").exists())
        self.assertFalse((path.parent / "queue_state.json").exists())

    def test_completed_runs_require_matching_provenance(self):
        path = configure.freeze_runs(self.data, self.results, self.experiment, ["10k"], [11])
        entry = configure.read_json(path)["runs"][0]
        output = Path(entry["run_dir"]) / "Evaluation/completion.json"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"finished": True, "smoke_only": False,
                                      "config_hash": entry["config_hash"], "data_hash": "frozen-data-identity"}))
        self.assertEqual(manager.classify(entry), "completed")
        output.write_text('{"finished":true,"config_hash":"wrong","data_hash":"wrong"}')
        with self.assertRaisesRegex(ValueError, "provenance"):
            manager.classify(entry)

    def test_failures_are_not_automatically_retried(self):
        path = configure.freeze_runs(self.data, self.results, self.experiment, ["10k"], [11])
        entry = configure.read_json(path)["runs"][0]
        output = Path(entry["run_dir"]) / "progress.json"
        output.parent.mkdir(parents=True)
        output.write_text('{"status":"failed"}')
        self.assertEqual(len(manager.inspect_runs(path, ["0"], 3)[2]), 0)
        self.assertEqual(len(manager.inspect_runs(path, ["0"], 3, retry_failed=True)[2]), 1)

    def test_unmaterialized_cache_cannot_be_configured(self):
        index_path = self.data / "index.json"
        index = configure.read_json(index_path)
        index["train"]["10k"]["ready"] = False
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "not ready"):
            configure.build_configs(self.data, self.results, self.experiment, ["10k"], [11])
        index["train"]["10k"]["ready"] = True
        index["train"]["10k"]["sha256"] = "not-a-digest"
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            configure.build_configs(self.data, self.results, self.experiment, ["10k"], [11])

    def test_unique_data_verified_once_and_mutation_rejected(self):
        path = configure.freeze_runs(self.data, self.results, self.experiment, ["10k"], [11, 22])
        entries = configure.read_json(path)["runs"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(manager.verify_bound_files(entries), 8)
        (self.data / "10k.pt").write_bytes(b"changed-labels-same-frozen-parent-ids")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "content changed"):
            manager.verify_bound_files(entries)

    def test_inherited_gpu_allocation_cannot_be_expanded(self):
        manager.validate_gpu_allocation(["2", "3"], {"CUDA_VISIBLE_DEVICES": "2,3"})
        manager.validate_gpu_allocation(["3"], {"CUDA_VISIBLE_DEVICES": "2,3"})
        with self.assertRaisesRegex(ValueError, "conflict"):
            manager.validate_gpu_allocation(["0", "1"], {"CUDA_VISIBLE_DEVICES": "2,3"})
        with self.assertRaisesRegex(ValueError, "conflict"):
            manager.validate_gpu_allocation(["0"], {"CUDA_VISIBLE_DEVICES": ""})

    def test_tmux_forwards_callers_cuda_allocation_with_space_paths(self):
        mocks = self.base / "mock executables"
        mocks.mkdir()
        capture = self.base / "captures"
        capture.mkdir()
        python_stub = mocks / "python with spaces"
        python_stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\nfrom pathlib import Path\n"
            "if sys.argv[1:2] == ['-c']:\n    print(str(Path(__file__).resolve()))\n"
            "else:\n    Path(os.environ['CAPTURE_DIR'], 'python.json').write_text(json.dumps("
            "{'args':sys.argv[1:], 'visible':os.environ.get('CUDA_VISIBLE_DEVICES'), "
            "'order':os.environ.get('CUDA_DEVICE_ORDER')}))\n")
        tmux_stub = mocks / "tmux"
        tmux_stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\nfrom pathlib import Path\n"
            "if sys.argv[1:2] == ['has-session']:\n    raise SystemExit(1)\n"
            "Path(os.environ['CAPTURE_DIR'], 'tmux.json').write_text(json.dumps(sys.argv[1:]))\n")
        python_stub.chmod(0o755)
        tmux_stub.chmod(0o755)
        arguments = ["--manifest", str(self.base / "results with spaces/runs.json"), "--gpus", "2,3"]
        for allocation, order in (("2,3", "PCI_BUS_ID"), (None, None)):
            with self.subTest(allocation=allocation):
                caller = dict(os.environ, PATH=str(mocks) + os.pathsep + os.environ["PATH"],
                              PYTHON_BIN=str(python_stub), CAPTURE_DIR=str(capture))
                for name, value in (("CUDA_VISIBLE_DEVICES", allocation), ("CUDA_DEVICE_ORDER", order)):
                    caller.pop(name, None)
                    if value is not None:
                        caller[name] = value
                subprocess.run(["bash", str(ROOT / "start_tmux.sh"), *arguments],
                               env=caller, check=True, capture_output=True, text=True)
                tmux_args = json.loads((capture / "tmux.json").read_text())
                # Execute only the captured launch command using harmless stubs,
                # under deliberately stale daemon variables. No actual tmux/GPU.
                launch = tmux_args[-1].split("; status=", 1)[0]
                stale = dict(caller, CUDA_VISIBLE_DEVICES="99", CUDA_DEVICE_ORDER="FASTEST_FIRST")
                subprocess.run(["bash", "-c", launch], env=stale, check=True, capture_output=True)
                recorded = json.loads((capture / "python.json").read_text())
                self.assertEqual(recorded["args"], [str(ROOT / "scripts/manage_runs.py"), *arguments])
                self.assertEqual(recorded["visible"], allocation)
                self.assertEqual(recorded["order"], order)

    def test_incomplete_zero_exit_is_not_retried(self):
        path = configure.freeze_runs(self.data, self.results, self.experiment, ["10k"], [11])
        entry = configure.read_json(path)["runs"][0]
        self.assertEqual(manager.classify(entry, {"status": "failed", "exit_code": 0}), "failed")
        self.assertEqual(manager.classify(entry, {"status": "interrupted", "exit_code": -9}), "pending")

    def test_queue_lock_prevents_duplicate_launcher(self):
        path = configure.freeze_runs(self.data, self.results, self.experiment, ["10k"], [11])
        with (path.parent / ".queue.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Another queue"):
                manager.run_queue(path, ["0"])

    def test_reject_duplicate_gpu_or_slots(self):
        with self.assertRaises(ValueError):
            manager.parse_gpus("0,0")
        self.assertEqual(manager.parse_gpus("1,3"), ["1", "3"])
        path = configure.freeze_runs(self.data, self.results, self.experiment, ["10k"], [11])
        with self.assertRaises(ValueError):
            manager.inspect_runs(path, ["0"], 0)

    def test_legacy_metadata_uses_case500(self):
        module_path = ROOT / "scripts/endpoint_training_v2/train_endpoint_run.py"
        spec = importlib.util.spec_from_file_location("portable_train", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = configure.build_configs(self.data, self.results, self.experiment, ["10k"], [11])[0]
        legacy = module.legacy_config(config)
        self.assertEqual(legacy["dataset"]["datasets"][0]["case_name"], "case500")
        self.assertEqual(legacy["functional"]["run_name"], "canos_case500_10k_seed11")
        self.assertEqual(legacy["optim"]["train_params"]["epochs"], 300)


if __name__ == "__main__":
    unittest.main()
