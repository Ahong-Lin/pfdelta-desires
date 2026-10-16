#!/usr/bin/env python3
"""One resumable CANOS-PF endpoint run; launched by the nine-slot queue.

The JSON config is immutable provenance. Dataset caches contain trusted local
PyG (data, slices) payloads. No data generation, splitting, or remote I/O occurs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import signal
import sys
import time


STOP_SIGNAL = None


def request_stop(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum


class StopRequested(Exception):
    pass


def check_stop():
    if STOP_SIGNAL is not None:
        raise StopRequested(f"Received signal {STOP_SIGNAL}")


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def config_hash(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(torch, path, payload):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path, text):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_dataset_class(torch):
    from torch_geometric.data import InMemoryDataset, HeteroData
    from torch_geometric.data.separate import separate

    class CachedDataset(InMemoryDataset):
        def __init__(self, cache_path):
            super().__init__(root=None, log=False)
            payload = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
            if not isinstance(payload, tuple) or len(payload) not in (2, 3):
                raise ValueError(f"Unsupported PyG cache: {cache_path}")
            data, self.slices = payload[:2]
            if isinstance(data, dict):
                data_class = payload[2] if len(payload) == 3 else HeteroData
                data = data_class.from_dict(data)
            self._data = data
            self._data_list = None
            self.case_name = "case118"

        def get(self, idx):
            # InMemoryDataset.get normally caches one Python graph per sample.
            # Nine concurrent jobs must retain only the mmap tensors, not that cache.
            if self.len() == 1:
                import copy
                return copy.copy(self._data)
            return separate(
                cls=self._data.__class__, batch=self._data, idx=idx,
                slice_dict=self.slices, decrement=False,
            )

    return CachedDataset


def verify_cache(dataset, split, config, np):
    """Check all inexpensive collated metadata, not merely first/last graphs."""
    expected = int(config[f"expected_{split}_count"])
    if len(dataset) != expected:
        raise ValueError(f"{split}: loaded {len(dataset)}, expected {expected}")
    metadata = dataset._data
    parents = metadata.load_setting_id.detach().cpu().numpy().reshape(-1)
    rhos = metadata.rho.detach().cpu().numpy().reshape(-1)
    if len(parents) != expected or len(rhos) != expected:
        raise ValueError(f"{split}: graph metadata lengths do not match graph count")
    if not np.isin(rhos, [0.0, 1.0]).all():
        raise ValueError(f"{split}: non-endpoint rho value in endpoint-only dataset")
    order = np.lexsort((rhos, parents))
    ordered_parents, ordered_rhos = parents[order], rhos[order]
    if expected % 2 or not np.array_equal(ordered_parents[::2], ordered_parents[1::2]):
        raise ValueError(f"{split}: endpoints are not paired by parent")
    if not ((ordered_rhos[::2] == 0).all() and (ordered_rhos[1::2] == 1).all()):
        raise ValueError(f"{split}: every parent must have exactly one rho=0 and one rho=1")
    if len(np.unique(ordered_parents)) != expected // 2:
        raise ValueError(f"{split}: duplicated parent pairs")
    parent_file = config.get(f"{split}_parent_ids")
    if parent_file:
        expected_parents = np.load(parent_file, allow_pickle=False).reshape(-1)
        if not np.array_equal(np.sort(expected_parents), ordered_parents[::2]):
            raise ValueError(f"{split}: cache parent IDs differ from frozen parent manifest")
    # The legacy loss averages each output component over nodes/branches. With
    # fixed counts within each component its batch mean is a graph-wise mean.
    sizes = {}
    for component, field in [("PV", "y"), ("PQ", "y"), ("slack", "y"),
                             (("bus", "branch", "bus"), "edge_label")]:
        boundaries = dataset.slices[component][field].detach().cpu().numpy()
        counts = np.diff(boundaries)
        if len(counts) != expected or not (counts == counts[0]).all() or counts[0] <= 0:
            raise ValueError(f"{split}: variable/empty component sizes need per-graph training loss")
        sizes[str(component)] = int(counts[0])
    return sizes


def legacy_config(config):
    """Keep model/checkpoint configuration readable by the existing tooling."""
    datasets = []
    for split in ("train", "val", "test"):
        cache = Path(config[f"{split}_cache"]).resolve()
        datasets.append({
            "name": "pfdeltaRhoCANOS", "model": "CANOS", "case_name": "case118",
            "add_bus_type": True, "root_dir": str(cache.parent),
            "processed_subdir": ".", "split": split,
            "task": "case118_endpoint_training_v2",
        })
    return {
        "config_type": "experiment", "dataset": {"datasets": datasets},
        "functional": {
            "seed": config["seed"], "cpu": config.get("device", "cuda:0") == "cpu",
            "checkpoint": True, "trainer_name": "gnn_trainer",
            "run_location": str(config["run_dir"]),
            "run_name": f"canos_case118_{config['scale']}_seed{config['seed']}",
            "is_debug": False,
        },
        "model": {
            "name": "canos_pf", "dataset": "_include_",
            "hidden_dim": config.get("hidden_dim", 128),
            "k_steps": config.get("k_steps", 20), "include_sent_messages": True,
        },
        "optim": {
            "optimizer": {"name": "Adam", "lr": config.get("lr", 0.0005), "weight_decay": 0.0},
            "lr_scheduler": {
                "name": "SequentialLR", "milestones": [5],
                "schedulers": [
                    {"name": "LinearLR", "start_factor": 0.01, "end_factor": 1.0, "total_iters": 5},
                    {"name": "StepLR", "step_size": 4, "gamma": 0.9},
                ],
            },
            "train_params": {
                "batch_size": config.get("batch_size", 64), "drop_last": False,
                "epochs": config.get("epochs", 300), "num_workers": 0, "pin_memory": False,
                "train_loss": [{"name": "canos_pf_mse"}],
            },
            "val_params": {
                "batch_size": config.get("val_batch_size", 50), "num_workers": 0,
                "pin_memory": False, "report_every": config.get("val_every", 2),
                "val_loss": [{"name": "canos_pf_mse"}],
            },
        },
        "provenance": {"experiment_id": config["experiment_id"], "data_hash": config["data_hash"],
                       "config_hash": config_hash(config)},
    }


def run(config_path, smoke_steps=0):
    import numpy as np
    import torch
    import yaml
    from torch_geometric.loader import DataLoader

    config = json.loads(Path(config_path).read_text())
    required = (
        "repo_root", "train_cache", "val_cache", "test_cache", "run_dir", "scale", "seed",
        "experiment_id", "data_hash", "expected_train_count", "expected_val_count", "expected_test_count",
    )
    for key in required:
        if key not in config:
            raise ValueError(f"Missing config key: {key}")
    if config.get("num_workers", 0) != 0:
        raise ValueError("This reproducible nine-slot setup requires num_workers=0")
    if smoke_steps < 0:
        raise ValueError("smoke_steps must be non-negative")
    cfg_hash = config_hash(config)
    run_dir = Path(config["run_dir"]).resolve()
    if smoke_steps:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = run_dir / "smoke_checks" / f"{stamp}_pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another process owns run lock: {run_dir}") from exc
    # Keep this file object open until run() returns.
    for name in ("Model", "Evaluation", "Logs"):
        (run_dir / name).mkdir(exist_ok=True)
    logger = logging.getLogger(f"endpoint.{os.getpid()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "Logs" / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    progress_path = run_dir / "progress.json"
    completion_path = run_dir / "Evaluation" / "completion.json"
    identity = {
        "experiment_id": config["experiment_id"], "config_hash": cfg_hash,
        "data_hash": config["data_hash"], "scale": config["scale"], "seed": config["seed"],
    }
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
    if completion and completion.get("finished"):
        for key in ("config_hash", "data_hash"):
            if completion.get(key) != identity[key]:
                raise RuntimeError(f"Completed run has different {key}; refusing to overwrite")
        logger.info("Already completed: %s", run_dir)
        return 0
    stored_config = run_dir / "Model" / "run_config.json"
    if stored_config.exists() and config_hash(json.loads(stored_config.read_text())) != cfg_hash:
        raise RuntimeError("Existing run_config.json differs; use a new run directory")
    atomic_json(stored_config, config)
    atomic_text(run_dir / "Model" / "config.yaml", yaml.safe_dump(legacy_config(config), sort_keys=False))
    atomic_json(progress_path, {**identity, "status": "loading", "pid": os.getpid(), "updated_at": timestamp()})

    repo = Path(config["repo_root"]).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from core.models.canos_pf import CANOS_PF
    from core.utils.pf_losses_utils import CANOS_PF_MSE
    import torch_geometric

    source_files = [
        repo / "core/models/canos_pf.py", repo / "core/models/canos_utils.py",
        repo / "core/utils/pf_losses_utils.py", Path(__file__).resolve(),
        Path(__file__).resolve().with_name("evaluate_endpoint.py"),
    ]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files if path.is_file()}
    atomic_json(run_dir / "Model" / "provenance.json", {
        **identity, "python_version": sys.version, "torch_version": str(torch.__version__),
        "torch_geometric_version": str(torch_geometric.__version__),
        "numpy_version": str(np.__version__), "source_sha256": source_hashes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_runtime_version": torch.version.cuda, "recorded_at": timestamp(),
    })

    threads = int(config.get("torch_threads", 4))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    device = torch.device(config.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU training")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Scatter reductions on CUDA can still have small nondeterministic differences.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    CachedDataset = make_dataset_class(torch)
    datasets = {}
    component_sizes = {}
    for split in ("train", "val"):
        datasets[split] = CachedDataset(config[f"{split}_cache"])
        component_sizes[split] = verify_cache(datasets[split], split, config, np)
    train_data, val_data = datasets["train"], datasets["val"]
    # Every graph in this paired endpoint dataset must have fixed component sizes.
    # Under that construction, graph-weighted batch loss is the sample mean MSE.
    reference = {name: train_data[0][name].y.shape for name in ("PV", "PQ", "slack")}
    for data in (train_data[-1], val_data[0], val_data[-1]):
        if any(data[name].y.shape != shape for name, shape in reference.items()):
            raise ValueError("Variable bus-type sizes require per-graph loss aggregation")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=int(config.get("batch_size", 64)), shuffle=True,
        drop_last=False, num_workers=0, pin_memory=False, generator=generator,
    )
    if smoke_steps > len(train_loader):
        raise ValueError("smoke_steps exceeds the number of training batches")
    # Give eval loaders independent RNG so validation frequency does not change
    # subsequent training permutations or the model's torch RNG state.
    val_generator = torch.Generator().manual_seed(seed + 1_000_003)
    val_loader = DataLoader(
        val_data, batch_size=int(config.get("val_batch_size", 50)), shuffle=False,
        drop_last=False, num_workers=0, pin_memory=False, generator=val_generator,
    )
    model = CANOS_PF(
        train_data, hidden_dim=int(config.get("hidden_dim", 128)),
        include_sent_messages=True, k_steps=int(config.get("k_steps", 20)),
    ).to(device)
    init_hash = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 0.0005)), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5),
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9),
        ],
        milestones=[5],
    )
    criterion = CANOS_PF_MSE()
    epochs = int(config.get("epochs", 300))
    val_every = int(config.get("val_every", 2))
    if epochs < 1 or val_every < 1:
        raise ValueError("epochs and val_every must be positive")
    next_epoch, train_steps, training_seconds = 0, 0, 0.0
    best_val, best_epoch, best_state = None, None, None
    history = []
    last_path, best_path = run_dir / "Model" / "last.pt", run_dir / "Model" / "best.pt"
    compatible_model = run_dir / "Model" / "model.pt"
    if not os.path.lexists(compatible_model):
        compatible_model.symlink_to("best.pt")
    elif not compatible_model.is_symlink() or os.readlink(compatible_model) != "best.pt":
        raise RuntimeError("Model/model.pt exists and is not the expected best.pt symlink")

    def rng_state():
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "train_loader_generator": generator.get_state(),
            "val_loader_generator": val_generator.get_state(),
        }

    def restore_rng(states):
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch_cpu"].cpu())
        if device.type == "cuda" and states["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in states["torch_cuda"]])
        generator.set_state(states["train_loader_generator"].cpu())
        val_generator.set_state(states["val_loader_generator"].cpu())

    def checkpoint_payload():
        return {
            **identity, "format_version": 1, "next_epoch": next_epoch,
            "model_state_dict": cpu_state(model), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "rng_state": rng_state(),
            "history": history, "best_val_mse": best_val, "best_epoch": best_epoch,
            "best_model_state_dict": best_state, "initial_weight_hash": init_hash,
            "train_steps": train_steps, "training_seconds": training_seconds,
            "saved_at": timestamp(), "smoke_only": bool(smoke_steps),
        }

    if last_path.exists() and not smoke_steps:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        for key in ("config_hash", "data_hash"):
            if checkpoint.get(key) != identity[key]:
                raise RuntimeError(f"Checkpoint {key} differs; refusing to resume")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        next_epoch = int(checkpoint["next_epoch"])
        history = checkpoint["history"]
        best_val, best_epoch = checkpoint["best_val_mse"], checkpoint["best_epoch"]
        best_state = checkpoint["best_model_state_dict"]
        init_hash = checkpoint["initial_weight_hash"]
        train_steps = int(checkpoint["train_steps"])
        training_seconds = float(checkpoint["training_seconds"])
        restore_rng(checkpoint["rng_state"])
        if best_state is not None:
            atomic_torch_save(torch, best_path, best_state)
        logger.info("Resuming epoch %d/%d at step %d", next_epoch + 1, epochs, train_steps)
        del checkpoint
    else:
        # Epoch-zero checkpoint permits exact recovery if the first epoch stops.
        atomic_torch_save(torch, last_path, checkpoint_payload())
    logger.info("Run %s seed=%d train=%d val=%d initial_hash=%s device=%s",
                config["scale"], seed, len(train_data), len(val_data), init_hash, device)

    def progress(status, epoch, **extra):
        atomic_json(progress_path, {
            **identity, "status": status, "epoch": epoch, "epochs_total": epochs,
            "train_step": train_steps, "best_val_mse": best_val, "best_epoch": best_epoch,
            "training_seconds": training_seconds, "pid": os.getpid(),
            "updated_at": timestamp(), "initial_weight_hash": init_hash, **extra,
        })

    def validation(max_batches=None):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                check_stop()
                batch = batch.to(device)
                loss = criterion(model(batch), batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite validation loss in batch {index}")
                graphs = int(batch.num_graphs)
                total += float(loss) * graphs
                count += graphs
                if max_batches is not None and index + 1 >= max_batches:
                    break
        if not count:
            raise RuntimeError("Empty validation loader")
        return total / count, count

    try:
        for epoch_index in range(next_epoch, epochs):
            check_stop()
            start = time.perf_counter()
            progress("training", epoch_index + 1)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            total, count = 0.0, 0
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            for batch_index, batch in enumerate(train_loader):
                check_stop()
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch), batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite train loss, epoch={epoch_index+1} batch={batch_index}")
                loss.backward()
                # A finite loss does not guarantee finite gradients. clip norm
                # infinity checks them without changing the training algorithm.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
                optimizer.step()
                graphs = int(batch.num_graphs)
                total += float(loss.detach()) * graphs
                count += graphs
                train_steps += 1
                if smoke_steps and batch_index + 1 >= smoke_steps:
                    break
            check_stop()
            if not smoke_steps and count != len(train_data):
                raise RuntimeError(f"Epoch consumed {count} graphs, expected {len(train_data)}")
            val_mse, val_count = None, None
            if smoke_steps or (epoch_index + 1) % val_every == 0 or epoch_index + 1 == epochs:
                progress("validating", epoch_index + 1)
                val_mse, val_count = validation(max_batches=1 if smoke_steps else None)
                if not smoke_steps and val_count != len(val_data):
                    raise RuntimeError("Validation count mismatch")
                if best_val is None or val_mse < best_val:
                    best_val, best_epoch, best_state = val_mse, epoch_index + 1, cpu_state(model)
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            training_seconds += elapsed
            next_epoch = epoch_index + 1
            row = {
                "epoch": next_epoch, "train_mse": total / count, "val_mse": val_mse,
                "train_graph_count": count, "val_graph_count": val_count, "learning_rate": epoch_lr,
                "train_steps": train_steps, "epoch_seconds": elapsed,
                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
                "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
            }
            history.append(row)
            # last.pt is the authority: it stores best weights too, so a crash
            # between these atomic saves cannot leave an inconsistent best model.
            atomic_torch_save(torch, last_path, checkpoint_payload())
            if best_state is not None:
                atomic_torch_save(torch, best_path, best_state)
            atomic_json(run_dir / "Evaluation" / "val_history.json", history)
            progress("epoch_complete", next_epoch, last_epoch_seconds=elapsed)
            logger.info("epoch=%d/%d train_mse=%.8g val_mse=%s lr=%.8g seconds=%.1f steps=%d peak_MiB=%.1f",
                        next_epoch, epochs, row["train_mse"], f"{val_mse:.8g}" if val_mse is not None else "-",
                        epoch_lr, elapsed, train_steps, row["gpu_peak_allocated_mib"])
            if smoke_steps:
                saved = torch.load(last_path, map_location="cpu", weights_only=False)
                model.load_state_dict(saved["model_state_dict"], strict=True)
                if state_hash(model.state_dict()) != state_hash(saved["model_state_dict"]):
                    raise RuntimeError("Checkpoint reload hash mismatch")
                report = {**identity, **row, "smoke_only": True, "finished": False,
                          "requested_smoke_steps": smoke_steps, "checkpoint_reload_verified": True,
                          "run_dir": str(run_dir), "initial_weight_hash": init_hash}
                atomic_json(run_dir / "smoke_report.json", report)
                progress("smoke_complete", next_epoch, last_epoch_seconds=elapsed)
                logger.info("Smoke complete (not a completed experiment): %s", run_dir)
                return 0

        check_stop()
        if best_state is None:
            raise RuntimeError("Training ended without a best validation checkpoint")
        model.load_state_dict(best_state, strict=True)
        progress("testing", epochs)
        # Imports the adjacent evaluator, which has no fixed test-size assumption.
        from evaluate_endpoint import evaluate
        test_data = CachedDataset(config["test_cache"])
        verify_cache(test_data, "test", config, np)
        parent_to_line = {}
        if config.get("parent_manifest"):
            with Path(config["parent_manifest"]).open() as handle:
                for line in handle:
                    record = json.loads(line)
                    parent_to_line[int(record.get("load_setting_id", record["parent_id"]))] = int(record["branch_id"])
        test_start = time.perf_counter()
        metrics = evaluate(
            model, test_data, device=device, batch_size=int(config.get("val_batch_size", 50)),
            outdir=run_dir / "Evaluation", parent_to_line=parent_to_line, num_workers=0,
        )
        check_stop()
        if metrics.get("num_samples") != len(test_data):
            raise RuntimeError("Evaluator did not cover the complete fixed test set")
        atomic_json(run_dir / "Evaluation" / "test_metrics.json", metrics)
        complete = {
            **identity, "finished": True, "smoke_only": False, "best_val_mse": best_val,
            "best_epoch": best_epoch, "epochs_completed": next_epoch, "train_steps": train_steps,
            "training_seconds": training_seconds, "test_seconds": time.perf_counter() - test_start,
            "initial_weight_hash": init_hash, "best_weight_hash": state_hash(best_state),
            "train_count": len(train_data), "val_count": len(val_data), "test_count": len(test_data),
            "completed_at": timestamp(), "test_metrics": metrics,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": str(torch.__version__),
        }
        atomic_json(completion_path, complete)
        progress("completed", epochs, last_epoch_seconds=history[-1]["epoch_seconds"])
        logger.info("COMPLETED best_epoch=%d best_val_mse=%.8g run=%s", best_epoch, best_val, run_dir)
        return 0
    except StopRequested:
        # Partial-epoch optimizer updates are deliberately not checkpointed.
        # Restart restores the last committed epoch and its exact RNG state.
        progress("interrupted", min(next_epoch + 1, epochs), signal=STOP_SIGNAL,
                 resume_from_epoch=next_epoch + 1)
        logger.warning("Stopped by signal %s; resume uses committed checkpoint at completed epoch %d",
                       STOP_SIGNAL, next_epoch)
        return 128 + int(STOP_SIGNAL or signal.SIGTERM)
    except BaseException as exc:
        progress("failed", min(next_epoch + 1, epochs), error=repr(exc))
        logger.exception("Run failed; no completion marker written")
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Immutable JSON run configuration")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="Only N optimizer steps and one validation batch, in a separate smoke directory")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    return run(args.config, args.smoke_steps)


if __name__ == "__main__":
    sys.exit(main())
