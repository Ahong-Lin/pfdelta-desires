import argparse
import copy
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import find_run, load_trainer


RHO_DATASET_BY_BASE = {
    "pfdeltaCANOS": "pfdeltaRhoCANOS",
    "pfdeltaRhoCANOS": "pfdeltaRhoCANOS",
    "pfdeltaPFNet": "pfdeltaRhoPFNet",
    "pfdeltaRhoPFNet": "pfdeltaRhoPFNet",
    "pfdeltaGNS": "pfdeltaRhoGNS",
    "pfdeltaRhoGNS": "pfdeltaRhoGNS",
}

PBL_MODEL_BY_MODEL = {
    "canos_pf": "CANOS",
    "flowcan": "CANOS",
    "flowcan_std_attn": "CANOS",
    "flowcan_no_attn": "CANOS",
    "powerflownet": "PFNet",
    "graph_neural_solver": "GNS",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained PFDelta models on a PFDelta-rho test split. "
            "Use this to compare rho-trained runs against original no-outage runs "
            "on the same sudden branch-disconnection dataset."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Run directory or run name. Can be passed more than once. "
            "Examples: runs/.../run_folder or canos_case30_rho_task1_1_260706_142034."
        ),
    )
    target.add_argument(
        "--root",
        help=(
            "Run group under runs/, e.g. canos_case30_rho_task_1_1. "
            "Every direct child containing config.yaml/model.pt is evaluated."
        ),
    )
    parser.add_argument(
        "--rho-root",
        default="data_generation_rho/case30/rho_task_1_1_single_edge",
        help="Root directory of the generated PFDelta-rho dataset.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Rho split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Evaluation device. CPU is safest when training is running.",
    )
    parser.add_argument(
        "--output",
        default="test_errors_rho.json",
        help="JSON file where results are accumulated.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional CSV path for a flat per-run summary.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["pbl", "config"],
        default="pbl",
        help=(
            "pbl evaluates only universal power balance; config reuses the run's "
            "configured val losses on the rho test split."
        ),
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Reprocess the rho dataset before evaluating.",
    )
    parser.add_argument(
        "--no-by-rho",
        action="store_true",
        help="Skip per-rho grouped metrics.",
    )
    return parser.parse_args()


def resolve_run(path_or_name):
    candidate = Path(path_or_name)
    if candidate.exists():
        return candidate
    found = find_run(path_or_name)
    if found == -1:
        raise FileNotFoundError(f"Could not find run: {path_or_name}")
    return Path(found)


def discover_runs(args):
    if args.root:
        root = Path(args.root)
        if not root.exists():
            root = Path("runs") / args.root
        if not root.exists():
            raise FileNotFoundError(f"Could not find root: {args.root}")
        runs = [
            child
            for child in sorted(root.iterdir())
            if child.is_dir()
            and (child / "config.yaml").exists()
            and (child / "model.pt").exists()
        ]
        if not runs:
            raise FileNotFoundError(f"No finished-looking runs found under {root}")
        return runs
    return [resolve_run(run) for run in args.run]


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_config(run_dir):
    import yaml

    with open(run_dir / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def infer_dataset_name(base_name, model_name):
    if base_name in RHO_DATASET_BY_BASE:
        return RHO_DATASET_BY_BASE[base_name]
    if model_name in {"canos_pf", "flowcan", "flowcan_std_attn", "flowcan_no_attn"}:
        return "pfdeltaRhoCANOS"
    if model_name == "powerflownet":
        return "pfdeltaRhoPFNet"
    if model_name == "graph_neural_solver":
        return "pfdeltaRhoGNS"
    raise ValueError(f"Cannot infer rho dataset for model={model_name}, dataset={base_name}")


def make_pbl_val_loss(config):
    model_name = config["model"]["name"]
    pbl_model = PBL_MODEL_BY_MODEL.get(model_name)
    if pbl_model is None:
        raise ValueError(f"No PBL model mapping for model={model_name}")
    return [{"name": "universal_power_balance", "model": pbl_model}]


def build_eval_config(run_dir, args):
    config = read_config(run_dir)
    config = copy.deepcopy(config)

    config["functional"]["is_debug"] = True
    config["functional"]["checkpoint"] = False
    config["functional"]["cpu"] = args.device == "cpu"
    config["functional"]["run_location"] = str(run_dir)

    base_dataset = copy.deepcopy(config["dataset"]["datasets"][0])
    model_name = config["model"]["name"]
    base_dataset["name"] = infer_dataset_name(base_dataset.get("name"), model_name)
    base_dataset["root_dir"] = args.rho_root
    base_dataset["split"] = args.split
    base_dataset["task"] = "rho_1.1"
    base_dataset["force_reload"] = bool(args.force_reload)
    if base_dataset["name"] == "pfdeltaRhoCANOS":
        base_dataset["model"] = "CANOS"
        base_dataset["add_bus_type"] = True
        base_dataset.pop("transform", None)
        base_dataset.pop("pre_transform", None)
    elif base_dataset["name"] == "pfdeltaRhoPFNet":
        base_dataset["model"] = "PFNet"
        base_dataset["add_bus_type"] = False
    elif base_dataset["name"] == "pfdeltaRhoGNS":
        base_dataset["model"] = "GNS"
        base_dataset["add_bus_type"] = False
        base_dataset.pop("transform", None)
        base_dataset.pop("pre_transform", None)

    config["dataset"] = {"datasets": [base_dataset]}

    config["optim"]["train_params"]["batch_size"] = args.batch_size
    config["optim"]["val_params"]["batch_size"] = args.batch_size
    if args.loss_mode == "pbl":
        config["optim"]["val_params"]["val_loss"] = make_pbl_val_loss(config)
    config["optim"]["val_params"].pop("early_stop", None)

    if args.device == "cuda":
        config["functional"]["cpu"] = False

    return config


def get_relative_pbl(loss_fn):
    relative_pbl = getattr(loss_fn, "relative_power_balance_mean", None)
    load_scale = getattr(loss_fn, "load_scale_mean", None)
    relative_pbl_max_load = getattr(loss_fn, "relative_power_balance_max_load_mean", None)
    max_load_scale = getattr(loss_fn, "max_load_scale_mean", None)
    return {
        "Relative PBL Mean": relative_pbl.item() if relative_pbl is not None else None,
        "PBL Load Scale": load_scale.item() if load_scale is not None else None,
        "Relative PBL Mean / Max Load": (
            relative_pbl_max_load.item()
            if relative_pbl_max_load is not None
            else None
        ),
        "PBL Max Load Scale": (
            max_load_scale.item() if max_load_scale is not None else None
        ),
    }


def summarize_dataset(dataset):
    rhos = []
    load_ids = []
    for data in dataset:
        if hasattr(data, "rho"):
            rhos.append(float(data.rho.view(-1)[0].item()))
        if hasattr(data, "load_setting_id"):
            load_ids.append(int(data.load_setting_id.view(-1)[0].item()))
    rho_counts = {}
    for rho in rhos:
        key = f"{rho:.6g}"
        rho_counts[key] = rho_counts.get(key, 0) + 1
    return {
        "num_samples": len(dataset),
        "rho_counts": rho_counts,
        "num_load_settings": len(set(load_ids)),
    }


def batch_num_graphs(data):
    num_graphs = getattr(data, "num_graphs", None)
    if num_graphs is not None:
        return int(num_graphs)
    if hasattr(data["bus"], "batch"):
        return int(data["bus"].batch.max().item()) + 1
    return 1


def maybe_item(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)


def run_eval_dataloader(trainer, dataloader):
    trainer.model.eval()
    weighted_losses = [0.0] * len(trainer.val_loss)
    total_graphs = 0
    inference_speed = 0.0
    pbl_extras = {
        "PBL L2": 0.0,
        "Relative PBL Mean": 0.0,
        "PBL Load Scale": 0.0,
        "Relative PBL Mean / Max Load": 0.0,
        "PBL Max Load Scale": 0.0,
    }
    pbl_extra_weights = {key: 0 for key in pbl_extras}
    pbl_max = None

    with torch.no_grad():
        for data in dataloader:
            data = copy.deepcopy(data).to(trainer.device)
            weight = batch_num_graphs(data)
            total_graphs += weight

            t0 = time.perf_counter()
            outputs = trainer.model(data)
            inference_speed += time.perf_counter() - t0

            for i, loss_func in enumerate(trainer.val_loss):
                loss = loss_func(outputs, data)
                weighted_losses[i] += loss.item() * weight

                if trainer.val_loss_names[i] != "PBL Mean":
                    continue

                current_max = maybe_item(getattr(loss_func, "power_balance_max", None))
                if current_max is not None:
                    pbl_max = current_max if pbl_max is None else max(pbl_max, current_max)

                extra_values = {
                    "PBL L2": getattr(loss_func, "power_balance_l2", None),
                    **get_relative_pbl(loss_func),
                }
                for key, value in extra_values.items():
                    value = maybe_item(value)
                    if value is None:
                        continue
                    pbl_extras[key] += value * weight
                    pbl_extra_weights[key] += weight

    if total_graphs == 0:
        raise ValueError("Evaluation dataloader is empty.")

    result = {
        name: weighted_losses[i] / total_graphs
        for i, name in enumerate(trainer.val_loss_names[:-1])
    }
    result["Inference speed (per batch)"] = inference_speed
    if pbl_max is not None:
        result["PBL Max"] = pbl_max
    for key, total in pbl_extras.items():
        if pbl_extra_weights[key] > 0:
            result[key] = total / pbl_extra_weights[key]
    return result


def run_overall_eval(trainer):
    return run_eval_dataloader(trainer, trainer.dataloaders[0])


def run_by_rho_eval(trainer, batch_size):
    dataset = trainer.datasets[0]
    if not dataset or not hasattr(dataset[0], "rho"):
        return {}

    by_rho = {}
    groups = {}
    for idx, data in enumerate(dataset):
        rho = float(data.rho.view(-1)[0].item())
        groups.setdefault(rho, []).append(idx)

    dataloader_class = trainer.get_dataloader_class()
    for rho, indices in sorted(groups.items()):
        subset = dataset.index_select(indices)
        dataloader = dataloader_class(subset, batch_size=batch_size, shuffle=False)
        by_rho[f"{rho:.6g}"] = {
            "num_samples": len(indices),
            "metrics": run_eval_dataloader(trainer, dataloader),
        }
    return by_rho


def evaluate_run(run_dir, args):
    config = build_eval_config(run_dir, args)
    trainer = load_trainer(config)
    overall = run_overall_eval(trainer)
    result = {
        "run_dir": str(run_dir),
        "model": config["model"]["name"],
        "seed": config["functional"].get("seed"),
        "rho_root": args.rho_root,
        "split": args.split,
        "loss_mode": args.loss_mode,
        "dataset": summarize_dataset(trainer.datasets[0]),
        "overall": overall,
    }
    if not args.no_by_rho:
        result["by_rho"] = run_by_rho_eval(trainer, args.batch_size)
    return result


def average_metric(rows, metric):
    values = [
        row["overall"][metric]
        for row in rows
        if metric in row["overall"] and row["overall"][metric] is not None
    ]
    if not values:
        return None
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def write_outputs(args, rows):
    output = Path(args.output)
    if output.exists():
        with open(output, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    key = f"{args.rho_root}:{args.split}"
    all_results[key] = {
        "runs": rows,
        "summary": {
            metric: average_metric(rows, metric)
            for metric in [
                "PBL Mean",
                "PBL Max",
                "PBL L2",
                "Relative PBL Mean",
                "Relative PBL Mean / Max Load",
            ]
        },
    }
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)

    if args.csv_output:
        fieldnames = [
            "run_dir",
            "model",
            "seed",
            "split",
            "num_samples",
            "rho_counts",
            "PBL Mean",
            "PBL Max",
            "PBL L2",
            "Relative PBL Mean",
            "Relative PBL Mean / Max Load",
            "Inference speed (per batch)",
        ]
        with open(args.csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                out = {
                    "run_dir": row["run_dir"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "split": row["split"],
                    "num_samples": row["dataset"]["num_samples"],
                    "rho_counts": json.dumps(row["dataset"]["rho_counts"]),
                }
                out.update({k: row["overall"].get(k) for k in fieldnames if k in row["overall"]})
                writer.writerow(out)


def main():
    args = parse_args()
    run_dirs = discover_runs(args)
    rows = []
    for run_dir in run_dirs:
        print(f"\nEvaluating rho test split for {run_dir}")
        print("-" * 80)
        row = evaluate_run(run_dir, args)
        rows.append(row)
        print(json.dumps(row["dataset"], indent=2))
        print(json.dumps(row["overall"], indent=2))

    print("\nSummary")
    print("-" * 80)
    for metric in ["PBL Mean", "PBL Max", "PBL L2", "Relative PBL Mean"]:
        summary = average_metric(rows, metric)
        if summary is not None:
            print(
                f"{metric}: {summary['mean']:.6g} +- {summary['std']:.6g} "
                f"(n={summary['n']})"
            )
    write_outputs(args, rows)
    print(f"\nSaved JSON results to {args.output}")
    if args.csv_output:
        print(f"Saved CSV results to {args.csv_output}")


if __name__ == "__main__":
    main()
