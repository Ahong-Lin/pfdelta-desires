import argparse
import copy
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.datasets.data_stats import pfnet_pfdata_stats
from core.utils.main_utils import load_registry
from core.utils.pf_losses_utils import PowerBalanceLoss
from core.utils.registry import registry


MODEL_INFO = {
    "canos_pf": {"family": "CANOS", "label": "CANOS-PF"},
    "flowcan": {"family": "CANOS", "label": "FlowCAN"},
    "flowcan_no_attn": {"family": "CANOS", "label": "FlowCANNoAttn"},
    "powerflownet": {"family": "PFNet", "label": "PFNet"},
    "graph_neural_solver": {"family": "GNS", "label": "GNS-S"},
}

GRID_KEYS = ["n", "n-1", "n-2", "close2inf"]
RAW_TEST_SPLITS = [
    ("n", "feasible_n"),
    ("n-1", "feasible_n-1"),
    ("n-2", "feasible_n-2"),
    ("c2i-n", "near infeasible_n"),
    ("c2i-n1", "near infeasible_n-1"),
    ("c2i-n2", "near infeasible_n-2"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate finished PFDelta runs on test splits and export paper-ready tables."
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Case name, e.g. case30 or case118.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["1.1", "2.2", "2.3"],
        help="Tasks to include, e.g. --tasks 1.1 2.2 2.3",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["canos_pf", "pfnet", "gns", "flowcan"],
        help="Model groups to include: canos_pf, pfnet, gns, flowcan",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Requested seeds. Missing ones are skipped with a warning.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Project runs root.",
    )
    parser.add_argument(
        "--data-root",
        default="data/pfdelta_data",
        help="PFDelta data root used inside configs.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Evaluation device. cpu is safest on shared servers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used for test evaluation.",
    )
    parser.add_argument(
        "--reference-task",
        default="1.1",
        help="Task used for ACC/GEN/VAR/TRD derived tables.",
    )
    parser.add_argument(
        "--eff-full-task",
        default="1.1",
        help="Reference task used for EFF. Default: 1.1, so task2.2/task2.3 are compared against task1.1.",
    )
    parser.add_argument(
        "--strict-seeds",
        action="store_true",
        help="Fail if any requested seed is missing instead of skipping it.",
    )
    return parser.parse_args()


def task_str(task):
    return str(task)


def task_token(task):
    return task_str(task).replace(".", "_")


def normalize_model_selector(model_name):
    model_name = model_name.lower()
    aliases = {
        "canos": "canos_pf",
        "canos-pf": "canos_pf",
        "pfnet": "powerflownet",
        "gns": "graph_neural_solver",
        "gns-s": "graph_neural_solver",
        "flowcan": "flowcan",
    }
    return aliases.get(model_name, model_name)


def model_label(model_name):
    return MODEL_INFO[model_name]["label"]


def model_family(model_name):
    return MODEL_INFO[model_name]["family"]


def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def maybe_float(v):
    if isinstance(v, float):
        return v
    if isinstance(v, int):
        return float(v)
    return float(str(v))


def run_case_from_config(config):
    datasets = config["dataset"]["datasets"]
    return str(datasets[0]["case_name"])


def run_task_from_config(config):
    datasets = config["dataset"]["datasets"]
    return task_str(datasets[0]["task"])


def run_seed_from_config(config):
    return int(config["functional"]["seed"])


def run_model_from_config(config):
    return config["model"]["name"]


def summary_finished(summary):
    return bool(summary.get("finished", False))


def discover_candidate_runs(runs_root, case_name):
    runs_root = Path(runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Could not find runs directory: {runs_root}")

    run_dirs = set()
    case_root = runs_root / case_name
    if case_root.exists():
        run_dirs.update(p for p in case_root.iterdir() if p.is_dir())

    for summary_path in runs_root.glob("**/summary.json"):
        run_dirs.add(summary_path.parent)

    return sorted(run_dirs)


def parse_eval_run_metadata(run_dir):
    parts = list(run_dir.parts)
    eval_parent = None
    for part in parts:
        if part.startswith("eval_case"):
            eval_parent = part
    if eval_parent is None:
        return None

    match = re.match(
        r"eval_(case\d+)_(canos|pfnet|gns|flowcan)_(\d+)_(\d+)$",
        eval_parent,
    )
    if not match:
        return None

    case_name, model_alias, major, minor = match.groups()
    seed_match = re.search(r"_seed(\d+)$", run_dir.name)
    if not seed_match:
        return None

    return {
        "case": case_name,
        "model": normalize_model_selector(model_alias),
        "task": f"{major}.{minor}",
        "seed": int(seed_match.group(1)),
    }


def is_eval_run(run_dir):
    return parse_eval_run_metadata(run_dir) is not None


def prefer_run(new_dir, old_dir):
    if old_dir is None:
        return True
    new_eval = is_eval_run(new_dir)
    old_eval = is_eval_run(old_dir)
    if new_eval != old_eval:
        return new_eval

    new_mtime = (new_dir / "summary.json").stat().st_mtime
    old_mtime = (old_dir / "summary.json").stat().st_mtime
    return new_mtime >= old_mtime


def choose_latest_runs(args):
    requested_models = {normalize_model_selector(m) for m in args.models}
    requested_tasks = {task_str(t) for t in args.tasks}
    requested_seeds = set(args.seeds)

    discovered = {}
    for run_dir in discover_candidate_runs(args.runs_root, args.case):
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.yaml"
        model_path = run_dir / "model.pt"
        if not (summary_path.exists() and config_path.exists() and model_path.exists()):
            continue

        summary = read_json(summary_path)
        if not summary_finished(summary):
            continue

        config = read_yaml(config_path)
        eval_meta = parse_eval_run_metadata(run_dir)
        case_name = eval_meta["case"] if eval_meta else run_case_from_config(config)
        task = eval_meta["task"] if eval_meta else run_task_from_config(config)
        seed = eval_meta["seed"] if eval_meta else run_seed_from_config(config)
        model_name = run_model_from_config(config)
        model_selector = eval_meta["model"] if eval_meta else model_name

        if case_name != args.case:
            continue
        if task not in requested_tasks:
            continue
        if model_selector not in requested_models:
            continue
        if seed not in requested_seeds:
            continue

        key = (task, model_name, seed)
        if prefer_run(run_dir, discovered.get(key)):
            discovered[key] = run_dir

    missing = []
    for task in sorted(requested_tasks):
        for model_name in sorted(requested_models):
            for seed in sorted(requested_seeds):
                if (task, model_name, seed) not in discovered:
                    missing.append((task, model_name, seed))

    if missing and args.strict_seeds:
        lines = [
            f"missing task={task} model={model_label(model)} seed={seed}"
            for task, model, seed in missing
        ]
        raise RuntimeError("Missing requested finished runs:\n" + "\n".join(lines))

    if missing:
        print("WARNING: some requested finished runs are missing and will be skipped:")
        for task, model_name, seed in missing:
            print(f"  - task={task} model={model_label(model_name)} seed={seed}")

    if not discovered:
        raise RuntimeError("No matching finished runs were found.")

    return discovered


def prepare_eval_config(run_dir, args):
    config = read_yaml(run_dir / "config.yaml")
    config["functional"]["is_debug"] = True
    config["functional"]["checkpoint"] = False

    run_location = config["functional"].get("run_location", "")
    if run_location.startswith("runs/"):
        run_location = run_location[len("runs/") :]
    config["functional"]["run_location"] = run_location

    if args.device == "cpu":
        config["functional"]["cpu"] = True
    elif args.device == "cuda":
        config["functional"]["cpu"] = False
    else:
        config["functional"]["cpu"] = not torch.cuda.is_available()

    base_dataset = copy.deepcopy(config["dataset"]["datasets"][0])
    base_dataset["root_dir"] = args.data_root

    case_name = str(base_dataset["case_name"])
    eval_datasets = []
    for _, suffix in RAW_TEST_SPLITS:
        dataset_cfg = copy.deepcopy(base_dataset)
        dataset_cfg["split"] = f"separate_{case_name}_test_{suffix}"
        eval_datasets.append(dataset_cfg)
    config["dataset"] = {"datasets": eval_datasets}
    config["optim"]["train_params"]["batch_size"] = args.batch_size
    config["optim"]["val_params"]["batch_size"] = args.batch_size
    return config


def build_trainer_for_run(run_dir, args):
    config = prepare_eval_config(run_dir, args)
    trainer = load_trainer_from_config(config, run_dir)
    return trainer, config


def load_trainer_from_config(config, run_dir):
    load_registry()
    trainer_name = config["functional"]["trainer_name"]
    trainer_class = registry.get_trainer_class(trainer_name)
    trainer = trainer_class(config)

    run_location = Path(run_dir)
    with open(run_location / "train.json", "r") as f:
        trainer.train_errors = json.load(f)
    with open(run_location / "val.json", "r") as f:
        trainer.val_errors = json.load(f)

    model_location = run_location / "model.pt"
    model_params = torch.load(model_location, map_location="cpu")
    trainer.model.load_state_dict(model_params)
    trainer.model.eval()
    return trainer


def branch_flows_from_predictions(V_pred, theta_pred, edge_index, edge_attr):
    src, dst = edge_index
    r, x, bs, tau, theta_shift = edge_attr
    y = 1 / (r + 1j * x)
    y_real = torch.real(y)
    y_imag = torch.imag(y)

    delta_theta_src = theta_pred[src] - theta_pred[dst]
    delta_theta_dst = theta_pred[dst] - theta_pred[src]

    P_flow_src = (
        V_pred[src]
        * V_pred[dst]
        / tau
        * (
            -y_real * torch.cos(delta_theta_src - theta_shift)
            - y_imag * torch.sin(delta_theta_src - theta_shift)
        )
        + y_real * (V_pred[src] / tau) ** 2
    )
    P_flow_dst = (
        V_pred[dst]
        * V_pred[src]
        / tau
        * (
            -y_real * torch.cos(delta_theta_dst + theta_shift)
            - y_imag * torch.sin(delta_theta_dst + theta_shift)
        )
        + y_real * V_pred[dst] ** 2
    )
    Q_flow_src = (
        V_pred[src]
        * V_pred[dst]
        / tau
        * (
            -y_real * torch.sin(delta_theta_src - theta_shift)
            + y_imag * torch.cos(delta_theta_src - theta_shift)
        )
        - (y_imag + bs / 2) * (V_pred[src] / tau) ** 2
    )
    Q_flow_dst = (
        V_pred[dst]
        * V_pred[src]
        / tau
        * (
            -y_real * torch.sin(delta_theta_dst + theta_shift)
            + y_imag * torch.cos(delta_theta_dst + theta_shift)
        )
        - (y_imag + bs / 2) * V_pred[dst] ** 2
    )
    return P_flow_src, Q_flow_src, P_flow_dst, Q_flow_dst


def paper_pbl_metrics(outputs, data, family):
    pbl = PowerBalanceLoss(family)
    pred = pbl.collect_model_predictions(family, data, outputs)
    V_pred, theta_pred, Pnet, Qnet = pred["predictions"]
    r, x, bs, tau, theta_shift = pred["edge_attr"]
    edge_index = data["bus", "branch", "bus"].edge_index
    src, dst = edge_index
    shunt_g = data["bus"].shunt[:, 0]
    shunt_b = data["bus"].shunt[:, 1]

    delta_P, delta_Q = PowerBalanceLoss.calculate_PBL(
        V_pred,
        theta_pred,
        Pnet,
        Qnet,
        r,
        x,
        bs,
        tau,
        theta_shift,
        shunt_g,
        shunt_b,
        src,
        dst,
    )
    mismatch = torch.sqrt(delta_P.square() + delta_Q.square())
    pbl_values = global_mean_pool(mismatch, data["bus"].batch)

    pfs, qfs, pfd, qfd = branch_flows_from_predictions(
        V_pred, theta_pred, edge_index, (r, x, bs, tau, theta_shift)
    )

    apparent_src = torch.sqrt(pfs.square() + qfs.square())
    apparent_dst = torch.sqrt(pfd.square() + qfd.square())
    incoming = torch.zeros_like(V_pred).scatter_add_(0, dst, apparent_dst)
    outgoing = torch.zeros_like(V_pred).scatter_add_(0, src, apparent_src)
    denom = torch.maximum(incoming, outgoing)

    node_rpbl = torch.where(
        denom > 0,
        (mismatch / denom.clamp_min(1e-8)) * 100.0,
        torch.zeros_like(mismatch),
    )
    return {
        "PBL": pbl_values,
        "rPBL (%)": global_mean_pool(node_rpbl, data["bus"].batch),
    }


def _sum_abs(x):
    return torch.abs(x).sum().item()


def _sum_sq(x):
    return x.square().sum().item()


def _component_metric_totals(pred, true, eps=1e-8):
    abs_err = torch.abs(pred - true)
    abs_true = torch.abs(true)
    valid = abs_true > eps
    return {
        "abs_err_sum": abs_err.sum().item(),
        "abs_true_sum": abs_true.sum().item(),
        "ape_sum": (abs_err[valid] / abs_true[valid]).sum().item(),
        "ape_count": int(valid.sum().item()),
        "sq_err_sum": _sum_sq(pred - true),
        "sq_err_count": int(pred.numel()),
    }


def align_angles_by_graph_mean(pred, true, batch, mask):
    aligned = pred.clone()
    for graph_id in torch.unique(batch):
        graph_mask = (batch == graph_id) & mask
        if graph_mask.any():
            offset = (pred[graph_mask] - true[graph_mask]).mean()
            aligned[batch == graph_id] = pred[batch == graph_id] - offset
    return aligned


def metric_components_for_outputs(outputs, data, model_name, align_gns_theta=True):
    if model_name in {"canos_pf", "flowcan", "flowcan_no_attn"}:
        pq_pred = outputs["PQ"]
        pq_true = data["PQ"].y
        pv_idx = data["PV", "PV_link", "bus"].edge_index[1]
        pv_bus_pred = outputs["bus"][pv_idx]
        pv_true = data["PV"].y
        slack_pred = outputs["slack"]
        slack_true = data["slack"].y

        comps = {
            "V": (pq_pred[:, 1], pq_true[:, 1]),
            "theta_pq": (pq_pred[:, 0], pq_true[:, 0]),
            "theta_pv": (pv_bus_pred[:, 0], pv_true[:, 1]),
            "P": (slack_pred[:, 0], slack_true[:, 0]),
            "Q_pv": (outputs["PV"][:, 0], pv_true[:, 0]),
            "Q_slack": (slack_pred[:, 1], slack_true[:, 1]),
        }
        return comps

    if model_name == "powerflownet":
        device = outputs.device
        case_name = data.case_name[0]
        mean = pfnet_pfdata_stats[case_name]["mean"]["bus"]["y"].to(device)
        std = pfnet_pfdata_stats[case_name]["std"]["bus"]["y"].to(device)
        pred = outputs * std + mean
        true = data["bus"].y * std + mean
        bus_type = data["bus"].bus_type
        pq_idx = (bus_type == 1).nonzero(as_tuple=True)[0]
        pv_idx = (bus_type == 2).nonzero(as_tuple=True)[0]
        slack_idx = (bus_type == 3).nonzero(as_tuple=True)[0]

        comps = {
            "V": (pred[pq_idx, 0], true[pq_idx, 0]),
            "theta_pq": (pred[pq_idx, 1], true[pq_idx, 1]),
            "theta_pv": (pred[pv_idx, 1], true[pv_idx, 1]),
            "P": (pred[slack_idx, 2], true[slack_idx, 2]),
            "Q_pv": (pred[pv_idx, 3], true[pv_idx, 3]),
            "Q_slack": (pred[slack_idx, 3], true[slack_idx, 3]),
        }
        return comps

    if model_name == "graph_neural_solver":
        bus_type = data["bus"].bus_type
        pq_idx = (bus_type == 1).nonzero(as_tuple=True)[0]
        pv_idx = (bus_type == 2).nonzero(as_tuple=True)[0]
        slack_idx = (bus_type == 3).nonzero(as_tuple=True)[0]

        true_v = data["bus"].bus_voltages[:, 1]
        true_theta = data["bus"].bus_voltages[:, 0]
        pred_theta = data["bus"].theta
        if align_gns_theta:
            angle_mask = bus_type != 3
            pred_theta = align_angles_by_graph_mean(
                pred_theta,
                true_theta,
                data["bus"].batch,
                angle_mask,
            )
        true_pnet = data["bus"].bus_gen[:, 0] - data["bus"].bus_demand[:, 0]
        true_qnet = data["bus"].bus_gen[:, 1] - data["bus"].bus_demand[:, 1]
        pred_pnet = data["bus"].pg - data["bus"].pd
        pred_qnet = data["bus"].qg - data["bus"].qd

        comps = {
            "V": (data["bus"].v[pq_idx], true_v[pq_idx]),
            "theta_pq": (pred_theta[pq_idx], true_theta[pq_idx]),
            "theta_pv": (pred_theta[pv_idx], true_theta[pv_idx]),
            "P": (pred_pnet[slack_idx], true_pnet[slack_idx]),
            "Q_pv": (pred_qnet[pv_idx], true_qnet[pv_idx]),
            "Q_slack": (pred_qnet[slack_idx], true_qnet[slack_idx]),
        }
        return comps

    raise KeyError(f"Unsupported model for metric extraction: {model_name}")


def init_metric_accumulator():
    acc = {
        "pbl_sum": 0.0,
        "rpbl_sum": 0.0,
        "num_graphs": 0,
    }
    for prefix in ["overall", "V", "theta", "P", "Q", "raw_overall", "raw_theta"]:
        acc[f"{prefix}_abs_err_sum"] = 0.0
        acc[f"{prefix}_abs_true_sum"] = 0.0
        acc[f"{prefix}_ape_sum"] = 0.0
        acc[f"{prefix}_ape_count"] = 0
        acc[f"{prefix}_sq_err_sum"] = 0.0
        acc[f"{prefix}_sq_err_count"] = 0
    return acc


def _update_group_totals(acc, prefix, keys, comps):
    abs_err_sum = 0.0
    abs_true_sum = 0.0
    ape_sum = 0.0
    ape_count = 0
    sq_err_sum = 0.0
    sq_err_count = 0

    for key in keys:
        totals = _component_metric_totals(*comps[key])
        abs_err_sum += totals["abs_err_sum"]
        abs_true_sum += totals["abs_true_sum"]
        ape_sum += totals["ape_sum"]
        ape_count += totals["ape_count"]
        sq_err_sum += totals["sq_err_sum"]
        sq_err_count += totals["sq_err_count"]

    acc[f"{prefix}_abs_err_sum"] += abs_err_sum
    acc[f"{prefix}_abs_true_sum"] += abs_true_sum
    acc[f"{prefix}_ape_sum"] += ape_sum
    acc[f"{prefix}_ape_count"] += ape_count
    acc[f"{prefix}_sq_err_sum"] += sq_err_sum
    acc[f"{prefix}_sq_err_count"] += sq_err_count


def update_accumulator(acc, comps, pbl_metrics, raw_comps=None):
    component_keys = {
        "V": ["V"],
        "theta": ["theta_pq", "theta_pv"],
        "P": ["P"],
        "Q": ["Q_pv", "Q_slack"],
    }

    for group_name, keys in component_keys.items():
        _update_group_totals(acc, group_name, keys, comps)

    all_keys = [key for keys in component_keys.values() for key in keys]
    _update_group_totals(acc, "overall", all_keys, comps)

    raw_source = raw_comps or comps
    _update_group_totals(acc, "raw_overall", all_keys, raw_source)
    _update_group_totals(acc, "raw_theta", component_keys["theta"], raw_source)

    acc["pbl_sum"] += pbl_metrics["PBL"].sum().item()
    acc["rpbl_sum"] += pbl_metrics["rPBL (%)"].sum().item()
    acc["num_graphs"] += int(pbl_metrics["PBL"].numel())


def finalize_accumulator(acc):
    def pct(num, den):
        return 100.0 * num / den if den > 0 else float("nan")

    def mean(sum_value, count):
        return sum_value / count if count > 0 else float("nan")

    return {
        "WAPE (%)": pct(acc["overall_abs_err_sum"], acc["overall_abs_true_sum"]),
        "WAPE_V (%)": pct(acc["V_abs_err_sum"], acc["V_abs_true_sum"]),
        "WAPE_theta (%)": pct(acc["theta_abs_err_sum"], acc["theta_abs_true_sum"]),
        "WAPE_P (%)": pct(acc["P_abs_err_sum"], acc["P_abs_true_sum"]),
        "WAPE_Q (%)": pct(acc["Q_abs_err_sum"], acc["Q_abs_true_sum"]),
        "WAPE_raw (%)": pct(acc["raw_overall_abs_err_sum"], acc["raw_overall_abs_true_sum"]),
        "WAPE_theta_raw (%)": pct(
            acc["raw_theta_abs_err_sum"], acc["raw_theta_abs_true_sum"]
        ),
        "MAPE (%)": pct(acc["overall_ape_sum"], acc["overall_ape_count"]),
        "MAPE_V (%)": pct(acc["V_ape_sum"], acc["V_ape_count"]),
        "MAPE_theta (%)": pct(acc["theta_ape_sum"], acc["theta_ape_count"]),
        "MAPE_P (%)": pct(acc["P_ape_sum"], acc["P_ape_count"]),
        "MAPE_Q (%)": pct(acc["Q_ape_sum"], acc["Q_ape_count"]),
        "MAPE_raw (%)": pct(acc["raw_overall_ape_sum"], acc["raw_overall_ape_count"]),
        "MAPE_theta_raw (%)": pct(
            acc["raw_theta_ape_sum"], acc["raw_theta_ape_count"]
        ),
        "MSE": mean(acc["overall_sq_err_sum"], acc["overall_sq_err_count"]),
        "MSE_V": mean(acc["V_sq_err_sum"], acc["V_sq_err_count"]),
        "MSE_theta": mean(acc["theta_sq_err_sum"], acc["theta_sq_err_count"]),
        "MSE_P": mean(acc["P_sq_err_sum"], acc["P_sq_err_count"]),
        "MSE_Q": mean(acc["Q_sq_err_sum"], acc["Q_sq_err_count"]),
        "MSE_raw": mean(acc["raw_overall_sq_err_sum"], acc["raw_overall_sq_err_count"]),
        "MSE_theta_raw": mean(
            acc["raw_theta_sq_err_sum"], acc["raw_theta_sq_err_count"]
        ),
        "PBL": acc["pbl_sum"] / acc["num_graphs"] if acc["num_graphs"] > 0 else float("nan"),
        "rPBL (%)": acc["rpbl_sum"] / acc["num_graphs"] if acc["num_graphs"] > 0 else float("nan"),
    }


@torch.no_grad()
def evaluate_one_dataloader(trainer, model_name, family, dataloader):
    trainer.model.eval()
    acc = init_metric_accumulator()
    for batch in tqdm(dataloader, leave=False):
        batch = copy.deepcopy(batch).to(trainer.device)
        if model_name == "graph_neural_solver":
            outputs = trainer.model(batch, return_data=False)
        else:
            outputs = trainer.model(batch)

        comps = metric_components_for_outputs(outputs, batch, model_name)
        raw_comps = None
        if model_name == "graph_neural_solver":
            raw_comps = metric_components_for_outputs(
                outputs,
                batch,
                model_name,
                align_gns_theta=False,
            )
        pbl_metrics = paper_pbl_metrics(outputs, batch, family)
        update_accumulator(acc, comps, pbl_metrics, raw_comps=raw_comps)
    return finalize_accumulator(acc)


def average_close2inf(seed_metrics):
    keys = list(seed_metrics["c2i-n"].keys())
    result = {}
    for key in keys:
        result[key] = (
            seed_metrics["c2i-n"][key]
            + seed_metrics["c2i-n1"][key]
            + seed_metrics["c2i-n2"][key]
        ) / 3.0
    return result


def evaluate_run(run_dir, args):
    trainer, config = build_trainer_for_run(run_dir, args)
    model_name = run_model_from_config(config)
    family = model_family(model_name)
    seed_metrics = {}
    for grid_name, dataloader in zip(
        [name for name, _ in RAW_TEST_SPLITS],
        trainer.dataloaders,
    ):
        seed_metrics[grid_name] = evaluate_one_dataloader(
            trainer,
            model_name,
            family,
            dataloader,
        )
    seed_metrics["close2inf"] = average_close2inf(seed_metrics)
    return seed_metrics


def summarize_runs(selected_runs, args):
    per_seed_rows = []
    nested = {}
    for (task, model_name, seed), run_dir in sorted(selected_runs.items()):
        print(
            f"\nEvaluating case={args.case} task={task} model={model_label(model_name)} "
            f"seed={seed} from {run_dir.name}"
        )
        metrics = evaluate_run(run_dir, args)
        nested.setdefault(task, {}).setdefault(model_name, {})[seed] = {
            "run_dir": str(run_dir),
            "metrics": metrics,
        }
        for grid in GRID_KEYS:
            row = {
                "Case": args.case,
                "Task": task,
                "Model": model_label(model_name),
                "Model Key": model_name,
                "Seed": seed,
                "Grid Attribute": grid,
                "Run Dir": str(run_dir),
            }
            row.update(metrics[grid])
            per_seed_rows.append(row)
    return nested, pd.DataFrame(per_seed_rows)


def summarize_over_seeds(per_seed_df):
    metric_cols = [
        "WAPE (%)",
        "WAPE_V (%)",
        "WAPE_theta (%)",
        "WAPE_P (%)",
        "WAPE_Q (%)",
        "WAPE_raw (%)",
        "WAPE_theta_raw (%)",
        "MAPE (%)",
        "MAPE_V (%)",
        "MAPE_theta (%)",
        "MAPE_P (%)",
        "MAPE_Q (%)",
        "MAPE_raw (%)",
        "MAPE_theta_raw (%)",
        "MSE",
        "MSE_V",
        "MSE_theta",
        "MSE_P",
        "MSE_Q",
        "MSE_raw",
        "MSE_theta_raw",
        "PBL",
        "rPBL (%)",
    ]
    grouped = per_seed_df.groupby(["Case", "Task", "Model", "Model Key", "Grid Attribute"], sort=True)
    rows = []
    for keys, group in grouped:
        row = {
            "Case": keys[0],
            "Task": keys[1],
            "Model": keys[2],
            "Model Key": keys[3],
            "Grid Attribute": keys[4],
            "Seeds": int(group["Seed"].nunique()),
            "Seed List": ",".join(str(x) for x in sorted(group["Seed"].unique())),
        }
        for col in metric_cols:
            row[f"Mean {col}"] = group[col].mean()
            row[f"SD {col}"] = 0.0 if len(group) == 1 else group[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Task", "Grid Attribute", "Model"])


def derive_acc_gen_var(summary_df, reference_task):
    sub = summary_df[summary_df["Task"] == reference_task].copy()
    if sub.empty:
        return {}

    acc = sub[sub["Grid Attribute"] == "n"].copy()
    acc["View"] = "ACC"

    base_map = {
        row["Model"]: row for _, row in acc.iterrows()
    }

    def with_ratios(grid_name, view_name):
        part = sub[sub["Grid Attribute"] == grid_name].copy()
        if part.empty:
            return part
        rho_wape = []
        rho_mape = []
        rho_mse = []
        rho_pbl = []
        rho_rpbl = []
        for _, row in part.iterrows():
            base = base_map.get(row["Model"])
            if base is None:
                rho_wape.append(float("nan"))
                rho_mape.append(float("nan"))
                rho_mse.append(float("nan"))
                rho_pbl.append(float("nan"))
                rho_rpbl.append(float("nan"))
            else:
                rho_wape.append(base["Mean WAPE (%)"] / row["Mean WAPE (%)"])
                rho_mape.append(base["Mean MAPE (%)"] / row["Mean MAPE (%)"])
                rho_mse.append(base["Mean MSE"] / row["Mean MSE"])
                rho_pbl.append(base["Mean PBL"] / row["Mean PBL"])
                rho_rpbl.append(base["Mean rPBL (%)"] / row["Mean rPBL (%)"])
        part["rho_WAPE"] = rho_wape
        part["rho_MAPE"] = rho_mape
        part["rho_MSE"] = rho_mse
        part["rho_PBL"] = rho_pbl
        part["rho_rPBL"] = rho_rpbl
        part["View"] = view_name
        return part

    gen_n1 = with_ratios("n-1", "GEN-N1")
    gen_n2 = with_ratios("n-2", "GEN-N2")

    var_df = acc[
        [
            "Case",
            "Task",
            "Model",
            "Grid Attribute",
            "Seeds",
            "Seed List",
            "Mean WAPE_V (%)",
            "SD WAPE_V (%)",
            "Mean WAPE_theta (%)",
            "SD WAPE_theta (%)",
            "Mean WAPE_P (%)",
            "SD WAPE_P (%)",
            "Mean WAPE_Q (%)",
            "SD WAPE_Q (%)",
            "Mean MAPE_V (%)",
            "SD MAPE_V (%)",
            "Mean MAPE_theta (%)",
            "SD MAPE_theta (%)",
            "Mean MAPE_P (%)",
            "SD MAPE_P (%)",
            "Mean MAPE_Q (%)",
            "SD MAPE_Q (%)",
            "Mean MSE_V",
            "SD MSE_V",
            "Mean MSE_theta",
            "SD MSE_theta",
            "Mean MSE_P",
            "SD MSE_P",
            "Mean MSE_Q",
            "SD MSE_Q",
        ]
    ].copy()
    var_df["View"] = "VAR"

    return {
        "acc": acc,
        "gen_n1": gen_n1,
        "gen_n2": gen_n2,
        "var": var_df,
    }


def derive_eff(summary_df, full_task):
    if full_task is None:
        return pd.DataFrame()
    sub = summary_df[summary_df["Task"].isin([task_str(full_task), "2.2", "2.3"])].copy()
    if sub.empty:
        return pd.DataFrame()

    base = sub[(sub["Task"] == task_str(full_task)) & (sub["Grid Attribute"] == "n")]
    base_map = {row["Model"]: row for _, row in base.iterrows()}
    rows = []
    for task, tag in [("2.2", "67"), ("2.3", "33")]:
        part = sub[(sub["Task"] == task) & (sub["Grid Attribute"] == "n")]
        for _, row in part.iterrows():
            ref = base_map.get(row["Model"])
            if ref is None:
                continue
            rows.append(
                {
                    "Case": row["Case"],
                    "Model": row["Model"],
                    "Task": row["Task"],
                    "Data Ratio Tag": tag,
                    "eta_WAPE": ref["Mean WAPE (%)"] / row["Mean WAPE (%)"],
                    "eta_MAPE": ref["Mean MAPE (%)"] / row["Mean MAPE (%)"],
                    "eta_MSE": ref["Mean MSE"] / row["Mean MSE"],
                    "eta_PBL": ref["Mean PBL"] / row["Mean PBL"],
                    "eta_rPBL": ref["Mean rPBL (%)"] / row["Mean rPBL (%)"],
                    "Reference Task": full_task,
                    "Seeds": row["Seeds"],
                }
            )
    return pd.DataFrame(rows)


def derive_trd(summary_df, reference_task):
    acc = summary_df[
        (summary_df["Task"] == reference_task) & (summary_df["Grid Attribute"] == "n")
    ].copy()
    if acc.empty:
        return pd.DataFrame()

    rows = []
    for _, row_i in acc.iterrows():
        d = 0
        for _, row_j in acc.iterrows():
            if row_i["Model"] == row_j["Model"]:
                continue
            no_worse = (
                row_j["Mean WAPE (%)"] <= row_i["Mean WAPE (%)"]
                and row_j["Mean rPBL (%)"] <= row_i["Mean rPBL (%)"]
            )
            strictly_better = (
                row_j["Mean WAPE (%)"] < row_i["Mean WAPE (%)"]
                or row_j["Mean rPBL (%)"] < row_i["Mean rPBL (%)"]
            )
            if no_worse and strictly_better:
                d += 1
        rows.append(
            {
                "Case": row_i["Case"],
                "Task": row_i["Task"],
                "Model": row_i["Model"],
                "Mean WAPE (%)": row_i["Mean WAPE (%)"],
                "Mean MAPE (%)": row_i["Mean MAPE (%)"],
                "Mean PBL": row_i["Mean PBL"],
                "Mean rPBL (%)": row_i["Mean rPBL (%)"],
                "Dominance Count": d,
            }
        )
    trd = pd.DataFrame(rows)
    trd["Average Dominance Count"] = trd["Dominance Count"].mean()
    return trd.sort_values("Dominance Count")


def save_outputs(args, nested, per_seed_df, summary_df, derived_tables):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"paper_eval_{args.case}_tasks_" + "_".join(task_token(t) for t in args.tasks)

    per_seed_json = out_dir / f"{prefix}_per_seed.json"
    with open(per_seed_json, "w") as f:
        json.dump(nested, f, indent=2)

    per_seed_csv = out_dir / f"{prefix}_per_seed.csv"
    per_seed_df.to_csv(per_seed_csv, index=False)

    summary_csv = out_dir / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    written = [per_seed_json, per_seed_csv, summary_csv]

    for name, df in derived_tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            path = out_dir / f"{prefix}_{name}.csv"
            df.to_csv(path, index=False)
            written.append(path)

    return written


def print_brief_summary(summary_df, reference_task, eff_df, trd_df):
    print("\nFinished aggregated summary:")
    cols = [
        "Task",
        "Grid Attribute",
        "Model",
        "Mean WAPE (%)",
        "Mean MAPE (%)",
        "Mean PBL",
        "Mean rPBL (%)",
        "Seeds",
    ]
    print(summary_df[cols].round(4).to_string(index=False))

    acc = summary_df[
        (summary_df["Task"] == reference_task) & (summary_df["Grid Attribute"] == "n")
    ]
    if not acc.empty:
        print(f"\nACC preview for task {reference_task}:")
        print(
            acc[
                [
                    "Model",
                    "Mean WAPE (%)",
                    "Mean MAPE (%)",
                    "Mean PBL",
                    "Mean rPBL (%)",
                    "Seeds",
                ]
            ].round(4).to_string(index=False)
        )

    if not eff_df.empty:
        print("\nEFF preview:")
        print(eff_df.round(4).to_string(index=False))

    if not trd_df.empty:
        print("\nTRD preview:")
        print(trd_df.round(4).to_string(index=False))


def main():
    args = parse_args()
    args.tasks = [task_str(t) for t in args.tasks]
    args.reference_task = task_str(args.reference_task)
    if args.eff_full_task is not None:
        args.eff_full_task = task_str(args.eff_full_task)

    selected_runs = choose_latest_runs(args)
    nested, per_seed_df = summarize_runs(selected_runs, args)
    summary_df = summarize_over_seeds(per_seed_df)

    derived = derive_acc_gen_var(summary_df, args.reference_task)
    eff_df = derive_eff(summary_df, args.eff_full_task)
    trd_df = derive_trd(summary_df, args.reference_task)
    if not eff_df.empty:
        derived["eff"] = eff_df
    if not trd_df.empty:
        derived["trd"] = trd_df

    written = save_outputs(args, nested, per_seed_df, summary_df, derived)
    print_brief_summary(summary_df, args.reference_task, eff_df, trd_df)

    print("\nWrote files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
