#!/usr/bin/env python3
"""Evaluate CANOS-PF on a frozen cached endpoint set, without legacy trainers.

MSE follows CANOS_PF_MSE: the sum of PV, PQ, slack and branch component
means. Per-graph values are checked against the repository implementation
on each batch. PBL reuses the repository's PowerBalanceLoss implementation.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from core.utils.pf_losses_utils import CANOS_PF_MSE, PowerBalanceLoss


BRANCH = ("bus", "branch", "bus")
METRICS = (
    "canos_pf_mse", "mse_PQ", "mse_PV", "mse_Slack", "mse_Branch",
    "PBL Mean", "PBL Max", "PBL L2", "Relative PBL Mean",
    "Relative PBL Mean / Max Load",
)


def _graph_mean(values, graph_ids, n_graphs):
    if values.ndim != 1:
        raise ValueError("Expected one scalar per node/branch")
    count = torch.bincount(graph_ids, minlength=n_graphs).to(values.dtype)
    if bool((count == 0).any()):
        raise ValueError("A graph has an empty CANOS metric component")
    total = values.new_zeros(n_graphs).scatter_add_(0, graph_ids, values)
    return total / count


def _graph_max(values, graph_ids, n_graphs):
    result = values.new_full((n_graphs,), float("-inf"))
    result.scatter_reduce_(0, graph_ids, values, reduce="amax", include_self=True)
    return result


def _summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    summary = {"num_samples": len(rows)}
    summary["num_parent_settings"] = len({row["load_setting_id"] for row in rows})
    for key in METRICS:
        values = [row[key] for row in rows if key in row]
        if values:
            summary[key] = max(values) if key == "PBL Max" else math.fsum(values) / len(values)
    return summary


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_csv(path, rows, fields):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, outdir, parent_to_line, num_workers=0):
    """Return and save overall, rho and outage-line metrics plus every record.

    ``parent_to_line`` maps integer load_setting_id to the original branch ID;
    it must include all evaluation parents. No cached/raw data are modified.
    Both CPU and CUDA are supported. This is evaluation only, not model
    selection: pass the checkpoint already selected using validation MSE.
    """
    device = torch.device(device)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    mse = CANOS_PF_MSE()
    pbl = PowerBalanceLoss("CANOS")
    model.eval()
    rows = []
    inference_seconds = 0.0
    evaluation_start = time.perf_counter()
    seen = set()
    for data in loader:
        data = data.clone().to(device)
        n = int(data.num_graphs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model(data)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - start

        metrics = {}
        for component, node_type in (("PQ", "PQ"), ("PV", "PV"), ("Slack", "slack")):
            squared_error = (output[node_type] - data[node_type].y).square().mean(dim=1)
            metrics["mse_" + component] = _graph_mean(squared_error, data[node_type].batch, n)
        branch_ids = data["bus"].batch[data[BRANCH].edge_index[0]]
        branch_error = (output["edge_preds"] - data[BRANCH].edge_label).square().mean(dim=1)
        metrics["mse_Branch"] = _graph_mean(branch_error, branch_ids, n)
        metrics["canos_pf_mse"] = sum(metrics.values())
        legacy_mse = mse(output, data)
        if not torch.allclose(metrics["canos_pf_mse"].mean(), legacy_mse, rtol=2e-5, atol=1e-7):
            raise ValueError("Per-graph MSE does not match CANOS_PF_MSE; inspect varying component sizes")

        pbl(output, data)
        squared_mismatch = pbl.delta_P.square() + pbl.delta_Q.square()
        mismatch = squared_mismatch.sqrt()
        bus_ids = data["bus"].batch
        metrics["PBL Mean"] = _graph_mean(mismatch, bus_ids, n)
        metrics["PBL Max"] = _graph_max(mismatch, bus_ids, n)
        metrics["PBL L2"] = squared_mismatch.new_zeros(n).scatter_add_(0, bus_ids, squared_mismatch).sqrt()
        if hasattr(data["bus"], "bus_demand"):
            demand = torch.linalg.vector_norm(data["bus"].bus_demand, dim=-1)
            demand_mean = _graph_mean(demand, bus_ids, n).clamp_min(1e-8)
            demand_max = _graph_max(demand, bus_ids, n).clamp_min(1e-8)
            metrics["Relative PBL Mean"] = metrics["PBL Mean"] / demand_mean
            metrics["Relative PBL Mean / Max Load"] = metrics["PBL Mean"] / demand_max
        if not torch.allclose(metrics["PBL Mean"].mean(), pbl.power_balance_mean, rtol=2e-5, atol=1e-7):
            raise ValueError("Per-graph PBL does not match the repository PBL")
        for name, values in metrics.items():
            if values.shape != (n,) or not bool(torch.isfinite(values).all()):
                raise ValueError(f"Invalid/nonfinite evaluation values in {name}")
        host_metrics = {name: values.detach().cpu().tolist() for name, values in metrics.items()}
        load_ids = data.load_setting_id.view(-1).detach().cpu().tolist()
        rhos = data.rho.view(-1).detach().cpu().tolist()
        sample_ids = (data.rho_sample_id.view(-1).detach().cpu().tolist()
                      if hasattr(data, "rho_sample_id") else [None] * n)
        if not (len(load_ids) == len(rhos) == len(sample_ids) == n):
            raise ValueError("Invalid batched parent/rho metadata")
        for i in range(n):
            parent = int(load_ids[i])
            rho = float(rhos[i])
            if rho not in (0.0, 1.0):
                raise ValueError(f"Expected endpoints only; got rho={rho}")
            if parent not in parent_to_line:
                raise KeyError(f"Parent {parent} missing from frozen line mapping")
            key = (parent, rho)
            if key in seen:
                raise ValueError(f"Duplicate evaluation parent/rho: {key}")
            seen.add(key)
            rows.append({
                "load_setting_id": parent, "rho": rho,
                "rho_sample_id": None if sample_ids[i] is None else int(sample_ids[i]),
                "branch_id": int(parent_to_line[parent]),
                **{name: values[i] for name, values in host_metrics.items()},
            })
    if len(rows) != len(dataset):
        raise ValueError(f"Evaluation saw {len(rows)} graphs, expected {len(dataset)}")
    if any((parent, 1.0 - rho) not in seen for parent, rho in seen):
        raise ValueError("Evaluation set contains an unpaired endpoint")

    grouped_rho, grouped_line, grouped_line_rho = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        grouped_rho[f"{row['rho']:g}"].append(row)
        grouped_line[str(row["branch_id"])].append(row)
        grouped_line_rho[(row["branch_id"], row["rho"])].append(row)
    summary = {
        "num_samples": len(rows),
        "num_parent_settings": len({row["load_setting_id"] for row in rows}),
        "overall": _summarize(rows),
        "by_rho": {key: _summarize(value) for key, value in sorted(grouped_rho.items())},
        "by_line": {key: _summarize(value) for key, value in sorted(grouped_line.items(), key=lambda item: int(item[0]))},
        "inference_seconds": inference_seconds,
        "evaluation_seconds": time.perf_counter() - evaluation_start,
        "batch_size": batch_size,
        "definitions": {
            "canos_pf_mse": "Mean over graphs of MSE_PQ + MSE_PV + MSE_Slack + MSE_Branch; checked against CANOS_PF_MSE per batch",
            "PBL Mean": "Mean over graphs of mean over buses of sqrt(delta_P^2 + delta_Q^2)",
            "PBL Max": "Maximum bus mismatch over the specified set (not average graph maximum)",
            "PBL L2": "Mean graph L2 norm of concatenated active and reactive bus mismatches",
            "Relative PBL Mean": "Mean graph PBL Mean divided by that graph's mean bus demand magnitude (clamped at 1e-8)",
            "Relative PBL Mean / Max Load": "Mean graph PBL Mean divided by that graph's maximum bus demand magnitude (clamped at 1e-8)",
        },
    }
    _write_csv(outdir / "per_sample.csv", rows,
               ["load_setting_id", "rho_sample_id", "branch_id", "rho", *METRICS])
    line_rows = [{"branch_id": line, "rho": rho, **_summarize(value)}
                 for (line, rho), value in sorted(grouped_line_rho.items())]
    _write_csv(outdir / "by_line_rho.csv", line_rows,
               ["branch_id", "rho", "num_samples", "num_parent_settings", *METRICS])
    _write_json(outdir / "summary.json", summary)
    return summary
