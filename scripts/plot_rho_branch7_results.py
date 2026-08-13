#!/usr/bin/env python3
"""Plot rho branch7 experiment training curves and rho=0 test PBL bars."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS = [
    ("60L-80R", "canos_case30_rho_branch7_60L80R", "rho_branch7_60L80R_test_pbl.csv"),
    ("80L-60R", "canos_case30_rho_branch7_80L60R", "rho_branch7_80L60R_test_pbl.csv"),
    ("100L-48R", "canos_case30_rho_branch7_100L48R", "rho_branch7_100L48R_test_pbl.csv"),
]

COLORS = {
    "60L-80R": "#2563eb",
    "80L-60R": "#059669",
    "100L-48R": "#dc2626",
}


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def latest_run_dir(runs_root: Path, run_group: str) -> Path:
    group = runs_root / run_group
    candidates = [
        child
        for child in group.iterdir()
        if child.is_dir()
        and (child / "train.json").exists()
        and (child / "summary.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run directory found under {group}")
    return max(candidates, key=lambda p: (p / "summary.json").stat().st_mtime)


def first_numeric_metric(row: dict) -> str:
    for key, value in row.items():
        if key == "Inference speed (per batch)":
            continue
        try:
            float(value)
            return key
        except (TypeError, ValueError):
            continue
    raise KeyError(f"No numeric metric found in row keys: {list(row)}")


def get_curve(run_dir: Path, filename: str, preferred: str | None = None):
    data = read_json(run_dir / filename)
    keys = sorted(data.keys(), key=lambda x: int(x))
    first = data[keys[0]]
    first_row = first[0] if isinstance(first, list) else first
    metric = preferred if preferred and preferred in first_row else first_numeric_metric(first_row)

    xs, ys = [], []
    for key in keys:
        row = data[key]
        row = row[0] if isinstance(row, list) else row
        if metric not in row:
            continue
        xs.append(int(key) + 1)
        ys.append(float(row[metric]))
    return xs, ys, metric


def to_float(value):
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def plot_training(args, run_dirs: dict[str, Path], output_dir: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    summary_rows = []

    for label, _, _ in EXPERIMENTS:
        xs, ys, metric = get_curve(run_dirs[label], "train.json", args.train_metric)
        summary = read_json(run_dirs[label] / "summary.json")
        best_epoch = int(summary.get("best_point", 0)) + 1

        ax.plot(xs, ys, label=label, color=COLORS[label], linewidth=2.1)
        summary_rows.append(
            {
                "Experiment": label,
                "Run dir": str(run_dirs[label]),
                "Train metric": metric,
                "Epochs": xs[-1],
                "Best epoch": best_epoch,
                "Initial train loss": ys[0],
                "Final train loss": ys[-1],
                "Min train loss": min(ys),
            }
        )

    ylabel = summary_rows[0]["Train metric"]
    ax.set_title("CANOS rho branch7 training loss, case30")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    if args.log_train:
        ax.set_yscale("log")
    fig.tight_layout()

    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output_dir / f"rho_branch7_training_loss_overlay.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "rho_branch7_training_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def read_test_rows(results_dir: Path):
    rows = []
    for label, _, csv_name in EXPERIMENTS:
        path = results_dir / csv_name
        with path.open(newline="") as f:
            loaded = list(csv.DictReader(f))
        if len(loaded) != 1:
            raise ValueError(f"Expected one row in {path}, got {len(loaded)}")
        row = {"Experiment": label, **loaded[0]}
        rows.append(row)
    return rows


def plot_test(args, rows: list[dict], output_dir: Path):
    wanted = [
        "Experiment",
        "run_dir",
        "num_samples",
        "rho_counts",
        "PBL Mean",
        "PBL Max",
        "PBL L2",
        "Relative PBL Mean",
        "Relative PBL Mean / Max Load",
        "Inference speed (per batch)",
    ]
    with (output_dir / "rho_branch7_test_pbl_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wanted)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in wanted})

    labels = [row["Experiment"] for row in rows]
    pbl_mean = [to_float(row.get("PBL Mean")) for row in rows]
    pbl_max = [to_float(row.get("PBL Max")) for row in rows]
    rel = [to_float(row.get("Relative PBL Mean")) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    bar_colors = [COLORS[label] for label in labels]

    axes[0].bar(labels, pbl_mean, color=bar_colors, width=0.62)
    axes[0].set_title("rho=0 test PBL Mean")
    axes[0].set_ylabel("PBL Mean")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(pbl_mean):
        if value is not None:
            axes[0].text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)

    metric2 = rel if any(value is not None for value in rel) else pbl_max
    metric2_name = "Relative PBL Mean" if any(value is not None for value in rel) else "PBL Max"
    axes[1].bar(labels, metric2, color=bar_colors, width=0.62)
    axes[1].set_title(f"rho=0 test {metric2_name}")
    axes[1].set_ylabel(metric2_name)
    axes[1].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(metric2):
        if value is not None:
            axes[1].text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("CANOS rho branch7 test comparison, case30", y=1.03)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output_dir / f"rho_branch7_test_pbl_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--results-dir", default="rho_test_results")
    parser.add_argument("--output-dir", default="rho_test_results/figures")
    parser.add_argument("--train-metric", default=None)
    parser.add_argument("--log-train", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    runs_root = (root / args.runs_root).resolve()
    results_dir = (root / args.results_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        label: latest_run_dir(runs_root, run_group)
        for label, run_group, _ in EXPERIMENTS
    }

    train_summary = plot_training(args, run_dirs, output_dir)
    test_rows = read_test_rows(results_dir)
    plot_test(args, test_rows, output_dir)

    print("Wrote figures and summaries to", output_dir)
    print("\nTraining summary")
    for row in train_summary:
        print(row)
    print("\nTest summary")
    for row in test_rows:
        print(
            {
                "Experiment": row.get("Experiment"),
                "num_samples": row.get("num_samples"),
                "rho_counts": row.get("rho_counts"),
                "PBL Mean": row.get("PBL Mean"),
                "PBL Max": row.get("PBL Max"),
                "Relative PBL Mean": row.get("Relative PBL Mean"),
            }
        )


if __name__ == "__main__":
    main()
