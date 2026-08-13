#!/usr/bin/env python3
"""Plot branch-7 rho experiment training curves and PBL test comparisons."""

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
    ("rho=1", "canos_case30_rho_pf_branch7_B_rho1_baseline_41976"),
    ("endpoint", "canos_case30_rho_pf_branch7_endpoint_mix_41976"),
    ("branch7-rho6", "canos_case30_rho_pf_branch7_B_rho6"),
    ("branch7-rho11", "canos_case30_rho_pf_branch7_B_rho11"),
    ("multi6-disjoint", "canos_case30_rho_pf_branchset6_B_disjoint_rho6_41976"),
    ("multi6-shared", "canos_case30_rho_pf_branchset6_B_shared_rho6_41976"),
    ("allbranch", "canos_case30_rho_pf_allbranch_B_balanced_rho6_41976"),
]

TESTS = ["T7-2000", "T6-2000", "T34-2000"]

COLORS = {
    "rho=1": "#6b7280",
    "endpoint": "#d97706",
    "branch7-rho6": "#2563eb",
    "branch7-rho11": "#059669",
    "multi6-disjoint": "#7c3aed",
    "multi6-shared": "#db2777",
    "allbranch": "#dc2626",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_run_dir(runs_root: Path, run_group: str) -> Path:
    group = runs_root / run_group
    candidates = [
        child
        for child in group.iterdir()
        if child.is_dir()
        and (child / "model.pt").exists()
        and (child / "summary.json").exists()
        and (child / "train.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete run found under {group}")
    return max(candidates, key=lambda path: (path / "summary.json").stat().st_mtime)


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


def get_curve(run_dir: Path, preferred: str | None = None):
    data = read_json(run_dir / "train.json")
    keys = sorted(data, key=lambda item: int(item))
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
        return None if math.isnan(out) else out
    except (TypeError, ValueError):
        return None


def read_test_rows(results_dir: Path) -> list[dict]:
    rows = []
    for test_name in TESTS:
        csv_path = results_dir / f"{test_name}_pbl_summary.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                run_dir = row["run_dir"]
                label = None
                for exp_label, group in EXPERIMENTS:
                    if group in run_dir:
                        label = exp_label
                        break
                if label is None:
                    label = Path(run_dir).parent.name
                rows.append({"Test": test_name, "Experiment": label, **row})
    return rows


def plot_training(args, runs_root: Path, output_dir: Path) -> list[dict]:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    summary_rows = []

    for label, group in EXPERIMENTS:
        run_dir = latest_run_dir(runs_root, group)
        xs, ys, metric = get_curve(run_dir, args.train_metric)
        summary = read_json(run_dir / "summary.json")
        best_epoch = int(summary.get("best_point", 0)) + 1
        ax.plot(xs, ys, label=label, color=COLORS[label], linewidth=1.9)
        summary_rows.append(
            {
                "Experiment": label,
                "Run dir": str(run_dir),
                "Train metric": metric,
                "Epochs": xs[-1],
                "Best epoch": best_epoch,
                "Initial train loss": ys[0],
                "Final train loss": ys[-1],
                "Min train loss": min(ys),
            }
        )

    ax.set_title("CANOS case30 branch7 rho experiments: training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(summary_rows[0]["Train metric"])
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    if args.log_train:
        ax.set_yscale("log")
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output_dir / f"training_loss_all_candidates.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "training_loss_all_candidates_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def plot_pbl(rows: list[dict], output_dir: Path) -> None:
    summary_fields = [
        "Test",
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
    rows.sort(
        key=lambda row: (
            TESTS.index(row["Test"]),
            [label for label, _ in EXPERIMENTS].index(row["Experiment"]),
        )
    )
    with (output_dir / "pbl_all_candidates_all_tests_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in summary_fields})

    labels = [label for label, _ in EXPERIMENTS]
    x = list(range(len(labels)))
    width = 0.24

    for metric, stem in [
        ("PBL Mean", "pbl_mean_all_tests"),
        ("Relative PBL Mean", "relative_pbl_mean_all_tests"),
    ]:
        fig, ax = plt.subplots(figsize=(12.0, 5.4))
        for test_idx, test_name in enumerate(TESTS):
            values = []
            for label in labels:
                match = next(row for row in rows if row["Test"] == test_name and row["Experiment"] == label)
                values.append(to_float(match.get(metric)))
            offsets = [pos + (test_idx - 1) * width for pos in x]
            ax.bar(offsets, values, width=width, label=test_name, alpha=0.88)
            for xpos, value in zip(offsets, values):
                if value is not None:
                    ax.text(xpos, value, f"{value:.2g}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_title(f"CANOS case30 branch7 rho experiments: {metric}")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_design_table(output_dir: Path) -> None:
    rows = [
        {
            "Experiment": "rho=1",
            "Training design": "41976 branch7 rho=1 samples",
            "Seen branches": "7",
            "Train rho": "1",
        },
        {
            "Experiment": "endpoint",
            "Training design": "20988 loads x branch7 x {rho=0,rho=1}",
            "Seen branches": "7",
            "Train rho": "0,1",
        },
        {
            "Experiment": "branch7-rho6",
            "Training design": "6996 loads x branch7 x rho={0,0.2,...,1}",
            "Seen branches": "7",
            "Train rho": "0,0.2,0.4,0.6,0.8,1",
        },
        {
            "Experiment": "branch7-rho11",
            "Training design": "3816 loads x branch7 x rho=0:0.1:1",
            "Seen branches": "7",
            "Train rho": "0:0.1:1",
        },
        {
            "Experiment": "multi6-disjoint",
            "Training design": "6996 loads, each load assigned to one of 6 branches, rho6",
            "Seen branches": "4,5,7,20,21,23",
            "Train rho": "0,0.2,0.4,0.6,0.8,1",
        },
        {
            "Experiment": "multi6-shared",
            "Training design": "1166 loads x 6 branches x rho6",
            "Seen branches": "4,5,7,20,21,23",
            "Train rho": "0,0.2,0.4,0.6,0.8,1",
        },
        {
            "Experiment": "allbranch",
            "Training design": "6996 load-branch pairs balanced across 34 branches x rho6",
            "Seen branches": "34 normal-loop branches",
            "Train rho": "0,0.2,0.4,0.6,0.8,1",
        },
    ]
    with (output_dir / "candidate_design_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--results-dir", default="rho_test_results/branch7_diversity_2000")
    parser.add_argument("--output-dir", default="rho_test_results/branch7_diversity_2000/figures")
    parser.add_argument("--train-metric", default=None)
    parser.add_argument("--log-train", action="store_true")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_design_table(output_dir)
    training_rows = plot_training(args, runs_root, output_dir)
    pbl_rows = read_test_rows(results_dir)
    plot_pbl(pbl_rows, output_dir)

    print(f"Wrote figures to {output_dir}")
    print("Training final losses:")
    for row in training_rows:
        print(row["Experiment"], row["Final train loss"])
    print("PBL rows:")
    for row in pbl_rows:
        print(row["Test"], row["Experiment"], row.get("PBL Mean"), row.get("Relative PBL Mean"))


if __name__ == "__main__":
    main()
