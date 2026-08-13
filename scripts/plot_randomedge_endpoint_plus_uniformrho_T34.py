#!/usr/bin/env python3
"""Plot random-edge endpoint plus uniform-rho T34 comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = [
    (
        "endpoint-42k",
        "runs/canos_case30_rho_pf_endpoint_pfdelta_style_randomedge_unpaired_41976/"
        "canos_case30_rho_pf_endpoint_pfdelta_style_randomedge_unpaired_41976_ep300_260725_064017",
    ),
    (
        "+rho 5k",
        "runs/canos_case30_rho_pf_endpoint_plus_uniformrho_extra5k_46976/"
        "canos_case30_rho_pf_endpoint_plus_uniformrho_extra5k_46976_ep300_260725_162442",
    ),
    (
        "+rho 10k",
        "runs/canos_case30_rho_pf_endpoint_plus_uniformrho_extra10k_51976/"
        "canos_case30_rho_pf_endpoint_plus_uniformrho_extra10k_51976_ep300_260725_162444",
    ),
    (
        "+rho 20k",
        "runs/canos_case30_rho_pf_endpoint_plus_uniformrho_extra20k_61976/"
        "canos_case30_rho_pf_endpoint_plus_uniformrho_extra20k_61976_ep300_260725_162446",
    ),
]

COLORS = {
    "endpoint-42k": "#4b5563",
    "+rho 5k": "#2563eb",
    "+rho 10k": "#059669",
    "+rho 20k": "#dc2626",
}

TEST_FILES = {
    "T34-rho0": "T34-rho0_pbl_summary.csv",
    "T34-rho1": "T34-rho1_pbl_summary.csv",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_metric_from_row(row: dict) -> str:
    for key, value in row.items():
        try:
            float(value)
            return key
        except (TypeError, ValueError):
            continue
    raise KeyError(f"No numeric train metric in row: {row}")


def read_train_curve(run_dir: Path) -> tuple[list[int], list[float], str]:
    data = read_json(run_dir / "train.json")
    keys = sorted(data, key=lambda key: int(key))
    first = data[keys[0]]
    if isinstance(first, list):
        first = first[0]
    metric = "canos_pf_mse" if "canos_pf_mse" in first else train_metric_from_row(first)
    xs, ys = [], []
    for key in keys:
        row = data[key]
        if isinstance(row, list):
            row = row[0]
        if metric in row:
            xs.append(int(key) + 1)
            ys.append(float(row[metric]))
    return xs, ys, metric


def label_for_run(run_dir: str) -> str:
    for label, needle in MODELS:
        if needle in run_dir:
            return label
    for label, needle in MODELS:
        if Path(needle).parent.name in run_dir or Path(needle).name in run_dir:
            return label
    return Path(run_dir).parent.name


def read_test_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for test_name, filename in TEST_FILES.items():
        with (results_dir / filename).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["test"] = test_name
                row["label"] = label_for_run(row["run_dir"])
                rows.append(row)
    return rows


def write_combined_summary(rows: list[dict], output_dir: Path) -> None:
    fields = [
        "test",
        "label",
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
    with (output_dir / "combined_T34_pbl_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_test_bars(rows: list[dict], output_dir: Path) -> None:
    labels = [label for label, _ in MODELS]
    tests = list(TEST_FILES)
    x = list(range(len(labels)))
    width = 0.34

    for metric, stem, ylabel in [
        ("PBL Mean", "T34_pbl_mean_comparison", "PBL Mean"),
        ("Relative PBL Mean", "T34_relative_pbl_mean_comparison", "Relative PBL Mean"),
    ]:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        for test_idx, test_name in enumerate(tests):
            values = []
            for label in labels:
                match = next(row for row in rows if row["test"] == test_name and row["label"] == label)
                values.append(float(match[metric]))
            positions = [pos + (test_idx - 0.5) * width for pos in x]
            color = "#0f766e" if test_name == "T34-rho0" else "#7c3aed"
            ax.bar(positions, values, width=width, label=test_name, color=color, alpha=0.86)
            for xpos, value in zip(positions, values):
                ax.text(xpos, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)

        ax.set_title("Random-edge endpoint baseline + random-rho augmentation")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_training(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    summary_rows = []
    for label, run_dir_text in MODELS:
        run_dir = Path(run_dir_text)
        xs, ys, metric = read_train_curve(run_dir)
        summary = read_json(run_dir / "summary.json")
        best_epoch = int(summary.get("best_point", 0)) + 1
        ax.plot(xs, ys, label=label, color=COLORS[label], linewidth=2.0)
        summary_rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "metric": metric,
                "epochs": xs[-1],
                "best_epoch": best_epoch,
                "initial_train_loss": ys[0],
                "final_train_loss": ys[-1],
                "min_train_loss": min(ys),
            }
        )
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(summary_rows[0]["metric"])
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"training_loss_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "training_loss_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="rho_test_results/randomedge_endpoint_plus_uniformrho_T34")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_test_rows(results_dir)
    write_combined_summary(rows, output_dir)
    plot_test_bars(rows, output_dir)
    plot_training(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
