#!/usr/bin/env python3
"""Plot branch-7-only baseline tests on T7-rho0 and T7-rho1."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS = [
    (
        "rho=1",
        "canos_case30_rho_pf_branch7_B_rho1_baseline_41976",
        "41976 samples, rho=1 only",
    ),
    (
        "01-42k",
        "canos_case30_rho_pf_branch7_endpoint_mix_41976",
        "41976 samples, rho in {0,1}",
    ),
    (
        "01-14k",
        "canos_case30_rho_pf_branch7_B_rho6_endpoints_13992",
        "13992 samples, same loads as rho6, rho in {0,1}",
    ),
    (
        "rho6",
        "canos_case30_rho_pf_branch7_B_rho6",
        "41976 samples, rho={0,0.2,0.4,0.6,0.8,1}",
    ),
    (
        "rho11",
        "canos_case30_rho_pf_branch7_B_rho11",
        "41976 samples, rho=0:0.1:1",
    ),
]

TESTS = [
    ("T7-rho0", "T7-rho0_pbl_summary.csv"),
    ("T7-rho1", "T7-rho1_pbl_summary.csv"),
]

COLORS = {
    "rho=1": "#6b7280",
    "01-42k": "#d97706",
    "01-14k": "#f59e0b",
    "rho6": "#2563eb",
    "rho11": "#059669",
}


def to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def label_for_run(run_dir: str) -> str:
    for label, token, _ in EXPERIMENTS:
        if token in run_dir:
            return label
    return Path(run_dir).parent.name


def read_rows(input_dir: Path) -> list[dict]:
    rows = []
    for test_name, filename in TESTS:
        with (input_dir / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"Test": test_name, "Experiment": label_for_run(row["run_dir"]), **row})
    order = {label: idx for idx, (label, _, _) in enumerate(EXPERIMENTS)}
    test_order = {test: idx for idx, (test, _) in enumerate(TESTS)}
    rows.sort(key=lambda row: (test_order[row["Test"]], order[row["Experiment"]]))
    return rows


def write_summary(rows: list[dict], output_dir: Path) -> None:
    fields = [
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
    with (output_dir / "branch7_only_T7_rho0_rho1_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_design(output_dir: Path) -> None:
    with (output_dir / "branch7_only_design_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["Experiment", "Training design"])
        writer.writeheader()
        for label, _, design in EXPERIMENTS:
            writer.writerow({"Experiment": label, "Training design": design})


def plot_metric(rows: list[dict], output_dir: Path, metric: str, stem: str) -> None:
    labels = [label for label, _, _ in EXPERIMENTS]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=False)

    for ax, (test_name, _) in zip(axes, TESTS):
        test_rows = [row for row in rows if row["Test"] == test_name]
        values_by_label = {row["Experiment"]: to_float(row.get(metric)) for row in test_rows}
        values = [values_by_label[label] for label in labels]
        bars = ax.bar(
            labels,
            values,
            color=[COLORS[label] for label in labels],
            alpha=0.9,
            width=0.68,
        )
        ax.set_title(test_name)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=24)
        for bar, value in zip(bars, values):
            if value is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    fig.suptitle(f"Branch 7 only: {metric} on rho=0 and rho=1 tests", y=1.02)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="rho_test_results/rho_baseline_test_branch7_only")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_dir)
    write_summary(rows, output_dir)
    write_design(output_dir)
    plot_metric(rows, output_dir, "PBL Mean", "branch7_only_pbl_mean")
    plot_metric(rows, output_dir, "Relative PBL Mean", "branch7_only_relative_pbl_mean")
    plot_metric(rows, output_dir, "PBL Max", "branch7_only_pbl_max")

    print(f"Wrote branch7-only figures to {output_dir}")


if __name__ == "__main__":
    main()
