#!/usr/bin/env python3
"""Plot per-test comparison panels for branch7 rho diversity experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PANELS = {
    "T7-2000": {
        "title": "T7-2000: branch7 rho=0",
        "purpose": "Branch7-specific interpolation and endpoint baselines",
        "experiments": ["rho=1", "endpoint", "branch7-rho6", "branch7-rho11"],
    },
    "T6-2000": {
        "title": "T6-2000: six-branch rho=0",
        "purpose": "Branch diversity on the 6-branch test set",
        "experiments": ["rho=1", "branch7-rho6", "multi6-disjoint", "multi6-shared", "allbranch"],
    },
    "T34-2000": {
        "title": "T34-2000: all normal-loop branches rho=0",
        "purpose": "Cross-branch generalization",
        "experiments": ["rho=1", "branch7-rho6", "multi6-disjoint", "multi6-shared", "allbranch"],
    },
}

COLORS = {
    "rho=1": "#6b7280",
    "endpoint": "#d97706",
    "branch7-rho6": "#2563eb",
    "branch7-rho11": "#059669",
    "multi6-disjoint": "#7c3aed",
    "multi6-shared": "#db2777",
    "allbranch": "#dc2626",
}


def to_float(value: str | None) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return None if math.isnan(out) else out
    except ValueError:
        return None


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(test_name: str, panel: dict, rows: list[dict], output_dir: Path) -> list[dict]:
    selected = []
    by_experiment = {
        row["Experiment"]: row
        for row in rows
        if row["Test"] == test_name
    }
    for experiment in panel["experiments"]:
        if experiment not in by_experiment:
            raise KeyError(f"Missing row for {test_name}/{experiment}")
        selected.append(by_experiment[experiment])

    labels = [row["Experiment"] for row in selected]
    pbl = [to_float(row.get("PBL Mean")) for row in selected]
    rel = [to_float(row.get("Relative PBL Mean")) for row in selected]
    colors = [COLORS[label] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for ax, values, metric in [
        (axes[0], pbl, "PBL Mean"),
        (axes[1], rel, "Relative PBL Mean"),
    ]:
        ax.bar(labels, values, color=colors, width=0.64)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            if value is not None:
                ax.text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=18)

    fig.suptitle(f"{panel['title']}\n{panel['purpose']}", y=1.08)
    fig.tight_layout()

    stem = test_name.replace("-", "_") + "_selected_candidates"
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return [
        {
            "Test": test_name,
            "Purpose": panel["purpose"],
            "Experiment": row["Experiment"],
            "PBL Mean": row.get("PBL Mean", ""),
            "Relative PBL Mean": row.get("Relative PBL Mean", ""),
            "PBL Max": row.get("PBL Max", ""),
            "Run dir": row.get("run_dir", ""),
        }
        for row in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="rho_test_results/branch7_diversity_2000/figures/pbl_all_candidates_all_tests_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="rho_test_results/branch7_diversity_2000/figures/per_test",
    )
    args = parser.parse_args()

    summary = Path(args.summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(summary)
    selected_rows = []
    for test_name, panel in PANELS.items():
        selected_rows.extend(plot_panel(test_name, panel, rows, output_dir))

    write_csv(
        output_dir / "per_test_selected_candidates_summary.csv",
        selected_rows,
        ["Test", "Purpose", "Experiment", "PBL Mean", "Relative PBL Mean", "PBL Max", "Run dir"],
    )

    print(f"Wrote per-test plots to {output_dir}")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
