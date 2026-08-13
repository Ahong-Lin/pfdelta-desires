#!/usr/bin/env python3
"""Plot PBL bars from a test_error_rho CSV output."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_RULES = [
    ("60L80R", "60L-80R"),
    ("80L60R", "80L-60R"),
    ("100L48R", "100L-48R"),
    ("rho1_baseline", "rho=1 baseline"),
]

ORDER = ["60L-80R", "80L-60R", "100L-48R", "rho=1 baseline"]
COLORS = {
    "60L-80R": "#2563eb",
    "80L-60R": "#059669",
    "100L-48R": "#dc2626",
    "rho=1 baseline": "#7c3aed",
}


def infer_label(row: dict) -> str:
    text = " ".join(str(value) for value in row.values())
    for needle, label in LABEL_RULES:
        if needle in text:
            return label
    return row.get("run_dir", "unknown")


def to_float(value):
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return None if math.isnan(out) else out
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="rho_test_results/figures")
    parser.add_argument("--stem", default="rho_branch7_random13_all4_test_pbl")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    for row in rows:
        row["Experiment"] = infer_label(row)
    rows.sort(key=lambda row: ORDER.index(row["Experiment"]) if row["Experiment"] in ORDER else 999)

    summary_fields = [
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
    summary_csv = output_dir / f"{args.stem}_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in summary_fields})

    labels = [row["Experiment"] for row in rows]
    pbl_mean = [to_float(row.get("PBL Mean")) for row in rows]
    pbl_max = [to_float(row.get("PBL Max")) for row in rows]
    rel = [to_float(row.get("Relative PBL Mean")) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colors = [COLORS.get(label, "#6b7280") for label in labels]

    axes[0].bar(labels, pbl_mean, color=colors, width=0.62)
    axes[0].set_title("rho=0 random-13 test PBL Mean")
    axes[0].set_ylabel("PBL Mean")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(pbl_mean):
        if value is not None:
            axes[0].text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)

    metric2 = rel if any(value is not None for value in rel) else pbl_max
    metric2_name = "Relative PBL Mean" if any(value is not None for value in rel) else "PBL Max"
    axes[1].bar(labels, metric2, color=colors, width=0.62)
    axes[1].set_title(f"rho=0 random-13 test {metric2_name}")
    axes[1].set_ylabel(metric2_name)
    axes[1].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(metric2):
        if value is not None:
            axes[1].text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("CANOS branch7 rho=0 random-13 test comparison, case30", y=1.03)
    fig.tight_layout()

    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output_dir / f"{args.stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {summary_csv}")
    for ext in ["png", "svg", "pdf"]:
        print(output_dir / f"{args.stem}.{ext}")
    print("Summary rows:")
    for row in rows:
        print({field: row.get(field) for field in summary_fields})


if __name__ == "__main__":
    main()
