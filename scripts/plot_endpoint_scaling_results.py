#!/usr/bin/env python3
"""Plot fixed-test endpoint scaling results for the case30 CANOS runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIZES = [6_000, 10_000, 20_000, 40_000, 45_000, 50_000, 80_000, 160_000, 320_000]
TESTS = ["T34-final-rho0", "T34-final-rho1"]
TEST_STYLES = {
    "T34-final-rho0": {"label": "T34 final rho=0 (N-1 outage)", "color": "#b91c1c"},
    "T34-final-rho1": {"label": "T34 final rho=1 (normal topology)", "color": "#2563eb"},
}
COLORS = ["#475569", "#0284c7", "#059669", "#d97706", "#7c3aed", "#db2777"]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def complete_run(runs_root: Path, size: int) -> Path:
    group = runs_root / f"canos_case30_rho_scaling_endpoint_{size // 1000}k"
    candidates = [
        path
        for path in group.iterdir()
        if path.is_dir()
        and (path / "model.pt").is_file()
        and (path / "summary.json").is_file()
        and (path / "train.json").is_file()
        and (path / "val.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No finished run found under {group}")
    return max(candidates, key=lambda path: (path / "summary.json").stat().st_mtime)


def train_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    records = read_json(run_dir / "train.json")
    points = sorted(records, key=lambda key: int(key))
    xs, ys = [], []
    for point in points:
        row = records[point]
        row = row[0] if isinstance(row, list) else row
        if "canos_pf_mse" in row:
            xs.append(int(point) + 1)
            ys.append(float(row["canos_pf_mse"]))
    return xs, ys


def val_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    records = read_json(run_dir / "val.json")
    points = sorted(records, key=lambda key: int(key))
    xs, ys = [], []
    for point in points:
        rows = records[point]
        row = rows[0] if isinstance(rows, list) else rows
        if "canos_pf_mse" in row:
            xs.append(int(point) + 1)
            ys.append(float(row["canos_pf_mse"]))
    return xs, ys


def scale_label(size: int) -> str:
    return f"{size // 1000}k"


def size_from_run_dir(run_dir: str, sizes: list[int]) -> int:
    for size in sizes:
        if f"endpoint_{size // 1000}k" in run_dir:
            return size
    raise ValueError(f"Cannot infer train size from {run_dir}")


def read_test_results(results_dir: Path, sizes: list[int]) -> list[dict]:
    rows: list[dict] = []
    for test in TESTS:
        path = results_dir / f"{test}_pbl_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation summary: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["test"] = test
                row["train_samples"] = size_from_run_dir(row["run_dir"], sizes)
                rows.append(row)
    return rows


def metric(value: str | None) -> float:
    if value in {None, ""}:
        return math.nan
    return float(value)


def write_summary(rows: list[dict], run_dirs: dict[int, Path], sizes: list[int], output_dir: Path) -> None:
    fields = [
        "test",
        "train_samples",
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
    with (output_dir / "final_test_pbl_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda row: (TESTS.index(row["test"]), row["train_samples"])):
            writer.writerow({field: row.get(field, "") for field in fields})

    curve_fields = [
        "train_samples",
        "run_dir",
        "best_epoch",
        "final_train_mse",
        "min_train_mse",
        "best_val_mse",
    ]
    with (output_dir / "training_curve_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        for size in sizes:
            run_dir = run_dirs[size]
            _, train = train_curve(run_dir)
            _, val = val_curve(run_dir)
            summary = read_json(run_dir / "summary.json")
            writer.writerow(
                {
                    "train_samples": size,
                    "run_dir": run_dir,
                    "best_epoch": int(summary["best_point"]) + 1,
                    "final_train_mse": train[-1],
                    "min_train_mse": min(train),
                    "best_val_mse": min(val),
                }
            )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_test(rows: list[dict], sizes: list[int], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for axis, metric_name in zip(axes, ["PBL Mean", "Relative PBL Mean"]):
        for test in TESTS:
            test_rows = sorted((row for row in rows if row["test"] == test), key=lambda row: row["train_samples"])
            values = [metric(row[metric_name]) for row in test_rows]
            style = TEST_STYLES[test]
            axis.plot(
                sizes,
                values,
                marker="o",
                markersize=5.5,
                linewidth=2.1,
                color=style["color"],
                label=style["label"],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes, [scale_label(size) for size in sizes])
        axis.tick_params(axis="x", labelrotation=35, labelsize=8)
        axis.set_xlabel("Training samples (rho=0/1 endpoint pairs)")
        axis.set_ylabel(metric_name)
        axis.set_title(f"Final test: {metric_name} (lower is better)")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8.5)
    fig.suptitle("CANOS case30 endpoint scaling on fixed 34-branch held-out loads", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "final_test_pbl_vs_training_size")


def plot_relative_pbl(rows: list[dict], sizes: list[int], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.7))
    for test in TESTS:
        test_rows = sorted((row for row in rows if row["test"] == test), key=lambda row: row["train_samples"])
        style = TEST_STYLES[test]
        axis.plot(
            sizes,
            [metric(row["Relative PBL Mean"]) for row in test_rows],
            marker="o",
            markersize=6,
            linewidth=2.25,
            color=style["color"],
            label=style["label"],
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes, [scale_label(size) for size in sizes])
    axis.tick_params(axis="x", labelrotation=35, labelsize=8)
    axis.set_xlabel("Training samples (rho=0/1 endpoint pairs)")
    axis.set_ylabel("Relative PBL Mean")
    axis.set_title("Final test relative PBL vs training size (lower is better)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "final_test_relative_pbl_vs_training_size")


def plot_training_curves(run_dirs: dict[int, Path], sizes: list[int], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharex=True)
    color_map = plt.get_cmap("tab10")
    for index, size in enumerate(sizes):
        color = COLORS[index] if index < len(COLORS) else color_map(index % color_map.N)
        xs, train = train_curve(run_dirs[size])
        val_xs, val = val_curve(run_dirs[size])
        label = scale_label(size)
        axes[0].plot(xs, train, color=color, linewidth=1.75, label=label)
        axes[1].plot(val_xs, val, color=color, marker="o", markersize=2.8, linewidth=1.55, label=label)

    for axis, title, ylabel in [
        (axes[0], "Training loss", "Train CANOS PF MSE"),
        (axes[1], "Fixed validation loss", "Validation CANOS PF MSE"),
    ]:
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(title="Train size", frameon=False)
    fig.suptitle("CANOS case30 endpoint scaling learning curves", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "training_and_validation_loss")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="rho_test_results/endpoint_scaling_T34_final")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in SIZES),
        help="Comma-separated endpoint training-record counts to include.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    if not sizes or any(size <= 0 or size % 2 for size in sizes):
        raise ValueError("--sizes must contain positive even integers")
    sizes = sorted(set(sizes))
    run_dirs = {size: complete_run(Path(args.runs_root), size) for size in sizes}
    rows = read_test_results(results_dir, sizes)

    write_summary(rows, run_dirs, sizes, output_dir)
    plot_final_test(rows, sizes, output_dir)
    plot_relative_pbl(rows, sizes, output_dir)
    plot_training_curves(run_dirs, sizes, output_dir)
    print(f"Wrote endpoint scaling figures to {output_dir}")


if __name__ == "__main__":
    main()
