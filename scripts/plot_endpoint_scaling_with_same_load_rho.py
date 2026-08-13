#!/usr/bin/env python3
"""Plot fixed T34 scaling results with same-load uniform-rho augmentation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TESTS = ("T34-final-rho0", "T34-final-rho1")
TEST_LABELS = {
    "T34-final-rho0": "T34 held-out rho=0 (N-1 outage)",
    "T34-final-rho1": "T34 held-out rho=1 (normal topology)",
}
METRICS = ("PBL Mean", "Relative PBL Mean")
# The 45k run is an optimization outlier and is intentionally omitted from
# the main scaling comparison. Its raw result remains in the source CSV.
ENDPOINT_SIZES = (6_000, 10_000, 20_000, 40_000, 50_000, 80_000, 160_000, 320_000)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def endpoint_rows(endpoint_dir: Path) -> list[dict]:
    sources = {
        "T34-final-rho0": (endpoint_dir / "T34-final-rho0_pbl_summary.csv",),
        "T34-final-rho1": (endpoint_dir / "T34-final-rho1_pbl_summary.csv",),
    }
    rows: list[dict] = []
    pattern = re.compile(r"endpoint_(\d+)k")
    for test, paths in sources.items():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing endpoint evaluation: {path}")
            for source_row in read_csv(path):
                match = pattern.search(source_row["run_dir"])
                if match is None:
                    raise ValueError(f"Cannot infer endpoint size from {source_row['run_dir']}")
                size = int(match.group(1)) * 1_000
                if size not in ENDPOINT_SIZES:
                    continue
                row = dict(source_row)
                row.update(
                    {
                        "test": test,
                        "variant": "endpoint-only",
                        "endpoint_samples": size,
                        "midrho_samples": 0,
                        "total_train_samples": size,
                    }
                )
                rows.append(row)
    for test in TESTS:
        test_sizes = {int(row["total_train_samples"]) for row in rows if row["test"] == test}
        if test_sizes != set(ENDPOINT_SIZES):
            raise ValueError(f"Endpoint rows for {test} are incomplete: {sorted(test_sizes)}")
    return rows


def rho_rows(rho_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for test in TESTS:
        path = rho_dir / f"{test}_pbl_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing rho-augmentation evaluation: {path}")
        for source_row in read_csv(path):
            if "uniformrho10k" in source_row["run_dir"]:
                midrho_samples = 10_000
            else:
                # This figure isolates the 40k + 10k mid-rho comparison.
                continue
            row = dict(source_row)
            row.update(
                {
                    "test": test,
                    "variant": "same-load-uniform-rho",
                    "endpoint_samples": 40_000,
                    "midrho_samples": midrho_samples,
                    "total_train_samples": 40_000 + midrho_samples,
                }
            )
            rows.append(row)
    return rows


def metric(row: dict, name: str) -> float:
    return float(row[name])


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_metrics(rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True)
    for column, test in enumerate(TESTS):
        endpoint = sorted(
            (row for row in rows if row["test"] == test and row["variant"] == "endpoint-only"),
            key=lambda row: int(row["total_train_samples"]),
        )
        rho = sorted(
            (row for row in rows if row["test"] == test and row["variant"] == "same-load-uniform-rho"),
            key=lambda row: int(row["total_train_samples"]),
        )
        endpoint_40k = next(row for row in endpoint if int(row["total_train_samples"]) == 40_000)
        rho_path = [endpoint_40k] + rho

        for row_index, metric_name in enumerate(METRICS):
            axis = axes[row_index, column]
            axis.plot(
                [int(row["total_train_samples"]) for row in endpoint],
                [metric(row, metric_name) for row in endpoint],
                color="#2563eb",
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                label="Endpoint only" if row_index == 0 and column == 0 else None,
            )
            axis.plot(
                [int(row["total_train_samples"]) for row in rho_path],
                [metric(row, metric_name) for row in rho_path],
                color="#9333ea",
                linestyle="--",
                marker="D",
                linewidth=2.0,
                markersize=5.5,
                label="40k endpoint + same-load mid-rho" if row_index == 0 and column == 0 else None,
            )
            for row in rho:
                total = int(row["total_train_samples"])
                label = f"40k+rho{int(row['midrho_samples']) // 1000}k"
                axis.annotate(
                    label,
                    (total, metric(row, metric_name)),
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="#6b21a8",
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(ENDPOINT_SIZES, [f"{size // 1000}k" for size in ENDPOINT_SIZES])
            axis.tick_params(axis="x", labelrotation=35, labelsize=8)
            axis.grid(True, alpha=0.25)
            axis.set_title(TEST_LABELS[test])
            axis.set_ylabel(f"{metric_name}\n(lower is better)")
            if row_index == 1:
                axis.set_xlabel("Total training samples")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Case30 fixed T34_6800: endpoint scaling vs same-load uniform-rho augmentation", y=1.07)
    fig.tight_layout()
    save_figure(fig, output_dir, "endpoint_scaling_vs_same_load_uniformrho_T34_final_6800")


def curve(run_dir: Path, split: str) -> tuple[list[int], list[float]]:
    payload = read_json(run_dir / f"{split}.json")
    xs, ys = [], []
    for key in sorted(payload, key=int):
        value = payload[key]
        row = value[0] if isinstance(value, list) else value
        if "canos_pf_mse" in row:
            xs.append(int(key) + 1)
            ys.append(float(row["canos_pf_mse"]))
    return xs, ys


def plot_learning_curves(rows: list[dict], output_dir: Path) -> None:
    unique_runs = {row["run_dir"]: row for row in rows}
    ordered = sorted(
        unique_runs.values(),
        key=lambda row: (int(row["endpoint_samples"]), int(row["midrho_samples"])),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    endpoint_colors = plt.get_cmap("Blues")
    for index, row in enumerate(ordered):
        is_rho = row["variant"] == "same-load-uniform-rho"
        if is_rho:
            color = "#9333ea" if int(row["midrho_samples"]) == 5_000 else "#7e22ce"
            label = f"40k+rho{int(row['midrho_samples']) // 1000}k"
            linestyle = "--"
        else:
            color = endpoint_colors(0.35 + 0.5 * index / max(len(ENDPOINT_SIZES) - 1, 1))
            label = f"endpoint {int(row['total_train_samples']) // 1000}k"
            linestyle = "-"
        run_dir = Path(row["run_dir"])
        for axis, split, title in zip(axes, ("train", "val"), ("Training loss", "Fixed endpoint validation loss")):
            xs, ys = curve(run_dir, split)
            axis.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.4, label=label)
            axis.set_yscale("log")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("CANOS PF MSE")
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Case30 learning curves: endpoint scaling and same-load mid-rho augmentation", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "training_and_validation_loss_with_same_load_uniformrho")


def write_summary(rows: list[dict], output_dir: Path) -> None:
    fields = [
        "test",
        "variant",
        "endpoint_samples",
        "midrho_samples",
        "total_train_samples",
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
    with (output_dir / "combined_T34_final_6800_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda row: (TESTS.index(row["test"]), int(row["total_train_samples"]))):
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-dir", default="rho_test_results/endpoint_scaling_T34_final")
    parser.add_argument(
        "--rho-dir",
        default="rho_test_results/endpoint_scaling_with_same_load_uniformrho_T34_final_6800",
    )
    parser.add_argument(
        "--output-dir",
        default="rho_test_results/endpoint_scaling_with_same_load_uniformrho_T34_final_6800",
    )
    parser.add_argument(
        "--skip-learning-curves",
        action="store_true",
        help="Only write the final-test comparison when run directories are unavailable locally.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = endpoint_rows(Path(args.endpoint_dir)) + rho_rows(Path(args.rho_dir))
    write_summary(rows, output_dir)
    plot_final_metrics(rows, output_dir)
    if not args.skip_learning_curves:
        plot_learning_curves(rows, output_dir)
    print(f"Wrote figures and summary to {output_dir}")


if __name__ == "__main__":
    main()
