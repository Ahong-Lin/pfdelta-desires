import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

MODEL_ORDER = ["gns", "pfnet", "canos", "flowcan"]
MODEL_LABELS = {
    "gns": "GNS",
    "pfnet": "PFNet",
    "canos": "CANOS",
    "flowcan": "FlowCAN",
}
MODEL_COLORS = {
    "gns": "#D55E00",
    "pfnet": "#0072B2",
    "canos": "#009E73",
    "flowcan": "#CC79A7",
}
BASE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot case118 100-epoch learning curves with best-so-far padding."
    )
    parser.add_argument("--selected-runs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=99)
    return parser.parse_args()


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sorted_epoch_items(data):
    return [(int(epoch), data[epoch]) for epoch in sorted(data.keys(), key=int)]


def val_pbl_curve(run_dir):
    data = read_json(run_dir / "val.json")
    curve = {}
    for epoch, row in sorted_epoch_items(data):
        if isinstance(row, dict):
            if "PBL Mean" in row:
                curve[epoch] = float(row["PBL Mean"])
            continue
        values = [
            float(item["PBL Mean"])
            for item in row
            if isinstance(item, dict) and "PBL Mean" in item
        ]
        if values:
            curve[epoch] = float(np.mean(values))
    return curve


def infer_training_loss_key(first_row):
    preferred = [
        "gns_last_loss",
        "canos_pf_mse+pf_constraint_violation",
        "Masked MSE, reg.",
        "Masked MSE",
        "MSELoss",
    ]
    for key in preferred:
        if key in first_row:
            return key
    for key in first_row:
        if "PBL" not in key:
            return key
    return next(iter(first_row))


def training_loss_curve(run_dir):
    data = read_json(run_dir / "train.json")
    first_epoch = sorted(data.keys(), key=int)[0]
    metric_key = infer_training_loss_key(data[first_epoch])
    curve = {
        epoch: float(row[metric_key])
        for epoch, row in sorted_epoch_items(data)
        if metric_key in row
    }
    return curve, metric_key


def pad_best_so_far(curve, max_epoch):
    padded = []
    best = None
    for epoch in range(max_epoch + 1):
        if epoch in curve:
            value = curve[epoch]
            best = value if best is None else min(best, value)
        elif best is None:
            future_epochs = [candidate for candidate in curve if candidate > epoch]
            if future_epochs:
                first_future = min(future_epochs)
                best = curve[first_future]
            else:
                best = np.nan
        padded.append(best)
    return np.asarray(padded, dtype=float)


def grouped_curves(selected_runs, metric, max_epoch):
    groups = {model: [] for model in MODEL_ORDER}
    details = []

    for item in selected_runs:
        model = item["model"]
        run_dir = Path(item["run_dir"])
        if metric == "learning_curve":
            curve = val_pbl_curve(run_dir)
            metric_key = "Validation PBL Mean"
        else:
            curve, metric_key = training_loss_curve(run_dir)
        padded = pad_best_so_far(curve, max_epoch)
        groups[model].append(padded)
        details.append(
            {
                **item,
                "metric_key": metric_key,
                "first_epoch": min(curve) if curve else None,
                "last_epoch": max(curve) if curve else None,
                "best_epoch_in_metric": (
                    min(curve, key=curve.get) if curve else None
                ),
                "best_value_in_metric": min(curve.values()) if curve else None,
            }
        )
    return groups, details


def plot_metric(selected_runs, output_dir, max_epoch, metric):
    groups, details = grouped_curves(selected_runs, metric, max_epoch)
    epochs = np.arange(max_epoch + 1)

    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for model in MODEL_ORDER:
        curves = groups[model]
        if not curves:
            continue
        arr = np.vstack(curves)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        ax.plot(
            epochs + 1,
            mean,
            color=MODEL_COLORS[model],
            linewidth=3.0,
            label=MODEL_LABELS[model],
        )
        ax.fill_between(
            epochs + 1,
            mean - std,
            mean + std,
            color=MODEL_COLORS[model],
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlim(1, max_epoch + 1)
    ax.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.0,
        labelspacing=0.5,
    )
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    if metric == "learning_curve":
        ax.set_ylabel("PBL Mean", fontsize=AXIS_LABEL_FONT_SIZE)
        basename = "case118_task1_1_ep100_learning_curve"
    else:
        ax.set_yscale("log")
        ax.set_ylabel("Training loss", fontsize=AXIS_LABEL_FONT_SIZE)
        basename = "case118_task1_1_ep100_training_loss_log"

    # Match the same-canvas reference figures: keep the axes box fixed across
    # linear/log y scales instead of letting tight_layout vary the margins.
    fig.subplots_adjust(left=0.155, right=0.944, bottom=0.185, top=0.937)
    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)

    details_path = output_dir / f"{basename}_curve_details.json"
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2)

    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {details_path}")


def main():
    args = parse_args()
    selected_runs = read_json(Path(args.selected_runs))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_metric(selected_runs, output_dir, args.max_epoch, "learning_curve")
    plot_metric(selected_runs, output_dir, args.max_epoch, "training_loss_log")


if __name__ == "__main__":
    main()
