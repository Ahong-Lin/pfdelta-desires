import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

MODEL_ORDER = ["pfnet", "gns", "canos", "flowcan"]
MODEL_LABELS = {
    "pfnet": "PFNet",
    "gns": "GNS-S",
    "canos": "CANOS-PF",
    "flowcan": "FlowCAN",
}
MODEL_COLORS = {
    "pfnet": "#4C72B0",
    "gns": "#FF8C1A",
    "canos": "#55A868",
    "flowcan": "#E15759",
}
DEFAULT_TASKS = ["task1_1", "task2_2", "task2_3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot aggregated learning curves across selected runs."
    )
    parser.add_argument(
        "--selected-runs",
        required=True,
        help="Path to selected_runs.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--metric",
        choices=["val_pbl_mean", "training_target"],
        required=True,
        help="Metric family to plot.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Cases to include, e.g. case30 case118.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Tasks to include, e.g. task1_1 task2_2 task2_3.",
    )
    parser.add_argument(
        "--yscale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Optional output basename without extension.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw a figure-level title.",
    )
    parser.add_argument(
        "--no-panel-title",
        action="store_true",
        help="Do not draw subplot titles.",
    )
    parser.add_argument(
        "--panel-font-size",
        type=float,
        default=13,
        help="Font size for subplot titles.",
    )
    parser.add_argument(
        "--label-font-size",
        type=float,
        default=10,
        help="Font size for axis labels.",
    )
    parser.add_argument(
        "--tick-font-size",
        type=float,
        default=10,
        help="Font size for tick labels.",
    )
    parser.add_argument(
        "--legend-font-size",
        type=float,
        default=11,
        help="Font size for the legend.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Line width for mean curves.",
    )
    parser.add_argument(
        "--no-band",
        action="store_true",
        help="Do not draw standard-deviation bands.",
    )
    parser.add_argument(
        "--band-mode",
        choices=["linear", "log"],
        default="linear",
        help="How to compute standard-deviation bands.",
    )
    parser.add_argument(
        "--legend-location",
        choices=["figure_bottom", "upper_right"],
        default="figure_bottom",
        help="Where to draw the legend.",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Optional y-axis label override.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Optional figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Optional figure height in inches.",
    )
    parser.add_argument(
        "--no-tight-bbox",
        action="store_true",
        help="Save figures without bbox_inches='tight'.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional lower bound for the y-axis.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional upper bound for the y-axis.",
    )
    parser.add_argument(
        "--pad-policy",
        choices=["carry_last", "available_only"],
        default="carry_last",
        help="How to aggregate curves with different stopping epochs.",
    )
    parser.add_argument(
        "--extend-to-epoch",
        type=int,
        default=None,
        help="Carry each curve's last value forward up to this epoch.",
    )
    return parser.parse_args()


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def case_sort_key(case_name):
    suffix = case_name.replace("case", "")
    if suffix.isdigit():
        return int(suffix)
    return case_name


def task_sort_key(task_name):
    token = task_name.replace("task", "")
    parts = token.split("_")
    return tuple(int(part) for part in parts if part.isdigit())


def task_label(task_name):
    return "Task " + task_name.replace("task", "").replace("_", ".")


def infer_training_target_key(train_row, config):
    losses = config.get("optim", {}).get("train_params", {}).get("train_loss", [])
    candidate_keys = list(train_row.keys())
    first_loss = losses[0] if losses else None

    if isinstance(first_loss, str) and first_loss in train_row:
        return first_loss

    if isinstance(first_loss, dict):
        loss_name = first_loss.get("name")
        if loss_name == "combined_loss":
            combined_key = f"{first_loss.get('loss1')}+{first_loss.get('loss2')}"
            if combined_key in train_row:
                return combined_key
        torch_nn_name = first_loss.get("torch_nn_name")
        if torch_nn_name in train_row:
            return torch_nn_name
        if loss_name in train_row:
            return loss_name

    non_pbl_keys = [key for key in candidate_keys if key != "PBL Mean"]
    if len(non_pbl_keys) == 1:
        return non_pbl_keys[0]

    return candidate_keys[0]


def extract_val_pbl_mean(run_dir):
    val_data = read_json(run_dir / "val.json")
    curve = {}
    for epoch in sorted(val_data.keys(), key=int):
        epoch_data = val_data[epoch]
        if isinstance(epoch_data, dict):
            if "PBL Mean" in epoch_data:
                curve[int(epoch)] = float(epoch_data["PBL Mean"])
            continue

        pbl_values = [
            float(entry["PBL Mean"])
            for entry in epoch_data
            if isinstance(entry, dict) and "PBL Mean" in entry
        ]
        if pbl_values:
            curve[int(epoch)] = float(np.mean(pbl_values))

    return curve, "PBL Mean"


def extract_training_target(run_dir):
    train_data = read_json(run_dir / "train.json")
    config = read_json_like_yaml(run_dir / "config.yaml")
    first_epoch = sorted(train_data.keys(), key=int)[0]
    metric_key = infer_training_target_key(train_data[first_epoch], config)
    curve = {
        int(epoch): float(train_data[epoch][metric_key])
        for epoch in sorted(train_data.keys(), key=int)
        if metric_key in train_data[epoch]
    }
    return curve, metric_key


def read_json_like_yaml(path):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read config.yaml for training_target plots."
        ) from exc

    with open(path, "r") as f:
        return yaml.safe_load(f)


def aggregate_curves(curves, pad_policy, band_mode, extend_to_epoch=None):
    epochs = sorted({epoch for curve in curves for epoch in curve})
    if not curves:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if pad_policy == "carry_last":
        min_epoch = min(min(curve.keys()) for curve in curves)
        max_epoch = max(max(curve.keys()) for curve in curves)
        if extend_to_epoch is not None:
            max_epoch = max(max_epoch, extend_to_epoch)
        epochs = list(range(min_epoch, max_epoch + 1))

    means = []
    lower_bounds = []
    upper_bounds = []
    for epoch in epochs:
        values = []
        for curve in curves:
            if epoch in curve:
                values.append(curve[epoch])
                continue
            if pad_policy == "carry_last":
                previous_epochs = [prev_epoch for prev_epoch in curve if prev_epoch < epoch]
                if previous_epochs:
                    last_epoch = max(previous_epochs)
                    values.append(curve[last_epoch])
        values = np.array(values, dtype=float)
        mean = float(np.mean(values))
        means.append(mean)
        if band_mode == "log":
            safe_values = np.maximum(values, 1e-12)
            log_values = np.log10(safe_values)
            log_mean = float(np.mean(log_values))
            log_std = float(np.std(log_values))
            lower_bounds.append(float(10 ** (log_mean - log_std)))
            upper_bounds.append(float(10 ** (log_mean + log_std)))
        else:
            std = float(np.std(values))
            lower_bounds.append(mean - std)
            upper_bounds.append(mean + std)
    return (
        np.array(epochs),
        np.array(means),
        np.array(lower_bounds),
        np.array(upper_bounds),
    )


def load_grouped_runs(selected_runs, cases, tasks):
    grouped = {}
    for item in selected_runs:
        case_name = item["case"]
        task_name = item["task"]
        if cases is not None and case_name not in cases:
            continue
        if task_name not in tasks:
            continue
        key = (case_name, task_name, item["model"])
        grouped.setdefault(key, []).append(Path(item["run_dir"]))
    return grouped


def extract_metric_curves(run_dirs, metric_name):
    curves = []
    metric_keys = []
    for run_dir in run_dirs:
        if metric_name == "val_pbl_mean":
            curve, metric_key = extract_val_pbl_mean(run_dir)
        else:
            curve, metric_key = extract_training_target(run_dir)
        if curve:
            curves.append(curve)
            metric_keys.append(metric_key)
    return curves, metric_keys


def default_title(metric_name):
    if metric_name == "val_pbl_mean":
        return "Convergence Curves by Case and Task (PBL Mean)"
    return "Convergence Curves by Case and Task (Training Target)"


def default_ylabel(metric_name):
    if metric_name == "val_pbl_mean":
        return "Validation PBL Mean"
    return "Training Target"


def default_basename(cases, metric_name):
    case_token = "_".join(cases)
    return f"learning_curves_{case_token}_{metric_name}"


def main():
    args = parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")

    selected_runs = read_json(args.selected_runs)
    cases = args.cases
    if cases is None:
        discovered_cases = {item["case"] for item in selected_runs}
        cases = sorted(discovered_cases, key=case_sort_key)
    else:
        cases = sorted(set(args.cases), key=case_sort_key)

    tasks = sorted(set(args.tasks), key=task_sort_key)
    grouped_runs = load_grouped_runs(selected_runs, cases, tasks)

    nrows = len(cases)
    ncols = len(tasks)
    fig_width = args.fig_width if args.fig_width is not None else 6.2 * ncols
    fig_height = args.fig_height if args.fig_height is not None else 4.2 * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    legend_handles = []
    legend_labels = []
    metric_key_summary = {}

    for row_index, case_name in enumerate(cases):
        for col_index, task_name in enumerate(tasks):
            ax = axes[row_index][col_index]
            for model_name in MODEL_ORDER:
                run_dirs = grouped_runs.get((case_name, task_name, model_name), [])
                if not run_dirs:
                    continue

                curves, metric_keys = extract_metric_curves(run_dirs, args.metric)
                if not curves:
                    continue

                unique_metric_keys = sorted(set(metric_keys))
                metric_key_summary[f"{case_name}/{task_name}/{model_name}"] = (
                    unique_metric_keys
                )

                epochs, means, lower_bounds, upper_bounds = aggregate_curves(
                    curves,
                    args.pad_policy,
                    args.band_mode,
                    args.extend_to_epoch,
                )
                line = ax.plot(
                    epochs,
                    means,
                    color=MODEL_COLORS[model_name],
                    linewidth=args.line_width,
                    label=MODEL_LABELS[model_name],
                )[0]
                if not args.no_band:
                    ax.fill_between(
                        epochs,
                        np.maximum(lower_bounds, 1e-12) if args.yscale == "log" else lower_bounds,
                        upper_bounds,
                        color=MODEL_COLORS[model_name],
                        alpha=0.15,
                    )

                if MODEL_LABELS[model_name] not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(MODEL_LABELS[model_name])

            if not args.no_panel_title:
                ax.set_title(
                    f"{case_name.upper()} / {task_label(task_name)}",
                    fontsize=args.panel_font_size,
                )
            ax.set_xlabel("Epoch", fontsize=args.label_font_size)
            if col_index == 0:
                ax.set_ylabel(
                    args.ylabel or default_ylabel(args.metric),
                    fontsize=args.label_font_size,
                )
            ax.set_yscale(args.yscale)
            if args.ymin is not None or args.ymax is not None:
                ax.set_ylim(bottom=args.ymin, top=args.ymax)
            ax.tick_params(axis="both", labelsize=args.tick_font_size)
            ax.grid(True, alpha=0.25, linewidth=0.6)

    if not args.no_title:
        fig.suptitle(args.title or default_title(args.metric), fontsize=18, y=0.98)
    if legend_handles and args.legend_location == "figure_bottom":
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=args.legend_font_size,
        )
    elif legend_handles:
        axes[0][0].legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            frameon=True,
            fontsize=args.legend_font_size,
        )
    top_margin = 0.95 if not args.no_title else 0.98
    bottom_margin = 0.07 if args.legend_location == "figure_bottom" else 0.02
    fig.tight_layout(rect=(0, bottom_margin, 1, top_margin))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = args.basename or default_basename(cases, args.metric)
    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    metadata_path = output_dir / f"{basename}_metric_keys.json"

    if args.no_tight_bbox:
        fig.savefig(png_path, dpi=300)
        fig.savefig(svg_path)
    else:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    with open(metadata_path, "w") as f:
        json.dump(metric_key_summary, f, indent=2, sort_keys=True)

    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
