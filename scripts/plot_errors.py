import argparse
import glob
import json
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


MODEL_ORDER = [
    ("canos", "CANOS-PF"),
    ("pfnet", "PFNet"),
    ("gns", "GNS-S"),
    ("flowcan", "FlowCAN"),
]

PALETTE = {
    "CANOS-PF": "#7D4E85",
    "PFNet": "#0086A8",
    "GNS-S": "#D95F02",
    "FlowCAN": "#2CA25F",
}


def root_for(case_name, model_name, task):
    task_name = str(task).replace(".", "_")
    suffix = "_case30" if case_name == "case30" else ""
    return f"runs/{model_name.lower()}_task_{task_name}{suffix}"


def dict_to_df(d, model_name, metric_name):
    rows = []
    for key, values in d.items():
        for v in values:
            rows.append({
                "Grid Attribute": key,
                metric_name: v,
                "Model": model_name,
            })
    return pd.DataFrame(rows)


def df_max_with_sd(df, metric_name):
    grouped = df.groupby(["Grid Attribute", "Model"])[metric_name]
    mean = grouped.mean()
    sd = grouped.std().fillna(0.0)
    return (mean + sd).max()


def relative_pbl_values(model_results):
    if "Relative PBL Mean / Max Load" in model_results:
        return model_results["Relative PBL Mean / Max Load"], True

    if "PBL Max Load Scale" in model_results:
        relative = {}
        for key, values in model_results["PBL Mean"].items():
            scales = model_results["PBL Max Load Scale"][key]
            relative[key] = [
                pbl / max(scale, 1e-8)
                for pbl, scale in zip(values, scales)
            ]
        return relative, True

    if "Relative PBL Mean" in model_results:
        return model_results["Relative PBL Mean"], True

    if "PBL Load Scale" in model_results:
        relative = {}
        for key, values in model_results["PBL Mean"].items():
            scales = model_results["PBL Load Scale"][key]
            relative[key] = [
                pbl / max(scale, 1e-8)
                for pbl, scale in zip(values, scales)
            ]
        return relative, True

    return model_results["PBL Mean"], False


def parser():
    parser = argparse.ArgumentParser(
        description="Plots training summaries or test errors."
    )
    parser.add_argument(
        "--row",
        type=int,
        default=1,
        help="Picks which default task row to plot when --source test-errors is used.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="case30",
        choices=["case30", "case118"],
        help="Power grid case to plot.",
    )
    parser.add_argument(
        "--tasks",
        type=float,
        nargs="+",
        default=None,
        help="Specific tasks to plot, e.g. --tasks 1.1 2.2 2.3.",
    )
    parser.add_argument(
        "--source",
        choices=["summary", "test-errors"],
        default="summary",
        help=(
            "summary plots finished runs directly from runs/<case>/*/summary.json; "
            "test-errors uses test_errors_p_seeds.json."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Seeds to average when --source summary is used.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="PBL Mean",
        help="Metric to read from summary.json validation results.",
    )
    return parser.parse_args()


def tasks_for_args(args):
    if args.tasks is not None:
        return args.tasks
    if args.source == "summary":
        return [1.1, 2.2, 2.3]
    if args.row == 1:
        return [1.1, 1.2, 1.3]
    return [4.1, 4.2, 4.3]


def task_id(task):
    return str(task).replace(".", "_")


def summary_value(summary, metric_name):
    val_results = summary.get("val", [])
    if val_results and metric_name in val_results[-1]:
        return val_results[-1][metric_name]
    train_results = summary.get("train", {})
    if metric_name in train_results:
        return train_results[metric_name]
    raise KeyError(f"Could not find metric {metric_name!r} in summary.json")


def summary_rows_for(case_name, model_key, model_label, task, seeds, metric_name):
    rows = []
    missing = []
    pattern_base = os.path.join(
        "runs",
        case_name,
        f"{model_key}_{case_name}_task{task_id(task)}_seed{{seed}}_*",
        "summary.json",
    )

    for seed in seeds:
        matches = sorted(glob.glob(pattern_base.format(seed=seed)))
        if not matches:
            missing.append(seed)
            continue

        summary_path = matches[-1]
        if len(matches) > 1:
            print(
                f"WARNING: found {len(matches)} runs for "
                f"{model_label} case={case_name} task={task} seed={seed}; "
                f"using {summary_path}"
            )

        with open(summary_path, "r") as f:
            summary = json.load(f)

        if not summary.get("finished", False):
            print(f"WARNING: skipping unfinished run {summary_path}")
            continue

        rows.append({
            "Task": f"Task {task}",
            "Model": model_label,
            "Seed": seed,
            metric_name: summary_value(summary, metric_name),
            "Run": os.path.basename(os.path.dirname(summary_path)),
        })

    if missing:
        print(
            f"WARNING: missing seeds {missing} for "
            f"{model_label} case={case_name} task={task}"
        )

    return rows


def plot_from_summary(args, tasks):
    metric_name = args.metric
    rows = []

    for task in tasks:
        for model_key, model_label in MODEL_ORDER:
            rows.extend(
                summary_rows_for(
                    args.case,
                    model_key,
                    model_label,
                    task,
                    args.seeds,
                    metric_name,
                )
            )

    if not rows:
        raise RuntimeError(
            f"No finished summary.json files found for case={args.case}, "
            f"tasks={tasks}, seeds={args.seeds}."
        )

    results_df = pd.DataFrame(rows)
    print("\nRuns included in plot:")
    for (task, model), group in results_df.groupby(["Task", "Model"], sort=False):
        seeds = ", ".join(str(seed) for seed in sorted(group["Seed"].unique()))
        mean = group[metric_name].mean()
        std = group[metric_name].std()
        print(f"{task:>8}  {model:<8}  seeds: {seeds:<7} mean={mean:.6g} sd={std:.6g}")

    fontsize = 18
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.8))
    sns.barplot(
        data=results_df,
        x="Task",
        y=metric_name,
        hue="Model",
        hue_order=[label for _, label in MODEL_ORDER],
        ax=ax,
        palette=PALETTE,
        errorbar="sd",
        capsize=0.1,
        edgecolor="white",
        err_kws={"linewidth": 1.6},
    )
    sns.stripplot(
        data=results_df,
        x="Task",
        y=metric_name,
        hue="Model",
        hue_order=[label for _, label in MODEL_ORDER],
        ax=ax,
        palette=PALETTE,
        dodge=True,
        size=4,
        alpha=0.65,
        linewidth=0.5,
        edgecolor="white",
        legend=False,
    )

    ax.set_xlabel("")
    ax.set_ylabel(f"Validation {metric_name}", fontsize=fontsize - 2, fontweight="bold")
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.yaxis.grid(True)
    ax.set_title(f"{args.case} validation errors", fontsize=fontsize + 2, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    keep = len(MODEL_ORDER)
    ax.legend(
        handles[:keep],
        labels[:keep],
        title="Model",
        loc="upper left",
        fontsize=fontsize - 4,
        title_fontsize=fontsize - 4,
    )

    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    output_stem = (
        f"plot_errors_{args.case}_summary_task"
        f"{'_'.join(task_id(t) for t in tasks)}"
    )
    plt.savefig(f"figures/{output_stem}.svg")
    plt.savefig(f"figures/{output_stem}.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved figures/{output_stem}.svg")
    print(f"Saved figures/{output_stem}.png")


def plot_from_test_errors(args, tasks):
    with open("test_errors_p_seeds.json", "r") as f:
        all_results = json.load(f)

    n = len(tasks)
    fontsize = 22
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.3), sharey=True)
    if n == 1:
        axes = [axes]
    plot_dfs = []
    use_relative_metric = True

    for task in tasks:
        results = {}
        for model_key, model_label in MODEL_ORDER:
            root_name = root_for(args.case, model_key, task)
            if root_name not in all_results:
                raise KeyError(
                    f"Missing {root_name} in test_errors_p_seeds.json. "
                    f"Run scripts/test_error.py --root {root_name.removeprefix('runs/')} first."
                )
            values, is_relative = relative_pbl_values(all_results[root_name])
            results[model_label] = values
            use_relative_metric = use_relative_metric and is_relative

        metric_name = "PBL Mean / max(sqrt(pd^2 + qd^2))"

        results_df = pd.concat([
            dict_to_df(results[model_label], model_label, metric_name)
            for _, model_label in MODEL_ORDER
        ])
        plot_dfs.append((task, results_df, metric_name))

    ymax = max(df_max_with_sd(df, metric_name) for _, df, metric_name in plot_dfs)
    ymax = max(0.05, ymax * 1.25)

    for i, (task, results_df, metric_name) in enumerate(plot_dfs):
        ax = axes[i]

        sns.barplot(
            data=results_df,
            x="Grid Attribute",
            y=metric_name,
            hue="Model",
            hue_order=[label for _, label in MODEL_ORDER],
            ax=ax,
            palette=PALETTE,
            errorbar="sd",
            capsize=0.1,
            edgecolor="white",
            err_kws={"linewidth": 1.6},
        )

        ax.set_xlabel("")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["N", "N-1", "N-2", "C2I"], fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.set_ylabel(metric_name if i == 0 else "", fontsize=fontsize - 8, fontweight="bold")
        ax.yaxis.grid(True)
        if use_relative_metric:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.set_ylim(0, ymax)
        else:
            ax.set_yticks([0, 0.45, 0.9, 1.35, 1.8])
            ax.set_ylim(0, 1.8)
        ax.get_legend().remove()
        title_text = "Task 1.3 / 2.1" if task == 1.3 else f"Task {task}"
        ax.set_title(title_text, fontsize=fontsize, fontweight="bold")

    axes[0].legend(
        title="Model",
        loc="upper left",
        fontsize=fontsize - 6,
        title_fontsize=fontsize - 6,
    )

    os.makedirs("figures", exist_ok=True)
    plt.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.88, wspace=0.05)
    output_stem = f"plot_errors_{args.case}_task{'_'.join(task_id(t) for t in tasks)}"
    plt.savefig(f"figures/{output_stem}.svg")
    plt.savefig(f"figures/{output_stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig("figures/test.svg")
    plt.savefig("figures/plot_errors.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    args = parser()
    tasks = tasks_for_args(args)
    if args.source == "summary":
        plot_from_summary(args, tasks)
    else:
        plot_from_test_errors(args, tasks)
