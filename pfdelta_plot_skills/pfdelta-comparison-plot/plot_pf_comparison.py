#!/usr/bin/env python3
"""Portable endpoint / intermediate-rho comparisons. See README.md; no inference.

Derived from plot_ieee30_sample_type_comparison.py,
plot_case118_sample_type_comparison_current.py, comparison_figure_layout.py.
Only Matplotlib and its dependencies are required. Python >= 3.9.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1.0.0"
FIELDS = {
    "MSE": ("MSE", "canos_pf_mse"),
    "PBL": ("PBL", "PBL Mean"),
    "rPBL": ("rPBL", "Relative PBL Mean"),
    "MAPE": ("MAPE", "MAPE (%)"),
}
COLORS = ("#60a5fa", "#ec4899")
WARNINGS = []
SOURCES = {}
SKIPPED = []
DEFINITIONS = {}
IDENTITIES = set()


def warn(message):
    if message not in WARNINGS:
        WARNINGS.append(message)
        print("WARNING: " + message, file=sys.stderr)


def source(path):
    path = Path(path).resolve()
    if str(path) not in SOURCES:
        SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(source(path).read_text(encoding="utf-8-sig"))


def count(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmM]?)", str(value).strip())
    if not match:
        raise ValueError(f"Invalid count: {value!r}")
    number = float(match[1]) * {"": 1, "k": 1000, "m": 1000000}[match[2].lower()]
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"Noninteger count: {value!r}")
    return int(number)


def endpoint(value):
    text = str(value).strip().lower()
    if text in {"0", "0.0", "rho0", "rho=0"} or re.search(r"[-_]rho0$", text):
        return "rho0"
    if text in {"1", "1.0", "rho1", "rho=1"} or re.search(r"[-_]rho1$", text):
        return "rho1"
    raise ValueError(f"Expected rho=0 or rho=1 test; got {value!r}")


def metric_values(raw, requested):
    result = {}
    for name, aliases in FIELDS.items():
        if requested and name not in requested:
            continue
        present = [float(raw[a]) for a in aliases if raw.get(a) not in (None, "")]
        if present:
            if any(x != present[0] for x in present):
                raise ValueError(f"Conflicting aliases for {name}: {raw}")
            result[name] = present[0]
    if raw.get("value") not in (None, "") and not result:
        name = raw.get("metric")
        if not name and requested and len(requested) == 1:
            name = requested[0]
        if name not in FIELDS:
            raise ValueError("A generic 'value' column requires 'metric' or --metrics MSE (one metric).")
        if not requested or name in requested:
            result[name] = float(raw["value"])
    if any(not math.isfinite(x) or x < 0 for x in result.values()):
        raise ValueError(f"Metrics must be finite and nonnegative: {result}")
    return result


def normalize(raw, origin, args):
    ep = count(raw["endpoint_samples"])
    mid = count(raw.get("midrho_samples", 0))
    total = count(raw.get("total_train_samples") or ep + mid)
    if ep <= 0 or ep + mid != total:
        raise ValueError(f"endpoint + midrho != total, or empty endpoints: {origin}")
    test_text = str(raw.get("test", raw.get("rho", "")))
    test = endpoint(test_text)
    inferred_id = re.sub(r"[-_]rho[01]$", "", test_text) if test_text.lower() != test else ""
    if inferred_id == test_text:
        inferred_id = ""
    if inferred_id and raw.get("test_id") and raw["test_id"] != inferred_id:
        raise ValueError(f"test name and explicit test_id disagree: {origin}")
    test_id = raw.get("test_id") or inferred_id or args.test_id or "unspecified"
    if args.test_id and (raw.get("test_id") or inferred_id) and test_id != args.test_id:
        raise ValueError(f"test_id conflicts with --test-id: {origin}")
    method = str(raw.get("method") or ("endpoint" if mid == 0 else "mixed"))
    if mid == 0:
        method = "endpoint"
    return {"test_id": test_id, "test": test, "method": method,
            "endpoint_samples": ep, "midrho_samples": mid,
            "total_train_samples": total, "source": str(origin)}


def csv_records(paths, args, aggregated=False):
    records = []
    for path in paths:
        path = source(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"endpoint_samples", "midrho_samples", "seeds" if aggregated else "seed"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path}: missing {sorted(required - set(reader.fieldnames or ()))}, "
                                 "use --points-csv for already aggregated plot tables.")
            for line, raw in enumerate(reader, 2):
                origin = f"{path}:{line}"
                # Old plot tables duplicate endpoint anchors on the pink path.
                if aggregated and count(raw["midrho_samples"]) == 0 and (
                    raw.get("curve") == "Mixed Samples" or raw.get("point_role") == "endpoint anchor"
                ):
                    continue
                base = normalize(raw, origin, args)
                values = metric_values(raw, args.metrics)
                if not values:
                    continue
                if aggregated:
                    seeds = sorted(int(s) for s in re.split(r"[;,\s]+", raw["seeds"].strip()) if s)
                    if not seeds or len(seeds) != len(set(seeds)):
                        raise ValueError(f"Missing/duplicate aggregate seeds: {origin}")
                    if count(raw.get("n_seeds") or len(seeds)) != len(seeds):
                        raise ValueError(f"n_seeds mismatch: {origin}")
                    if args.seeds and set(seeds) != set(args.seeds):
                        raise ValueError(f"Cannot reselect seeds from preaggregated means: {origin}; use --csv.")
                    sd = raw.get("value_sd")
                    sd = float(sd) if sd not in (None, "") else None
                    if sd is not None and (not math.isfinite(sd) or sd < 0):
                        raise ValueError(f"Invalid standard deviation: {origin}")
                    for metric, value in values.items():
                        records.append({**base, "metric": metric, "value": value, "value_sd": sd,
                                        "seeds": seeds, "n_seeds": len(seeds)})
                else:
                    seed = count(raw["seed"])
                    for metric, value in values.items():
                        records.append({**base, "seed": seed, "metric": metric, "value": value})
    return records


def composition(config, done, run, mixed):
    total = done.get("train_count", config.get("expected_train_count"))
    ep, mid = config.get("endpoint_samples"), config.get("midrho_samples")
    if ep is None:
        ep, mid = done.get("endpoint_samples"), done.get("midrho_samples")
    if ep is not None and mid is not None:
        ep, mid = count(ep), count(mid)
    elif mixed:
        match = next((re.fullmatch(r"(\d+(?:\.\d+)?[kKmM]?)\+(\d+(?:\.\d+)?[kKmM]?)", part)
                      for part in reversed(run.parts)
                      if re.fullmatch(r"(\d+(?:\.\d+)?[kKmM]?)\+(\d+(?:\.\d+)?[kKmM]?)", part)), None)
        if not match:
            raise ValueError(f"Cannot infer mixed composition: {run}. Set endpoint_samples/midrho_samples "
                             "in run_config.json, use 40k+5k directory names, or export a CSV.")
        ep, mid = count(match[1]), count(match[2])
    else:
        ep, mid = count(total if total is not None else config.get("scale", done.get("scale", ""))), 0
    if mixed and mid <= 0:
        raise ValueError(f"--mixed-root contains an endpoint run: {run}")
    if not mixed and mid != 0:
        raise ValueError(f"--endpoint-root contains a mixed run: {run}; specify --mixed-root.")
    if total is not None and count(total) != ep + mid:
        raise ValueError(f"Composition disagrees with recorded train_count: {run}")
    return ep, mid


def result_records(args):
    records, found = [], set()
    roots = [(p, False) for p in args.endpoint_root] + [(p, True) for p in args.mixed_root]
    for root, mixed in roots:
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"Results directory does not exist: {root}")
        completions = sorted(root.rglob("completion.json"))
        # Include known run configs, so unfinished jobs are visible in metadata.
        runs = {p.parent.parent for p in completions if p.parent.name == "Evaluation"}
        runs.update(p.parent.parent for p in root.rglob("run_config.json") if p.parent.name == "Model")
        for run in sorted(runs):
            if run in found:
                raise ValueError(f"Overlapping input roots contain the same run: {run}")
            found.add(run)
            cf = run / "Evaluation/completion.json"
            if not cf.is_file():
                SKIPPED.append({"run": str(run), "reason": "no completion.json (unfinished)"})
                continue
            done = read_json(cf)
            if done.get("finished") is not True or done.get("smoke_only", False):
                SKIPPED.append({"run": str(run), "reason": "unfinished or smoke_only"})
                continue
            cfg = run / "Model/run_config.json"
            config = read_json(cfg) if cfg.is_file() else {}
            mf = run / "Evaluation/test_metrics.json"
            metrics = read_json(mf) if mf.is_file() else done.get("test_metrics")
            if not metrics or "by_rho" not in metrics:
                raise ValueError(f"Finished run lacks by_rho test metrics: {run}")
            ep, mid = composition(config, done, run, mixed)
            seed = config.get("seed", done.get("seed"))
            if seed is None:
                raise ValueError(f"Missing seed: {run}")
            if config.get("seed") is not None and done.get("seed") is not None and config["seed"] != done["seed"]:
                raise ValueError(f"Config/completion seed mismatch: {run}")
            if done.get("test_count") is not None and metrics.get("num_samples") != done["test_count"]:
                raise ValueError(f"Test count mismatch: {run}")
            fingerprint = config.get("test_sha256") or config.get("test_cache_sha256")
            test_cache = config.get("test_cache", "")
            if fingerprint:
                identity = "sha256:" + fingerprint
            elif test_cache:
                identity = "recorded_path:" + test_cache
            else:
                raise ValueError(f"No recorded test_cache or test hash: {run}; export CSV with test_id.")
            IDENTITIES.add(identity)
            for name, definition in metrics.get("definitions", {}).items():
                if name in DEFINITIONS and DEFINITIONS[name] != definition:
                    raise ValueError(f"Metric definition differs between runs: {name}")
                DEFINITIONS[name] = definition
            observed = set()
            endpoint_counts = []
            for rho, summary in metrics["by_rho"].items():
                test = endpoint(rho)
                if test in observed:
                    raise ValueError(f"Duplicate endpoint summary: {run}")
                observed.add(test)
                n_test = count(summary.get("num_samples", ""))
                if n_test <= 0:
                    raise ValueError(f"Empty endpoint test: {run}/{test}")
                endpoint_counts.append(n_test)
                raw = {"test": test, "test_id": config.get("test_id") or args.test_id or identity,
                       "endpoint_samples": ep, "midrho_samples": mid, "seed": seed,
                       "method": config.get("method", "mixed" if mixed else "endpoint")}
                base = normalize(raw, str(mf if mf.exists() else cf), args)
                values = metric_values(summary, args.metrics)
                if args.metrics and set(args.metrics) - set(values):
                    raise ValueError(f"{run}/{test}: missing requested metrics {set(args.metrics) - set(values)}")
                for metric, value in values.items():
                    records.append({**base, "seed": count(seed), "metric": metric, "value": value,
                                    "num_test_samples": n_test, "run_dir": str(run)})
            if observed != {"rho0", "rho1"}:
                raise ValueError(f"Both endpoint test summaries required: {run}")
            if sum(endpoint_counts) != metrics.get("num_samples"):
                raise ValueError(f"by_rho counts do not sum to overall test count: {run}")
    if len(IDENTITIES) > 1:
        raise ValueError("Different recorded test identities; do not merge Test sets. "
                         "If paths differ only due to relocation, export verified CSV with one test_id. "
                         + repr(sorted(IDENTITIES)))
    if any(s.startswith("recorded_path:") for s in IDENTITIES):
        warn("Test identity checked by recorded cache path/count, not by hashing the dataset contents.")
    if SKIPPED:
        warn(f"Skipped {len(SKIPPED)} unfinished/smoke runs; details in metadata.json.")
    return records


def group_key(row):
    return tuple(row[k] for k in ("metric", "test", "method", "endpoint_samples", "midrho_samples"))


def aggregate(records, args, preaggregated):
    records = [r for r in records if not args.sizes or r["total_train_samples"] in args.sizes]
    inventory = {group_key(r) for r in records}
    if args.seeds and not preaggregated:
        records = [r for r in records if r["seed"] in args.seeds]
    if not records:
        raise ValueError("No matching test results. Check input paths, metrics, sizes, and seeds.")
    ids = {r["test_id"] for r in records}
    if len(ids) != 1:
        raise ValueError(f"Multiple Test identities in one comparison: {sorted(ids)}")
    if ids == {"unspecified"}:
        warn("CSV has no test_id; dataset equivalence is unverified. Set --test-id after checking the split.")
    for test in ("rho0", "rho1"):
        counts = {r.get("num_test_samples") for r in records if r["test"] == test} - {None, ""}
        if len(counts) > 1:
            raise ValueError(f"Per-seed test sample counts differ for {test}: {counts}")
    groups = defaultdict(list)
    for key in inventory:
        groups[key] = []
    for row in records:
        groups[group_key(row)].append(row)
    points = []
    incomplete = set()
    for key, rows in sorted(groups.items()):
        if preaggregated:
            if len(rows) != 1:
                raise ValueError(f"Duplicate aggregate point {key}")
            point = dict(rows[0])
        else:
            seeds = sorted(r["seed"] for r in rows)
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"Duplicate seed/metric for {key}; select one run per seed. Sources: "
                                 + repr([r["source"] for r in rows]))
            if args.seeds and set(seeds) != set(args.seeds):
                message = f"Incomplete {key}: expected {args.seeds}, found {seeds}"
                if args.incomplete == "error":
                    raise ValueError(message)
                warn(message)
                if args.incomplete == "skip" or not rows:
                    incomplete.add((key[0], key[2], key[3], key[4]))
                    continue
            point = {k: v for k, v in rows[0].items() if k not in ("seed", "value", "source", "run_dir")}
            values = [r["value"] for r in rows]
            point.update(value=statistics.mean(values), value_sd=statistics.stdev(values) if len(values) > 1 else None,
                         seeds=seeds, n_seeds=len(seeds))
        points.append(point)
    # Drop both panels when one side is incomplete, keeping the comparison paired.
    points = [p for p in points if (p["metric"], p["method"], p["endpoint_samples"], p["midrho_samples"]) not in incomplete]
    expected_pairs = defaultdict(set)
    for point in points:
        key = (point["metric"], point["method"], point["endpoint_samples"], point["midrho_samples"])
        expected_pairs[key].add(point["test"])
    for key, tests in expected_pairs.items():
        if tests != {"rho0", "rho1"}:
            raise ValueError(f"Missing rho0/rho1 panel for {key}: {tests}")
        pair = [p for p in points if (p["metric"], p["method"], p["endpoint_samples"], p["midrho_samples"]) == key]
        if pair[0]["seeds"] != pair[1]["seeds"]:
            raise ValueError(f"rho0 and rho1 have different seeds: {key}")
    if not points:
        raise ValueError("No complete groups remain.")
    for metric in sorted({p["metric"] for p in points}):
        seed_sets = {tuple(p["seeds"]) for p in points if p["metric"] == metric}
        if len(seed_sets) > 1:
            warn(f"{metric}: seed sets vary across points; inspect plot_points.csv (no seed selection applied).")
        if args.metrics and metric not in args.metrics:
            raise AssertionError(metric)
    if args.metrics and set(args.metrics) - {p["metric"] for p in points}:
        raise ValueError(f"Requested metrics missing: {set(args.metrics) - {p['metric'] for p in points}}")
    # Native tests must have equal sizes across models within each endpoint.
    for test in ("rho0", "rho1"):
        counts = {p.get("num_test_samples") for p in points if p["test"] == test} - {None, ""}
        if len(counts) > 1:
            raise ValueError(f"Test sample counts differ for {test}: {counts}")
    keys = {group_key(p) for p in points}
    used = [r for r in records if group_key(r) in keys]
    return points, used, incomplete


def paths_for(points, metric, test):
    visible = [p for p in points if p["metric"] == metric and p["test"] == test]
    endpoints = sorted((p for p in visible if p["midrho_samples"] == 0), key=lambda p: p["total_train_samples"])
    paths = [("endpoint", None, endpoints)] if endpoints else []
    mixed = defaultdict(list)
    for p in visible:
        if p["midrho_samples"]:
            mixed[(p["method"], p["endpoint_samples"])].append(p)
    for (method, base), group in sorted(mixed.items()):
        anchors = [p for p in endpoints if p["endpoint_samples"] == base]
        if not anchors:
            warn(f"{metric}/{test}/{base}: no endpoint anchor; draw only observed mixed points.")
        elif any(p["seeds"] != anchors[0]["seeds"] for p in group):
            warn(f"{metric}/{base}: mixed and anchor seed sets differ; anchor is the baseline mean, not paired inference.")
        paths.append((method, base, anchors + sorted(group, key=lambda p: p["total_train_samples"])))
    return paths


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ";".join(map(str, v)) if isinstance(v, list) else v for k, v in row.items()})


def typography(args, mpl):
    from matplotlib import font_manager
    custom = []
    for path in args.font_file:
        font_manager.fontManager.addfont(str(path))
        custom.append(font_manager.FontProperties(fname=str(path)).get_name())
    installed = {f.name for f in font_manager.fontManager.ttflist}
    latin = "Times New Roman" if "Times New Roman" in installed else "DejaVu Serif"
    families = [latin]
    if args.language == "zh":
        from matplotlib.ft2font import FT2Font
        candidates = custom + ["SimSun", "Noto Serif CJK SC", "Noto Sans CJK SC", "Source Han Sans SC",
                               "Microsoft YaHei", "PingFang SC", "Songti SC"]
        chinese = next((f for f in candidates if f in installed and
                        all(ord(c) in FT2Font(font_manager.findfont(f, fallback_to_default=False)).get_charmap()
                            for c in "训练样本仅端点混合正常拓扑线路停运越低好")), None)
        if not chinese:
            raise ValueError("Chinese font missing. Use --language en, install Noto CJK, or --font-file /path/font.ttf.")
        families.append(chinese)
    elif custom:
        families = custom + families
    mpl.rcParams.update({"font.family": families, "font.size": 16, "font.weight": "normal",
                         "axes.titleweight": "normal", "axes.labelweight": "normal", "axes.titlesize": 16,
                         "axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 14,
                         "legend.fontsize": 15, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "path", "axes.unicode_minus": False})
    return families


def short_count(value):
    return f"{value / 1000:g}k" if value >= 1000 else str(value)


def draw(points, metric, args, incomplete, plt):
    from matplotlib.ticker import NullLocator, MaxNLocator
    zh = args.language == "zh"
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.75))
    # Fixed axes from the original comparison_figure_layout.py.
    for ax, left in zip(axes, (99.16, 99.16 + 469.44 + 84.592)):
        ax.set_position((left / 1152, 72.45 / 414, 469.44 / 1152, 218.592 / 414))
        ax.yaxis.set_label_coords(-55.1 / 469.44, 0.5)
    legend = {}
    for ax, test in zip(axes, ("rho0", "rho1")):
        paths = paths_for(points, metric, test)
        visible = [p for _, _, group in paths for p in group]
        ticks = sorted({p["total_train_samples"] for p in visible})
        if not visible:
            raise ValueError(f"No points for {metric}/{test}")
        ranges = []
        all_mixed = {(name, base) for name, base, _ in paths if base is not None}
        multiple_methods = len({name for name, _ in all_mixed}) > 1
        for name, base, group in paths:
            is_endpoint = base is None
            label = ("仅端点训练" if zh else "Endpoint only") if is_endpoint else ("混合训练" if zh else "Mixed samples")
            if not is_endpoint and multiple_methods:
                label += f" ({name})"
            color = COLORS[0 if is_endpoint else 1]
            marker = "o" if is_endpoint else "D"
            # Retain a gap when a known, incomplete group was excluded.
            missing = sorted(ep + mid for met, method, ep, mid in incomplete
                             if met == metric and ((is_endpoint and mid == 0) or
                                                   (not is_endpoint and method == name and ep == base)))
            xs, ys = [], []
            for point in group:
                x, y = point["total_train_samples"], point["value"]
                if args.yscale == "log2" and y <= 0:
                    raise ValueError(f"Log y requires positive {metric}; use --yscale linear for zero values.")
                if xs and any(xs[-1] < m < x for m in missing):
                    xs.append(float("nan")); ys.append(float("nan"))
                xs.append(x); ys.append(y); ranges.append(y)
            line, = ax.plot(xs, ys, color=color, marker=marker, linestyle="-" if is_endpoint else (0, (3.5, 2.2)),
                            linewidth=2.2, markersize=6.5 if is_endpoint else 6.3, markeredgewidth=0, zorder=3)
            legend.setdefault(label, line)
            if args.uncertainty == "sd":
                for p in group:
                    sd = p.get("value_sd")
                    if p["n_seeds"] < 2:
                        warn(f"{metric}: single-seed points have no estimable sample SD; no bar drawn.")
                        continue
                    if sd is None:
                        raise ValueError("Aggregate input has no SD. Use per-seed CSV or --uncertainty none.")
                    if args.yscale == "log2" and p["value"] - sd <= 0:
                        raise ValueError(f"{metric}: mean-SD <= 0; use --yscale linear with SD bars.")
                    ax.errorbar(p["total_train_samples"], p["value"], yerr=sd, color=color, fmt="none", capsize=3)
                    ranges.extend([p["value"] - sd, p["value"] + sd])
        if args.xscale == "log2":
            ax.set_xscale("log", base=2)
            ax.set_xlim(ticks[0] / 1.055, ticks[-1] * 1.055)
        else:
            pad = max((ticks[-1] - ticks[0]) * .04, ticks[-1] * .04)
            ax.set_xlim(max(0, ticks[0] - pad), ticks[-1] + pad)
        if args.yscale == "log2":
            ax.set_yscale("log", base=2)
            high, low = max(ranges), min(ranges)
            order = math.floor(math.log10(high))
            upper = next(n * 10 ** order for n in (1, 2, 4, 8, 10) if n * 10 ** order >= high)
            divisor = 2 if metric in ("PBL", "rPBL") else 4
            yticks = [upper]
            while yticks[-1] > low:
                yticks.append(yticks[-1] / divisor)
            # Bound tick density for extreme dynamic ranges, never discard data.
            if len(yticks) > 9:
                stride = math.ceil(len(yticks) / 9)
                yticks = sorted(set(yticks[::stride] + [yticks[-1]]), reverse=True)
            exponent = math.floor(math.log10(upper))
            scale = 10 ** exponent if exponent <= -3 or exponent >= 4 else 1
            ax.set_yticks(yticks, [f"{y / scale:.6g}" for y in yticks])
            if scale != 1:
                ax.text(0, 1.015, r"$\times 10^{" + str(exponent) + "}$", transform=ax.transAxes, fontsize=13)
            ax.set_ylim(yticks[-1] / 1.12, yticks[0] * 1.12)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        ax.set_xticks(ticks, [short_count(x) for x in ticks])
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", rotation=60, labelsize=11 if len(ticks) > 10 else 14, pad=2)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=.25)
        ax.set_title(("N-1 线路停运" if zh else "N-1 outage") if test == "rho0" else ("正常拓扑" if zh else "Normal topology"))
        ax.set_xlabel("训练样本数" if zh else "Training samples")
        name = "MAPE (%)" if metric == "MAPE" else metric
        ax.set_ylabel(name + ("（越低越好）" if zh else " (lower is better)"))
    title = args.title or (f"{args.case} | {metric}" + (f" | {args.test_id}" if args.test_id else ""))
    fig.suptitle(title, y=.975, fontsize=16)
    fig.legend(list(legend.values()), list(legend), loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=min(3, len(legend)), frameon=False)
    stem = re.sub(r"[^\w.-]+", "_", args.case) + f"_{metric}_comparison_{args.language}"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.output_dir / f"{stem}.{extension}", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"{metric}: {args.output_dir / (stem + '.png')}")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--case", required=True, help="Display/output name, e.g. case30, case118, case500")
    p.add_argument("--endpoint-root", "--results-root", action="append", type=Path, default=[])
    p.add_argument("--mixed-root", action="append", type=Path, default=[])
    p.add_argument("--csv", nargs="+", type=Path, default=[], help="Per-seed metric CSVs")
    p.add_argument("--points-csv", nargs="+", type=Path, default=[], help="Already aggregated historical plot tables")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metrics", default="all", help="all, or comma-separated MSE,PBL,rPBL,MAPE")
    p.add_argument("--seeds", help="Comma-separated seed filter and expected set")
    p.add_argument("--incomplete", choices=("error", "skip", "available"), default="error")
    p.add_argument("--sizes", help="Optional total-count filter, e.g. 40k,45k,50k,80k; retain anchors explicitly")
    p.add_argument("--test-id", help="Name of one verified fixed test split, e.g. T168-v2")
    p.add_argument("--language", choices=("en", "zh"), default="en")
    p.add_argument("--font-file", action="append", type=Path, default=[])
    p.add_argument("--uncertainty", choices=("none", "sd"), default="none")
    p.add_argument("--xscale", choices=("log2", "linear"), default="log2")
    p.add_argument("--yscale", choices=("log2", "linear"), default="log2")
    p.add_argument("--title")
    p.add_argument("--dpi", type=int, default=300)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    modes = [bool(args.endpoint_root or args.mixed_root), bool(args.csv), bool(args.points_csv)]
    if sum(modes) != 1:
        raise ValueError("Choose exactly one: Results roots, --csv, or --points-csv.")
    args.metrics = None if args.metrics == "all" else list(dict.fromkeys(args.metrics.split(",")))
    if args.metrics and set(args.metrics) - set(FIELDS):
        raise ValueError(f"Metrics must be one of {list(FIELDS)}")
    args.seeds = sorted(set(count(s) for s in args.seeds.split(","))) if args.seeds else None
    args.sizes = sorted(set(count(s) for s in args.sizes.split(","))) if args.sizes else None
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    aggregated = bool(args.points_csv)
    if modes[0]:
        records = result_records(args)
    else:
        records = csv_records(args.points_csv if aggregated else args.csv, args, aggregated)
    points, used, incomplete = aggregate(records, args, aggregated)
    import matplotlib
    matplotlib.use("Agg")  # Works on headless company servers.
    import matplotlib.pyplot as plt
    fonts = typography(args, matplotlib)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [name for name in FIELDS if any(p["metric"] == name for p in points)]
    for metric in metrics:
        draw(points, metric, args, incomplete, plt)
    write_csv(args.output_dir / "plot_points.csv", points)
    if not aggregated:
        write_csv(args.output_dir / "per_seed.csv", used)
    metadata = {
        "script_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else [str(x) if isinstance(x, Path) else x for x in v]
                      if isinstance(v, list) else v for k, v in vars(args).items()},
        "input_mode": "preaggregated historical means" if aggregated else "per-seed results",
        "input_sha256": SOURCES, "test_id": sorted({p["test_id"] for p in points}),
        "test_identity_evidence": sorted(IDENTITIES), "metric_definitions_from_evaluator": DEFINITIONS,
        "aggregation": "Existing means retained" if aggregated else "Arithmetic mean over listed seeds; sample SD ddof=1",
        "uncertainty": args.uncertainty, "seed_groups": [
            {k: p[k] for k in ("metric", "test", "method", "endpoint_samples", "midrho_samples", "seeds", "n_seeds")}
            for p in points], "skipped_runs": SKIPPED,
        "skipped_incomplete_groups": [list(x) for x in sorted(incomplete)], "warnings": WARNINGS,
        "fonts": fonts, "matplotlib_version": matplotlib.__version__, "python_version": sys.version,
        "style": {"canvas_inches": [16, 5.75], "endpoint_color": COLORS[0], "mixed_color": COLORS[1],
                  "anchor": "reuse observed endpoint mean; separate path for each endpoint base/method",
                  "original_scripts": ["plot_ieee30_sample_type_comparison.py",
                                       "plot_case118_sample_type_comparison_current.py", "comparison_figure_layout.py"]},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(points)} aggregate points; {len(used)} input rows; metrics={','.join(metrics)}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
