#!/usr/bin/env python3
"""Portable PF scaling-law fits. See SKILL.md; no inference.

Input adapters derived from the portable comparison tool; default log-space
fit reproduces plot_ieee30_combined_scaling.py for both IEEE 30 and 118.
Input adapters embedded from the comparison tool. NumPy, Matplotlib; SciPy for nonlinear fits. Python >= 3.9.
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
    if raw.get("endpoint_samples") in (None, ""):
        known = raw.get("variant") in {"strong-baseline", "endpoint-only", "endpoint-supplement"} or raw.get("category") in {"strong endpoint", "weak endpoint"}
        if not known or raw.get("midrho_samples") not in (None, "", "0", 0):
            raise ValueError(f"Missing sample composition: {origin}; specify endpoint_samples/midrho_samples.")
        raw = {**raw, "endpoint_samples": raw["total_train_samples"], "midrho_samples": 0}
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
            required = {"seeds" if aggregated else "seed"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path}: missing {sorted(required - set(reader.fieldnames or ()))}, "
                                 "use --points-csv for already aggregated plot tables.")
            for line, raw in enumerate(reader, 2):
                origin = f"{path}:{line}"
                # Old plot tables duplicate endpoint anchors on the pink path.
                if aggregated and count(raw.get("midrho_samples") or 0) == 0 and (
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


def predict_fit(fit, sizes):
    import numpy as np
    x = np.asarray(sizes, dtype=float) / fit["N_ref"]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        prediction = fit["C"] + fit["A"] * np.exp(-fit["b"] * np.log(x))
    if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0):
        raise ValueError("Fitted curve is nonfinite/nonpositive in the requested range.")
    return prediction


def r_squared(y, fitted):
    import numpy as np
    sse = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    # Constant targets have undefined R2; do not report an artificial perfect fit.
    tolerance = np.finfo(float).eps ** 2 * max(float(np.sum(y * y)), np.finfo(float).tiny)
    return (1 - sse / sst if sst > tolerance else None), sse


def fit_scaling(sizes, values, model="power", fit_space="log", n_ref=40000, max_exponent=10):
    """Fit scale means with equal size weights; C is not an empirical minimum."""
    import numpy as np
    x, y = np.asarray(sizes, dtype=float), np.asarray(values, dtype=float)
    minimum = 3 if model == "power" else 4
    if len(x) < minimum or len(set(x)) != len(x):
        raise ValueError(f"{model} requires at least {minimum} distinct sample sizes; got {len(x)}.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Fitting requires finite, strictly positive sample counts and metrics.")
    lx, ly = np.log(x / n_ref), np.log(y)
    design = np.column_stack((np.ones_like(lx), lx))
    intercept, slope = np.linalg.lstsq(design, ly, rcond=None)[0]
    notes, optimizer = [], {}
    if model == "power" and fit_space == "log":
        amplitude, exponent, floor = float(np.exp(intercept)), float(-slope), 0.0
        objective = "ordinary least squares of log(mean error); equal weight per training size"
    else:
        from scipy.optimize import least_squares
        scale = float(np.exp(np.mean(ly)))
        normalized = y / scale
        candidates = []
        is_offset = model == "offset"
        lower = [-50, 1e-8 if is_offset else -max_exponent] + ([0] if is_offset else [])
        upper = [50, max_exponent] + ([float("inf")] if is_offset else [])

        def residual(theta):
            # Clipping only keeps rejected trial steps numerically finite. Returned fits
            # are rechecked in the original, unclipped physical expression below.
            log_prediction = theta[0] - theta[1] * lx
            if is_offset:
                c = theta[2]
                log_prediction = np.logaddexp(log_prediction, np.log(c) if c > 0 else -np.inf)
            if fit_space == "log":
                return log_prediction - np.log(normalized)
            return np.exp(np.clip(log_prediction, -700, 300)) - normalized

        floors = [0, .25 * float(min(normalized)), .75 * float(min(normalized))] if is_offset else [0]
        exponents = [float(np.clip(-slope, 1e-4 if is_offset else -max_exponent * .9, max_exponent * .9)), .2, 1, 2]
        for initial_c in floors:
            for initial_b in exponents:
                initial_a = float(np.mean(np.log(np.maximum(normalized - initial_c, 1e-12)) + initial_b * lx))
                initial = [float(np.clip(initial_a, -49, 49)), float(np.clip(initial_b, lower[1] + 1e-6, upper[1] - 1e-6))]
                if is_offset:
                    initial.append(initial_c)
                result = least_squares(residual, initial, bounds=(lower, upper), x_scale="jac",
                                       max_nfev=5000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
                if result.success and np.all(np.isfinite(result.fun)):
                    candidates.append(result)
        if not candidates:
            raise ValueError(f"{model}/{fit_space}: nonlinear optimizer failed for every initial guess.")
        result = min(candidates, key=lambda r: float(np.sum(r.fun ** 2)))
        amplitude, exponent = float(np.exp(result.x[0]) * scale), float(result.x[1])
        floor = float(result.x[2] * scale) if is_offset else 0.0
        singular = np.linalg.svd(result.jac, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        if condition is not None and not math.isfinite(condition):
            condition = None
        optimizer = {"method": "scipy.optimize.least_squares, multiple initial guesses",
                     "successful_starts": len(candidates), "nfev": int(result.nfev),
                     "message": str(result.message), "jacobian_condition": condition,
                     "parameterization": "log(A/geometric_mean_y), b, C/geometric_mean_y (offset only)",
                     "b_bounds": [lower[1], upper[1]], "C_lower_bound": 0 if is_offset else None,
                     "log_scaled_A_bounds": [-50, 50], "active_mask": [int(v) for v in result.active_mask]}
        objective = f"nonlinear least squares of {'log error' if fit_space == 'log' else 'raw error (constant numerical rescaling)'}; equal weight per training size"
        if condition is None or condition > 1e8:
            notes.append("Ill-conditioned fit: parameters may be weakly identifiable.")
        if any(result.active_mask) or abs(exponent - max_exponent) < 1e-5:
            notes.append("A fitted parameter reached a numerical bound; inspect before interpreting the exponent.")
        if is_offset:
            if floor <= 1e-6 * min(y):
                notes.append("C is near zero; data do not demonstrate a positive error floor.")
            if len(x) < 6 or max(x) / min(x) < 4:
                notes.append("Few sizes or a narrow range: offset C and exponent b may trade off strongly.")
            near = [r for r in candidates if float(np.sum(r.fun ** 2)) <= float(np.sum(result.fun ** 2)) * 1.001 + 1e-9]
            if near and max(r.x[1] for r in near) - min(r.x[1] for r in near) > .1:
                notes.append("Near-equal objectives have different exponents across starts; fit is not well identified.")
    fit = {"model": model, "fit_space": fit_space, "A": amplitude, "b": exponent, "C": floor,
           "N_ref": float(n_ref), "formula": "E(N) = C + A * (N/N_ref)^(-b)",
           "point_count": len(x), "sample_min": int(min(x)), "sample_max": int(max(x)),
           "objective": objective, "optimizer": optimizer}
    prediction = predict_fit(fit, x)
    r2log, sselog = r_squared(ly, np.log(prediction))
    r2linear, sselinear = r_squared(y, prediction)
    fit.update(r_squared_log=r2log, r_squared_linear=r2linear, sse_log=sselog, sse_linear=sselinear,
               rmse_log=float(np.sqrt(sselog / len(x))), rmse_linear=float(np.sqrt(sselinear / len(x))))
    if exponent <= 0:
        notes.append("b <= 0: observed fit does not support error decay as sample count increases.")
    if r2log is None:
        notes.append("Constant log target: R2_log is undefined.")
    elif r2log < .8:
        notes.append("Low log-space goodness of fit (R2_log < 0.8); report the empirical points and fit range.")
    if max(x) / min(x) < 2:
        notes.append("Training-size range spans less than 2x; the estimated exponent is local.")
    fit["quality_notes"] = notes
    return fit


def series_groups(points, args):
    groups = defaultdict(list)
    for point in points:
        if point["midrho_samples"] == 0:
            if args.series in ("endpoint", "all"):
                groups[(point["metric"], point["test"], "endpoint", 0)].append(point)
        elif args.series in ("mixed", "all"):
            groups[(point["metric"], point["test"], point["method"], point["endpoint_samples"])].append(point)
    if args.include_anchor:
        for (metric, test, method, base), group in groups.items():
            if base:
                anchors = [p for p in points if p["metric"] == metric and p["test"] == test
                           and p["endpoint_samples"] == base and p["midrho_samples"] == 0]
                if not anchors:
                    raise ValueError(f"Missing requested endpoint anchor for {metric}/{test}/{base}.")
                if any(p["seeds"] != anchors[0]["seeds"] for p in group):
                    warn(f"{metric}/{test}/{base}: anchor and mixed seed sets differ.")
                group.insert(0, anchors[0])
    if not groups:
        raise ValueError("No series selected; check --series and input composition.")
    return {key: sorted(value, key=lambda p: p["total_train_samples"]) for key, value in sorted(groups.items())}


def fit_groups(groups, args):
    parameters, observations, curves = [], [], []
    import numpy as np
    for (metric, test, method, base), group in groups.items():
        fit_rows = [p for p in group if (args.fit_min is None or p["total_train_samples"] >= args.fit_min)
                    and (args.fit_max is None or p["total_train_samples"] <= args.fit_max)]
        models = ("power", "offset") if args.model == "both" else (args.model,)
        for model in models:
            identity = {"metric": metric, "test": test, "method": method, "endpoint_base": base}
            try:
                fit = fit_scaling([p["total_train_samples"] for p in fit_rows], [p["value"] for p in fit_rows],
                                  model, args.fit_space, args.n_ref, args.max_exponent)
            except ValueError as exc:
                raise ValueError(f"{identity}/{model}: {exc}") from exc
            fit = {**identity, **fit}
            parameters.append(fit)
            for note in fit["quality_notes"]:
                warn(f"{metric}/{test}/{method}/{base}/{model}: {note}")
            predictions = predict_fit(fit, [p["total_train_samples"] for p in fit_rows])
            for point, predicted in zip(fit_rows, predictions):
                observations.append({**identity, "model": model, "total_train_samples": point["total_train_samples"],
                                     "seeds": point["seeds"], "n_seeds": point["n_seeds"], "observed_mean": point["value"],
                                     "fitted_mean": float(predicted), "residual": point["value"] - float(predicted),
                                     "log_residual": math.log(point["value"]) - math.log(float(predicted))})
            grid = np.geomspace(fit["sample_min"], fit["sample_max"], 300)
            for n, value in zip(grid, predict_fit(fit, grid)):
                curves.append({**identity, "model": model, "total_train_samples": float(n), "fitted_mean": float(value)})
    return parameters, observations, curves


def render_scaling(metric, groups, fits, curves, args, plt):
    from matplotlib.ticker import NullLocator, LogLocator, FuncFormatter
    selected = {key: rows for key, rows in groups.items() if key[0] == metric}
    zh = args.language == "zh"
    panels = args.layout == "panels"
    fig, axes = plt.subplots(1, 2 if panels else 1, figsize=(14, 6.5) if panels else (9.2, 6.5), squeeze=False)
    axes = list(axes[0])
    colors = {"rho0": "#dc2626", "rho1": "#2563eb"}
    names = {"rho0": "N-1 线路停运" if zh else "N-1 outage", "rho1": "正常拓扑" if zh else "Normal topology"}
    group_names = sorted({(key[2], key[3]) for key in selected})
    line_handles, line_labels = [], []
    for key, rows in selected.items():
        _, test, method, base = key
        ax = axes[0 if not panels or test == "rho0" else 1]
        index = group_names.index((method, base))
        if not panels:
            color = colors[test] if len(group_names) == 1 else plt.get_cmap("tab10")((2 * index + int(test == "rho1")) % 10)
        else:
            color = "#60a5fa" if base == 0 else plt.get_cmap("tab10")((index + 5) % 10)
        label = names[test] if not panels else ("仅端点训练" if zh else "Endpoint only")
        if base:
            label = (names[test] + " | " if not panels else "") + f"{method}, {short_count(base)} base"
        xs, ys = [r["total_train_samples"] for r in rows], [r["value"] for r in rows]
        if args.yscale == "log2" and min(ys) <= 0:
            raise ValueError("Log y requires positive empirical points.")
        # Points are deliberately unconnected: absent/completed-later scales are not imputed.
        line, = ax.plot(xs, ys, "o" if base == 0 else "D", color=color, markersize=6, label=label, zorder=4)
        line_handles.append(line); line_labels.append(label)
        if args.uncertainty == "sd":
            for p in rows:
                if p["n_seeds"] < 2:
                    warn(f"{metric}: single-seed points have no estimable sample SD.")
                    continue
                sd = p.get("value_sd")
                if sd is None:
                    raise ValueError("SD unavailable in aggregated input; use --csv or --uncertainty none.")
                if args.yscale == "log2" and p["value"] - sd <= 0:
                    raise ValueError("mean-SD <= 0; use --yscale linear for SD bars.")
                ax.errorbar(p["total_train_samples"], p["value"], yerr=sd, fmt="none", color=color, capsize=3)
        matching = [f for f in fits if (f["metric"], f["test"], f["method"], f["endpoint_base"]) == key]
        for fit in matching:
            curve = [p for p in curves if (p["metric"], p["test"], p["method"], p["endpoint_base"]) == key and p["model"] == fit["model"]]
            fitted, = ax.plot([p["total_train_samples"] for p in curve], [p["fitted_mean"] for p in curve],
                             color=color, linestyle="--" if fit["model"] == "power" else ":", linewidth=1.8, zorder=2)
            r2 = "n/a" if fit["r_squared_log"] is None else f"{fit['r_squared_log']:.3f}"
            formula = f"E={fit['A']:.3g}(N/N_0)^{{{-fit['b']:.3f}}}"
            if fit["model"] == "offset":
                formula = f"E={fit['C']:.3g}+{fit['A']:.3g}(N/N_0)^{{{-fit['b']:.3f}}}"
            line_handles.append(fitted)
            context = f"{names[test]} / {'endpoint' if not base else method + '-' + short_count(base)} / " if panels or len(group_names) > 1 else ""
            line_labels.append(context + (rf"{fit['model']}: ${formula}$, $R^2_{{\log}}={r2}$" if r2 != "n/a" else f"{fit['model']}: ${formula}$, R2_log undefined"))
    for i, ax in enumerate(axes):
        points = [p for key, rows in selected.items() for p in rows if not panels or key[1] == ("rho0" if i == 0 else "rho1")]
        ticks = sorted({p["total_train_samples"] for p in points})
        if args.xscale == "log2":
            ax.set_xscale("log", base=2)
        if args.yscale == "log2":
            ax.set_yscale("log", base=2)
            ax.yaxis.set_major_locator(LogLocator(base=2, numticks=8))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.4g}"))
        ax.set_xticks(ticks, [short_count(n) for n in ticks])
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", rotation=40, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=.25)
        ax.set_xlabel("训练样本数" if zh else "Training samples", fontsize=12)
        ax.set_ylabel(("MAPE (%)" if metric == "MAPE" else metric) + ("（越低越好）" if zh else " (lower is better)"), fontsize=12)
        if panels:
            ax.set_title(names["rho0" if i == 0 else "rho1"], fontsize=12)
    title = args.title or f"{args.case} | {metric} | scaling fit"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, .04, 1, .95))
    # Legend is below the plotting area so long equations never obscure observations.
    fig.legend(line_handles, line_labels, loc="upper center", bbox_to_anchor=(.5, -.005), ncol=1, frameon=False, fontsize=10)
    fig.text(.5, .005, f"N0 = {short_count(args.n_ref)}; fit space: {args.fit_space}; curves only within fitted sizes", ha="center", fontsize=9)
    stem = re.sub(r"[^\w.-]+", "_", args.case) + f"_{metric}_scaling_{args.model}_{args.language}"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"{stem}.{extension}", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"{metric}: {args.output_dir / (stem + '.png')}")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--case", required=True)
    p.add_argument("--endpoint-root", "--results-root", action="append", type=Path, default=[])
    p.add_argument("--mixed-root", action="append", type=Path, default=[])
    p.add_argument("--csv", nargs="+", type=Path, default=[])
    p.add_argument("--points-csv", nargs="+", type=Path, default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metrics", default="all")
    p.add_argument("--seeds")
    p.add_argument("--incomplete", choices=("error", "skip", "available"), default="error")
    p.add_argument("--sizes", help="Total sample counts to retain in both plots and fits")
    p.add_argument("--test-id")
    p.add_argument("--language", choices=("en", "zh"), default="en")
    p.add_argument("--font-file", action="append", type=Path, default=[])
    p.add_argument("--uncertainty", choices=("none", "sd"), default="none")
    p.add_argument("--xscale", choices=("log2", "linear"), default="log2")
    p.add_argument("--yscale", choices=("log2", "linear"), default="log2")
    p.add_argument("--model", choices=("power", "offset", "both"), default="power")
    p.add_argument("--fit-space", choices=("log", "linear"), default="log")
    p.add_argument("--n-ref", type=count, default=40000)
    p.add_argument("--fit-min", type=count)
    p.add_argument("--fit-max", type=count)
    p.add_argument("--max-exponent", type=float, default=10, help="Nonlinear fit bound on b; log-OLS power has no bound")
    p.add_argument("--series", choices=("endpoint", "mixed", "all"), default="endpoint")
    p.add_argument("--include-anchor", action="store_true", help="Include matching endpoint baseline in each mixed fit")
    p.add_argument("--layout", choices=("overlay", "panels"), default="overlay")
    p.add_argument("--title")
    p.add_argument("--dpi", type=int, default=300)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sum(map(bool, (args.endpoint_root or args.mixed_root, args.csv, args.points_csv))) != 1:
        raise ValueError("Choose one input mode: native result directories, --csv, or --points-csv.")
    args.metrics = None if args.metrics == "all" else list(dict.fromkeys(args.metrics.split(",")))
    if args.metrics and set(args.metrics) - set(FIELDS):
        raise ValueError(f"Unknown metrics: {set(args.metrics) - set(FIELDS)}")
    args.seeds = sorted(set(count(s) for s in args.seeds.split(","))) if args.seeds else None
    args.sizes = sorted(set(count(s) for s in args.sizes.split(","))) if args.sizes else None
    if args.n_ref <= 0 or args.dpi <= 0 or not math.isfinite(args.max_exponent) or args.max_exponent <= .001:
        raise ValueError("N_ref, dpi, max_exponent must be positive (max_exponent > .001).")
    if args.fit_min is not None and args.fit_max is not None and args.fit_min > args.fit_max:
        raise ValueError("--fit-min exceeds --fit-max")
    aggregated = bool(args.points_csv)
    records = result_records(args) if args.endpoint_root or args.mixed_root else csv_records(args.points_csv if aggregated else args.csv, args, aggregated)
    if args.series == "endpoint":
        records = [r for r in records if r["midrho_samples"] == 0]
    elif args.series == "mixed" and not args.include_anchor:
        records = [r for r in records if r["midrho_samples"] > 0]
    points, used, incomplete = aggregate(records, args, aggregated)
    groups = series_groups(points, args)
    fits, observations, curves = fit_groups(groups, args)
    if args.series != "endpoint":
        warn("Mixed-series fits describe a fixed-endpoint augmentation path versus total records, not an endpoint-only scaling law.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fonts = typography(args, matplotlib)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in FIELDS:
        if any(key[0] == metric for key in groups):
            render_scaling(metric, groups, fits, curves, args, plt)
    drawn_keys = {group_key(p) for rows in groups.values() for p in rows}
    write_csv(args.output_dir / "plot_points.csv", [p for p in points if group_key(p) in drawn_keys])
    if not aggregated:
        write_csv(args.output_dir / "per_seed.csv", [p for p in used if group_key(p) in drawn_keys])
    write_csv(args.output_dir / "fit_parameters.csv", [{k: v for k, v in f.items() if k not in ("optimizer", "quality_notes")} for f in fits])
    write_csv(args.output_dir / "fit_points.csv", observations)
    write_csv(args.output_dir / "fit_curves.csv", curves)
    (args.output_dir / "fit_parameters.json").write_text(json.dumps(fits, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metadata = {"script_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {k: str(v) if isinstance(v, Path) else [str(x) if isinstance(x, Path) else x for x in v]
                              if isinstance(v, list) else v for k, v in vars(args).items()},
                "input_sha256": SOURCES, "input_mode": "preaggregated means" if aggregated else "per-seed metrics",
                "test_id": sorted({p["test_id"] for p in points}), "test_identity_evidence": sorted(IDENTITIES),
                "metric_definitions_from_evaluator": DEFINITIONS, "skipped_runs": SKIPPED,
                "skipped_incomplete_groups": [list(x) for x in sorted(incomplete)], "warnings": WARNINGS,
                "weights": "Arithmetic mean over seeds at each size; equal weight per size in the fit",
                "uncertainty": "SD bars, if requested, are between-seed variability; no fitted-parameter confidence intervals",
                "scope": "Descriptive fit on the stated data/range; no automatic outlier/seed selection or extrapolation",
                "fonts": fonts, "python_version": sys.version, "matplotlib_version": matplotlib.__version__}
    import numpy as np
    metadata["numpy_version"] = np.__version__
    if args.model != "power" or args.fit_space != "log":
        import scipy
        metadata["scipy_version"] = scipy.__version__
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(fits)} fits; {len(observations)} fitted observations; {len(curves)} curve points.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
