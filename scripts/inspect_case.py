#!/usr/bin/env python3
"""Read-only case inventory; standard library only. Generator uses PowerModels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

BUNDLE = Path(__file__).resolve().parents[1]


def matrix(text, name):
    match = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S)
    if not match:
        raise ValueError(f"Missing MATPOWER matrix: {name}")
    clean = re.sub(r"%[^\n]*", "", match.group(1))
    return [[float(value) for value in row.split()] for row in clean.split(";") if row.strip()]


def inventory(path):
    payload = path.read_bytes()
    text = payload.decode("utf-8")
    buses, branches, gens = [matrix(text, name) for name in ("bus", "branch", "gen")]
    active_buses = {int(row[0]) for row in buses if int(row[1]) != 4}

    def connected(off=-1):
        adj = {bus: [] for bus in active_buses}
        for i, row in enumerate(branches):
            if i == off or not int(row[10]):
                continue
            f, t = int(row[0]), int(row[1])
            if f not in adj or t not in adj:
                return False
            adj[f].append(t)
            adj[t].append(f)
        todo = [min(active_buses)]
        seen = set(todo)
        while todo:
            for nxt in adj[todo.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return seen == active_buses

    if not connected():
        raise ValueError("Base case is disconnected")
    eligible, bridges, unity_transformers = [], [], []
    for i, row in enumerate(branches):
        if not int(row[10]):
            continue
        is_bridge = not connected(i)
        if is_bridge:
            bridges.append(str(i + 1))
        # MATPOWER nonzero TAP explicitly marks transformers, including TAP=1.
        transformer = row[8] != 0
        if transformer and row[8] == 1:
            unity_transformers.append(str(i + 1))
        if not transformer and row[9] == 0 and not is_bridge and (row[2] != 0 or row[3] != 0):
            eligible.append(str(i + 1))
    return {
        "case_file": path.name,
        "case_sha256": hashlib.sha256(payload).hexdigest(),
        "bus_count": len(buses),
        "branch_count": len(branches),
        "active_branch_count": sum(int(row[10]) != 0 for row in branches),
        "active_bridge_count": len(bridges),
        "transformer_count": sum(row[8] != 0 for row in branches),
        "active_unity_tap_transformer_count": len(unity_transformers),
        "generator_count": len(gens),
        "active_generator_count": sum(int(row[7]) != 0 for row in gens),
        "eligible_branch_count": len(eligible),
        "eligible_branch_ids": eligible,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    path = Path(config["case_file"])
    if not path.is_absolute():
        path = BUNDLE / path
    result = inventory(path)
    count = result["eligible_branch_count"]
    train = config["train_endpoint_records"]
    val = 2 * count * config["val_parents_per_branch"]
    test = 2 * count * config["test_parents_per_branch"]
    result.update(train_endpoint_records=train, val_endpoint_records=val,
                  test_endpoint_records=test, total_endpoint_records=train + val + test,
                  total_parents=(train + val + test) // 2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
