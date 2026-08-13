#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/linhanjiang/pfdelta-main
cd "$repo_root"

protocol_root=data_generation_rho/case30/scaling_protocol_v1
bank_root=data_generation_rho/case30/rho_pf_bank_normal_loop_edges_10000_0p1
solved_root=data_generation_rho/case30/rho_pf_scaling_endpoint_master_newloads_10001_48800
master_root=data_generation_rho/case30/rho_pf_scaling_endpoint_master_80k
val_root=data_generation_rho/case30/rho_pf_scaling_val_endpoint_4000

if [[ ! -f "$solved_root/meta.json" ]]; then
  echo "Missing completed solver metadata: $solved_root/meta.json" >&2
  exit 1
fi

python3 data_generation_rho/build_fixed_endpoint_scaling.py assemble-master \
  --protocol-dir "$protocol_root" \
  --bank-root "$bank_root" \
  --solved-root "$solved_root" \
  --output-root "$master_root" \
  --clean

python3 data_generation_rho/build_fixed_endpoint_scaling.py assemble-training \
  --master-root "$master_root" \
  --val-root "$val_root" \
  --output-base data_generation_rho/case30 \
  --sizes 5000,10000,20000,40000,80000 \
  --clean

python3 - <<'PY'
import json
from pathlib import Path

base = Path("data_generation_rho/case30")
train_roots = [base / f"rho_pf_scaling_endpoint_{size}k" for size in (5, 10, 20, 40, 80)]
val_loads = {
    int(record["load_setting_id"])
    for record in json.load(open(base / "rho_pf_scaling_val_endpoint_4000/splits/val.json"))["records"]
}
test_loads = {
    int(record["load_setting_id"])
    for record in json.load(open(base / "rho_pf_scaling_T34_final_rho0_6800/splits/test.json"))["records"]
}
previous = set()
for root in train_roots:
    records = json.load(open(root / "splits/train.json"))["records"]
    loads = {int(record["load_setting_id"]) for record in records}
    rho0 = sum(float(record["rho"]) == 0.0 for record in records)
    rho1 = sum(float(record["rho"]) == 1.0 for record in records)
    assert len(loads) * 2 == len(records)
    assert rho0 == rho1 == len(loads)
    assert not (loads & val_loads)
    assert not (loads & test_loads)
    assert previous <= loads
    previous = loads
    print(f"verified {root.name}: records={len(records)} loads={len(loads)} rho0={rho0} rho1={rho1}")
PY
