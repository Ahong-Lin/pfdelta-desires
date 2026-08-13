#!/usr/bin/env bash
set -euo pipefail

cd /home/linhanjiang/pfdelta-main

while [[ ! -f data_generation_rho/case30/rho_pf_scaling_endpoint_master_160k/meta.json ]]; do
  sleep 60
done

if [[ -e data_generation_rho/case30/n_extra_88801_168800 ]]; then
  echo "Unexpected existing normal-load output: data_generation_rho/case30/n_extra_88801_168800" >&2
  exit 1
fi

~/apps/julia-1.9.4/bin/julia --project=data_generation_rho \
  data_generation_rho/create_case30_normal_pf_samples.jl \
  --output-dir data_generation_rho/case30/n_extra_88801_168800 \
  --num-loads 80000 \
  --seed 20260802

python3 data_generation_rho/prepare_case30_scaling_normal_sources.py \
  --source-range data/pfdelta_data/case30/n/raw:1:56000:1 \
  --source-range data_generation_rho/case30/n_extra_56001_76000/raw:56001:76000:1 \
  --source-range data_generation_rho/case30/n_extra_76001_88800/raw:76001:88800:1 \
  --source-range data_generation_rho/case30/n_extra_88801_168800/raw:88801:168800:1 \
  --output-dir data_generation_rho/case30/n_scaling_canonical_1_168800

python3 data_generation_rho/build_fixed_endpoint_scaling.py make-schedule \
  --start-load-id 88801 \
  --end-load-id 168800 \
  --seed 20260802 \
  --output data_generation_rho/case30/scaling_protocol_v1/plans/train_extension_320k_88801_168800.jsonl \
  --force

~/apps/julia-1.9.4/bin/julia --project=data_generation_rho \
  data_generation_rho/create_planned_endpoint_master_from_samples.jl \
  --source-dir data_generation_rho/case30/n_scaling_canonical_1_168800 \
  --plan data_generation_rho/case30/scaling_protocol_v1/plans/train_extension_320k_88801_168800.jsonl \
  --output-dir data_generation_rho/case30/rho_pf_scaling_endpoint_extension_88801_168800 \
  --min-load-id 88801 \
  --max-load-id 168800 \
  --seed 20260802 \
  --clean \
  --print-every 100

python3 data_generation_rho/extend_endpoint_master.py \
  --base-master data_generation_rho/case30/rho_pf_scaling_endpoint_master_160k \
  --extension-root data_generation_rho/case30/rho_pf_scaling_endpoint_extension_88801_168800 \
  --output-root data_generation_rho/case30/rho_pf_scaling_endpoint_master_320k \
  --clean

python3 data_generation_rho/build_fixed_endpoint_scaling.py assemble-training \
  --master-root data_generation_rho/case30/rho_pf_scaling_endpoint_master_320k \
  --val-root data_generation_rho/case30/rho_pf_scaling_val_endpoint_4000 \
  --output-base data_generation_rho/case30 \
  --sizes 320000 \
  --clean
