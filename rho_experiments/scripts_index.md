# Rho Script Index

## Data Generation and Dataset Assembly

- `data_generation_rho/create_rho_bank.jl`
  Generates coarse rho banks over normal loop branches.

- `data_generation_rho/create_rho_bank_usable_loads.jl`
  Generates coarse rho banks while accepting only usable load settings.

- `data_generation_rho/refine_rho_bank.jl`
  Refines complete coarse curves from `0:0.1:1` to `0:0.01:1`.

- `data_generation_rho/make_rho_experiment_splits.py`
  Builds CANOS-ready fixed-branch experiment datasets:
  `60L80R`, `80L60R`, and `100L48R`.

- `data_generation_rho/make_rho1_baseline.py`
  Builds the rho-aware normal-operation baseline where train/val samples have
  `rho=1` and test uses the same branch7 `rho=0` outage split.

## Training Configs

- `core/configs/canos_case30_rho_branch7_60L80R.yaml`
- `core/configs/canos_case30_rho_branch7_80L60R.yaml`
- `core/configs/canos_case30_rho_branch7_100L48R.yaml`
- `core/configs/canos_case30_rho_branch7_rho1_baseline_4800.yaml`

## Evaluation and Plotting

- `scripts/test_error_rho.py`
  Evaluates trained runs on a PFDelta-rho split and reports PBL metrics.

- `scripts/plot_rho_branch7_results.py`
  Plots the three rho-continuation training curves and three-way test PBL bars.

- `scripts/plot_rho_test_pbl_bars.py`
  Plots PBL bars from a `test_error_rho.py` CSV, including the rho=1 baseline.

- `scripts/build_rho0_random_test_root.py`
  Builds a small random rho=0 test root. Use with care: random samples can
  overlap with training load settings unless explicitly excluded.
