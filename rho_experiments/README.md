# PFDelta Rho Experiments

This directory is an index for rho-continuation experiments. It does not replace
the main code/data locations; it collects the important notes, result tables, and
figures so the experiment can be understood without searching the whole repo.

## Contents

- `branch7_case30/`: IEEE case30, branch 7, rho-continuation experiment.
- `scripts_index.md`: where the rho generation, training, testing, and plotting
  scripts live in the main repository.

## Main Repo Locations

- Rho data generation code: `data_generation_rho/`
- Rho dataset reader: `core/datasets/pfdelta_rho_dataset.py`
- CANOS rho configs: `core/configs/canos_case30_rho_branch7_*.yaml`
- Rho evaluation/plot scripts: `scripts/*rho*.py`
- Raw local result pull: `rho_test_results/`

## Current Result Snapshot

The current branch7 case30 result snapshot is copied under:

```text
rho_experiments/branch7_case30/results/
```

The original copies remain in `rho_test_results/`.
