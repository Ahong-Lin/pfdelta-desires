# Case30 Branch7 Rho Experiment

## Question

Can a CANOS model trained on continuous line-availability parameter `rho` better
generalize to a sudden branch outage (`rho=0`) than a rho-aware baseline trained
only on normal operation (`rho=1`)?

## Setup

- Case: IEEE `case30`
- Controlled branch: `branch_id=7`
- Test condition: branch7 outage, `rho=0`
- Test size: 13 held-out load settings
- Model: CANOS
- Epochs: 300

Every branch has an edge-level `rho` attribute. The controlled branch changes
with rho; uncontrolled branches have `rho=1`.

For `rho > 0`, the controlled branch uses:

```text
br_status = 1
br_r = br_r_base / rho
br_x = br_x_base / rho
g_fr = g_fr_base * rho
b_fr = b_fr_base * rho
g_to = g_to_base * rho
b_to = b_to_base * rho
```

For `rho=0`, the branch is disconnected by `br_status=0`.

## Training Sets

The three rho-continuation experiments all use 4800 train samples but trade off
load diversity against rho density:

| Experiment | Train loads | Rho values per train load | Train samples |
|---|---:|---:|---:|
| 60L-80R | 60 | 80 | 4800 |
| 80L-60R | 80 | 60 | 4800 |
| 100L-48R | 100 | 48 | 4800 |

Validation uses 10 load settings with full `0:0.01:1` curves. Test uses 13
held-out load settings at `rho=0`.

The baseline is:

| Experiment | Train samples | Train rho | Test |
|---|---:|---:|---|
| rho=1 baseline | 4800 | 1 only | same branch7 `rho=0` held-out test |

## Result Summary

Final four-model held-out `rho=0` test result:

| Experiment | Test samples | PBL Mean | PBL Max | Relative PBL Mean |
|---|---:|---:|---:|---:|
| 60L-80R | 13 | 0.06622 | 0.63473 | 0.49516 |
| 80L-60R | 13 | 0.03737 | 0.20400 | 0.27939 |
| 100L-48R | 13 | 0.03473 | 0.24537 | 0.25972 |
| rho=1 baseline | 13 | 0.13147 | 1.54522 | 0.98208 |

The rho-continuation models outperform the rho=1-only baseline on the same
`rho=0` outage test. Among the three rho-continuation choices, `100L-48R` has
the lowest PBL Mean and Relative PBL Mean on this test.

## Files

Figures:

- `results/figures/rho_branch7_all4_heldout13_test_pbl.png`
- `results/figures/rho_branch7_training_loss_overlay.png`

Tables:

- `results/tables/rho_branch7_all4_heldout13_test_pbl_summary.csv`
- `results/tables/rho_branch7_training_summary.csv`
- `results/tables/rho_branch7_all4_heldout13_pbl.csv`

Original local result location:

```text
rho_test_results/
```

Server run groups:

```text
runs/canos_case30_rho_branch7_60L80R
runs/canos_case30_rho_branch7_80L60R
runs/canos_case30_rho_branch7_100L48R
runs/canos_case30_rho_branch7_rho1_baseline_4800
```
