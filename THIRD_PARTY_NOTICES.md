# Third-party notices

This migration bundle contains third-party source code and a network case. Their original notices are retained. This bundle is an adapted experiment runner; it is not an official upstream release.

## PFDelta source

The files under `data_generation/src/` were copied from the locally audited PFDelta/case118 source snapshot and preserve PFDelta's existing adaptations. PFDelta's repository license is MIT, copyright (c) 2025 Machine Learning & Optimization for Sustainable Systems Lab (MOSSLab); its complete text is in `licenses/LICENSE`.

The copied OPFLearn-derived files also retain the original OPFLearn notices below. The MIT notice does not replace those notices.

Local changes for this migration, dated 2026-10-14:

- `src/create_samples_distributed.jl`: keep and join every distributed Future before the next source batch; drain post-termination output queues; propagate worker failures and stop workers after a failed batch; respect the requested worker count.
- `src/tasks.jl`: return unsuccessful attempts to replenish sampling; use one dedicated sampler with sole ownership of the request queue; check DONE during batch publication; delegate shutdown queue draining to the coordinator to avoid competing-consumer races.
- The new files outside `src/` implement case500 endpoint generation, audits, splitting and migration commands. They are local additions, not part of the upstream benchmark specification.

## OPFLearn

Source: <https://github.com/NREL/OPFLearn.jl>.

Copyright (c) 2021, Alliance for Sustainable Energy, LLC. The adapted source headers identify the BSD 3-Clause License. Its complete upstream text is in `licenses/LICENSE-OPFLearn` (also covering references to `LICENSE-OPFLearn` in source headers).

## PGLib 500-bus case

File: `data_generation/pglib/pglib_opf_case500_goc.m`, PGLib-OPF v23.07, unchanged case data and attribution header.

Source collection: <https://github.com/power-grid-lib/pglib-opf>. The file header identifies this as a synthetic case developed for ARPA-e Grid Optimization Competition, Challenge 1.

Copyright (c) 2020 by A.B. Birchfield, H. Li, J. Wert, J. Yeo, T. Xu and T.J. Overbye. Licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The complete license is included in `licenses/LICENSE-CC-BY-4.0.txt`.

The network file requests that publications using these data cite:

A.B. Birchfield, T. Xu, K.M. Gegner, K.S. Shetye, T.J. Overbye, “Grid Structural Characteristics as Validation Criteria for Synthetic Networks,” IEEE Transactions on Power Systems, 2017.

## Installed dependencies

`data_generation/Project.toml`, `Manifest.toml`, and `requirements.txt` identify dependencies installed separately. Their own licenses continue to apply; this bundle does not relicense them or include their binaries. License-file provenance and SHA-256 values are in `licenses/sources.json`.
