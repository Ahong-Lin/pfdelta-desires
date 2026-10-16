# Sources and licenses

`core/` comes from the user's running PFDelta/CANOS-PF case118 experiment on the liuy server, downloaded on 2026-10-15. These retained source files are unchanged; hashes are recorded in `SOURCE_PROVENANCE.json`.

The PFDelta code is distributed under the MIT license, Copyright (c) 2025 Machine Learning & Optimization for Sustainable Systems Lab (MOSSLab). The complete license is included in `LICENSE`. Original author comments in CANOS-PF and the Facebook/Pythia attribution in `core/utils/registry.py` are retained.

`reference/liuy_endpoint_v2/` preserves the exact original experiment scripts for comparison. These reference scripts contain original case118 paths and are not the case500 launch entrypoints.

The active `scripts/endpoint_training_v2/` derives from those experiment scripts. Changes for case500 concern configurable case identity and portable data/configuration paths; the model, loss, optimizer, scheduler, batching and validation/test selection protocol are retained. New preparation/configuration/queue scripts connect the existing case500 generation output to this training protocol. A source diff is supplied in `reference/TRAINER_ADAPTATION.patch`.

PyTorch, PyTorch Geometric, NumPy, PyYAML and tqdm are external dependencies installed from their distributions; their source packages, environments, datasets and trained weights are not bundled here.
