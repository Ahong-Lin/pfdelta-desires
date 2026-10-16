#!/usr/bin/env python3
"""Check training imports, package versions and visible CUDA devices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', help='Allow CPU-only development checks.')
    parser.add_argument('--allow-version-drift', action='store_true',
                        help='Report version differences without treating them as an installation failure.')
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch_geometric
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.models.canos_pf import CANOS_PF
    from core.utils.pf_losses_utils import CANOS_PF_MSE, PowerBalanceLoss
    from core.datasets.pfdelta_rho_dataset import PFDeltaRhoCANOS

    expected = {'python': '3.11', 'torch': '2.1.2', 'torch_geometric': '2.6.1',
                'numpy': '1.26.3', 'PyYAML': '6.0.3'}
    found = {'python': f'{sys.version_info.major}.{sys.version_info.minor}',
             'torch': str(torch.__version__).split('+')[0],
             'torch_geometric': torch_geometric.__version__, 'numpy': np.__version__,
             'PyYAML': yaml.__version__}
    deviations = {k: {'expected': v, 'found': found[k]} for k, v in expected.items() if found[k] != v}
    cuda = torch.cuda.is_available()
    devices = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            # Trigger a real device operation rather than relying only on the driver probe.
            probe = torch.ones(1, device=f'cuda:{i}').sum().item()
            devices.append({'visible_index': i, 'name': properties.name,
                            'memory_GiB': round(properties.total_memory / 2**30, 2),
                            'compute_capability': list(torch.cuda.get_device_capability(i)),
                            'device_probe': probe})
    print(json.dumps({'versions': found, 'version_differences': deviations,
                      'cuda_available': cuda, 'cuda_runtime': torch.version.cuda,
                      'devices': devices, 'model_and_reader_imports': 'passed'}, indent=2))
    if deviations and not args.allow_version_drift:
        raise SystemExit('Version mismatch: recreate the pinned environment or explicitly acknowledge drift.')
    if not cuda and not args.cpu:
        raise SystemExit('CUDA unavailable; check GPU visibility, driver and PyTorch installation.')


if __name__ == '__main__':
    main()
