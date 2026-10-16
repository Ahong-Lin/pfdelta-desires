#!/usr/bin/env python3
"""Summarize completed endpoint experiments, retaining every seed including weak runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results_root.glob('*/seed*/Evaluation/completion.json')):
        completion = json.loads(path.read_text())
        if not completion.get('finished') or completion.get('smoke_only'):
            continue
        metrics = completion['test_metrics']
        overall = metrics.get('overall', metrics)
        rows.append({'scale': completion['scale'], 'seed': completion['seed'],
                     'MSE': overall['canos_pf_mse'], 'PBL_Mean': overall['PBL Mean'],
                     'best_epoch': completion['best_epoch'],
                     'best_val_mse': completion['best_val_mse'],
                     'test_count': completion['test_count'],
                     'data_hash': completion['data_hash']})
    rows.sort(key=lambda r: (int(str(r['scale']).rstrip('k')) * (1000 if str(r['scale']).endswith('k') else 1), r['seed']))
    grouped = []
    for scale in dict.fromkeys(row['scale'] for row in rows):
        members = [r for r in rows if r['scale'] == scale]
        if len({r['seed'] for r in members}) != len(members):
            raise ValueError(f'Duplicate seed in {scale}')
        if len({r['data_hash'] for r in members}) != 1:
            raise ValueError(f'Different data hashes within {scale}; do not average incompatible runs')
        entry = {'scale': scale, 'completed_seeds': ','.join(str(r['seed']) for r in members),
                 'n_seeds': len(members)}
        for key in ['MSE', 'PBL_Mean']:
            values = [r[key] for r in members]
            entry[f'{key}_mean'] = statistics.mean(values)
            entry[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else ''
        grouped.append(entry)
    output = args.results_root / '_summary'
    write_csv(output / 'per_seed.csv', rows,
              ['scale','seed','MSE','PBL_Mean','best_epoch','best_val_mse','test_count','data_hash'])
    write_csv(output / 'by_scale.csv', grouped,
              ['scale','completed_seeds','n_seeds','MSE_mean','MSE_std','PBL_Mean_mean','PBL_Mean_std'])
    print(json.dumps({'completed_runs': len(rows), 'output': str(output.resolve())}, indent=2))


if __name__ == '__main__':
    main()
