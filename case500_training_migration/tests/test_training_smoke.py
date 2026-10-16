"""CPU integration check using synthetic 3-bus fixtures, not real PF labels."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
from test_case500_data import make_source, prep
from configure_runs import freeze_runs


class TrainingIntegrationTests(unittest.TestCase):
    def test_two_epoch_cpu_train_checkpoint_full_test_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source, cache, results = base / 'source', base / 'cache', base / 'Results'
            make_source(source)
            prep.prepare(source, cache, scales='all')
            experiment = json.loads((ROOT / 'configs/experiment.json').read_text())
            # Integration-only overrides; the distributed formal config is never modified.
            experiment.update(scales=['4', '8'], seeds=[11], epochs=2, device='cpu', torch_threads=1)
            manifest_path = freeze_runs(cache, results, experiment, scales=['8'])
            manifest = json.loads(manifest_path.read_text())
            entry = manifest['runs'][0]
            command = [sys.executable, str(ROOT / 'scripts/endpoint_training_v2/train_endpoint_run.py'),
                       '--config', entry['config']]
            process = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stdout[-5000:] + process.stderr[-5000:])
            run = Path(entry['run_dir'])
            completion = json.loads((run / 'Evaluation/completion.json').read_text())
            self.assertTrue(completion['finished'])
            self.assertFalse(completion['smoke_only'])
            self.assertEqual(completion['epochs_completed'], 2)
            self.assertEqual(completion['train_count'], 8)
            self.assertEqual(completion['test_count'], 4)
            metrics = completion['test_metrics']
            metrics = metrics.get('overall', metrics)
            self.assertTrue(math.isfinite(metrics['canos_pf_mse']))
            self.assertTrue(math.isfinite(metrics['PBL Mean']))
            self.assertTrue((run / 'Model/model.pt').is_symlink())
            first_completion = (run / 'Evaluation/completion.json').read_bytes()
            repeat = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(repeat.returncode, 0, repeat.stdout + repeat.stderr)
            self.assertEqual(first_completion, (run / 'Evaluation/completion.json').read_bytes())
            summary = subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_results.py'),
                                      '--results-root', str(results)], capture_output=True, text=True, timeout=10)
            self.assertEqual(summary.returncode, 0, summary.stdout + summary.stderr)
            self.assertTrue((results / '_summary/per_seed.csv').is_file())
            self.assertEqual(len((results / '_summary/per_seed.csv').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
