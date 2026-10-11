#!/usr/bin/env python3
"""Run the complete 80-group matrix sequentially; never overwrite verified results."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'results/local_runs')
    parser.add_argument('--e1_dir', type=Path, default=ROOT.parent / 'E1')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '.full_matrix.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    groups = [(m, 'cuda', b, None) for m in ('nepco', 'trage') for b in (1, 2, 4, 8, 16, 32, 64, 128)]
    groups += [(m, 'cpu', b, c) for m in ('nepco', 'trage') for c in (8, 16, 32, 64)
               for b in (1, 2, 4, 8, 16, 32, 64, 128)]
    summary = []
    for model, device, batch, cores in groups:
        command = [sys.executable, str(ROOT / 'benchmark_inference.py'), '--model', model,
                   '--device', device, '--batch_size', str(batch), '--gpu_index', str(args.gpu_index),
                   '--output_dir', str(args.output_dir), '--e1_dir', str(args.e1_dir)]
        if cores:
            command += ['--cpu_cores', str(cores)]
        while True:
            result = subprocess.run(command)
            if result.returncode == 75:
                print('Resource busy; waiting 30 seconds before retrying the entire configuration.', flush=True)
                time.sleep(30)
                continue
            result.check_returncode()
            break
        name = {'nepco': 'Nepco', 'trage': 'TraGe'}[model]
        suffix = f'c{cores}-b{batch}' if cores else f'b{batch}'
        path = args.output_dir / name / ('cpu' if cores else 'gpu') / f'{suffix}.json'
        raw = json.loads(path.read_text())
        fields = ('model', 'device', 'batch_size', 'cpu_physical_cores', 'samples',
                  'mean_ms_per_sample', 'std_ms_per_sample', 'mean_batch_ms',
                  'mean_full_batch_ms', 'mean_samples_per_second')
        summary.append({key: raw[key] for key in fields})
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        with (args.output_dir / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
    print('Completed all 80 full-test configurations.', flush=True)


if __name__ == '__main__':
    main()
