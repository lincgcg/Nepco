#!/usr/bin/env python3
"""Measure one full DataCon FP32 forward-latency configuration using E1 models."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)


class ResourceBusy(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def physical_cpu_set(count):
    """Balanced socket mapping, one allowed logical CPU per physical core."""
    topology = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
        socket = int((base / 'physical_package_id').read_text())
        core = int((base / 'core_id').read_text())
        topology.setdefault((socket, core), cpu)
    sockets = {}
    for (socket, core), cpu in sorted(topology.items()):
        sockets.setdefault(socket, []).append(cpu)
    selected = []
    while len(selected) < count:
        changed = False
        for socket in sorted(sockets):
            if sockets[socket] and len(selected) < count:
                selected.append(sockets[socket].pop(0))
                changed = True
        if not changed:
            raise RuntimeError(f'Requires {count} allowed physical CPU cores; insufficient capacity')
    return selected


def gpu_uuid(index):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    for line in output.splitlines():
        number, uuid = [part.strip() for part in line.split(',')]
        if int(number) == index:
            return uuid
    raise RuntimeError('Requested GPU index is unavailable')


def check_resources(device, selected_uuid):
    # Fail closed: an unavailable resource query must not silently permit timing.
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    for line in output.splitlines():
        uuid, pid = [part.strip() for part in line.split(',')]
        if int(pid) != os.getpid() and (device == 'cpu' or uuid == selected_uuid):
            raise ResourceBusy('External GPU compute activity; no result accepted')


def load_e1(e1, name):
    """Called in a fresh process so the two uer runtimes cannot contaminate each other."""
    import torch
    from argparse import Namespace
    art = e1 / 'artifacts/DataCon'
    test = art / 'test_dataset.tsv'
    reference = e1 / 'results/DataCon'
    if name == 'Nepco':
        spec = importlib.util.spec_from_file_location('nepco_inference', e1 / 'reproduce.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from uer.utils import str2tokenizer
        settings = json.loads((art / 'model_config.json').read_text())
        settings.update(vocab_path=str(e1.parent / 'vocab/hex_vocab.txt'), spm_model_path=None,
                        do_lower_case='true', labels_num=10, pooling='max', embedding=['word'],
                        seq_length=128, test_path=str(test), max_samples=None)
        args = Namespace(**settings)
        args.tokenizer = str2tokenizer['bert'](args)
        rows = mod.read_dataset(args)
        model = mod.Classifier(args)
        model_path = art / 'finetuned_model.bin'
        mod.load_checkpoint(model, model_path)
        forward = model.infer
    else:
        runtime = e1 / 'TraGe'
        sys.path.insert(0, str(runtime))
        from classifier import Classifier
        from dataset import read_dataset
        from uer.utils.tokenizers import BertTokenizer
        args = Namespace(**json.loads((art / 'TraGe/model_config.json').read_text())['model'])
        args.vocab_path = str(runtime / 'vocab/encryptd_vocab.txt')
        args.spm_model_path = None
        args.do_lower_case = 'true'
        args.tokenizer = BertTokenizer(args)
        rows = read_dataset(args, test)
        model = Classifier(args)
        model_path = art / 'TraGe/finetuned_model.bin'
        try:
            state = torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state, strict=True)
        forward = model.forward
        reference /= 'TraGe'
    with (reference / 'predictions.tsv').open() as handle:
        expected = [int(row['prediction']) for row in csv.DictReader(handle, delimiter='\t')]
    if len(rows) != 2857 or len(expected) != 2857:
        raise ValueError('This benchmark requires the full accepted 2857-row DataCon test set')
    return model, forward, rows, expected, model_path, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('nepco', 'trage'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--batch_size', type=int, choices=BATCHES, default=1)
    parser.add_argument('--cpu_cores', type=int, choices=(8, 16, 32, 64), default=8)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--e1_dir', type=Path, default=ROOT.parent / 'E1')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'results/local_runs')
    parser.add_argument('--check_only', action='store_true',
                        help='Functional first/tail batch check only; no latency result is produced')
    args = parser.parse_args()
    name = {'nepco': 'Nepco', 'trage': 'TraGe'}[args.model]
    cores = args.cpu_cores if args.device == 'cpu' else 8
    suffix = f'c{cores}-b{args.batch_size}' if args.device == 'cpu' else f'b{args.batch_size}'
    output = args.output_dir / name / ('cpu' if args.device == 'cpu' else 'gpu') / f'{suffix}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    # Shared across original/new entry points for this E2 checkout.
    lock = (ROOT / '.latency.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a completed configuration')
    selected_uuid = gpu_uuid(args.gpu_index) if args.device == 'cuda' else None
    check_resources(args.device, selected_uuid)
    selected = physical_cpu_set(cores)
    os.sched_setaffinity(0, set(selected))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(cores)
    os.environ['OMP_PROC_BIND'] = 'true'
    os.environ['OMP_PLACES'] = 'threads'
    os.environ['CUDA_VISIBLE_DEVICES'] = selected_uuid if selected_uuid else ''
    import torch
    torch.set_num_threads(cores)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    e1 = args.e1_dir.resolve()
    model, forward, rows, expected, model_path, test = load_e1(e1, name)
    manifest = json.loads((ROOT / 'full_benchmark_manifest.json').read_text())
    if sha(test) != manifest['test_sha256'] or sha(model_path) != manifest['model_sha256'][name]:
        raise ValueError('E1 model/test hash differs from the verified inputs')
    device = torch.device(args.device)
    model.to(device).eval()  # Refresh Nepco cache after device placement.
    if not all(param.dtype == torch.float32 for param in model.parameters()):
        raise ValueError('Expected FP32 parameters')
    src = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
    seg = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
    batch = args.batch_size
    batches = [(src[i:i + batch], seg[i:i + batch]) for i in range(0, 2857, batch)]
    sizes = [s.shape[0] for s, _ in batches]
    if args.check_only:
        with torch.inference_mode():
            for index in sorted({0, len(batches) - 1}):
                check_resources(args.device, selected_uuid)
                s, g = batches[index]
                logits = forward(s, g)
                offset = index * batch
                if not torch.isfinite(logits).all().item() or logits.argmax(1).cpu().tolist() != expected[offset:offset + len(s)]:
                    raise ValueError('Functional first/tail batch prediction check failed')
        check_resources(args.device, selected_uuid)
        print(f'{name} {args.device}: functional first/tail batch check passed; no timing result produced')
        return
    events = None
    if args.device == 'cuda':
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        events[0].record(); events[1].record(); torch.cuda.synchronize()
    last_check = 0.0

    def resource_guard(force=False):
        nonlocal last_check
        if force or time.monotonic() - last_check >= 1:
            check_resources(args.device, selected_uuid)
            last_check = time.monotonic()

    def run_pass(timed, pass_id):
        times, gpu_times, predictions = [], [], []
        resource_guard(True)
        with torch.inference_mode():
            for s, g in batches:
                resource_guard()
                if events:
                    torch.cuda.synchronize()
                    events[0].record()
                begin = time.perf_counter()
                logits = forward(s, g)
                if events:
                    events[1].record()
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - begin) * 1000
                if timed:
                    times.append(elapsed)
                    if events:
                        gpu_times.append(events[0].elapsed_time(events[1]))
                if not torch.isfinite(logits).all().item():
                    raise ValueError('Nonfinite output')
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        resource_guard(True)
        if predictions != expected:
            raise ValueError('Predictions do not exactly match the E1 reference')
        print(f'{name}: {"measurement" if timed else "warmup"} {pass_id} complete (2857 samples)', flush=True)
        if not timed:
            return
        if not all(math.isfinite(value) and value > 0 for value in times):
            raise ValueError('Invalid raw timing')
        total = sum(times) / 1000
        result = dict(pass_id=pass_id, samples=2857, batches=len(batches), last_batch_size=sizes[-1],
                      total_forward_seconds=total, mean_batch_ms=statistics.mean(times),
                      mean_full_batch_ms=statistics.mean(t for t, n in zip(times, sizes) if n == batch),
                      amortized_ms_per_sample=sum(times) / 2857, samples_per_second=2857 / total,
                      batch_latency_ms=times, predictions_match=True)
        result['pass'] = result.pop('pass_id')
        if events:
            result['cuda_event_batch_ms'] = gpu_times
        return result

    run_pass(False, 1)
    measured = [run_pass(True, index) for index in range(1, 6)]
    values = [entry['amortized_ms_per_sample'] for entry in measured]
    result = dict(status='complete', model=name, device=args.device, dataset='DataCon', samples=2857,
                  batch_size=batch, cpu_physical_cores=cores if args.device == 'cpu' else None,
                  precision='float32', warmup_full_passes=1, measured_full_passes=5,
                  timing_scope='synchronized logits-forward host-wall only',
                  mean_ms_per_sample=statistics.mean(values), median_ms_per_sample=statistics.median(values),
                  std_ms_per_sample=statistics.stdev(values), min_ms_per_sample=min(values),
                  max_ms_per_sample=max(values), all_predictions_match=True, passes=measured)
    for key in ('mean_batch_ms', 'mean_full_batch_ms'):
        result[key] = statistics.mean(entry[key] for entry in measured)
    result['mean_samples_per_second'] = statistics.mean(entry['samples_per_second'] for entry in measured)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    print(f'{name}: {result["mean_ms_per_sample"]:.6f} ms/sample', flush=True)


if __name__ == '__main__':
    try:
        main()
    except ResourceBusy as error:
        print(str(error), file=sys.stderr)
        sys.exit(75)
