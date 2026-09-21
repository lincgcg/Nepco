#!/usr/bin/env python3
"""Measurement utilities for the Nepco CPU latency smoke test."""

import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch


def parse_cpu_cores(spec):
    """Parse a Linux taskset-style list such as ``0-7`` or ``0,2,4,6``."""
    cores = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid CPU range: {part}")
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    unique = tuple(sorted(set(cores)))
    if len(unique) != 8:
        raise ValueError(
            f"CPU_CORES must select exactly 8 logical cores; got {spec!r}"
        )
    return unique


CPU_CORES = parse_cpu_cores(os.environ.get("CPU_CORES", "0-7"))
THREADS = len(CPU_CORES)
BATCH_SIZE = 1
WARMUP_PASSES = 1
REPETITIONS = 5


def configure_cpu():
    """Lock the process and PyTorch to the selected eight-core protocol."""
    requested = set(CPU_CORES)
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("This benchmark requires Linux CPU affinity support.")
    os.sched_setaffinity(0, requested)
    actual = set(os.sched_getaffinity(0))
    if actual != requested:
        raise RuntimeError(f"CPU affinity mismatch: expected {requested}, got {actual}")

    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return sorted(actual)


def strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_strict(model, model_path):
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(strip_module_prefix(state_dict), strict=True)


def read_test_tensors(args, constants, max_samples=None):
    pad_id = args.tokenizer.convert_tokens_to_ids([constants.PAD_TOKEN])[0]
    src_rows, seg_rows = [], []

    with open(args.test_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "text_a" not in reader.fieldnames:
            raise ValueError("The test TSV must contain a text_a column.")

        for row in reader:
            src = args.tokenizer.convert_tokens_to_ids(
                [constants.CLS_TOKEN]
                + args.tokenizer.tokenize(row["text_a"])
                + [constants.SEP_TOKEN]
            )
            seg = [1] * len(src)
            src = src[: args.seq_length]
            seg = seg[: args.seq_length]
            if len(src) < args.seq_length:
                padding = args.seq_length - len(src)
                src.extend([pad_id] * padding)
                seg.extend([0] * padding)
            src_rows.append(src)
            seg_rows.append(seg)
            if max_samples is not None and len(src_rows) >= max_samples:
                break

    if not src_rows:
        raise ValueError("No test samples were loaded.")
    return torch.tensor(src_rows, dtype=torch.long), torch.tensor(seg_rows, dtype=torch.long)


def timed_pass(model, src, seg):
    """Match the historical boundary: sum only model.infer calls."""
    total_seconds = 0.0
    model.eval()
    with torch.inference_mode():
        for offset in range(0, src.size(0), BATCH_SIZE):
            src_batch = src[offset: offset + BATCH_SIZE]
            seg_batch = seg[offset: offset + BATCH_SIZE]
            started = time.perf_counter()
            model.infer(src_batch, seg_batch)
            total_seconds += time.perf_counter() - started
    return total_seconds


def ci95_half_width(values):
    if len(values) < 2:
        return 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def run_benchmark(model_name, model, args, constants, output_dir, max_samples=None):
    affinity = configure_cpu()
    src, seg = read_test_tensors(args, constants, max_samples=max_samples)

    for index in range(WARMUP_PASSES):
        timed_pass(model, src, seg)
        print(f"{model_name}: warm-up {index + 1}/{WARMUP_PASSES} complete", flush=True)

    totals = []
    for index in range(REPETITIONS):
        total = timed_pass(model, src, seg)
        totals.append(total)
        print(
            f"{model_name}: run {index + 1}/{REPETITIONS}, "
            f"total={total:.6f}s, per-sample={total * 1000 / src.size(0):.6f}ms",
            flush=True,
        )

    per_sample_ms = [total * 1000.0 / src.size(0) for total in totals]
    result = {
        "model": model_name,
        "batch_size": BATCH_SIZE,
        "samples": src.size(0),
        "threads": THREADS,
        "cpu_cores": affinity,
        "warmup_passes": WARMUP_PASSES,
        "repetitions": REPETITIONS,
        "timing_scope": "sum of model.infer calls only",
        "mean_ms_per_sample": statistics.mean(per_sample_ms),
        "ci95_half_ms_per_sample": ci95_half_width(per_sample_ms),
        "runs_ms_per_sample": per_sample_ms,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{model_name}.json"
    csv_path = output_dir / f"{model_name}.csv"
    payload = {
        "schema_version": 2,
        "model_path": str(Path(args.output_model_path).resolve()),
        "test_path": str(Path(args.test_path).resolve()),
        "results": [result],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "model", "batch_size", "samples", "threads", "cpu_cores",
            "warmup_passes", "repetitions", "mean_ms_per_sample",
            "ci95_half_ms_per_sample",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        row = dict(result)
        row["cpu_cores"] = ",".join(str(core) for core in affinity)
        writer.writerow({field: row[field] for field in fields})

    print(
        f"{model_name}: {result['mean_ms_per_sample']:.6f} ± "
        f"{result['ci95_half_ms_per_sample']:.6f} ms/sample (95% CI)"
    )
    return result
