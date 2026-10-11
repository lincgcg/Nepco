# E2: Nepco CPU latency smoke test

E2 is a functional smoke test for the Nepco CPU inference path. It loads the
fine-tuned DataCon checkpoint and test TSV already packaged by E1, runs Nepco
inference, and writes machine-readable timing results.

The smoke test intentionally answers one question: can the released Nepco
checkpoint and TSV be loaded and timed end to end? It is not intended to
produce a paper-grade host benchmark or an ET-BERT comparison.

## Fixed smoke-test protocol

- Model and data: `E1/artifacts/DataCon/finetuned_model.bin` and
  `E1/artifacts/DataCon/test_dataset.tsv`.
- Samples: first 8 test rows by default.
- Batch size: 1.
- CPU affinity: eight logical cores, default `0-7`, configurable with `CPU_CORES`.
- PyTorch/OMP/MKL/OpenBLAS threads: 8.
- Warm-up: one pass.
- Measurement: five passes.
- Timing boundary: the sum of `model.infer` calls only; checkpoint loading,
  TSV parsing, tokenization, tensor construction, and output writing are
  excluded.
- Statistic: mean milliseconds per sample and a 95% confidence-interval
  half-width across the five passes.

The checkpoint is loaded with `strict=True`, so an architecture mismatch fails
instead of silently benchmarking a partially loaded model.

## Run

From the AE package root:

```bash
bash E2/run_smoke_test.sh
```

The result files are:

```text
E2/results/DataCon/nepco.json
E2/results/DataCon/nepco.csv
```

`MAX_SAMPLES` may be increased for a longer timing check while keeping the
batch size and CPU-core assignment fixed:

```bash
MAX_SAMPLES=32 bash E2/run_smoke_test.sh
```

The script requires Linux `taskset`. `CPU_CORES` must select exactly eight
distinct logical cores available to the current process. For example:

```bash
CPU_CORES=8-15 bash E2/run_smoke_test.sh
```

Choose core IDs allowed on your machine; the default `0-7` is not available in
every restricted environment.

## Verified functional run

The packaged smoke test used logical cores 168–175 and completed successfully
with a mean of `2.271679 ms/sample`
and a 95% CI half-width of `0.201898 ms/sample` over the five eight-sample
passes. This value confirms that the timing path works; it should not be cited
as a full-dataset latency result.

## Full-test inference efficiency: Nepco and TraGe

The original smoke test above is unchanged in purpose. This additional experiment
compares Nepco and TraGe using the entire accepted DataCon test set from E1.
No model, vocabulary, or test dataset is duplicated in E2, and no training is run.

### Full-test protocol

- All 2857 test samples in every pass, including the final partial batch.
- Batch sizes: 1, 2, 4, 8, 16, 32, 64, and 128.
- CPU: 8, 16, 32, and 64 physical cores; one logical thread per physical core,
  balanced across available sockets, with intra-op threads equal to core count
  and inter-op threads set to 1. GPU runs use 8 host physical cores.
- FP32 eager inference, `eval()` and `inference_mode()`; no TF32, mixed precision,
  quantization, or compilation.
- One full warm-up pass and five full measured passes per configuration.
- Prebuilt device-local input batches. Only logits-forward calls are timed;
  loading, tokenization, input construction, transfers, argmax, and output writing
  are excluded. GPU host-wall timing synchronizes CUDA; CUDA event timings are
  also retained.
- Every warm-up and measured pass must exactly match E1 reference predictions.
- All 80 configurations run sequentially in separate processes. CPU measurement
  waits for all external GPU compute jobs to finish; GPU measurement requires the
  selected card to be free. Resource checks run between batches at approximately
  one-second intervals. If activity is detected, the current configuration is
  discarded and the matrix runner waits before retrying it from full warm-up.
  Other users' processes are never stopped.
- Primary metric: mean amortized milliseconds per sample across five passes.
  Standard deviations, per-pass throughput, batch timings, and tail sizes are
  retained in JSON. At batch sizes greater than 1, amortized ms/sample is not
  single-request response latency.

The new runner requires Linux affinity/topology support and `nvidia-smi`, including
for the CPU resource gate, plus the Python dependencies used by E1. It selects
physical cores dynamically; insufficient core capacity is an error, not a reason
to substitute SMT threads. Choose an available GPU with `--gpu_index`.

### Run the additional full-test experiment

From the AE package root, run a single configuration:

```bash
python3 E2/benchmark_inference.py --model nepco --device cpu --cpu_cores 8 --batch_size 16
python3 E2/benchmark_inference.py --model trage --device cuda --gpu_index 0 --batch_size 128
```

Run the complete 80-configuration matrix (GPU first, then CPU):

```bash
bash E2/run_benchmark.sh --gpu_index 0
```

New measurements go to `E2/results/local_runs/` by default and never overwrite
the packaged verified results. Completed configurations are not overwritten:
use a fresh `--output_dir` for another run. The matrix runner retries resource-busy
configurations only; other errors stop the run for inspection. `PYTHON_BIN` may
optionally select a Python interpreter for the shell entry point, without a
hard-coded environment path. Do not run old/new timing commands concurrently.

### Verified full-test results

All 80 configurations completed and passed an independent audit: exact matrix,
full sample counts and tails, reference prediction equality, raw timing and
aggregate arithmetic, result hashes, and unchanged E1/E2 inputs at measurement
time. Values below are **ms/sample**, averaged over five full-test passes;
lower is better. These results are separate from the eight-sample smoke test.

#### GPU

| Batch size | Nepco | TraGe |
|---:|---:|---:|
| 1 | 0.570781 | 4.807048 |
| 2 | 0.287564 | 2.439468 |
| 4 | 0.147001 | 1.531708 |
| 8 | 0.073163 | 1.082531 |
| 16 | 0.037462 | 0.817490 |
| 32 | 0.019976 | 0.745115 |
| 64 | 0.013070 | 0.685009 |
| 128 | 0.011993 | 0.625392 |

#### Nepco CPU

| Batch size | 8 cores | 16 cores | 32 cores | 64 cores |
|---:|---:|---:|---:|---:|
| 1 | 2.077645 | 2.307808 | 4.324003 | 5.273061 |
| 2 | 1.691219 | 2.547725 | 4.097549 | 4.627781 |
| 4 | 1.442090 | 1.700522 | 4.059684 | 4.142761 |
| 8 | 1.078516 | 1.196222 | 3.634029 | 4.316085 |
| 16 | 0.850767 | 2.073442 | 2.916396 | 2.171513 |
| 32 | 0.876924 | 2.483391 | 3.506448 | 3.127300 |
| 64 | 0.899153 | 2.568364 | 3.078269 | 3.912377 |
| 128 | 1.092309 | 2.561189 | 3.403550 | 4.134102 |

#### TraGe CPU

| Batch size | 8 cores | 16 cores | 32 cores | 64 cores |
|---:|---:|---:|---:|---:|
| 1 | 45.542681 | 34.362860 | 29.128429 | 29.979410 |
| 2 | 39.462596 | 24.402240 | 19.837938 | 18.855424 |
| 4 | 43.250825 | 29.952709 | 24.009978 | 20.974976 |
| 8 | 41.668792 | 27.285640 | 21.540364 | 16.124533 |
| 16 | 40.622424 | 26.239629 | 19.744437 | 16.326462 |
| 32 | 43.720686 | 30.123954 | 22.084880 | 19.650226 |
| 64 | 42.674548 | 29.063351 | 22.177259 | 19.752021 |
| 128 | 55.284424 | 38.444319 | 29.892895 | 29.359080 |

These are shared-host measurements, not exclusive-host or end-to-end latency
guarantees. GPU runs allowed external work on other cards; CPU runs required no
external GPU compute activity but did not reserve the entire host. One-second
polling cannot rule out subsecond interference. CPU affinity was balanced across
two sockets without explicit NUMA memory binding. Exact timings depend on
hardware, runtime, and load; the portable runner does not promise identical values
on another machine. Host identity and absolute server paths are not published.

### Additional result files

```text
E2/full_benchmark_manifest.json
E2/results/DataCon/full_benchmark/summary.csv
E2/results/DataCon/full_benchmark/summary.json
E2/results/DataCon/full_benchmark/Nepco/cpu/c8-b16.json
E2/results/DataCon/full_benchmark/Nepco/gpu/b128.json
E2/results/DataCon/full_benchmark/TraGe/cpu/c64-b8.json
E2/results/DataCon/full_benchmark/TraGe/gpu/b128.json
```

The model/device directories contain all 80 original configuration records.
`full_benchmark_manifest.json` records the protocol, relative input references,
input hashes, and hashes of every packaged result. The historical smoke test
continues to use its original scripts and `results/DataCon/nepco.*` files.
