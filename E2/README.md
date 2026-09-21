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
- CPU affinity: exactly eight fixed logical cores. The portable default is
  `0-7`; set `CPU_CORES` to use a different eight-core set.
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
PYTHON_BIN=python3 bash E2/run_smoke_test.sh
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

To repeat the verified h800 placement, run:

```bash
CPU_CORES=168-175 bash E2/run_smoke_test.sh
```

The script requires Linux `taskset`. `CPU_CORES` must name exactly eight
logical cores available to the current process.

## Verified functional run

The packaged smoke test was rerun on host `cstor`, pinned to logical cores
168–175, with Python 3.10.14 and PyTorch 2.3.0. It completed successfully with
a mean of `2.271679 ms/sample` and a 95% CI half-width of
`0.201898 ms/sample` over the five eight-sample passes. This value confirms
that the timing path works; it should not be cited as a full-dataset latency
result or compared across hosts.
