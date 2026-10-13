# E2: Efficiency Evaluation

E2 evaluates traffic-analysis latency under host CPU/GPU deployment (E2.1,
Fig. 9) and SmartNIC deployment (E2.2, Figs. 7–8 and TABLE IV), supporting
the efficiency claim (C2). E2.1 compares Nepco and TraGe on the complete
DataCon test set; E2.2 uses the Nepco BlueField-3 deployment workflow.

## E2.1: Host CPU/GPU Inference Latency

### Preparation

Download the DataCon artifacts following [Download E1 artifacts](../E1/artifacts/README.md).
This experiment reuses the Nepco and TraGe checkpoints and the complete
2,857-sample test TSV from E1 without duplicating artifacts or retraining.
The full matrix requires up to 64 available physical CPU cores and a
CUDA-capable NVIDIA GPU. Linux CPU-affinity/topology support and `nvidia-smi`
are required, including for the CPU resource gate.

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

The runner selects physical cores dynamically; insufficient core capacity is
an error, not a reason to substitute SMT threads. Choose an available GPU with
`--gpu_index`.

### Execution

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

### Results

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

### Result files

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

## E2.2: SmartNIC Analysis Latency

### Preparation

Prepare two connected NVIDIA BlueField-3 SmartNICs with DOCA/DPDK and
ONNX Runtime for C/C++. Configure device access, hugepages, and CPU affinity
on an isolated test link. Export the selected Nepco checkpoint from the
repository root, supplying its calibration training TSV and class count:

```bash
FINETUNED_MODEL_PATH=outputs/model_finetune/models/finetuned_model.bin \
TRAIN_PATH=outputs/finetune_data/datasets/train_dataset.tsv \
LABELS_NUM=15 \
bash scripts/4_latency_measure_SmartNIC/1_onnx_conversion/run_onnx_export.sh
```

Replace the example inputs and class count with those for the selected model.
The exporter requires a training TSV for INT8 calibration; the E1 test TSV
is not a substitute. Transfer the exported model, vocabulary, and deployment
sources to the receiver if using a separate system. The deployment uses
`outputs/onnx/output.preprocessed.onnx` by default.

### Execution

Run from the repository root on each corresponding system:

```bash
DEPLOY=scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment
```

On the receiver, export `ONNX_ROOT` (ONNX Runtime installation), `MODEL_PATH`,
`VOCAB_PATH`, `LABELS_NUM`, and `RECEIVER_PCI` for the testbed, then run:

```bash
bash "$DEPLOY/run_build_receiver.sh"
bash "$DEPLOY/run_receiver.sh"
```

On the sender, export `SENDER_PCI` and `SENDER_PORT_ID`, then run:

```bash
bash "$DEPLOY/run_build_sender.sh"
bash "$DEPLOY/run_sender.sh"
```

The runtime wrappers invoke `sudo` for device access. See the
[SmartNIC deployment instructions](../scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/README.md)
for the available settings.

### Results

The sender reports throughput; the receiver logs queue wait, input construction,
inference, and total analysis latency in `log.txt` under `OUTPUT_DIR`
(default `outputs/SmartNIC_deployment/receiver/`). These measurements support
the SmartNIC evaluation associated with Figs. 7–8 and TABLE IV. Separate
collection time when reporting post-collection analysis latency. Measurement
settings and aggregation for the corresponding figures and tables are outlined
in [E_Mapping/README.md](../E_Mapping/README.md). E2.1 provides a host-side
comparison without SmartNIC hardware.

## Optional quick CPU timing check

The original eight-sample Nepco smoke test remains available without changing
its scripts or protocol:

```bash
bash E2/run_smoke_test.sh
```

It uses the DataCon Nepco checkpoint, batch size 1, eight logical CPU cores,
one warm-up pass, and five measured passes. Only `model.infer` calls are timed;
loading, tokenization, tensor construction, and output writing are excluded.
The checkpoint is loaded with `strict=True`.

Linux `taskset` is required. `CPU_CORES` must select exactly eight available
logical cores; its default is `0-7`. For example:

```bash
CPU_CORES=8-15 bash E2/run_smoke_test.sh
```

Choose core IDs available on your machine. `MAX_SAMPLES` can increase the
sample count while keeping the batch size and core count fixed. Outputs are
`E2/results/DataCon/nepco.json` and `E2/results/DataCon/nepco.csv`.
The archived functional run reports `2.271679 ± 0.201898 ms/sample`
(mean and 95% CI half-width). This eight-sample check is separate from the
full-test E2.1 comparison above.
