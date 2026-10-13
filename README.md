# _Nepco_

Versatile yet Efficient Network Traffic Analysis: Offloading Network Foundation Model to SmartNIC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22893365.svg)](https://doi.org/10.5281/zenodo.22893365)

## Citation

The accepted manuscript is available on [arXiv](https://arxiv.org/abs/2508.02001).

```bibtex
@inproceedings{lin2025versatile,
  title={Versatile yet Efficient Network Traffic Analysis: Offloading Network Foundation Model to SmartNIC},
  author={Lin, Chungang and Meng, Xuying and Zuo, Tianyu and Zhang, Weiyao and Shen, Meng and Zhao, Ruijie and Che, Guanming and Meng, Ruiqi and Huang, Ziyue and Luo, Haitong and others},
  booktitle = {34th Annual Network and Distributed System Security Symposium (NDSS)},
  year = {2027},
  note = {To appear},
  url = {https://arxiv.org/abs/2508.02001}
}
```

## Introduction

This repository contains the artifacts needed to reproduce the main Nepco experiments. It includes the Nepco model implementation, the model configuration, the vocabulary, a released Nepco pre-trained checkpoint, and the scripts used for data processing, model training, latency measurement, SmartNIC deployment, and adversarial robustness evaluation.

The artifacts are organized as follows:

- `configs/`: Nepco model configuration, including `nepco_config.json`.
- `uer/`: the Nepco-only UER runtime used by the scripts.
- `vocab/`: the hexadecimal vocabulary used by Nepco.
- `models/`: released model resources, including `nepco_pretrained_model.bin`.
- `scripts/`: executable workflows for reproducing the experiments.
- `E1/`: versatility evaluation through Nepco and TraGe classification on nine datasets; fine-tuned checkpoints and test TSVs are downloaded separately.
- `E2/`: efficiency evaluation through full-test host CPU/GPU benchmarks for Nepco and TraGe, plus instructions for Nepco SmartNIC deployment.
- `E_Mapping/`: paper figure/table mappings, resource requirements, and example commands.


## Function Summary

The scripts for reproducing the Nepco experiments are provided in `scripts/`. They are divided into five functional parts, corresponding to the main artifact-evaluation workflows.

E1 and E2 evaluate the two core aspects of Nepco: versatility and efficiency. E1 reports Nepco and TraGe classification performance on nine complete test sets (TABLE III and TABLE VI). E2.1 compares their host CPU/GPU inference latency (Fig. 9), while E2.2 provides the Nepco SmartNIC deployment workflow (Figs. 7–8 and TABLE IV). E1 and E2.1 reuse released checkpoints without retraining; E2.2 requires two BlueField-3 SmartNICs. See [E_Mapping/README.md](E_Mapping/README.md) for the corresponding paper experiments.

`1_input_data_processing` provides the data preparation pipeline. It supports pre-training data generation by converting raw pcap/pcapng files into a hex-token traffic corpus and then building the pre-training dataset. It also supports fine-tuning data generation by converting labeled pcap directories into Nepco-compatible train/validation/test TSV files.

`2_model_train` provides the model training pipeline. It contains the Nepco pre-training workflow and the Nepco fine-tuning workflow. The pre-training workflow trains Nepco with the MLM objective, while the fine-tuning workflow loads a pre-trained Nepco checkpoint and trains the final traffic classifier on labeled downstream datasets.

`3_latency_measure_Host` provides host-side latency measurement. It measures Nepco inference latency on CPU and GPU using a fine-tuned checkpoint and a TSV test set. The CPU script reports latency over multiple batch sizes, and the GPU script reports latency for the configured batch size.

`4_latency_measure_SmartNIC` provides the SmartNIC deployment workflow. It first converts a fine-tuned Nepco checkpoint to ONNX format and then provides the BlueField-3 sender/receiver programs for online SmartNIC deployment and latency measurement.

`5_adversarial_robustness` provides the adversarial robustness workflow. It fine-tunes Nepco on raw data, computes gradient-based byte importance, constructs Top-X and Random-X randomized inputs under the same perturbation budget, evaluates the fixed raw model.

## Environment

Run all commands from the AE package root:

```bash
cd Nepco
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The Python workflows require PyTorch, Scapy, NumPy, scikit-learn, tqdm, and the ONNX packages listed in `requirements.txt`. The SmartNIC deployment scripts are intended for a BlueField-3 environment with DPDK and ONNX Runtime installed.

## 1. Input Data Processing

This part provides the input generation pipeline for Nepco. It converts raw packet captures into the 4-hex-character token representation used by `vocab/hex_vocab.txt`.

The pre-training pipeline produces a plain traffic corpus and then converts it to an MLM pre-training dataset. The fine-tuning pipeline converts labeled pcap directories into `train_dataset.tsv`, `valid_dataset.tsv`, and `test_dataset.tsv` files.

Important configurable parameters include:

- `pcap_dir` or `PCAP_DIR`: input pcap/pcapng directory.
- `packet_hex_chars`: maximum number of hex characters kept per packet.
- `token_nibbles`: number of hex characters per token. Use `4` for Nepco.
- `max_packets_per_flow`: maximum number of packets read from each pcap.
- `class_num` or `CLASS_NUM`: number of classes for fine-tuning data.
- `random_seed` or `RANDOM_SEED`: random seed for labeled data sampling.

### 1.1 Pre-training Corpus Generation

```bash
python3 scripts/1_input_data_processing/generate_pretrain_corpus.py \
  --pcap_dir /path/to/pretraining/pcaps \
  --output_corpus outputs/pretrain_data/traffic_corpus.txt \
  --suffixes .pcap,.pcapng \
  --packet_hex_chars 256 \
  --max_packets_per_flow 5 \
  --token_nibbles 4
```

By default, the corpus generator anonymizes MAC addresses, IP addresses, and transport ports. Use `--keep_addresses` only if address and port fields should be preserved.

The generated corpus can then be converted into the pre-training dataset:

```bash
python3 scripts/1_input_data_processing/build_pretrain_dataset.py \
  --corpus_path outputs/pretrain_data/traffic_corpus.txt \
  --dataset_path outputs/pretrain_data/dataset.pt \
  --vocab_path vocab/hex_vocab.txt \
  --processes_num 1 \
  --data_processor mlm \
  --seq_length 128 \
  --dup_factor 5 \
  --span_masking \
  --span_geo_prob 0.3 \
  --span_max_length 5
```

The same two steps can be executed with the wrapper script:

```bash
PCAP_DIR=/path/to/pretraining/pcaps \
bash scripts/1_input_data_processing/run_pretrain_data_generation.sh
```

After this step, the default output files are:

```text
outputs/pretrain_data/traffic_corpus.txt
outputs/pretrain_data/dataset.pt
```

If multiple corpus files need to be merged, use:

```bash
python3 scripts/1_input_data_processing/combine_corpus.py \
  --input_files corpus_1.txt corpus_2.txt corpus_3.txt \
  --output_file outputs/pretrain_data/traffic_corpus_all.txt \
  --shuffle \
  --seed 7
```

### 1.2 Fine-tuning Data Generation

The labeled pcap directory should contain one subdirectory per class. Class names are sorted and mapped to integer labels.

```bash
python3 scripts/1_input_data_processing/generate_finetune_dataset.py \
  --pcap_path /path/to/labeled/pcaps/ \
  --dataset_dir outputs/finetune_data/datasets/ \
  --middle_save_path outputs/finetune_data/cache/ \
  --class_num 15 \
  --random_seed 1 \
  --packet_hex_chars 256 \
  --token_nibbles 4 \
  --max_packets_per_flow 5
```

The same step can be executed with the wrapper script:

```bash
PCAP_DIR=/path/to/labeled/pcaps \
CLASS_NUM=15 \
RANDOM_SEED=1 \
bash scripts/1_input_data_processing/run_finetune_data_generation.sh
```

After this step, the default output files are:

```text
outputs/finetune_data/datasets/train_dataset.tsv
outputs/finetune_data/datasets/valid_dataset.tsv
outputs/finetune_data/datasets/test_dataset.tsv
outputs/finetune_data/datasets/nolabel_test_dataset.tsv
outputs/finetune_data/cache/dataset.json
outputs/finetune_data/cache/picked_file_record
outputs/finetune_data/cache/dataset/
```

The fine-tuning split uses an 8:1:1 train/validation/test ratio.

## 2. Model Train

This part provides Nepco model pre-training and fine-tuning. The pre-training workflow trains Nepco with the MLM objective. The fine-tuning workflow loads a Nepco pre-trained checkpoint and trains a classifier on labeled traffic data.

The model architecture of Nepco is specified by `configs/nepco_config.json`.
The main fine-tuning scripts default to a learning rate of `5e-4`. For different traffic analysis tasks, the `LEARNING_RATE` argument can be overridden after performing task-specific fine-tuning learning-rate search.

### 2.1 Model Pre-training

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/2_model_train/model_pretrain/pretrain.py \
  --dataset_path outputs/pretrain_data/dataset.pt \
  --vocab_path vocab/hex_vocab.txt \
  --config_path configs/nepco_config.json \
  --output_model_path outputs/model_pretrain/output_model.bin \
  --world_size 1 \
  --gpu_ranks 0 \
  --total_steps 100000 \
  --save_checkpoint_steps 10000 \
  --data_processor mlm \
  --embedding word \
  --remove_embedding_layernorm \
  --encoder Nepco \
  --target mlm \
  --mask fully_visible \
  --span_masking \
  --span_geo_prob 0.3 \
  --span_max_length 5 \
  --batch_size 512 \
  --learning_rate 1e-3
```

The same step can be executed with the wrapper script:

```bash
DATASET_PATH=outputs/pretrain_data/dataset.pt \
GPU_IDS=0 \
bash scripts/2_model_train/model_pretrain/run_pretrain.sh
```

After model pre-training, checkpoints are saved as:

```text
outputs/model_pretrain/output_model.bin-<step>
```

For AE reproduction, a released checkpoint is already provided at:

```text
models/nepco_pretrained_model.bin
```

### 2.2 Model Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/2_model_train/model_finetune/finetune.py \
  --pretrained_model_path models/nepco_pretrained_model.bin \
  --output_model_path outputs/model_finetune/models/finetuned_model.bin \
  --vocab_path vocab/hex_vocab.txt \
  --config_path configs/nepco_config.json \
  --train_path outputs/finetune_data/datasets/train_dataset.tsv \
  --dev_path outputs/finetune_data/datasets/valid_dataset.tsv \
  --test_path outputs/finetune_data/datasets/test_dataset.tsv \
  --epochs_num 10 \
  --batch_size 32 \
  --pooling max \
  --embedding word \
  --learning_rate 5e-4 \
  --seq_length 128 \
  --labels_num 15
```

The same step can be executed with the wrapper script:

```bash
PRETRAINED_MODEL_PATH=models/nepco_pretrained_model.bin \
TRAIN_PATH=outputs/finetune_data/datasets/train_dataset.tsv \
DEV_PATH=outputs/finetune_data/datasets/valid_dataset.tsv \
TEST_PATH=outputs/finetune_data/datasets/test_dataset.tsv \
LABELS_NUM=15 \
GPU_IDS=0 \
bash scripts/2_model_train/model_finetune/run_finetune.sh
```

After model fine-tuning, the default output files are:

```text
outputs/model_finetune/models/finetuned_model.bin
outputs/model_finetune/models/prf/prf.csv
```

`finetuned_model.bin` is the checkpoint with the best validation accuracy. `prf.csv` contains per-class precision, recall, and F1-score on the test set.

### 2.3 Few-shot Evaluation

`few_shot.py` samples up to a specified number of training examples per class and fine-tunes Nepco separately for each shot count. Only the training set is sampled; the validation set and the full supplied test set remain unchanged. Runs execute sequentially at one specified learning rate, without an automatic learning-rate search.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/2_model_train/model_finetune/few_shot.py \
  --pretrained_model_path models/nepco_pretrained_model.bin \
  --config_path configs/nepco_config.json \
  --vocab_path vocab/hex_vocab.txt \
  --train_path outputs/finetune_data/datasets/train_dataset.tsv \
  --dev_path outputs/finetune_data/datasets/valid_dataset.tsv \
  --test_path outputs/finetune_data/datasets/test_dataset.tsv \
  --labels_num 15 \
  --shot_list 5 25 50 250 500 2500 5000 \
  --shot_seed 3 \
  --learning_rate 5e-4 \
  --epochs_num 10 \
  --batch_size 32 \
  --seq_length 128 \
  --output_root outputs/few_shot
```

Set `--labels_num` to the dataset's class count. With a nominal reference of 5,000 examples per class, the listed shot counts correspond to the labeled-data budgets of 0.1%, 0.5%, 1%, 5%, 10%, 50%, and 100% used for the Figure 6 experiment. These are per-class sample budgets, not percentages of the supplied training TSV. If a class contains fewer training examples than requested, all available examples are used and the actual count is recorded. Add `--strict_shots` to reject such cases.

Use `--prepare_only` to generate the sampled training TSVs without starting model training. The default sampling seed is `3`; the existing fine-tuning entry uses model seed `7`.

Each shot count has a separate output directory:

```text
outputs/few_shot/shot_<count>/train_dataset.tsv
outputs/few_shot/shot_<count>/sampling.json
outputs/few_shot/shot_<count>/train.log
outputs/few_shot/shot_<count>/models/finetuned_model.bin
outputs/few_shot/shot_<count>/models/prf/prf.csv
```

`sampling.json` records the requested and actual per-class sample counts. The checkpoint is selected using validation accuracy and evaluated on the full supplied test set. With `--prepare_only`, only `train_dataset.tsv` and `sampling.json` are generated. Existing shot directories are not overwritten; use a new `--output_root` for another run.

## 3. Latency Measure Host

This part measures Nepco inference latency on a host CPU or GPU. The scripts load a fine-tuned Nepco checkpoint and run inference on a TSV test set.

The CPU script evaluates multiple batch sizes from `1` to `1024`. For each batch size, it repeats inference five times and reports the mean per-sample latency with a 95% confidence interval. The GPU script measures the specified batch size with five repeated runs.

### 3.1 CPU Latency

```bash
python3 scripts/3_latency_measure_Host/latency_cpu.py \
  --output_model_path outputs/model_finetune/models/finetuned_model.bin \
  --vocab_path vocab/hex_vocab.txt \
  --config_path configs/nepco_config.json \
  --test_path outputs/finetune_data/datasets/test_dataset.tsv \
  --batch_size 32 \
  --pooling max \
  --embedding word \
  --seq_length 128 \
  --labels_num 15
```

The same step can be executed with the wrapper script:

```bash
FINETUNED_MODEL_PATH=outputs/model_finetune/models/finetuned_model.bin \
TEST_PATH=outputs/finetune_data/datasets/test_dataset.tsv \
LABELS_NUM=15 \
bash scripts/3_latency_measure_Host/run_latency_cpu.sh
```

### 3.2 GPU Latency

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/3_latency_measure_Host/latency_gpu.py \
  --output_model_path outputs/model_finetune/models/finetuned_model.bin \
  --vocab_path vocab/hex_vocab.txt \
  --config_path configs/nepco_config.json \
  --test_path outputs/finetune_data/datasets/test_dataset.tsv \
  --batch_size 32 \
  --pooling max \
  --embedding word \
  --seq_length 128 \
  --labels_num 15
```

The same step can be executed with the wrapper script:

```bash
FINETUNED_MODEL_PATH=outputs/model_finetune/models/finetuned_model.bin \
TEST_PATH=outputs/finetune_data/datasets/test_dataset.tsv \
LABELS_NUM=15 \
GPU_IDS=0 \
bash scripts/3_latency_measure_Host/run_latency_gpu.sh
```

The latency scripts print timing results to the terminal. They also write a test-set classification summary to:

```text
outputs/model_finetune/models/prf/prf.csv
```

## 4. Latency Measure SmartNIC

This part provides the SmartNIC deployment workflow. It first converts a fine-tuned Nepco checkpoint into ONNX and then deploys the ONNX model in a BlueField-3 sender/receiver setup.

### 4.1 ONNX Conversion

The ONNX conversion script exports the Nepco classifier, preprocesses the ONNX graph, and performs static INT8 quantization using calibration samples from the training set.

```bash
python3 scripts/4_latency_measure_SmartNIC/1_onnx_conversion/onnx_export.py \
  --output_model_path outputs/model_finetune/models/finetuned_model.bin \
  --onnx_path outputs/onnx \
  --vocab_path vocab/hex_vocab.txt \
  --config_path configs/nepco_config.json \
  --train_path outputs/finetune_data/datasets/train_dataset.tsv \
  --pooling max \
  --embedding word \
  --seq_length 128 \
  --labels_num 15
```

The same step can be executed with the wrapper script:

```bash
FINETUNED_MODEL_PATH=outputs/model_finetune/models/finetuned_model.bin \
TRAIN_PATH=outputs/finetune_data/datasets/train_dataset.tsv \
LABELS_NUM=15 \
bash scripts/4_latency_measure_SmartNIC/1_onnx_conversion/run_onnx_export.sh
```

After ONNX conversion, the default output files are:

```text
outputs/onnx/output.onnx
outputs/onnx/output.preprocessed.onnx
outputs/onnx/output.static_int8.onnx
```

The SmartNIC receiver uses `output.preprocessed.onnx` by default.

### 4.2 SmartNIC Deployment

The deployment code is in:

```text
scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/
```

The sender generates dynamic flow-distribution traffic. The receiver collects up to five packets per flow, zeroes address and transport-port fields in the copied model input, encodes packets with Nepco's 4-hex-character vocabulary, runs ONNX inference, and records online latency statistics. The original 5-tuple is still preserved for flow lookup and hardware filter-rule installation.

Build the receiver:

```bash
MODEL_PATH=/absolute/path/to/output.preprocessed.onnx \
VOCAB_PATH=/absolute/path/to/hex_vocab.txt \
LABELS_NUM=15 \
ONNX_ROOT="${ONNX_ROOT:?Set ONNX_ROOT to your ONNX Runtime installation}" \
bash scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/run_build_receiver.sh
```

Build the sender:

```bash
bash scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/run_build_sender.sh
```

Run the receiver:

```bash
ONNX_ROOT="${ONNX_ROOT:?Set ONNX_ROOT to your ONNX Runtime installation}" \
RECEIVER_PCI=0000:03:00.0,dv_flow_en=2 \
bash scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/run_receiver.sh
```

Run the sender:

```bash
SENDER_PCI=0000:03:00.1 \
bash scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/run_sender.sh
```

The run scripts invoke `sudo` internally when launching the DPDK applications.

Important configurable parameters include:

- `MODEL_PATH`: ONNX model compiled into the receiver.
- `VOCAB_PATH`: Nepco vocabulary compiled into the receiver.
- `LABELS_NUM`: number of output classes.
- `ONNX_ROOT`: ONNX Runtime installation root on BF3.
- `MAX_PACKETS_PER_FLOW`: number of packets collected before inference. The default is `5`.
- `TARGET_BYTE_LEN`: number of packet bytes encoded by the receiver. The default is `128`.
- `SENDER_PCI` and `RECEIVER_PCI`: DPDK PCI addresses.

The receiver writes latency logs and flow records to:

```text
outputs/SmartNIC_deployment/receiver/log.txt
outputs/SmartNIC_deployment/receiver/flow_tuples.txt
```

The detailed BF3 build and run notes are also provided in:

```text
scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/README.md
```

## 5. Adversarial Robustness

This part evaluates Nepco under gradient-attribution-guided token randomization. It compares Top-X and Random-X perturbations under the same perturbation budget.

The workflow contains three stages:

1. Fine-tune Nepco on Clean Train and Clean Valid, then evaluate Clean Test.
2. Compute gradient-based token importance with the clean model, build Top-X and Random-X attacked test sets, and evaluate the fixed clean model.
3. Build attacked train/valid/test sets and perform attack-aware fine-tuning on each attacked dataset.

The default ratios are `1 2 3 4 5`, corresponding to 1% through 5% randomized tokens. The default attack seed is `1`.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/5_adversarial_robustness/adversarial_robustness.py \
  --pretrained_model_path models/nepco_pretrained_model.bin \
  --vocab_path vocab/hex_vocab.txt \
  --config_path configs/nepco_config.json \
  --train_path outputs/finetune_data/datasets/train_dataset.tsv \
  --dev_path outputs/finetune_data/datasets/valid_dataset.tsv \
  --test_path outputs/finetune_data/datasets/test_dataset.tsv \
  --output_root outputs/adversarial_robustness \
  --model_name Nepco \
  --dataset_name CIC-EVSE \
  --run_seed 01 \
  --ratios 1 2 3 4 5 \
  --attack_seeds 1 \
  --epochs_num 10 \
  --batch_size 32 \
  --pooling max \
  --embedding word \
  --learning_rate 5e-4 \
  --seq_length 128 \
  --labels_num 15 \
  --importance_target predicted \
  --importance_reduce mean \
  --replacement_mode hex \
  --attack_aware_init pretrained \
  --save_attacked_tsv \
  --save_importance
```

The same step can be executed with the wrapper script:

```bash
PRETRAINED_MODEL_PATH=models/nepco_pretrained_model.bin \
TRAIN_PATH=outputs/finetune_data/datasets/train_dataset.tsv \
DEV_PATH=outputs/finetune_data/datasets/valid_dataset.tsv \
TEST_PATH=outputs/finetune_data/datasets/test_dataset.tsv \
LABELS_NUM=15 \
GPU_IDS=0 \
bash scripts/5_adversarial_robustness/run_adversarial_robustness.sh
```

For each attack seed, the script writes one summary CSV:

```text
outputs/adversarial_robustness/<Dataset>/<Model>/<Run_Seed>/results/adv_summary_AttackSeed_<seed>.csv
```

## E1. Versatility Evaluation

E1.1 evaluates Nepco and E1.2 evaluates TraGe on the nine datasets in TABLE III and TABLE VI. Their fine-tuned checkpoints and test TSVs are distributed separately through Google Drive (approximately 5 GB). Before running E1 or E2.1, follow [Download E1 artifacts](E1/artifacts/README.md) and place the nine dataset folders directly under `E1/artifacts/`. No retraining is required. For example, the DataCon Nepco model and test TSV must be located at:

```text
E1/artifacts/DataCon/finetuned_model.bin
E1/artifacts/DataCon/test_dataset.tsv
```

Run the full 2,857-sample test set from the repository root:

```bash
bash E1/run_smoke_test.sh
```

The evaluator loads the checkpoint with `strict=True` and writes
`metrics.json`, per-class `prf.csv`, and `predictions.tsv` under
`E1/results/DataCon/`. The verified result is:

```text
Precision: 0.9920384649
Recall:    0.9912944099
Macro F1:  0.9916037393
Accuracy:  0.9866993350
Samples:   2857
```

E1 also provides full-test evaluation on all nine datasets for Nepco and TraGe:

```bash
bash E1/run_smoke_test.sh --all --device cpu
bash E1/run_trage.sh all --device cpu
```

The manifests record the configurations, verified metrics, and artifact hashes.
See [E1/README.md](E1/README.md) for both models' verified full-test results and
the inference commands. No training is performed by these entry points.

## E2. Efficiency Evaluation

### E2.1: Host CPU/GPU Inference Latency

E2.1 compares Nepco and TraGe using the complete 2,857-sample DataCon test set
from E1. Both models are evaluated at batch sizes 1, 2, 4, 8, 16, 32, 64,
and 128; CPU mode additionally uses 8, 16, 32, and 64 physical cores.
The 80 CPU/GPU configurations run sequentially with one warm-up pass and five
measured passes each. Only model-forward execution is timed.

Run individual configurations or the complete matrix from the repository root:

```bash
python3 E2/benchmark_inference.py --model nepco --device cpu --cpu_cores 8 --batch_size 16
python3 E2/benchmark_inference.py --model trage --device cuda --gpu_index 0 --batch_size 128
bash E2/run_benchmark.sh --gpu_index 0
```

See [E2/README.md](E2/README.md) for the full-test results, timing boundaries,
and resource conditions. The full-test runner selects CPU cores dynamically
and saves new measurements under `E2/results/local_runs/`, separately from
the reference results under `E2/results/DataCon/full_benchmark/`.
Linux CPU-affinity/topology support and `nvidia-smi` are required; the complete
matrix also requires 64 available physical CPU cores and a CUDA-capable GPU.

### E2.2: SmartNIC Analysis Latency

E2.2 uses two connected NVIDIA BlueField-3 SmartNICs with DOCA/DPDK and
ONNX Runtime to measure Nepco throughput and analysis latency. Follow
[E2.2 in E2/README.md](E2/README.md#e22-smartnic-analysis-latency) for preparation,
sender/receiver commands, and result locations. The ONNX export and deployment
workflows are provided in [Section 4](#4-latency-measure-smartnic) above.

The original eight-sample CPU smoke test remains available as an optional
timing-path check with `bash E2/run_smoke_test.sh`. It is separate from the
complete-test-set evaluation in E2.1.
