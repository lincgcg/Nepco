#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AE_ROOT}"

: "${FINETUNED_MODEL_PATH:?Set FINETUNED_MODEL_PATH to a Nepco fine-tuned checkpoint.}"
: "${TEST_PATH:?Set TEST_PATH to test_dataset.tsv.}"
: "${LABELS_NUM:?Set LABELS_NUM to the number of classes.}"

mkdir -p "$(dirname "${FINETUNED_MODEL_PATH}")" "$(dirname "${FINETUNED_MODEL_PATH}")/prf"

CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" python3 scripts/3_latency_measure_Host/latency_gpu.py \
  --output_model_path "${FINETUNED_MODEL_PATH}" \
  --vocab_path "${VOCAB_PATH:-vocab/hex_vocab.txt}" \
  --config_path "${CONFIG_PATH:-configs/nepco_config.json}" \
  --test_path "${TEST_PATH}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --pooling max \
  --embedding word \
  --seq_length "${SEQ_LENGTH:-128}" \
  --labels_num "${LABELS_NUM}"
