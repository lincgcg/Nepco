#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${AE_ROOT}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-models/nepco_pretrained_model.bin}"
if [[ ! -f "${PRETRAINED_MODEL_PATH}" ]]; then
  echo "Missing Nepco pretraining checkpoint: ${PRETRAINED_MODEL_PATH}" >&2
  echo "Set PRETRAINED_MODEL_PATH or place nepco_pretrained_model.bin under models/." >&2
  exit 1
fi
: "${TRAIN_PATH:?Set TRAIN_PATH to train_dataset.tsv.}"
: "${DEV_PATH:?Set DEV_PATH to valid_dataset.tsv.}"
: "${TEST_PATH:?Set TEST_PATH to test_dataset.tsv.}"
: "${LABELS_NUM:?Set LABELS_NUM to the number of classes.}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/model_finetune}"
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/models}"
PRF_DIR="${PRF_DIR:-${MODEL_DIR}/prf}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${MODEL_DIR}/finetuned_model.bin}"

mkdir -p "${MODEL_DIR}" "${PRF_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" python3 scripts/2_model_train/model_finetune/finetune.py \
  --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
  --output_model_path "${OUTPUT_MODEL_PATH}" \
  --vocab_path "${VOCAB_PATH:-vocab/hex_vocab.txt}" \
  --config_path "${CONFIG_PATH:-configs/nepco_config.json}" \
  --train_path "${TRAIN_PATH}" \
  --dev_path "${DEV_PATH}" \
  --test_path "${TEST_PATH}" \
  --epochs_num "${EPOCHS_NUM:-10}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --pooling max \
  --embedding word \
  --learning_rate "${LEARNING_RATE:-5e-4}" \
  --seq_length "${SEQ_LENGTH:-128}" \
  --labels_num "${LABELS_NUM}"
