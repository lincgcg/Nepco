#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${AE_ROOT}"

: "${FINETUNED_MODEL_PATH:?Set FINETUNED_MODEL_PATH to a Nepco fine-tuned checkpoint.}"
: "${TRAIN_PATH:?Set TRAIN_PATH to a train_dataset.tsv file for calibration.}"
: "${LABELS_NUM:?Set LABELS_NUM to the number of classes.}"

ONNX_PATH="${ONNX_PATH:-outputs/onnx}"
mkdir -p "${ONNX_PATH}"

python3 scripts/4_latency_measure_SmartNIC/1_onnx_conversion/onnx_export.py \
  --output_model_path "${FINETUNED_MODEL_PATH}" \
  --onnx_path "${ONNX_PATH}" \
  --vocab_path "${VOCAB_PATH:-vocab/hex_vocab.txt}" \
  --config_path "${CONFIG_PATH:-configs/nepco_config.json}" \
  --train_path "${TRAIN_PATH}" \
  --pooling max \
  --embedding word \
  --seq_length "${SEQ_LENGTH:-128}" \
  --labels_num "${LABELS_NUM}"
