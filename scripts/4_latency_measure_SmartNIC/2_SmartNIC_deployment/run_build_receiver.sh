#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ONNX_ROOT="${ONNX_ROOT:-/home/ubuntu/pre2/onnx}"
MODEL_PATH="${MODEL_PATH:-${AE_ROOT}/outputs/onnx/output.preprocessed.onnx}"
VOCAB_PATH="${VOCAB_PATH:-${AE_ROOT}/vocab/hex_vocab.txt}"
LABELS_NUM="${LABELS_NUM:-2}"

make -C "${SCRIPT_DIR}/rec" clean all \
  ONNX_ROOT="${ONNX_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  NUM_CLASSES="${LABELS_NUM}" \
  SEQ_LENGTH="${SEQ_LENGTH:-128}" \
  MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-5}" \
  TARGET_BYTE_LEN="${TARGET_BYTE_LEN:-128}"
