#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${AE_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" E1/reproduce.py \
  --output_model_path "${MODEL_PATH:-E1/artifacts/DataCon/finetuned_model.bin}" \
  --test_path "${TEST_PATH:-E1/artifacts/DataCon/test_dataset.tsv}" \
  --config_path "${CONFIG_PATH:-configs/nepco_config.json}" \
  --vocab_path "${VOCAB_PATH:-vocab/hex_vocab.txt}" \
  --labels_num 10 \
  --batch_size "${BATCH_SIZE:-32}" \
  --pooling max \
  --embedding word \
  --seq_length 128 \
  --results_dir "${RESULTS_DIR:-E1/results/DataCon}" \
  "$@"
