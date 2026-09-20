#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${AE_ROOT}"

DATASET_PATH="${DATASET_PATH:-outputs/pretrain_data/dataset.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/model_pretrain}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${OUTPUT_DIR}/output_model.bin}"
VOCAB_PATH="${VOCAB_PATH:-vocab/hex_vocab.txt}"
CONFIG_PATH="${CONFIG_PATH:-configs/nepco_config.json}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" python3 scripts/2_model_train/model_pretrain/pretrain.py \
  --dataset_path "${DATASET_PATH}" \
  --vocab_path "${VOCAB_PATH}" \
  --config_path "${CONFIG_PATH}" \
  --output_model_path "${OUTPUT_MODEL_PATH}" \
  --world_size "${WORLD_SIZE:-1}" \
  --gpu_ranks ${GPU_RANKS:-0} \
  --total_steps "${TOTAL_STEPS:-100000}" \
  --save_checkpoint_steps "${SAVE_CHECKPOINT_STEPS:-10000}" \
  --data_processor mlm \
  --embedding word \
  --remove_embedding_layernorm \
  --encoder Nepco \
  --target mlm \
  --mask fully_visible \
  --span_masking \
  --span_geo_prob "${SPAN_GEO_PROB:-0.3}" \
  --span_max_length "${SPAN_MAX_LENGTH:-5}" \
  --batch_size "${BATCH_SIZE:-512}" \
  --learning_rate "${LEARNING_RATE:-1e-3}" \
  "$@"
