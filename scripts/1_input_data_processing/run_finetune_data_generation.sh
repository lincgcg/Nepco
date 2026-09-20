#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AE_ROOT}"

: "${PCAP_DIR:?Set PCAP_DIR to the labeled pcap directory.}"
: "${CLASS_NUM:?Set CLASS_NUM to the number of classes.}"

DATASET_DIR="${DATASET_DIR:-outputs/finetune_data/datasets}"
MIDDLE_SAVE_PATH="${MIDDLE_SAVE_PATH:-outputs/finetune_data/cache}"
RANDOM_SEED="${RANDOM_SEED:-1}"
PACKET_HEX_CHARS="${PACKET_HEX_CHARS:-256}"
TOKEN_NIBBLES="${TOKEN_NIBBLES:-4}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-5}"

mkdir -p "${DATASET_DIR}" "${MIDDLE_SAVE_PATH}"

python3 scripts/1_input_data_processing/generate_finetune_dataset.py \
  --pcap_path "${PCAP_DIR%/}/" \
  --dataset_dir "${DATASET_DIR%/}/" \
  --middle_save_path "${MIDDLE_SAVE_PATH%/}/" \
  --class_num "${CLASS_NUM}" \
  --random_seed "${RANDOM_SEED}" \
  --packet_hex_chars "${PACKET_HEX_CHARS}" \
  --token_nibbles "${TOKEN_NIBBLES}" \
  --max_packets_per_flow "${MAX_PACKETS_PER_FLOW}"
