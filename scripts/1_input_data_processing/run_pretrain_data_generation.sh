#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${AE_ROOT}"

: "${PCAP_DIR:?Set PCAP_DIR to the raw pcap directory.}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/pretrain_data}"
CORPUS_PATH="${CORPUS_PATH:-${OUTPUT_DIR}/traffic_corpus.txt}"
DATASET_PATH="${DATASET_PATH:-${OUTPUT_DIR}/dataset.pt}"
VOCAB_PATH="${VOCAB_PATH:-vocab/hex_vocab.txt}"

mkdir -p "${OUTPUT_DIR}"

python3 scripts/1_input_data_processing/generate_pretrain_corpus.py \
  --pcap_dir "${PCAP_DIR}" \
  --output_corpus "${CORPUS_PATH}" \
  --packet_hex_chars "${PACKET_HEX_CHARS:-256}" \
  --max_packets_per_flow "${MAX_PACKETS_PER_FLOW:-5}" \
  --token_nibbles "${TOKEN_NIBBLES:-4}" \
  "$@"

python3 scripts/1_input_data_processing/build_pretrain_dataset.py \
  --corpus_path "${CORPUS_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --vocab_path "${VOCAB_PATH}" \
  --processes_num "${PROCESSES_NUM:-1}" \
  --data_processor mlm \
  --seq_length "${SEQ_LENGTH:-128}" \
  --span_masking \
  --span_geo_prob "${SPAN_GEO_PROB:-0.3}" \
  --span_max_length "${SPAN_MAX_LENGTH:-5}"
