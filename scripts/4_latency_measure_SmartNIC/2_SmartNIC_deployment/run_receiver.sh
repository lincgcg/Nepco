#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ONNX_ROOT="${ONNX_ROOT:-/home/ubuntu/pre2/onnx}"
RECEIVER_LCORES="${RECEIVER_LCORES:-0-15}"
RECEIVER_MEM_CHANNELS="${RECEIVER_MEM_CHANNELS:-4}"
RECEIVER_PCI="${RECEIVER_PCI:-0000:03:00.0,dv_flow_en=2}"
RECEIVER_FILE_PREFIX="${RECEIVER_FILE_PREFIX:-bf3_app}"
RECEIVER_PORT_ID="${RECEIVER_PORT_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${AE_ROOT}/outputs/SmartNIC_deployment/receiver}"

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

sudo env LD_LIBRARY_PATH="${ONNX_ROOT}/lib:${LD_LIBRARY_PATH:-}" \
  "${SCRIPT_DIR}/rec/analysis" \
  -l "${RECEIVER_LCORES}" \
  -n "${RECEIVER_MEM_CHANNELS}" \
  -a "${RECEIVER_PCI}" \
  --file-prefix="${RECEIVER_FILE_PREFIX}" \
  -- \
  -p "${RECEIVER_PORT_ID}"
