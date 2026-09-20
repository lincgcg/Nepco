#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SENDER_LCORES="${SENDER_LCORES:-0-15}"
SENDER_MEM_CHANNELS="${SENDER_MEM_CHANNELS:-4}"
SENDER_PCI="${SENDER_PCI:-0000:03:00.1}"

sudo "${SCRIPT_DIR}/send/dpdk_sender" \
  -l "${SENDER_LCORES}" \
  -n "${SENDER_MEM_CHANNELS}" \
  -a "${SENDER_PCI}" \
  --
