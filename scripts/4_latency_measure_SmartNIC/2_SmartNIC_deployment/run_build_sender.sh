#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

make -C "${SCRIPT_DIR}/send" clean all \
  PORT_ID="${SENDER_PORT_ID:-1}"
