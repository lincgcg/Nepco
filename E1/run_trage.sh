#!/usr/bin/env bash
set -euo pipefail
E1_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "$E1_DIR/reproduce_trage.py" "$@"
