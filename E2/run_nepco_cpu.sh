#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/DataCon}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
CPU_CORES="${CPU_CORES:-0-7}"

taskset -c "${CPU_CORES}" env \
  CPU_CORES="${CPU_CORES}" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_classifier_DCS_nowandb_measure_Nepco_CPU.py" \
  --output_dir "${RESULTS_DIR}" \
  --max_samples "${MAX_SAMPLES}" \
  "$@"
