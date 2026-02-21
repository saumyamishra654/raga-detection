#!/bin/bash
# Wrapper for driver.py with HPC-friendly environment activation.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

RAGA_CONDA_ENV="${RAGA_CONDA_ENV:-raga}"
RAGA_SKIP_ENV_ACTIVATE="${RAGA_SKIP_ENV_ACTIVATE:-0}"
RAGA_PYTHON_BIN="${RAGA_PYTHON_BIN:-python3}"

if [[ "${RAGA_SKIP_ENV_ACTIVATE}" != "1" ]]; then
  RAGA_CONDA_SH="${RAGA_CONDA_SH:-}"
  if [[ -z "${RAGA_CONDA_SH}" ]]; then
    for candidate in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "/opt/miniconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh" \
      "/usr/local/miniconda3/etc/profile.d/conda.sh"; do
      if [[ -f "${candidate}" ]]; then
        RAGA_CONDA_SH="${candidate}"
        break
      fi
    done
  fi

  if [[ -n "${RAGA_CONDA_SH}" && -f "${RAGA_CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${RAGA_CONDA_SH}"
    conda activate "${RAGA_CONDA_ENV}"
  else
    echo "[WARN] Conda activation skipped: no conda.sh found. Set RAGA_CONDA_SH or RAGA_SKIP_ENV_ACTIVATE=1." >&2
  fi
fi

if ! command -v "${RAGA_PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: ${RAGA_PYTHON_BIN}" >&2
  exit 127
fi

exec "${RAGA_PYTHON_BIN}" "${SCRIPT_DIR}/driver.py" "$@"
