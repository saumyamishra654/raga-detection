#!/bin/bash
# Launch local app for pipeline parameter tuning on macOS.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_URL="http://127.0.0.1:8765/app"

if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate raga
else
  echo "[WARN] Conda activation script not found at /opt/miniconda3/etc/profile.d/conda.sh"
  echo "[WARN] Ensure your Python environment is already active."
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg is not in PATH. Some pipeline modes may fail."
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "[WARN] ffprobe is not in PATH. Some pipeline modes may fail."
fi

echo "[INFO] Starting local app on ${APP_URL}"

if command -v open >/dev/null 2>&1; then
  (sleep 1; open "${APP_URL}") >/dev/null 2>&1 &
fi

cd "$SCRIPT_DIR"
python -m uvicorn local_app.server:app --host 127.0.0.1 --port 8765 --reload

