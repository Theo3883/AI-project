#!/usr/bin/env bash
set -euo pipefail

# Create and prepare a Python venv (CPU-only)
PY=python3
VENV_DIR=".venv"

echo "[1/3] Creating virtualenv at $VENV_DIR"
$PY -m venv "$VENV_DIR"

echo "[2/3] Activating venv and upgrading pip"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

echo "[3/3] Installing runtime deps (CPU-only llama-cpp, HF CLI)"
pip install --no-cache-dir llama-cpp-python transformers torch "huggingface_hub[cli]" tqdm rich pyyaml

cat <<EOF

Done.

Next:
  1) ./download_models.sh      # fetch GGUF models into ./models (CPU quantized)
  2) ./run_gui.sh              # launch the Tkinter app

EOF
