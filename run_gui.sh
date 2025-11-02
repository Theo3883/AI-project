#!/usr/bin/env bash
set -euo pipefail
VENV_DIR="${VENV_DIR:-.venv}"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python3 app.py
