#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
# Try hf first (newer huggingface_hub versions), fallback to huggingface-cli
HFC="${HFC:-}"
if [ -z "$HFC" ]; then
  if [ -x "$VENV_DIR/bin/hf" ]; then
    HFC="$VENV_DIR/bin/hf"
  elif [ -x "$VENV_DIR/bin/huggingface-cli" ]; then
    HFC="$VENV_DIR/bin/huggingface-cli"
  fi
fi

if [ -z "$HFC" ] || [ ! -x "$HFC" ]; then
  echo "ERROR: huggingface CLI not found (tried: $VENV_DIR/bin/hf and $VENV_DIR/bin/huggingface-cli)"
  echo "Run:  ./setup_env.sh"
  exit 1
fi

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

download_one () {
  local repo="$1"          # e.g., TheBloke/Mistral-7B-Instruct-v0.2-GGUF
  local pattern="$2"       # e.g., *Q4_K_M.gguf
  local localdir="$MODELS_DIR/${repo#*/}"  # models/Mistral-7B-Instruct-v0.2-GGUF

  echo "==> Downloading $repo ($pattern) to $localdir"
  "$HFC" download "$repo" --include "$pattern" --local-dir "$localdir"
  local gguf_file
  gguf_file=$(ls "$localdir"/*.gguf 2>/dev/null | head -n 1 || true)
  if [ -z "${gguf_file:-}" ]; then
    echo "ERROR: .gguf not found for $repo with pattern $pattern"
    exit 1
  fi
  echo "$gguf_file"
}

# Mistral 7B Instruct (used by app.py)
mistral_file=$(download_one "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" "*Q4_K_M.gguf")
cp -f "$mistral_file" "$MODELS_DIR/mistral-7b-instruct.Q4_K_M.gguf"

# CodeLlama 7B Instruct (not used by app.py - commented out)
# codellama_file=$(download_one "TheBloke/CodeLlama-7B-Instruct-GGUF" "*Q4_K_M.gguf")
# cp -f "$codellama_file" "$MODELS_DIR/codellama-7b-instruct.Q4_K_M.gguf"

# Download BERT QA model
echo
echo "==> Downloading BERT QA model (SQuAD fine-tuned)"
BERT_TARGET_DIR="$MODELS_DIR/bert-squad"
mkdir -p "$BERT_TARGET_DIR"

try_repo () {
  local repo="$1"
  echo "-> Trying: $repo"
  if "$HFC" download "$repo" --local-dir "$BERT_TARGET_DIR"; then
    echo "Downloaded $repo into $BERT_TARGET_DIR"
    return 0
  else
    echo "Failed: $repo"
    return 1
  fi
}

# Try a few well-known repos (first one that succeeds is used)
BERT_DOWNLOADED=false
try_repo "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad" && BERT_DOWNLOADED=true || true
if [ "$BERT_DOWNLOADED" = false ]; then
  try_repo "bert-large-uncased-whole-word-masking-finetuned-squad" && BERT_DOWNLOADED=true || true
fi
if [ "$BERT_DOWNLOADED" = false ]; then
  try_repo "deepset/bert-large-uncased-whole-word-masking-squad2" && BERT_DOWNLOADED=true || true
fi

if [ "$BERT_DOWNLOADED" = false ]; then
  echo "WARNING: Could not download BERT QA checkpoint. Continuing anyway..."
fi

echo
echo "Downloaded GGUF models:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "(none)"
echo
if [ -d "$BERT_TARGET_DIR" ] && [ -n "$(ls -A "$BERT_TARGET_DIR" 2>/dev/null)" ]; then
  echo "Downloaded BERT model to: $BERT_TARGET_DIR"
else
  echo "BERT model not downloaded (optional)"
fi
echo
echo "OK."
