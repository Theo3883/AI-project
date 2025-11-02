# AI Category Q&A Generator (Offline, Tkinter + llama.cpp)

This project provides a **Tkinter GUI** to select an AI category and automatically generate a **Question + Answer** locally using open LLMs running on CPU via **llama-cpp-python**.

## Features
- 12 AI categories (Bayesian inference, planning, game theory, CSP, etc.)
- Fully offline after model download
- Uses GGUF quantized models (default: Mistral 7B Instruct, CodeLlama 7B Instruct)
- CPU-only (works on Apple Silicon / M1 Max)
- Markdown output + Save to file

## Quickstart

```bash
# 1) Create env and install dependencies
./setup_env.sh

# 2) Download quantized models into ./models
./download_models.sh

# 3) Launch the Tkinter GUI
./run_gui.sh
```

## Models
- `models/mistral-7b-instruct.Q4_K_M.gguf` (general theory)
- `models/codellama-7b-instruct.Q4_K_M.gguf` (code/math heavy)

You can change which category uses which model by editing `CATEGORY_TO_MODEL_KEY` in `app.py`,
and add more models via `MODEL_REGISTRY` plus `download_models.sh`.

## Notes
- Make sure your Python build has Tkinter available (on macOS this is usually true).
- The scripts install only CPU dependencies; no GPU is required.
- If you prefer different quantization levels, modify the `*Q4_K_M.gguf` patterns in `download_models.sh`.
