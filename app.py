#!/usr/bin/env python3
"""
Main entry point for AI Category Q&A Generator.

Hybrid Tkinter app:
- Uses Mistral (via llama-cpp-python, local GGUF) to GENERATE a Question + Context for a chosen category.
- Uses a BERT QA checkpoint (Hugging Face Transformers, local directory) to ANSWER extractively from the context.

This app runs fully offline once the models are downloaded.
"""

import sys

# Check dependencies
try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not found. Install it inside your venv:\n"
          "  pip install --no-cache-dir llama-cpp-python\n"
          "Or run: ./setup_env.sh")
    sys.exit(1)

try:
    from transformers import pipeline
except ImportError:
    print("ERROR: transformers not found. Install it inside your venv:\n"
          "  pip install transformers torch\n"
          "Or run: ./setup_env.sh")
    sys.exit(1)

# Import application components
from models.llm_handler import LLMHandler
from models.bert_handler import BERTHandler
from services.qa_service import QAService
from gui import App

# Model paths
MISTRAL_GGUF_PATH = "models/mistral-7b-instruct.Q4_K_M.gguf"
BERT_LOCAL_DIR = "models/bert-squad"


def main():
    """Main entry point."""
    # Initialize handlers
    llm_handler = LLMHandler(MISTRAL_GGUF_PATH)
    bert_handler = BERTHandler(BERT_LOCAL_DIR)
    
    # Initialize service
    qa_service = QAService(llm_handler, bert_handler)
    
    # Initialize and run GUI
    app = App(qa_service)
    app.mainloop()


if __name__ == "__main__":
    main()
