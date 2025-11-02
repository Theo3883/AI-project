#!/usr/bin/env python3
"""
Hybrid Tkinter app:
- Uses Mistral (via llama-cpp-python, local GGUF) to GENERATE a Question + Context for a chosen category.
- Uses a BERT QA checkpoint (Hugging Face Transformers, local directory) to ANSWER extractively from the context.

Files expected after running the download scripts:
- models/mistral-7b-instruct.Q4_K_M.gguf
- models/bert-squad/   (a HF snapshot directory for a BERT fine-tuned on SQuAD)

Dependencies (installed by setup_env.sh):
- llama-cpp-python
- transformers
- torch
- huggingface_hub[cli]
- tk (usually bundled with python.org macOS installers)

This app runs fully offline once the models are downloaded.
"""

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime

# --- Generative (Mistral via llama.cpp) ---
try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None

# --- Extractive QA (BERT via Transformers) ---
try:
    from transformers import pipeline
except Exception as e:
    pipeline = None

# ------------------------------
# Categories (exactly as requested)
# ------------------------------
CATEGORIES = [
    "ai_type_for_system",
    "algorithm_selection",
    "bayesian_inference",
    "csp_arc_consistency",
    "game_theory_nash",
    "heuristic_admissibility_combo",
    "ontology_concepts",
    "partial_order_planning",
    "planning_strips_adl",
    "q_learning_update",
    "state_representation",
    "value_iteration_step",
]

# ------------------------------
# Model paths/registry
# ------------------------------
MISTRAL_GGUF_PATH = "models/mistral-7b-instruct.Q4_K_M.gguf"
BERT_LOCAL_DIR = "models/bert-squad"  # folder created by download_models.sh

SYSTEM_PROMPT = (
    "You are a concise, helpful AI tutor. "
    "Generate a single exam-quality question and a context containing all necessary facts to answer it. "
    "Structure the context so that complex answers (multiple items, calculations, lists) can be extracted. "
    "The context should contain implicit information that allows the answer to be deduced or extracted, but do NOT explicitly state the answer. "
    "When possible, structure information clearly (use bullet points, tables, or clear sections) to aid extraction."
)

VALIDATION_PROMPT = (
    "You are a helpful tutor. Given a question, context, and a proposed answer, "
    "determine if the answer is correct based on the context. Respond with only 'CORRECT' or 'INCORRECT' "
    "followed by a brief explanation (1-2 sentences)."
)

EXPANSION_PROMPT = (
    "You are a helpful tutor. Given a question, context, and a partial/extracted answer from a QA system, "
    "expand it into a complete, detailed answer that fully addresses the question. "
    "Use the context to provide all relevant details, calculations, or components needed for a complete answer."
)

CATEGORY_INSTRUCTIONS = {
    "ai_type_for_system":
        "Create exactly ONE question that asks the learner to classify an agent type. "
        "Provide a CONTEXT with a detailed scenario describing the agent's behavior, decision-making process, and characteristics. "
        "Include enough details so the agent type can be inferred, but do NOT explicitly state the type.",
    "algorithm_selection":
        "Create exactly ONE question asking to choose the most appropriate algorithm "
        "(BFS, DFS, UCS, A*, Greedy Best-First, Hill-Climbing, Simulated Annealing). "
        "Provide a CONTEXT describing a problem scenario with specific characteristics (optimality requirements, heuristic availability, constraints, etc.) "
        "that make one algorithm clearly most appropriate, but do NOT name the algorithm.",
    "bayesian_inference":
        "Create exactly ONE numeric Bayes question asking for P(H|E). "
        "Provide a CONTEXT with prior P(H), likelihoods P(E|H) and P(E|~H), and all numerical values needed to compute the posterior. "
        "Include the numbers but do NOT state the computed P(H|E).",
    "csp_arc_consistency":
        "Create ONE small CSP question (≤3 variables) asking for pruned domains after AC-3. "
        "Provide a CONTEXT listing initial domains and binary constraints. Show enough detail to deduce which values get pruned, but do NOT list the final domains.",
    "game_theory_nash":
        "Create ONE 2x2 game question asking for Nash equilibria. "
        "Provide a CONTEXT with the complete payoff matrix showing all strategy combinations and payoffs. "
        "Include the matrix so equilibria can be calculated, but do NOT state which strategy pairs are equilibria.",
    "heuristic_admissibility_combo":
        "Create ONE question asking whether max(h1,h2) or h1+h2 are admissible/consistent. "
        "Provide CONTEXT describing h1 and h2 properties (e.g., both admissible, both consistent, values, monotonicity). "
        "Include enough information to determine admissibility/consistency of the combinations, but do NOT state the conclusions.",
    "ontology_concepts":
        "Create ONE ontology question asking for a specific concept or relationship. "
        "Provide CONTEXT with relevant axioms, assertions, and relationships that entail the answer. "
        "Include the necessary facts but do NOT explicitly state the answer.",
    "partial_order_planning":
        "Create ONE partial-order planning question asking for a plan or ordering. "
        "Provide CONTEXT with initial state, goal state, and action descriptions (preconditions and effects). "
        "Include enough detail to construct a plan, but do NOT provide the plan or ordering.",
    "planning_strips_adl":
        "Create ONE STRIPS/ADL question asking for encoding or successor state. "
        "Provide CONTEXT with action schema (preconditions, add/delete effects) and initial state. "
        "Include all necessary information, but do NOT state the successor state or encoding.",
    "q_learning_update":
        "Create ONE Q-learning question asking for the updated Q(s,a). "
        "Provide CONTEXT with transition details: state s, action a, reward r, next state s', learning rate alpha, discount gamma, current Q(s,a), and max_a' Q(s',a'). "
        "Include all numerical values needed for the calculation, but do NOT show the computed update.",
    "state_representation":
        "Create ONE question asking for a factored state representation or number of states. "
        "Provide CONTEXT describing the domain variables and their possible values. "
        "Include enough detail to determine the state space, but do NOT state the representation or count.",
    "value_iteration_step":
        "Create ONE value-iteration question asking for V_{k+1} values. "
        "Provide CONTEXT with MDP details: rewards R(s), transition probabilities P(s'|s,a), discount gamma, and current value function V_k(s). "
        "Include all numbers needed to compute V_{k+1}, but do NOT show the updated values.",
}

def build_generation_messages(category: str):
    intr = CATEGORY_INSTRUCTIONS.get(category, "")
    task = (
        f"Category: {category}\n"
        f"{intr}\n\n"
        "Output format (Markdown):\n"
        "### Question\n"
        "- A single, clearly stated question that can be answered using information from the context.\n"
        "### Context\n"
        "- A well-structured context containing the problem description and all necessary facts.\n"
        "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
        "- Include all numerical values, relationships, and details needed to answer the question.\n"
        "- Structure information so that components of complex answers can be identified and extracted.\n"
        "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

def build_expansion_messages(question: str, context: str, partial_answer: str):
    """Build messages for Mistral to expand/complete BERT's extracted answer."""
    task = (
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Extracted Answer (partial): {partial_answer}\n\n"
        "Expand this extracted answer into a complete, detailed answer that fully addresses the question. "
        "Include all relevant details, calculations, multiple components, or complete information needed. "
        "If the extracted answer is incomplete or vague, use the context to provide the full answer."
    )
    return [
        {"role": "system", "content": EXPANSION_PROMPT},
        {"role": "user", "content": task},
    ]

def build_validation_messages(question: str, context: str, proposed_answer: str):
    """Build messages for Mistral to validate BERT's answer."""
    task = (
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Proposed Answer: {proposed_answer}\n\n"
        "Is the proposed answer correct based on the context? Respond with 'CORRECT' or 'INCORRECT' followed by a brief explanation."
    )
    return [
        {"role": "system", "content": VALIDATION_PROMPT},
        {"role": "user", "content": task},
    ]

# -------- LLM cache/loaders --------
_llm_cache = {}
_qa_pipe = None

def get_mistral() -> "Llama":
    if Llama is None:
        raise RuntimeError("llama-cpp-python is not installed. Run ./setup_env.sh")
    
    # Try the expected path first, then search subdirectories
    model_path = MISTRAL_GGUF_PATH
    if not os.path.exists(model_path):
        # Search for .gguf files in models subdirectories
        models_dir = "models"
        found = False
        if os.path.isdir(models_dir):
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".gguf") and "mistral" in file.lower():
                        model_path = os.path.join(root, file)
                        found = True
                        break
                if found:
                    break
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Mistral model not found at {MISTRAL_GGUF_PATH} or in subdirectories. Run ./download_models.sh"
        )
    
    if "mistral" in _llm_cache:
        return _llm_cache["mistral"]
    # Optimize for speed: larger batch size, more threads, efficient context
    cpu_count = os.cpu_count() or 8
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,  # Further reduced for faster processing (sufficient for most Q&A)
        n_threads=max(6, cpu_count),  # Use more threads for parallel processing
        n_batch=1024,  # Larger batch for faster processing
        n_gpu_layers=0,   # CPU-only (Apple Silicon friendly)
        verbose=False,
        use_mlock=True,  # Lock memory to prevent swapping (faster)
        use_mmap=True,  # Memory map for faster loading
        n_predict=256,  # Default prediction length for faster completion
    )
    _llm_cache["mistral"] = llm
    return llm

def get_qa_pipeline():
    global _qa_pipe, pipeline
    if pipeline is None:
        raise RuntimeError("transformers is not installed. Run ./setup_env.sh")
    if _qa_pipe is not None:
        return _qa_pipe
    
    # Try the expected path first, then search subdirectories
    bert_dir = BERT_LOCAL_DIR
    if not os.path.isdir(bert_dir):
        # Search for BERT model directories in models subdirectories
        models_dir = "models"
        found = False
        if os.path.isdir(models_dir):
            for root, dirs, files in os.walk(models_dir):
                for dir_name in dirs:
                    # Look for directories with "bert" in the name
                    if "bert" in dir_name.lower() and os.path.isfile(os.path.join(root, dir_name, "config.json")):
                        bert_dir = os.path.join(root, dir_name)
                        found = True
                        break
                if found:
                    break
    
    if not os.path.isdir(bert_dir):
        raise FileNotFoundError(
            f"BERT QA model directory not found at {BERT_LOCAL_DIR} or in subdirectories. Run ./download_models.sh"
        )
    _qa_pipe = pipeline("question-answering", model=bert_dir, tokenizer=bert_dir, device=-1)
    return _qa_pipe

# --------- Simple parser for Question/Context ---------
def parse_q_and_context(markdown_text: str):
    # Expect "### Question" then "### Context". Be forgiving.
    q = ""
    ctx = ""
    # Normalize line endings
    text = markdown_text.replace("\r\n", "\n")
    # Find sections
    q_match = re.search(r"^\s*###\s*Question\s*$", text, flags=re.IGNORECASE|re.MULTILINE)
    c_match = re.search(r"^\s*###\s*Context\s*$", text, flags=re.IGNORECASE|re.MULTILINE)
    if q_match:
        q_start = q_match.end()
        if c_match:
            q = text[q_start:c_match.start()].strip()
            ctx = text[c_match.end():].strip()
        else:
            q = text[q_start:].strip()
            ctx = ""
    else:
        # Fallback: try to split on first heading
        parts = re.split(r"^\s*###\s*", text, flags=re.MULTILINE)
        if len(parts) >= 2:
            # crude fallback
            q = parts[1].strip()
    return q, ctx

# ------------------------------
# Tkinter GUI
# ------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Category: Generate (Mistral) + Answer (BERT)")
        self.geometry("1000x760")

        self.category_var = tk.StringVar(value=CATEGORIES[0])
        self.temperature_var = tk.DoubleVar(value=0.6)
        self.max_tokens_var = tk.IntVar(value=512)

        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Category:").grid(row=0, column=0, sticky="w")
        self.cb = ttk.Combobox(top, values=CATEGORIES, textvariable=self.category_var, width=35, state="readonly")
        self.cb.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(top, text="Temp:").grid(row=0, column=2, sticky="e")
        ttk.Entry(top, width=6, textvariable=self.temperature_var).grid(row=0, column=3, padx=6, sticky="w")

        ttk.Label(top, text="Max tokens:").grid(row=0, column=4, sticky="e")
        ttk.Entry(top, width=8, textvariable=self.max_tokens_var).grid(row=0, column=5, padx=6, sticky="w")

        self.btn_generate = ttk.Button(top, text="Generate & Answer", command=self.on_generate_and_answer)
        self.btn_generate.grid(row=0, column=6, padx=10)

        self.btn_save = ttk.Button(top, text="Save Output", command=self.on_save, state="disabled")
        self.btn_save.grid(row=0, column=7, padx=6)

        self.out = ScrolledText(self, wrap=tk.WORD, font=("Menlo", 12))
        self.out.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        self._thread = None

    def on_generate_and_answer(self):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("Busy", "Please wait for the current task to finish.")
            return

        cat = self.category_var.get()
        try:
            temp = float(self.temperature_var.get())
        except Exception:
            temp = 0.6
        try:
            max_toks = int(self.max_tokens_var.get())
        except Exception:
            max_toks = 512

        self.out.delete("1.0", tk.END)
        self.out.insert(tk.END, f"# Category: {cat}\n\n")
        self.status.set("Generating question + context with Mistral (CPU)...")

        def work():
            try:
                # Step 1: Generate Question + Context
                llm = get_mistral()
                messages = build_generation_messages(cat)
                res = llm.create_chat_completion(
                    messages=messages,
                    temperature=max(0.0, min(2.0, temp)),
                    top_p=0.9,  # Slightly lower for faster generation
                    max_tokens=min(512, max_toks),  # Reduced default max for speed
                    repeat_penalty=1.1,
                )
                gen_text = res["choices"][0]["message"]["content"].strip()
                q, ctx = parse_q_and_context(gen_text)
                if not q or not ctx:
                    raise RuntimeError("Failed to parse '### Question' and '### Context' from model output.")

                # Step 2: Answer with BERT QA
                self.status.set("Answering with BERT (extractive QA, CPU)...")
                qa = get_qa_pipeline()
                pred = qa({"question": q, "context": ctx})
                bert_raw_answer = pred.get("answer", "").strip()
                score = float(pred.get("score", 0.0))
                
                # Clean up common formatting issues
                bert_raw_answer = bert_raw_answer.replace("`", "").strip()
                
                # Validate answer quality
                confidence_warning = ""
                if score < 0.1:
                    confidence_warning = "\n⚠️ **Low confidence score** - answer may be unreliable."
                elif score < 0.3:
                    confidence_warning = "\n⚠️ **Moderate confidence** - answer should be verified."
                
                # Step 2.5: Expand BERT's answer with Mistral for complex answers
                self.status.set("Expanding answer with Mistral (CPU)...")
                expansion_messages = build_expansion_messages(q, ctx, bert_raw_answer)
                expansion_res = llm.create_chat_completion(
                    messages=expansion_messages,
                    temperature=0.1,  # Very low temperature for fastest, most focused responses
                    top_p=0.8,  # Lower top_p for faster, more deterministic responses
                    max_tokens=256,  # Reduced for faster expansion
                    repeat_penalty=1.1,
                )
                bert_expanded_answer = expansion_res["choices"][0]["message"]["content"].strip()
                
                # Step 3: Validate expanded answer with Mistral
                self.status.set("Validating answer with Mistral (CPU)...")
                validation_messages = build_validation_messages(q, ctx, bert_expanded_answer)
                validation_res = llm.create_chat_completion(
                    messages=validation_messages,
                    temperature=0.1,  # Very low temperature for fastest validation
                    top_p=0.8,  # Lower top_p for faster responses
                    max_tokens=128,  # Reduced for faster validation (just need CORRECT/INCORRECT + brief explanation)
                    repeat_penalty=1.1,
                )
                validation_text = validation_res["choices"][0]["message"]["content"].strip()

                final_md = (
                    f"## Generated by Mistral\n\n{gen_text}\n\n"
                    f"## Extractive Answer (BERT - Raw)\n\n"
                    f"**Raw Answer:** {bert_raw_answer}\n\n"
                    f"**Confidence:** {score:.3f}{confidence_warning}\n\n"
                    f"## Expanded Answer (Mistral)\n\n"
                    f"{bert_expanded_answer}\n\n"
                    f"## Validation by Mistral\n\n{validation_text}\n"
                )
            except Exception as e:
                final_md = f"ERROR: {e}"
            self.after(0, lambda: self._display(final_md))

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _display(self, text: str):
        self.out.insert(tk.END, text + "\n")
        self.btn_save.configure(state="normal")
        self.status.set("Done.")

    def on_save(self):
        content = self.out.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("Nothing to save", "Generate output first.")
            return
        os.makedirs("outputs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        cat = self.category_var.get()
        path = os.path.join("outputs", f"{cat}_{ts}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        self.status.set(f"Saved to {path}")
        messagebox.showinfo("Saved", f"Saved to {path}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
