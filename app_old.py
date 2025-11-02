#!/usr/bin/env python3
"""
Tkinter GUI to generate a self-contained Question + Answer for a selected AI category,
using local GGUF models via llama-cpp-python (CPU-only).

Default models (paths) expected after running download_models.sh:
- models/mistral-7b-instruct.Q4_K_M.gguf
- models/codellama-7b-instruct.Q4_K_M.gguf

Requires: llama-cpp-python (Tkinter typically ships with Python on macOS).
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime

try:
    from llama_cpp import Llama
except Exception as e:
    print("llama-cpp-python not found. Install it inside your venv:\n"
          "  pip install --no-cache-dir llama-cpp-python\n")
    raise

# ------------------------------
# Categories (exactly as you listed)
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
# Model registry (local GGUF files)
# You can change these paths if you download different quantizations.
# ------------------------------
MODEL_REGISTRY = {
    "mistral": {
        "path": "models/mistral-7b-instruct.Q4_K_M.gguf",
        "desc": "Mistral 7B Instruct (Q4_K_M, CPU-only)"
    },
    "codellama": {
        "path": "models/codellama-7b-instruct.Q4_K_M.gguf",
        "desc": "CodeLlama 7B Instruct (Q4_K_M, CPU-only)"
    },
    # Example of adding another model later:
    # "llama2": {
    #     "path": "models/llama-2-13b-chat.Q4_K_M.gguf",
    #     "desc": "Llama 2 13B Chat (Q4_K_M, CPU-only)"
    # },
}

# Map categories to model keys. Anything not listed uses "mistral" by default.
CATEGORY_TO_MODEL_KEY = {
    # Code/math-heavy -> CodeLlama
    "q_learning_update": "codellama",
    "value_iteration_step": "codellama",
    # Everything else -> Mistral (default)
}

# ------------------------------
# Category-specific instructions to guide the LLM
# Keep short and concrete so answers are reliable.
# ------------------------------
CATEGORY_INSTRUCTIONS = {
    "ai_type_for_system":
        "Create a short scenario and ask the learner to classify the agent type "
        "(simple reflex, model-based reflex, goal-based, utility-based, or learning). "
        "Then give the correct type with a one-paragraph justification.",

    "algorithm_selection":
        "Pose a tiny search or optimization problem and ask which algorithm is most appropriate "
        "(choose among BFS, DFS, UCS, A*, Greedy Best-First, Hill-Climbing, Simulated Annealing). "
        "Then give the choice plus a brief rationale.",

    "bayesian_inference":
        "Provide a small numeric example with a prior P(H) and likelihoods P(E|H), P(E|~H). "
        "Ask for P(H|E). Then compute and report the posterior (3 decimal places) with a short explanation.",

    "csp_arc_consistency":
        "Give a CSP with three variables (domains size ≤ 3) and a couple of binary constraints. "
        "Ask to run one AC-3 pass (all arcs initially queued) and report the pruned domains.",

    "game_theory_nash":
        "Provide a 2×2 payoff matrix (row, column player). Ask for pure-strategy Nash equilibria; "
        "if none, compute the mixed equilibrium with probabilities (2–3 decimals).",

    "heuristic_admissibility_combo":
        "Describe a search problem with two admissible heuristics h1 and h2. Ask whether max(h1,h2) "
        "and h1+h2 are admissible/consistent for A*, and explain briefly why.",

    "ontology_concepts":
        "Define a tiny domain and ask for 3 axioms (e.g., subClassOf, disjointWith, domain/range) and 2 assertions. "
        "Then state one inference the ontology entails.",

    "partial_order_planning":
        "Provide initial state, goal, and 2–3 actions (preconditions/effects). Ask to sketch a partial-order plan: "
        "list steps, ordering constraints, and one causal link.",

    "planning_strips_adl":
        "Ask to encode one action in STRIPS or ADL with preconditions, add/delete effects. "
        "Give an initial state and ask for the successor state after applying that action once.",

    "q_learning_update":
        "Provide a single transition (s, a, r, s'), learning rate α, discount γ, current Q(s,a), "
        "and max_a' Q(s',a'). Ask to compute the updated Q(s,a).",

    "state_representation":
        "Give a small domain and ask to propose a factored state representation (variables and domains) "
        "and compute the total number of states.",

    "value_iteration_step":
        "Define a tiny MDP (2–3 states) with rewards and transition probabilities. Provide V_k values "
        "and γ, and ask to compute V_{k+1} for each state (2–3 decimals).",
}

# ------------------------------
# Prompt template (Markdown output)
# ------------------------------
SYSTEM_PROMPT = (
    "You are a concise, helpful AI tutor. Create clear, exam-quality items and give correct answers."
)

def build_messages(category: str, answer_also: bool = True):
    instr = CATEGORY_INSTRUCTIONS.get(category, "")
    task = (
        f"Create exactly ONE original question in the category: {category}.\n"
        f"{instr}\n\n"
        "Output format (Markdown):\n"
        "### Question\n"
        "- State the problem clearly and concisely.\n"
    )
    if answer_also:
        task += (
            "### Answer\n"
            "- Provide the correct answer with a brief explanation (3–6 sentences). "
            "Use formulas or small tables only if essential.\n"
        )
    else:
        task += "(Stop here. Do NOT include an answer.)\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

# ------------------------------
# LLM loader/cache
# ------------------------------
_LOADED = {}

def get_llm_for_category(category: str) -> Llama:
    key = CATEGORY_TO_MODEL_KEY.get(category, "mistral")
    model_meta = MODEL_REGISTRY[key]
    model_path = model_meta["path"]
    
    # If the expected path doesn't exist, try to find the model in subdirectories
    if not os.path.exists(model_path):
        # Search for .gguf files in models subdirectories
        models_dir = "models"
        found = False
        if os.path.isdir(models_dir):
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".gguf"):
                        # Try to match by key in filename
                        if key == "mistral" and "mistral" in file.lower():
                            model_path = os.path.join(root, file)
                            found = True
                            break
                        elif key == "codellama" and ("codellama" in file.lower() or "code" in file.lower()):
                            model_path = os.path.join(root, file)
                            found = True
                            break
                if found:
                    break
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_meta['path']}\n"
            f"Run:  ./download_models.sh   to fetch GGUF models into ./models"
        )
    if key in _LOADED:
        return _LOADED[key]
    # CPU-only: n_gpu_layers = 0
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=max(2, (os.cpu_count() or 8)),
        n_batch=256,
        n_gpu_layers=0,   # force CPU on Apple Silicon
        verbose=False,
    )
    _LOADED[key] = llm
    return llm

# ------------------------------
# Tkinter GUI
# ------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Category Q&A Generator (Offline)")
        self.geometry("900x700")

        self.category_var = tk.StringVar(value=CATEGORIES[0])
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.max_tokens_var = tk.IntVar(value=512)
        self.include_answer_var = tk.BooleanVar(value=True)

        # Top frame: controls
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Category:").grid(row=0, column=0, sticky="w")
        self.category_cb = ttk.Combobox(top, values=CATEGORIES, textvariable=self.category_var, width=30, state="readonly")
        self.category_cb.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(top, text="Temperature:").grid(row=0, column=2, sticky="e")
        self.temp_entry = ttk.Entry(top, width=6, textvariable=self.temperature_var)
        self.temp_entry.grid(row=0, column=3, padx=6, sticky="w")

        ttk.Label(top, text="Max tokens:").grid(row=0, column=4, sticky="e")
        self.max_entry = ttk.Entry(top, width=6, textvariable=self.max_tokens_var)
        self.max_entry.grid(row=0, column=5, padx=6, sticky="w")

        self.answer_chk = ttk.Checkbutton(top, text="Include Answer", variable=self.include_answer_var)
        self.answer_chk.grid(row=0, column=6, padx=10, sticky="w")

        self.gen_btn = ttk.Button(top, text="Generate", command=self.on_generate_clicked)
        self.gen_btn.grid(row=0, column=7, padx=10)

        self.save_btn = ttk.Button(top, text="Save Output", command=self.on_save_clicked, state="disabled")
        self.save_btn.grid(row=0, column=8, padx=6)

        # Output
        self.output = ScrolledText(self, wrap=tk.WORD, font=("Menlo", 12))
        self.output.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.status = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self, textvariable=self.status, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self._current_thread = None

    def on_generate_clicked(self):
        if self._current_thread and self._current_thread.is_alive():
            messagebox.showinfo("Busy", "Please wait for the current generation to finish.")
            return

        cat = self.category_var.get()
        try:
            temp = float(self.temperature_var.get())
        except Exception:
            temp = 0.7
        try:
            max_toks = int(self.max_tokens_var.get())
        except Exception:
            max_toks = 512
        inc_ans = bool(self.include_answer_var.get())

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"# Generating for category: {cat}\n\n")
        self.status.set("Loading model and generating... (CPU)")

        def run():
            try:
                llm = get_llm_for_category(cat)
                msgs = build_messages(cat, answer_also=inc_ans)
                res = llm.create_chat_completion(
                    messages=msgs,
                    temperature=max(0.0, min(2.0, temp)),
                    top_p=0.95,
                    max_tokens=max(64, min(2048, max_toks)),
                    repeat_penalty=1.1,
                )
                text = res["choices"][0]["message"]["content"].strip()
            except Exception as e:
                text = f"ERROR: {e}"
            self.after(0, lambda: self._display_text(text))

        self._current_thread = threading.Thread(target=run, daemon=True)
        self._current_thread.start()

    def _display_text(self, text: str):
        self.output.insert(tk.END, text + "\n")
        self.save_btn.configure(state="normal")
        self.status.set("Done.")

    def on_save_clicked(self):
        content = self.output.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("Nothing to save", "Generate content first.")
            return
        os.makedirs("outputs", exist_ok=True)
        from datetime import datetime
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
