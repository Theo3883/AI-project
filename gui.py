"""Tkinter GUI application for AI Category Q&A Generator."""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from services.qa_service import QAService
from strategies.categories import CategoryStrategyFactory
from utils.markdown_renderer import MarkdownRenderer


class App(tk.Tk):
    """Main application GUI."""
    
    def __init__(self, qa_service: QAService):
        """
        Initialize the GUI application.
        
        Args:
            qa_service: QA service instance
        """
        super().__init__()
        self.qa_service = qa_service
        self.title("AI Category: Generate (Mistral) + Answer (BERT)")
        self.geometry("1000x760")
        
        # Get all available categories
        self.categories = CategoryStrategyFactory.get_all_categories()
        
        # UI Variables
        self.category_var = tk.StringVar(value=self.categories[0] if self.categories else "")
        
        self._setup_ui()
        self._thread = None
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Top frame with controls
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(top, text="Category:").grid(row=0, column=0, sticky="w")
        self.category_cb = ttk.Combobox(
            top,
            values=self.categories,
            textvariable=self.category_var,
            width=40,
            state="readonly"
        )
        self.category_cb.grid(row=0, column=1, padx=6, pady=4, sticky="w")
        
        self.btn_generate = ttk.Button(top, text="Generate & Answer", command=self.on_generate_and_answer)
        self.btn_generate.grid(row=0, column=2, padx=10)
        
        self.btn_save = ttk.Button(top, text="Save Output", command=self.on_save, state="disabled")
        self.btn_save.grid(row=0, column=3, padx=6)
        
        # Output text area with markdown renderer
        self.output = ScrolledText(self, wrap=tk.WORD, font=("Menlo", 12), bg="#ffffff", fg="#2c3e50")
        self.output.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.markdown_renderer = MarkdownRenderer(self.output)
        
        # Status bar
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_generate_and_answer(self):
        """Handle generate and answer button click."""
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("Busy", "Please wait for the current task to finish.")
            return
        
        category = self.category_var.get()
        
        self.output.delete("1.0", tk.END)
        self.markdown_renderer.render(f"# Category: {category}\n\n")
        self.status.set("Generating question + context with Mistral (CPU)...")
        
        def work():
            try:
                result = self.qa_service.generate_question_and_answer(
                    category_name=category,
                    temperature=0.6,  # Default temperature
                    max_tokens=512    # Default max tokens
                )
                formatted_output = self.qa_service.format_output(result)
            except Exception as e:
                formatted_output = f"ERROR: {e}"
            
            self.after(0, lambda: self._display(formatted_output))
        
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
    
    def _display(self, text: str):
        """Display the result in the output area with markdown rendering."""
        # Store raw markdown for saving
        self._last_output = text
        # Render markdown with formatting
        self.markdown_renderer.render(text)
        self.btn_save.configure(state="normal")
        self.status.set("Done.")
    
    def on_save(self):
        """Handle save button click."""
        if not hasattr(self, '_last_output'):
            messagebox.showinfo("Nothing to save", "Generate output first.")
            return
        
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        category = self.category_var.get()
        path = os.path.join("outputs", f"{category}_{timestamp}.md")
        
        # Save the raw markdown text
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._last_output + "\n")
        
        self.status.set(f"Saved to {path}")
        messagebox.showinfo("Saved", f"Saved to {path}")

