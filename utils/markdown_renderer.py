"""Markdown renderer for Tkinter ScrolledText widget."""

import re
from typing import List, Tuple


class MarkdownRenderer:
    """Renders markdown to formatted Tkinter text."""
    
    def __init__(self, text_widget):
        """
        Initialize renderer.
        
        Args:
            text_widget: Tkinter ScrolledText widget
        """
        self.text = text_widget
        self._setup_tags()
    
    def _setup_tags(self):
        """Set up text formatting tags."""
        # Headings
        self.text.tag_config("h1", font=("Menlo", 18, "bold"), foreground="#2c3e50", spacing1=10, spacing3=5)
        self.text.tag_config("h2", font=("Menlo", 16, "bold"), foreground="#34495e", spacing1=8, spacing3=4)
        self.text.tag_config("h3", font=("Menlo", 14, "bold"), foreground="#34495e", spacing1=6, spacing3=3)
        
        # Bold text
        self.text.tag_config("bold", font=("Menlo", 12, "bold"), foreground="#2c3e50")
        
        # Regular text
        self.text.tag_config("normal", font=("Menlo", 12))
        
        # Answer text (highlighted)
        self.text.tag_config("answer", font=("Menlo", 12, "bold"), foreground="#27ae60", background="#ecf0f1")
        
        # Confidence score
        self.text.tag_config("confidence", font=("Menlo", 11), foreground="#3498db")
        self.text.tag_config("warning", font=("Menlo", 11, "italic"), foreground="#e74c3c")
        
        # Validation
        self.text.tag_config("correct", font=("Menlo", 12, "bold"), foreground="#27ae60")
        self.text.tag_config("incorrect", font=("Menlo", 12, "bold"), foreground="#e74c3c")
        
        # Section separators
        self.text.tag_config("separator", font=("Menlo", 11), foreground="#95a5a6", background="#ecf0f1")
    
    def render(self, markdown_text: str):
        """
        Render markdown text to the widget.
        
        Args:
            markdown_text: Markdown formatted text
        """
        self.text.delete("1.0", "end")
        
        lines = markdown_text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Headings
            if line.startswith('### '):
                self._insert_heading(line[4:].strip(), "h3")
            elif line.startswith('## '):
                self._insert_heading(line[3:].strip(), "h2")
            elif line.startswith('# '):
                self._insert_heading(line[2:].strip(), "h1")
            
            # Horizontal rule
            elif line.strip() == '---' or line.strip() == '***':
                self._insert_separator()
            
            # Tables (check if next line has | and is not empty)
            elif '|' in line:
                # Check if this is a table row (next line is separator or another row)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # If next line is separator (|--|) or another row with |, it's a table
                    if (next_line.startswith('|') and ('-' in next_line or '|' in next_line)) or \
                       (i + 2 < len(lines) and '|' in lines[i + 2]):
                        i = self._render_table(lines, i)
                    else:
                        self._render_text_line(line)
                else:
                    self._render_text_line(line)
            
            # Bold text (lines with **)
            elif '**' in line:
                self._render_bold_line(line)
            
            # Empty lines
            elif not line.strip():
                self.text.insert("end", "\n", "normal")
            
            # Regular text
            else:
                self._render_text_line(line)
            
            i += 1
    
    def _insert_heading(self, text: str, tag: str):
        """Insert a heading."""
        self.text.insert("end", f"{text}\n", tag)
        self.text.insert("end", "\n", "normal")
    
    def _insert_separator(self):
        """Insert a horizontal separator line."""
        self.text.insert("end", "─" * 80 + "\n", "separator")
        self.text.insert("end", "\n", "normal")
    
    def _render_bold_line(self, line: str):
        """Render a line with bold text."""
        parts = re.split(r'(\*\*[^*]+\*\*)', line)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                # Bold text
                bold_text = part[2:-2]
                self.text.insert("end", bold_text, "bold")
            else:
                self.text.insert("end", part, "normal")
        self.text.insert("end", "\n", "normal")
    
    def _render_text_line(self, line: str):
        """Render a regular text line with inline formatting."""
        # Handle bold text inline
        parts = re.split(r'(\*\*[^*]+\*\*)', line)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                bold_text = part[2:-2]
                # Special handling for answers
                if "Answer:" in bold_text or "Raw Answer:" in bold_text:
                    self.text.insert("end", bold_text.replace("Answer:", "Answer:").replace("Raw Answer:", "Raw Answer:"), "answer")
                else:
                    self.text.insert("end", bold_text, "bold")
            elif "Confidence:" in part:
                # Highlight confidence scores
                conf_match = re.search(r'Confidence:\s*([\d.]+)', part)
                if conf_match:
                    before = part[:part.index("Confidence:")]
                    after = part[part.index("Confidence:") + len("Confidence:"):]
                    self.text.insert("end", before, "normal")
                    self.text.insert("end", "Confidence: ", "bold")
                    self.text.insert("end", conf_match.group(1), "confidence")
                    self.text.insert("end", after, "normal")
                else:
                    self.text.insert("end", part, "normal")
            elif "⚠️" in part or "WARNING" in part.upper() or "Low confidence" in part or "Moderate confidence" in part:
                self.text.insert("end", part, "warning")
            elif part.strip().startswith("CORRECT"):
                self.text.insert("end", part, "correct")
            elif part.strip().startswith("INCORRECT"):
                self.text.insert("end", part, "incorrect")
            else:
                self.text.insert("end", part, "normal")
        self.text.insert("end", "\n", "normal")
    
    def _render_table(self, lines: List[str], start_idx: int) -> int:
        """
        Render a markdown table.
        
        Args:
            lines: All lines
            start_idx: Starting index
            
        Returns:
            Index after table
        """
        table_lines = []
        i = start_idx
        
        # Collect table rows
        while i < len(lines) and '|' in lines[i]:
            if lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
            i += 1
        
        if len(table_lines) < 2:
            return start_idx
        
        # Parse header
        header = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
        
        # Skip separator row
        data_rows = []
        for row_line in table_lines[2:]:
            row = [cell.strip() for cell in row_line.split('|')[1:-1]]
            if row and any(cell for cell in row):
                data_rows.append(row)
        
        # Render table
        if header:
            # Header
            self.text.insert("end", "┌", "normal")
            for j, cell in enumerate(header):
                width = max(len(cell), max(len(row[j]) for row in data_rows if j < len(row)) if data_rows else 0)
                self.text.insert("end", "─" * (width + 2), "normal")
                if j < len(header) - 1:
                    self.text.insert("end", "┬", "normal")
            self.text.insert("end", "┐\n", "normal")
            
            # Header cells
            self.text.insert("end", "│", "normal")
            for j, cell in enumerate(header):
                width = max(len(cell), max(len(row[j]) for row in data_rows if j < len(row)) if data_rows else 0)
                self.text.insert("end", f" {cell:<{width}} ", "bold")
                if j < len(header) - 1:
                    self.text.insert("end", "│", "normal")
            self.text.insert("end", "│\n", "normal")
            
            # Separator
            self.text.insert("end", "├", "normal")
            for j, cell in enumerate(header):
                width = max(len(cell), max(len(row[j]) for row in data_rows if j < len(row)) if data_rows else 0)
                self.text.insert("end", "─" * (width + 2), "normal")
                if j < len(header) - 1:
                    self.text.insert("end", "┼", "normal")
            self.text.insert("end", "┤\n", "normal")
            
            # Data rows
            for row in data_rows:
                self.text.insert("end", "│", "normal")
                for j, cell in enumerate(row):
                    width = max(len(header[j]), max(len(row2[j]) for row2 in data_rows if j < len(row2)) if data_rows else 0)
                    self.text.insert("end", f" {cell:<{width}} ", "normal")
                    if j < len(row) - 1:
                        self.text.insert("end", "│", "normal")
                self.text.insert("end", "│\n", "normal")
            
            # Footer
            self.text.insert("end", "└", "normal")
            for j, cell in enumerate(header):
                width = max(len(cell), max(len(row[j]) for row in data_rows if j < len(row)) if data_rows else 0)
                self.text.insert("end", "─" * (width + 2), "normal")
                if j < len(header) - 1:
                    self.text.insert("end", "┴", "normal")
            self.text.insert("end", "┘\n", "normal")
            
            self.text.insert("end", "\n", "normal")
        
        return i - 1

