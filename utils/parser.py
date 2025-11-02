"""Markdown parsing utilities."""

import re


from typing import Tuple

def parse_question_and_context(markdown_text: str) -> Tuple[str, str]:
    """
    Parse question and context from markdown text.
    
    Args:
        markdown_text: Markdown text with ### Question and ### Context sections
        
    Returns:
        Tuple of (question, context) strings
    """
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
            q = parts[1].strip()
    
    return q, ctx

