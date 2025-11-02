"""Prompt building utilities."""

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


def build_expansion_messages(question: str, context: str, partial_answer: str) -> list:
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


def build_validation_messages(question: str, context: str, proposed_answer: str) -> list:
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

