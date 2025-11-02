"""Service layer for Q&A generation and processing."""

from typing import Dict, Any
from models.llm_handler import LLMHandler
from models.bert_handler import BERTHandler
from strategies.base import CategoryStrategy
from strategies.categories import CategoryStrategyFactory
from utils.parser import parse_question_and_context
from utils.prompts import SYSTEM_PROMPT, build_expansion_messages, build_validation_messages


class QAService:
    """Service for generating questions and answers using LLM and BERT."""
    
    def __init__(self, llm_handler: LLMHandler, bert_handler: BERTHandler):
        """
        Initialize QA service.
        
        Args:
            llm_handler: LLM handler instance
            bert_handler: BERT handler instance
        """
        self.llm_handler = llm_handler
        self.bert_handler = bert_handler
    
    def generate_question_and_answer(
        self,
        category_name: str,
        temperature: float = 0.6,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Generate question, context, and answer for a category.
        
        Args:
            category_name: Name of the category
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with 'question', 'context', 'raw_answer', 'expanded_answer', 'validation', 'confidence', 'warnings'
        """
        # Get strategy for category
        strategy = CategoryStrategyFactory.create(category_name)
        
        # Step 1: Generate Question + Context
        messages = strategy.build_generation_messages(SYSTEM_PROMPT)
        res = self.llm_handler.generate(
            messages=messages,
            temperature=max(0.0, min(2.0, temperature)),
            top_p=0.9,
            max_tokens=min(512, max_tokens),
            repeat_penalty=1.1,
        )
        gen_text = res["choices"][0]["message"]["content"].strip()
        question, context = parse_question_and_context(gen_text)
        
        if not question or not context:
            raise RuntimeError("Failed to parse '### Question' and '### Context' from model output.")
        
        # Step 2: Answer with BERT QA
        bert_result = self.bert_handler.answer(question, context)
        bert_raw_answer = bert_result["answer"]
        score = bert_result["score"]
        
        # Confidence warnings
        confidence_warning = ""
        if score < 0.1:
            confidence_warning = "\n⚠️ **Low confidence score** - answer may be unreliable."
        elif score < 0.3:
            confidence_warning = "\n⚠️ **Moderate confidence** - answer should be verified."
        
        # Step 2.5: Expand BERT's answer with Mistral
        expansion_messages = build_expansion_messages(question, context, bert_raw_answer)
        expansion_res = self.llm_handler.generate(
            messages=expansion_messages,
            temperature=0.1,
            top_p=0.8,
            max_tokens=256,
            repeat_penalty=1.1,
        )
        bert_expanded_answer = expansion_res["choices"][0]["message"]["content"].strip()
        
        # Step 3: Validate expanded answer with Mistral
        validation_messages = build_validation_messages(question, context, bert_expanded_answer)
        validation_res = self.llm_handler.generate(
            messages=validation_messages,
            temperature=0.1,
            top_p=0.8,
            max_tokens=128,
            repeat_penalty=1.1,
        )
        validation_text = validation_res["choices"][0]["message"]["content"].strip()
        
        return {
            "category": category_name,
            "generated_text": gen_text,
            "question": question,
            "context": context,
            "raw_answer": bert_raw_answer,
            "expanded_answer": bert_expanded_answer,
            "validation": validation_text,
            "confidence": score,
            "confidence_warning": confidence_warning,
        }
    
    def format_output(self, result: Dict[str, Any]) -> str:
        """Format the result as markdown output with better structure."""
        output = f"## Generated Question & Context\n\n"
        output += f"{result['generated_text']}\n\n"
        output += "─" * 80 + "\n\n"
        
        output += f"## Extracted Answer (BERT)\n\n"
        output += f"**Raw Answer:** {result['raw_answer']}\n\n"
        output += f"**Confidence Score:** {result['confidence']:.3f}{result['confidence_warning']}\n\n"
        output += "─" * 80 + "\n\n"
        
        output += f"## Expanded Answer\n\n"
        output += f"{result['expanded_answer']}\n\n"
        output += "─" * 80 + "\n\n"
        
        output += f"## Validation\n\n"
        output += f"{result['validation']}\n"
        
        return output

