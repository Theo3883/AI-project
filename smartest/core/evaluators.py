from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from .models import Question, QuestionType, EvaluationResult
from ..services.text_processing import TextProcessor


class AnswerEvaluator(ABC):
    def __init__(self, text_processor: TextProcessor) -> None:
        self.text_processor = text_processor

    @abstractmethod
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        ...


class StrategyEvaluator(AnswerEvaluator):
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)
        score = self.text_processor.keyword_score(norm_correct, norm_user)

        feedback = (
            f"Raspunsul tau este evaluat la {score:.1f}%. "
            f"Strategia recomandata este: {question.correct_answer}."
        )

        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={},
        )


class NashEvaluator(AnswerEvaluator):
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)
        score = self.text_processor.keyword_score(norm_correct, norm_user)

        feedback = (
            f"Raspunsul tau este evaluat la {score:.1f}%. "
            f"Echilibrul Nash asteptat: {question.correct_answer}."
        )

        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={},
        )


class CspEvaluator(AnswerEvaluator):
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)

        correct_pairs = self.text_processor.extract_assignments(norm_correct)
        user_pairs = self.text_processor.extract_assignments(norm_user)

        total = max(len(correct_pairs), 1)
        matches = 0
        for var, val in correct_pairs.items():
            if var in user_pairs and user_pairs[var] == val:
                matches += 1

        structural_score = 100.0 * matches / total
        text_score = self.text_processor.keyword_score(norm_correct, norm_user)
        score = 0.7 * structural_score + 0.3 * text_score

        feedback = (
            f"Ai {matches} variabile corect din {total}. "
            f"Scor total: {score:.1f}%. Asignarea de referinta este: {question.correct_answer}."
        )

        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={"matches": matches, "total": total},
        )


class MinimaxEvaluator(AnswerEvaluator):
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)

        correct_numbers = self.text_processor.extract_numbers(norm_correct)
        user_numbers = self.text_processor.extract_numbers(norm_user)

        score = 0.0
        if correct_numbers and len(correct_numbers) == len(user_numbers):
            matches = sum(1 for a, b in zip(correct_numbers, user_numbers) if a == b)
            score = 100.0 * matches / len(correct_numbers)
        else:
            score = self.text_processor.keyword_score(norm_correct, norm_user)

        feedback = (
            f"Scor: {score:.1f}%. "
            f"Raspuns de referinta: {question.correct_answer}."
        )

        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={},
        )


class MDPEvaluator(AnswerEvaluator):
    """Evaluates answers for MDP problems (Value/Policy Iteration)."""
    
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        # Check if the correct answer expects both value and policy
        correct_has_value = "V(" in question.correct_answer
        correct_has_policy = "politica" in question.correct_answer.lower() or "policy" in question.correct_answer.lower()
        
        # Check what user provided
        user_has_value = "V(" in user_answer or any(char.isdigit() for char in user_answer)
        user_has_policy = "politica" in user_answer.lower() or "policy" in user_answer.lower() or \
                         any(action in user_answer.lower() for action in ["up", "down", "left", "right", "sus", "jos", "stanga", "dreapta"])
        
        # Extract numerical values from user answer
        user_numbers = self.text_processor.extract_numbers(user_answer)
        correct_numbers = self.text_processor.extract_numbers(question.correct_answer)
        
        # Compare values with tolerance
        value_score = self._compare_with_tolerance(user_numbers, correct_numbers, tolerance=0.05)
        
        # Compare policy
        policy_score = self._compare_policy(user_answer, question.correct_answer) if user_has_policy else 0.0
        
        # Calculate final score based on what was expected and provided
        if correct_has_value and correct_has_policy:
            # Both value and policy expected
            if not user_has_value or not user_has_policy:
                # Penalize if user didn't provide both
                if user_has_value and not user_has_policy:
                    score = value_score * 0.5  # Only 50% for partial answer
                    feedback = f"Scor: {score:.1f}%. Ai furnizat doar valoarea, lipseste politica! Raspuns complet asteptat: {question.correct_answer}"
                elif user_has_policy and not user_has_value:
                    score = policy_score * 0.5  # Only 50% for partial answer
                    feedback = f"Scor: {score:.1f}%. Ai furnizat doar politica, lipseste valoarea! Raspuns complet asteptat: {question.correct_answer}"
                else:
                    score = 0.0
                    feedback = f"Scor: {score:.1f}%. Raspuns incomplet. Raspuns asteptat: {question.correct_answer}"
            else:
                # Both provided, weight them
                score = 0.6 * value_score + 0.4 * policy_score
                feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        elif correct_has_value:
            # Only value expected
            score = value_score
            feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        elif correct_has_policy:
            # Only policy expected
            score = policy_score
            feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        else:
            # Fallback
            score = value_score
            feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        
        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=user_answer,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={}
        )
    
    def _compare_with_tolerance(self, user_nums: list, correct_nums: list, tolerance: float = 0.05) -> float:
        """Compare numbers with floating point tolerance."""
        if not correct_nums:
            return self.text_processor.keyword_score(
                str(correct_nums), str(user_nums)
            )
        
        if len(user_nums) != len(correct_nums):
            return 0.0
        
        matches = 0
        for u, c in zip(user_nums, correct_nums):
            if abs(u - c) <= tolerance * abs(c) if c != 0 else abs(u - c) <= tolerance:
                matches += 1
        
        return 100.0 * matches / len(correct_nums)
    
    def _compare_policy(self, user_answer: str, correct_answer: str) -> float:
        """Compare policy actions."""
        user_lower = user_answer.lower()
        correct_lower = correct_answer.lower()
        
        actions = ["up", "down", "left", "right", "sus", "jos", "stanga", "dreapta"]
        
        user_actions = [a for a in actions if a in user_lower]
        correct_actions = [a for a in actions if a in correct_lower]
        
        if not correct_actions:
            return 100.0
        
        # Check if any user action matches correct action
        # (allowing for Romanian/English equivalents)
        equivalents = {
            "up": "sus", "down": "jos", "left": "stanga", "right": "dreapta",
            "sus": "up", "jos": "down", "stanga": "left", "dreapta": "right"
        }
        
        for user_action in user_actions:
            if user_action in correct_actions:
                return 100.0
            equiv = equivalents.get(user_action)
            if equiv and equiv in correct_actions:
                return 100.0
        
        return 0.0


class RLEvaluator(AnswerEvaluator):
    """Evaluates answers for RL problems (Q-learning, TD-learning)."""
    
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        # Extract numerical values from user answer
        user_numbers = self.text_processor.extract_numbers(user_answer)
        correct_numbers = self.text_processor.extract_numbers(question.correct_answer)
        
        # For RL, we're more lenient with numerical precision
        numerical_score = self._compare_with_tolerance(user_numbers, correct_numbers, tolerance=0.1)
        
        # Also check for policy if present
        policy_score = 0.0
        if "politica" in user_answer.lower() or "policy" in user_answer.lower() or "π" in user_answer:
            policy_score = self._compare_policy(user_answer, question.correct_answer)
        
        # Check for Q-values format
        q_format_score = 0.0
        if "q(" in user_answer.lower():
            q_format_score = 20.0  # Bonus for correct format
        
        # Combine scores
        if policy_score > 0:
            score = 0.5 * numerical_score + 0.4 * policy_score + 0.1 * q_format_score
        else:
            score = 0.9 * numerical_score + 0.1 * q_format_score
        
        feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        
        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=user_answer,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={}
        )
    
    def _compare_with_tolerance(self, user_nums: list, correct_nums: list, tolerance: float = 0.1) -> float:
        """Compare numbers with floating point tolerance."""
        if not correct_nums:
            return 50.0  # Partial credit if no numbers expected
        
        if len(user_nums) < len(correct_nums):
            return 0.0
        
        matches = 0
        for i, c in enumerate(correct_nums):
            if i < len(user_nums):
                u = user_nums[i]
                if abs(u - c) <= tolerance * abs(c) if c != 0 else abs(u - c) <= tolerance:
                    matches += 1
        
        return 100.0 * matches / len(correct_nums)
    
    def _compare_policy(self, user_answer: str, correct_answer: str) -> float:
        """Compare policy actions."""
        user_lower = user_answer.lower()
        correct_lower = correct_answer.lower()
        
        actions = ["up", "down", "left", "right", "sus", "jos", "stanga", "dreapta"]
        
        user_actions = [a for a in actions if a in user_lower]
        correct_actions = [a for a in actions if a in correct_lower]
        
        if not correct_actions:
            return 100.0
        
        # Check if actions overlap
        equivalents = {
            "up": "sus", "down": "jos", "left": "stanga", "right": "dreapta",
            "sus": "up", "jos": "down", "stanga": "left", "dreapta": "right"
        }
        
        matches = 0
        for correct_action in correct_actions:
            if correct_action in user_actions:
                matches += 1
            else:
                equiv = equivalents.get(correct_action)
                if equiv and equiv in user_actions:
                    matches += 1
        
        return 100.0 * matches / len(correct_actions) if correct_actions else 100.0


class EvaluatorFactory:
    def __init__(self, text_processor: TextProcessor) -> None:
        self.text_processor = text_processor
        self._mapping: Dict[QuestionType, type[AnswerEvaluator]] = {
            QuestionType.STRATEGY_SELECTION: StrategyEvaluator,
            QuestionType.NASH_EQUILIBRIUM: NashEvaluator,
            QuestionType.CSP_COMPLETION: CspEvaluator,
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxEvaluator,
            QuestionType.VALUE_ITERATION: MDPEvaluator,
            QuestionType.POLICY_ITERATION: MDPEvaluator,
            QuestionType.Q_LEARNING: RLEvaluator,
            QuestionType.TD_LEARNING: RLEvaluator,
            QuestionType.RL_PARAMETERS: RLEvaluator,
            QuestionType.MDP_COMPARISON: MDPEvaluator,
        }

    def get_evaluator(self, q_type: QuestionType) -> AnswerEvaluator:
        cls = self._mapping[q_type]
        return cls(self.text_processor)
