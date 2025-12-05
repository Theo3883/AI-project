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


class EvaluatorFactory:
    def __init__(self, text_processor: TextProcessor) -> None:
        self.text_processor = text_processor
        self._mapping: Dict[QuestionType, type[AnswerEvaluator]] = {
            QuestionType.STRATEGY_SELECTION: StrategyEvaluator,
            QuestionType.NASH_EQUILIBRIUM: NashEvaluator,
            QuestionType.CSP_COMPLETION: CspEvaluator,
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxEvaluator,
        }

    def get_evaluator(self, q_type: QuestionType) -> AnswerEvaluator:
        cls = self._mapping[q_type]
        return cls(self.text_processor)
