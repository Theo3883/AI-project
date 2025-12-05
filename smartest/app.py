from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .core.models import Question, QuestionType, EvaluationResult
from .core.generators import QuestionFactory
from .core.evaluators import EvaluatorFactory
from .services.text_processing import TextProcessor
from .services.pdf_service import PdfService


@dataclass
class GeneratedTest:
    questions: List[Question]


class SmarTestApp:
    def __init__(self) -> None:
        self.text_processor = TextProcessor()
        self.pdf_service = PdfService()
        self.question_factory = QuestionFactory()
        self.evaluator_factory = EvaluatorFactory(self.text_processor)
        self._questions: List[Question] = []
        self._current_test: Optional[GeneratedTest] = None

    def generate_questions(self, question_types: List[QuestionType], count: int, difficulty: str = "medium") -> List[Question]:
        self._questions = []
        i = 0
        while len(self._questions) < count:
            q_type = question_types[i % len(question_types)]
            generator = self.question_factory.get_generator(q_type)
            question = generator.generate(difficulty=difficulty)
            self._questions.append(question)
            i += 1
        self._current_test = GeneratedTest(self._questions)
        return self._questions

    @property
    def questions(self) -> List[Question]:
        return list(self._questions)

    def build_test(self, questions: Optional[List[Question]] = None) -> GeneratedTest:
        if questions is None:
            questions = self._questions
        self._current_test = GeneratedTest(questions=list(questions))
        return self._current_test

    def evaluate_answer(self, question: Question, user_answer_text: str) -> EvaluationResult:
        evaluator = self.evaluator_factory.get_evaluator(question.q_type)
        return evaluator.evaluate(question, user_answer_text)

    def export_question_pdf(self, question: Question, file_path: str) -> None:
        self.pdf_service.export_single_question(question, file_path)

    def export_test_pdf(self, questions: List[Question], file_path: str) -> None:
        self.pdf_service.export_test(questions, file_path)

    def export_evaluation_pdf(self, question: Question, user_answer: str, evaluation: EvaluationResult, file_path: str) -> None:
        self.pdf_service.export_evaluation(question, user_answer, evaluation, file_path)

    def extract_text_from_pdf(self, file_path: str) -> str:
        return self.pdf_service.extract_text(file_path)

    def run(self) -> None:
        from .gui.main_window import run_gui
        run_gui(self)
