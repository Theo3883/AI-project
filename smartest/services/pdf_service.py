from __future__ import annotations

from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader

from ..core.models import Question, EvaluationResult


class PdfService:
    def export_single_question(self, question: Question, file_path: str) -> None:
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4
        text_obj = c.beginText(40, height - 50)
        text_obj.setFont("Helvetica", 12)
        text_obj.textLine(question.title)
        text_obj.textLine("")
        for line in question.text.split("\n"):
            text_obj.textLine(line)
        c.drawText(text_obj)
        c.showPage()
        c.save()

    def export_test(self, questions: List[Question], file_path: str) -> None:
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4
        for idx, q in enumerate(questions, start=1):
            text_obj = c.beginText(40, height - 50)
            text_obj.setFont("Helvetica", 12)
            text_obj.textLine(f"Intrebarea {idx}: {q.title}")
            text_obj.textLine("")
            for line in q.text.split("\n"):
                text_obj.textLine(line)
            c.drawText(text_obj)
            c.showPage()
        c.save()

    def export_evaluation(
        self,
        question: Question,
        user_answer: str,
        evaluation: EvaluationResult,
        file_path: str,
    ) -> None:
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4
        text_obj = c.beginText(40, height - 50)
        text_obj.setFont("Helvetica", 12)

        text_obj.textLine(f"Intrebare: {question.title}")
        text_obj.textLine("")

        text_obj.textLine("Enunt:")
        for line in question.text.split("\n"):
            text_obj.textLine(line)

        text_obj.textLine("")

        text_obj.textLine("Raspuns student:")
        for line in user_answer.split("\n"):
            text_obj.textLine(line)

        text_obj.textLine("")
        text_obj.textLine(f"Scor: {evaluation.score:.1f}%")
        text_obj.textLine("Feedback:")
        for line in evaluation.feedback.split("\n"):
            text_obj.textLine(line)

        text_obj.textLine("")
        text_obj.textLine("Raspuns de referinta:")
        for line in evaluation.correct_answer.split("\n"):
            text_obj.textLine(line)

        c.drawText(text_obj)
        c.showPage()
        c.save()

    def extract_text(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
        return "\n".join(texts)
