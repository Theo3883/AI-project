from __future__ import annotations

from typing import List

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QTabWidget,
)

from ..core.models import Question, QuestionType
from ..app import SmarTestApp
from .qa_panel import QAPanel


class MainWindow(QMainWindow):
    def __init__(self, app_facade: SmarTestApp) -> None:
        super().__init__()
        self.app_facade = app_facade
        self.questions: List[Question] = []
        self.current_index: int = -1

        self.setWindowTitle("SmarTest – Trainer pentru examenul de AI")
        self.resize(1000, 700)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        # Create main layout with tabs
        main_layout = QVBoxLayout(central)
        
        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Tab 1: Question Generator (existing functionality)
        generator_tab = QWidget()
        tabs.addTab(generator_tab, "Question Generator")
        root_layout = QHBoxLayout(generator_tab)

        left_panel = QVBoxLayout()
        root_layout.addLayout(left_panel, 1)

        cfg_group = QGroupBox("Generare intrebari")
        cfg_layout = QVBoxLayout(cfg_group)

        self.spin_count = QSpinBox()
        self.spin_count.setMinimum(1)
        self.spin_count.setMaximum(20)
        self.spin_count.setValue(4)
        cfg_layout.addWidget(QLabel("Numar de intrebari:"))
        cfg_layout.addWidget(self.spin_count)

        # Classic problem types
        self.chk_strategy = QCheckBox("Strategie pentru problema (N-Queens, etc.)")
        self.chk_nash = QCheckBox("Echilibru Nash (joc 2x2)")
        self.chk_csp = QCheckBox("CSP cu Backtracking + MRV")
        self.chk_minimax = QCheckBox("MinMax cu Alpha-Beta")
        
        # MDP and Reinforcement Learning types
        self.chk_value_iteration = QCheckBox("Value Iteration (MDP)")
        self.chk_policy_iteration = QCheckBox("Policy Iteration (MDP)")
        self.chk_qlearning = QCheckBox("Q-learning (RL)")
        self.chk_tdlearning = QCheckBox("TD-learning (RL)")
        self.chk_rl_params = QCheckBox("Parametri RL (alpha, gamma, epsilon)")
        
        # Add all checkboxes to layout
        for chk in [self.chk_strategy, self.chk_nash, self.chk_csp, self.chk_minimax,
                    self.chk_value_iteration, self.chk_policy_iteration, 
                    self.chk_qlearning, self.chk_tdlearning, self.chk_rl_params]:
            chk.setChecked(True)
            cfg_layout.addWidget(chk)

        self.btn_generate = QPushButton("Genereaza intrebari")
        self.btn_generate.clicked.connect(self._on_generate)
        cfg_layout.addWidget(self.btn_generate)

        left_panel.addWidget(cfg_group)

        self.list_questions = QListWidget()
        self.list_questions.currentRowChanged.connect(self._on_question_selected)
        left_panel.addWidget(QLabel("Intrebari generate:"))
        left_panel.addWidget(self.list_questions, 1)

        btn_export_test = QPushButton("Exporta testul in PDF...")
        btn_export_test.clicked.connect(self._on_export_test)
        left_panel.addWidget(btn_export_test)

        right_panel = QVBoxLayout()
        root_layout.addLayout(right_panel, 2)

        self.lbl_title = QLabel("Nicio intrebare inca.")
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        right_panel.addWidget(self.lbl_title)

        right_panel.addWidget(QLabel("Enunt:"))
        self.txt_question = QTextEdit()
        self.txt_question.setReadOnly(True)
        right_panel.addWidget(self.txt_question, 2)

        right_panel.addWidget(QLabel("Raspunsul tau:"))
        self.txt_answer = QTextEdit()
        right_panel.addWidget(self.txt_answer, 2)

        btn_row = QHBoxLayout()
        self.btn_check = QPushButton("Evalueaza raspunsul")
        self.btn_check.clicked.connect(self._on_evaluate)
        btn_row.addWidget(self.btn_check)

        self.btn_load_pdf = QPushButton("Incarca raspuns din PDF...")
        self.btn_load_pdf.clicked.connect(self._on_load_pdf)
        btn_row.addWidget(self.btn_load_pdf)

        right_panel.addLayout(btn_row)

        right_panel.addWidget(QLabel("Feedback & raspuns corect:"))
        self.txt_feedback = QTextEdit()
        self.txt_feedback.setReadOnly(True)
        right_panel.addWidget(self.txt_feedback, 2)

        btn_export_eval = QPushButton("Exporta feedback pentru intrebare in PDF...")
        btn_export_eval.clicked.connect(self._on_export_evaluation)
        right_panel.addWidget(btn_export_eval)
        
        # Tab 2: Q&A Problem Solver
        qa_tab = QAPanel(self.app_facade)
        tabs.addTab(qa_tab, "Rezolvator Probleme Q&A")

    def _selected_types(self) -> List[QuestionType]:
        types: List[QuestionType] = []
        # Classic types
        if self.chk_strategy.isChecked():
            types.append(QuestionType.STRATEGY_SELECTION)
        if self.chk_nash.isChecked():
            types.append(QuestionType.NASH_EQUILIBRIUM)
        if self.chk_csp.isChecked():
            types.append(QuestionType.CSP_COMPLETION)
        if self.chk_minimax.isChecked():
            types.append(QuestionType.MINIMAX_ALPHA_BETA)
        # MDP and RL types
        if self.chk_value_iteration.isChecked():
            types.append(QuestionType.VALUE_ITERATION)
        if self.chk_policy_iteration.isChecked():
            types.append(QuestionType.POLICY_ITERATION)
        if self.chk_qlearning.isChecked():
            types.append(QuestionType.Q_LEARNING)
        if self.chk_tdlearning.isChecked():
            types.append(QuestionType.TD_LEARNING)
        if self.chk_rl_params.isChecked():
            types.append(QuestionType.RL_PARAMETERS)
        return types

    def _current_question(self) -> Question | None:
        if 0 <= self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def _on_generate(self) -> None:
        q_types = self._selected_types()
        if not q_types:
            QMessageBox.warning(self, "Atentie", "Selecteaza cel putin un tip de intrebare.")
            return

        count = int(self.spin_count.value())
        self.questions = self.app_facade.generate_questions(q_types, count)
        self.list_questions.clear()
        for q in self.questions:
            item = QListWidgetItem(f"[{q.q_type.name}] {q.title}")
            self.list_questions.addItem(item)
        if self.questions:
            self.list_questions.setCurrentRow(0)

    def _on_question_selected(self, row: int) -> None:
        self.current_index = row
        q = self._current_question()
        if not q:
            self.lbl_title.setText("Nicio intrebare selectata.")
            self.txt_question.clear()
            return
        self.lbl_title.setText(q.title)
        self.txt_question.setPlainText(q.text)
        self.txt_answer.clear()
        self.txt_feedback.clear()

    def _on_evaluate(self) -> None:
        q = self._current_question()
        if not q:
            return
        user_answer = self.txt_answer.toPlainText().strip()
        if not user_answer:
            QMessageBox.information(self, "Info", "Introdu un raspuns inainte de evaluare.")
            return
        result = self.app_facade.evaluate_answer(q, user_answer)
        fb_lines = [
            f"Scor: {result.score:.1f}%",
            "",
            "Feedback:",
            result.feedback,
            "",
            "Raspuns de referinta:",
            q.correct_answer,
        ]
        self.txt_feedback.setPlainText("\n".join(fb_lines))

    def _on_load_pdf(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Alege un PDF cu raspunsul",
            filter="PDF files (*.pdf)",
        )
        if not file_path:
            return
        try:
            text = self.app_facade.extract_text_from_pdf(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Eroare", f"Nu am putut citi PDF-ul: {exc}")
            return
        self.txt_answer.setPlainText(text)

    def _on_export_test(self) -> None:
        if not self.questions:
            QMessageBox.information(self, "Info", "Nu exista intrebari generate.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salveaza testul in PDF",
            filter="PDF files (*.pdf)",
        )
        if not file_path:
            return
        try:
            self.app_facade.export_test_pdf(self.questions, file_path)
            QMessageBox.information(self, "Succes", "Testul a fost salvat.")
        except Exception as exc:
            QMessageBox.critical(self, "Eroare", f"Nu am putut scrie PDF-ul: {exc}")

    def _on_export_evaluation(self) -> None:
        q = self._current_question()
        if not q:
            return
        user_answer = self.txt_answer.toPlainText().strip()
        if not user_answer:
            QMessageBox.information(self, "Info", "Nu exista raspuns pentru aceasta intrebare.")
            return
        result = self.app_facade.evaluate_answer(q, user_answer)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salveaza feedback-ul in PDF",
            filter="PDF files (*.pdf)",
        )
        if not file_path:
            return
        try:
            self.app_facade.export_evaluation_pdf(q, user_answer, result, file_path)
            QMessageBox.information(self, "Succes", "Feedback-ul a fost salvat.")
        except Exception as exc:
            QMessageBox.critical(self, "Eroare", f"Nu am putut scrie PDF-ul: {exc}")


def run_gui(app_facade: SmarTestApp) -> None:
    import sys
    qt_app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(app_facade)
    win.show()
    qt_app.exec()
