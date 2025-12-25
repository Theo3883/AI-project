from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QMessageBox,
)
from PySide6.QtCore import Qt

if TYPE_CHECKING:
    from ..app import SmarTestApp


class QAPanel(QWidget):
    """Widget for the Q&A problem solver interface."""
    
    def __init__(self, app_facade: SmarTestApp) -> None:
        super().__init__()
        self.app_facade = app_facade
        self._build_ui()
    
    def _build_ui(self) -> None:
        """Build the Q&A interface."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Rezolvator de Probleme AI - Sistem Q&A")
        header.setStyleSheet("font-weight: bold; font-size: 18px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Instructions
        instructions = QLabel(
            "Descrie problema ta mai jos. "
            "Sistemul va detecta automat tipul de problema si va furniza o solutie."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Input section
        input_group = QGroupBox("Descrierea Problemei")
        input_layout = QVBoxLayout(input_group)
        
        input_layout.addWidget(QLabel("Introdu problema ta:"))
        self.txt_input = QTextEdit()
        self.txt_input.setPlaceholderText(
            "Exemplu: Gaseste echilibrul Nash intr-un joc 2x2 cu payoff-uri: "
            "(3,2) si (1,4) pe primul rand, (2,3) si (4,1) pe al doilea rand."
        )
        self.txt_input.setMinimumHeight(120)
        input_layout.addWidget(self.txt_input)
        
        # Buttons row
        btn_row = QHBoxLayout()
        
        self.btn_solve = QPushButton("Rezolva Problema")
        self.btn_solve.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_solve.clicked.connect(self._on_solve)
        btn_row.addWidget(self.btn_solve)
        
        self.btn_examples = QPushButton("Arata Exemple")
        self.btn_examples.clicked.connect(self._on_show_examples)
        btn_row.addWidget(self.btn_examples)
        
        self.btn_clear = QPushButton("Sterge Tot")
        self.btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self.btn_clear)
        
        input_layout.addLayout(btn_row)
        layout.addWidget(input_group)
        
        # Output section
        output_group = QGroupBox("Solutie")
        output_layout = QVBoxLayout(output_group)
        
        # Detected type
        output_layout.addWidget(QLabel("Tip de Problema Detectat:"))
        self.txt_detected_type = QTextEdit()
        self.txt_detected_type.setReadOnly(True)
        self.txt_detected_type.setMaximumHeight(40)
        output_layout.addWidget(self.txt_detected_type)
        
        # Extracted parameters
        output_layout.addWidget(QLabel("Parametrii Extrasi:"))
        self.txt_params = QTextEdit()
        self.txt_params.setReadOnly(True)
        self.txt_params.setMaximumHeight(120)
        output_layout.addWidget(self.txt_params)
        
        # Solution
        output_layout.addWidget(QLabel("Solutie:"))
        self.txt_solution = QTextEdit()
        self.txt_solution.setReadOnly(True)
        self.txt_solution.setMaximumHeight(100)
        output_layout.addWidget(self.txt_solution)
        
        # Explanation
        output_layout.addWidget(QLabel("Explicatie:"))
        self.txt_explanation = QTextEdit()
        self.txt_explanation.setReadOnly(True)
        output_layout.addWidget(self.txt_explanation, 1)
        
        layout.addWidget(output_group, 1)
    
    def _on_solve(self) -> None:
        """Handle solve button click."""
        question_text = self.txt_input.toPlainText().strip()
        
        if not question_text:
            QMessageBox.information(
                self,
                "Nicio intrare",
                "Te rog introdu mai intai o descriere a problemei."
            )
            return
        
        # Clear previous output
        self.txt_detected_type.clear()
        self.txt_params.clear()
        self.txt_solution.clear()
        self.txt_explanation.clear()
        
        # Get answer from the system
        response = self.app_facade.answer_question(question_text)
        
        if response.success:
            # Display successful result
            self.txt_detected_type.setPlainText(
                f"{response.detected_type} (incredere: {response.confidence:.0%})"
            )
            self.txt_detected_type.setStyleSheet("background-color: #e8f5e9;")
            
            self.txt_params.setPlainText(response.extracted_params)
            self.txt_solution.setPlainText(response.solution)
            self.txt_explanation.setPlainText(response.explanation)
        else:
            # Display error
            self.txt_detected_type.setPlainText("Eroare la parsarea problemei")
            self.txt_detected_type.setStyleSheet("background-color: #ffebee;")
            
            self.txt_params.setPlainText("Nu s-au putut extrage parametrii")
            self.txt_solution.setPlainText("Nicio solutie disponibila")
            self.txt_explanation.setPlainText(
                f"Eroare: {response.error_message}\n\n"
                "Te rog reformuleaza intrebarea sau apasa 'Arata Exemple' "
                "pentru a vedea formatul asteptat."
            )
    
    def _on_show_examples(self) -> None:
        """Show example questions in a message box."""
        examples = self.app_facade.get_example_questions()
        
        message_parts = ["Iata exemple de intrebari pentru fiecare tip de problema:\n"]
        
        for problem_type, questions in examples.items():
            message_parts.append(f"\n{problem_type}:")
            for i, question in enumerate(questions, 1):
                message_parts.append(f"  {i}. {question}")
        
        message = "\n".join(message_parts)
        
        # Create a scrollable message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Exemple de Intrebari")
        msg_box.setText("Exemple pentru fiecare tip de problema:")
        msg_box.setDetailedText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec()
    
    def _on_clear(self) -> None:
        """Clear all input and output fields."""
        self.txt_input.clear()
        self.txt_detected_type.clear()
        self.txt_params.clear()
        self.txt_solution.clear()
        self.txt_explanation.clear()
        self.txt_detected_type.setStyleSheet("")

