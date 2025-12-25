from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .problem_parser import ProblemParser, ParsedProblem
from ..core.models import QuestionType
from ..core.solvers import SolverFactory


@dataclass
class QAResponse:
    """Response from the Q&A system."""
    success: bool
    detected_type: str
    detected_type_enum: QuestionType | None
    confidence: float
    extracted_params: str
    solution: str
    explanation: str
    error_message: str | None = None


class QAService:
    """Service that orchestrates question answering: parsing -> solving -> formatting."""
    
    def __init__(self) -> None:
        self.parser = ProblemParser()
        self.solver_factory = SolverFactory()
    
    def answer_question(self, question_text: str) -> QAResponse:
        """
        Process a user's question and return a solution.
        
        Args:
            question_text: Natural language description of the problem
            
        Returns:
            QAResponse with solution or error information
        """
        try:
            # Step 1: Parse the question
            parsed = self.parser.parse(question_text)
            
            # Step 2: Get the appropriate solver
            solver = self.solver_factory.get_solver(parsed.question_type)
            
            # Step 3: Solve the problem
            solution_data = solver.solve(parsed.data)
            
            # Step 4: Format the response
            return self._format_response(parsed, solution_data)
            
        except ValueError as e:
            # Parsing or validation error
            error_msg = str(e)
            
            # For debugging: if not the standard multi-problem error, show more info
            if error_msg == "Nu stiu sa raspund acum":
                explanation = (
                    "Am detectat multiple tipuri de probleme in intrebarea ta. "
                    "Te rog sa intrebi despre un singur tip de problema la un moment dat:\n"
                    "- Nash Equilibrium SAU\n"
                    "- CSP SAU\n"
                    "- Minimax Alpha-Beta SAU\n"
                    "- Strategy Selection"
                )
            else:
                explanation = ""
            
            return QAResponse(
                success=False,
                detected_type="Unknown",
                detected_type_enum=None,
                confidence=0.0,
                extracted_params="",
                solution="",
                explanation=explanation,
                error_message=error_msg
            )
        except Exception as e:
            # Unexpected error
            return QAResponse(
                success=False,
                detected_type="Unknown",
                detected_type_enum=None,
                confidence=0.0,
                extracted_params="",
                solution="",
                explanation="",
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _format_response(self, parsed: ParsedProblem, solution_data: Dict[str, Any]) -> QAResponse:
        """Format the solution based on problem type."""
        
        if parsed.question_type == QuestionType.NASH_EQUILIBRIUM:
            return self._format_nash_response(parsed, solution_data)
        elif parsed.question_type == QuestionType.CSP_COMPLETION:
            return self._format_csp_response(parsed, solution_data)
        elif parsed.question_type == QuestionType.MINIMAX_ALPHA_BETA:
            return self._format_minimax_response(parsed, solution_data)
        elif parsed.question_type == QuestionType.STRATEGY_SELECTION:
            return self._format_strategy_response(parsed, solution_data)
        else:
            return QAResponse(
                success=False,
                detected_type=parsed.question_type.name,
                detected_type_enum=parsed.question_type,
                confidence=parsed.confidence,
                extracted_params="",
                solution="",
                explanation="",
                error_message=f"Unsupported question type: {parsed.question_type.name}"
            )
    
    def _format_nash_response(self, parsed: ParsedProblem, solution_data: Dict[str, Any]) -> QAResponse:
        """Format Nash equilibrium solution."""
        game = parsed.data["game"]
        equilibria = solution_data.get("equilibria", [])
        solution_text = solution_data.get("text", "")
        
        # Format extracted parameters
        params_lines = ["Matricea Jocului:"]
        params_lines.append(f"  Strategii Rand: {', '.join(game.row_strategies)}")
        params_lines.append(f"  Strategii Coloana: {', '.join(game.col_strategies)}")
        params_lines.append("  Payoff-uri (Jucator Rand, Jucator Coloana):")
        for i, row in enumerate(game.payoffs):
            row_str = "    " + "  ".join(str(payoff) for payoff in row)
            params_lines.append(row_str)
        
        # Format solution
        if equilibria:
            solution_parts = ["Echilibru Nash gasit:"]
            for i, j in equilibria:
                solution_parts.append(
                    f"  • ({game.row_strategies[i]}, {game.col_strategies[j]}) "
                    f"cu payoff-uri {game.payoffs[i][j]}"
                )
            solution = "\n".join(solution_parts)
        else:
            solution = "Nu exista echilibru Nash pur in acest joc."
        
        return QAResponse(
            success=True,
            detected_type="Echilibru Nash",
            detected_type_enum=QuestionType.NASH_EQUILIBRIUM,
            confidence=parsed.confidence,
            extracted_params="\n".join(params_lines),
            solution=solution,
            explanation=solution_text
        )
    
    def _format_csp_response(self, parsed: ParsedProblem, solution_data: Dict[str, Any]) -> QAResponse:
        """Format CSP solution."""
        csp = parsed.data["csp"]
        solution = solution_data.get("solution")
        
        # Format extracted parameters
        params_lines = ["Problema CSP:"]
        params_lines.append(f"  Variabile: {', '.join(csp.variables.keys())}")
        
        # Show domains
        sample_var = next(iter(csp.variables.values()))
        params_lines.append(f"  Domeniu: {', '.join(str(v) for v in sample_var.domain)}")
        
        # Show constraints
        if csp.constraints:
            params_lines.append("  Constrangeri:")
            for c in csp.constraints:
                params_lines.append(f"    • {c.var1} {c.relation} {c.var2}")
        
        # Show partial assignment
        if csp.partial_assignment:
            params_lines.append("  Asignare Partiala:")
            for var, val in csp.partial_assignment.items():
                params_lines.append(f"    • {var} = {val}")
        
        # Format solution
        if solution:
            solution_lines = ["Solutie gasita:"]
            for var in sorted(solution.keys()):
                solution_lines.append(f"  • {var} = {solution[var]}")
            solution_text = "\n".join(solution_lines)
        else:
            solution_text = "Nu s-a gasit solutie. CSP-ul ar putea fi nesatisfiabil."
        
        explanation = (
            "Solutie obtinuta folosind Backtracking cu euristica MRV (Minimum Remaining Values) "
            "si Forward Checking. Algoritmul exploreaza spatiul de cautare "
            "in mod sistematic, alegand variabilele cu cele mai putine valori valide ramase."
        )
        
        return QAResponse(
            success=True,
            detected_type="CSP (Problema de Satisfacere a Constrangerilor)",
            detected_type_enum=QuestionType.CSP_COMPLETION,
            confidence=parsed.confidence,
            extracted_params="\n".join(params_lines),
            solution=solution_text,
            explanation=explanation
        )
    
    def _format_minimax_response(self, parsed: ParsedProblem, solution_data: Dict[str, Any]) -> QAResponse:
        """Format Minimax with Alpha-Beta solution."""
        root_value = solution_data.get("root_value")
        leaf_count = solution_data.get("leaf_count")
        
        # Count total leaves in tree
        def count_leaves(node):
            if not node.children:
                return 1
            return sum(count_leaves(child) for child in node.children)
        
        root = parsed.data["root"]
        total_leaves = count_leaves(root)
        
        # Format extracted parameters
        params_lines = ["Arbore de Joc:"]
        params_lines.append(f"  Total noduri frunza: {total_leaves}")
        params_lines.append("  Structura arbore: Arbore binar cu niveluri MAX/MIN alternate")
        
        # Try to extract leaf values for display
        def get_leaf_values(node):
            if not node.children:
                return [node.value]
            values = []
            for child in node.children:
                values.extend(get_leaf_values(child))
            return values
        
        leaf_values = get_leaf_values(root)
        params_lines.append(f"  Valori frunze: {leaf_values}")
        
        # Format solution
        solution = (
            f"Valoare Minimax la radacina: {root_value}\n"
            f"Noduri frunza evaluate: {leaf_count} din {total_leaves}\n"
            f"Noduri frunza taiate (pruned): {total_leaves - leaf_count}"
        )
        
        explanation = (
            "Algoritmul Minimax cu Alpha-Beta pruning a fost aplicat depth-first de la stanga la dreapta. "
            "Valorile Alpha si Beta au fost mentinute la fiecare nod pentru a detecta situatii in care explorarea "
            "ulterioara nu ar afecta rezultatul final. "
            f"Optimizarea prin pruning a redus numarul de evaluari de la {total_leaves} la {leaf_count}, "
            f"economisind {total_leaves - leaf_count} evaluari."
        )
        
        return QAResponse(
            success=True,
            detected_type="Minimax cu Alpha-Beta Pruning",
            detected_type_enum=QuestionType.MINIMAX_ALPHA_BETA,
            confidence=parsed.confidence,
            extracted_params="\n".join(params_lines),
            solution=solution,
            explanation=explanation
        )
    
    def _format_strategy_response(self, parsed: ParsedProblem, solution_data: Dict[str, Any]) -> QAResponse:
        """Format strategy selection solution."""
        problem = parsed.data["problem"]
        recommended = solution_data.get("recommended_strategy", "Necunoscut")
        explanation = solution_data.get("explanation", "")
        
        # Format extracted parameters
        params_lines = ["Tip de Problema:"]
        params_lines.append(f"  Problema identificata: {problem}")
        
        # Format solution
        solution = f"Strategie recomandata: {recommended}"
        
        return QAResponse(
            success=True,
            detected_type="Selectie Strategie",
            detected_type_enum=QuestionType.STRATEGY_SELECTION,
            confidence=parsed.confidence,
            extracted_params="\n".join(params_lines),
            solution=solution,
            explanation=explanation
        )
    
    def get_example_questions(self) -> Dict[str, List[str]]:
        """Return example questions for each problem type."""
        return {
            "Echilibru Nash": [
                "Gaseste echilibrul Nash intr-un joc 2x2 cu payoff-uri: (3,2) si (1,4) pe primul rand, (2,3) si (4,1) pe al doilea rand.",
                "Am un joc cu payoff-uri (5,5) (0,10) pe randul 1 si (10,0) (1,1) pe randul 2. Care este echilibrul Nash?",
                "Analizeaza acest joc pentru echilibru Nash: Rand 1: (4,3) (2,5), Rand 2: (3,4) (5,2)"
            ],
            "CSP": [
                "Rezolva un CSP de colorare a grafului cu variabile A, B, C si constrangeri A!=B, B!=C, A!=C folosind culorile rosu, albastru, verde. A este deja rosu.",
                "Am un CSP cu variabile X, Y, Z cu domeniu [1, 2, 3]. Constrangeri: X!=Y, Y!=Z, X!=Z. X este 1.",
                "Problema colorare graf: 4 variabile A, B, C, D. Culori: rosu, albastru, verde. Constrangeri: A!=B, A!=C, B!=C, C!=D. Partial: A=rosu"
            ],
            "Minimax Alpha-Beta": [
                "Aplica minimax cu alpha-beta pruning pe un arbore de joc cu valori frunze: [3, 5, 2, 9, 7, 4, 8, 1]",
                "Arbore de joc cu 8 frunze: 6, 2, 8, 4, 5, 9, 3, 7. Gaseste valoarea minimax cu alpha-beta.",
                "Minimax alpha-beta pe arbore cu frunze [10, 5, 7, 3, 12, 8, 2, 14]"
            ],
            "Selectie Strategie": [
                "Care este cea mai buna strategie pentru rezolvarea problemei n-queens?",
                "Ce algoritm ar trebui sa folosesc pentru Turnurile din Hanoi?",
                "Recomanda o strategie pentru problema de colorare a grafului.",
                "Ce abordare functioneaza pentru problema calului (knight's tour)?"
            ]
        }

