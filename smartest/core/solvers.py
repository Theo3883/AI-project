from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from .models import QuestionType, NashGame, CspInstance, GameTreeNode


class Solver(ABC):
    @abstractmethod
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ...


class StrategySolver(Solver):
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        problem = data.get("problem", "")
        
        strategy_map = {
            "n-queens": "Backtracking",
            "hanoi": "DFS",
            "graph-coloring": "Backtracking cu MRV si Forward Checking",
            "knights-tour": "Backtracking",
        }
        
        recommended_strategy = strategy_map.get(problem, "Backtracking")
        
        explanation_map = {
            "n-queens": (
                "Problema N-Queens este o problema de satisfacere a constrangerilor "
                "care necesita explorare sistematica a spatiului de solutii. "
                "Backtracking este ideal pentru ca permite revenirea la stari anterioare "
                "cand o asignare partiala nu poate fi extinsa la o solutie completa."
            ),
            "hanoi": (
                "Problema Turnurilor din Hanoi are o structura recursiva naturala. "
                "DFS (Depth-First Search) este potrivit pentru ca exploreaza complet "
                "o ramura a spatiului de stari inainte de a trece la alta, "
                "ceea ce corespunde naturii recursive a problemei."
            ),
            "graph-coloring": (
                "Graph Coloring este o problema CSP clasica. Backtracking cu MRV "
                "(Minimum Remaining Values) si Forward Checking optimizeaza cautarea "
                "prin alegerea variabilelor cu cele mai putine valori posibile si "
                "prin eliminarea anticipata a valorilor inconsistente."
            ),
            "knights-tour": (
                "Knight's Tour este o problema de cautare care necesita explorare "
                "sistematica. Backtracking permite revenirea la stari anterioare cand "
                "o secventa de mutari nu poate fi extinsa la un tur complet."
            ),
        }
        
        explanation = explanation_map.get(problem, "Strategia recomandata pentru aceasta problema.")
        
        return {
            "recommended_strategy": recommended_strategy,
            "explanation": explanation,
        }


class NashSolver(Solver):
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        game: NashGame = data["game"]
        
        equilibria = []
        
        # Verificam fiecare combinatie de strategii pentru echilibru Nash
        for i, row_strategy in enumerate(game.row_strategies):
            for j, col_strategy in enumerate(game.col_strategies):
                row_payoff, col_payoff = game.payoffs[i][j]
                
                # Verificam daca jucatorul 1 (row) are o deviatie profitabila
                row_has_deviation = False
                for k in range(len(game.row_strategies)):
                    if k != i and game.payoffs[k][j][0] > row_payoff:
                        row_has_deviation = True
                        break
                
                # Verificam daca jucatorul 2 (col) are o deviatie profitabila
                col_has_deviation = False
                for k in range(len(game.col_strategies)):
                    if k != j and game.payoffs[i][k][1] > col_payoff:
                        col_has_deviation = True
                        break
                
                # Daca niciun jucator nu are deviatie profitabila, este echilibru Nash
                if not row_has_deviation and not col_has_deviation:
                    equilibria.append((i, j))
        
        if equilibria:
            i, j = equilibria[0]
            text = (
                f"Exista echilibru Nash pur: ({game.row_strategies[i]}, {game.col_strategies[j]}). "
                f"Niciun jucator nu are motiv sa devieze unilateral de la aceasta combinatie de strategii."
            )
        else:
            text = "Nu exista echilibru Nash pur. Cel putin un jucator are o deviatie profitabila de la orice combinatie de strategii."
        
        return {
            "equilibria": equilibria,
            "text": text,
        }


class CspSolver(Solver):
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        csp: CspInstance = data["csp"]
        
        # Implementare simpla de backtracking cu MRV si forward checking
        solution = {}
        
        def is_consistent(var: str, val: Any, assignment: Dict[str, Any]) -> bool:
            for constraint in csp.constraints:
                if constraint.var1 == var and constraint.var2 in assignment:
                    if constraint.relation == "!=":
                        if assignment[constraint.var2] == val:
                            return False
                elif constraint.var2 == var and constraint.var1 in assignment:
                    if constraint.relation == "!=":
                        if assignment[constraint.var1] == val:
                            return False
            return True
        
        def select_unassigned_variable(assignment: Dict[str, Any]) -> str | None:
            # MRV: alege variabila cu cele mai putine valori ramase
            unassigned = [v for v in csp.variables.keys() if v not in assignment]
            if not unassigned:
                return None
            
            best_var = None
            min_remaining = float('inf')
            
            for var in unassigned:
                remaining = sum(
                    1 for val in csp.variables[var].domain
                    if is_consistent(var, val, assignment)
                )
                if remaining < min_remaining:
                    min_remaining = remaining
                    best_var = var
            
            return best_var
        
        def backtrack(assignment: Dict[str, Any]) -> Dict[str, Any] | None:
            if len(assignment) == len(csp.variables):
                return assignment
            
            var = select_unassigned_variable(assignment)
            if var is None:
                return None
            
            for val in csp.variables[var].domain:
                if is_consistent(var, val, assignment):
                    assignment[var] = val
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                    del assignment[var]
            
            return None
        
        # Pornim de la asignarea partiala
        solution = dict(csp.partial_assignment)
        solution = backtrack(solution)
        
        return {
            "solution": solution,
        }


class MinimaxAlphaBetaSolver(Solver):
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        root: GameTreeNode = data["root"]
        leaf_count = [0]
        
        def minimax_alpha_beta(node: GameTreeNode, alpha: int, beta: int, is_max: bool) -> int:
            if not node.children:
                leaf_count[0] += 1
                return node.value if node.value is not None else 0
            
            if is_max:
                value = float('-inf')
                for child in node.children:
                    value = max(value, minimax_alpha_beta(child, alpha, beta, child.is_max_player))
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
                return value
            else:
                value = float('inf')
                for child in node.children:
                    value = min(value, minimax_alpha_beta(child, alpha, beta, child.is_max_player))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
                return value
        
        root_value = minimax_alpha_beta(root, float('-inf'), float('inf'), root.is_max_player)
        
        return {
            "root_value": root_value,
            "leaf_count": leaf_count[0],
        }


class SolverFactory:
    def __init__(self) -> None:
        self._mapping: Dict[QuestionType, type[Solver]] = {
            QuestionType.STRATEGY_SELECTION: StrategySolver,
            QuestionType.NASH_EQUILIBRIUM: NashSolver,
            QuestionType.CSP_COMPLETION: CspSolver,
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxAlphaBetaSolver,
        }
    
    def get_solver(self, q_type: QuestionType) -> Solver:
        cls = self._mapping[q_type]
        return cls()

