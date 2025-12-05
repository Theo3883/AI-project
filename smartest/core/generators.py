from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from .models import (
    Question,
    QuestionType,
    NashGame,
    CspInstance,
    CspVariable,
    CspConstraint,
    GameTreeNode,
)
from .solvers import SolverFactory


class QuestionGenerator(ABC):
    """Template Method: defineste interfata pentru generatori de intrebari."""

    _id_counter = itertools.count(1)

    def __init__(self) -> None:
        self.solver_factory = SolverFactory()

    @abstractmethod
    def generate(self, difficulty: str = "medium") -> Question:
        raise NotImplementedError

    def _next_id(self) -> int:
        return next(self._id_counter)


class StrategyQuestionGenerator(QuestionGenerator):
    """Genereaza o problema concreta + intrebare de tip "alege strategia potrivita"."""

    def _nqueens_instance(self) -> Dict[str, Any]:
        n = random.randint(4, 8)
        desc = (
            f"Consideram problema N-Queens pe o tabla de sah de dimensiune {n}x{n}. "
            f"Trebuie plasate {n} regine astfel incat niciuna sa nu atace alta regina "
            "(nici pe linie, nici pe coloana, nici pe diagonale)."
        )
        return {"name": "n-queens", "n": n, "description": desc}

    def _hanoi_instance(self) -> Dict[str, Any]:
        pegs = random.choice([3, 4])
        discs = random.randint(3, 8)
        desc = (
            f"Consideram problema Turnurilor din Hanoi generalizat cu {pegs} tije si {discs} discuri. "
            "Initial toate discurile sunt pe tija 1 (in ordine de la cel mai mare jos la cel mai mic sus) "
            "si trebuie mutate pe tija finala, respectand regulile (nu se poate pune un disc mai mare peste unul mai mic)."
        )
        return {"name": "hanoi", "pegs": pegs, "discs": discs, "description": desc}

    def _graph_coloring_instance(self) -> Dict[str, Any]:
        num_nodes = random.randint(4, 6)
        nodes = [chr(ord("A") + i) for i in range(num_nodes)]
        possible_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                possible_edges.append((nodes[i], nodes[j]))
        random.shuffle(possible_edges)
        num_edges = random.randint(num_nodes, min(len(possible_edges), num_nodes + 2))
        edges = sorted(possible_edges[:num_edges])
        colors = random.choice([3, 4])
        edges_str = ", ".join(f"({u},{v})" for u, v in edges)
        desc = (
            f"Avem un graf neorientat cu nodurile: {', '.join(nodes)}. "
            f"Muchiile grafului sunt: {edges_str}. "
            f"Se cere sa se coloreze graful folosind cel mult {colors} culori astfel incat "
            "doua noduri adiacente sa nu aiba aceeasi culoare."
        )
        return {
            "name": "graph-coloring",
            "nodes": nodes,
            "edges": edges,
            "colors": colors,
            "description": desc,
        }

    def _knights_tour_instance(self) -> Dict[str, Any]:
        size = random.choice([5, 6, 8])
        start_row = random.randint(1, size)
        start_col = random.randint(1, size)
        desc = (
            f"Consideram problema Knight's Tour pe o tabla {size}x{size}. "
            f"Calul porneste din patratul cu coordonatele ({start_row}, {start_col}) "
            "folosind mutarile standard de cal in sah. "
            "Se cere gasirea unui tur in care calul viziteaza fiecare patrat o singura data."
        )
        return {
            "name": "knights-tour",
            "size": size,
            "start": (start_row, start_col),
            "description": desc,
        }

    def generate(self, difficulty: str = "medium") -> Question:
        # alegem aleator una dintre cele 4 probleme si generam o instanta
        choice = random.choice(["n-queens", "hanoi", "graph-coloring", "knights-tour"])
        if choice == "n-queens":
            instance = self._nqueens_instance()
        elif choice == "hanoi":
            instance = self._hanoi_instance()
        elif choice == "graph-coloring":
            instance = self._graph_coloring_instance()
        else:
            instance = self._knights_tour_instance()

        problem_name = instance["name"]

        solver = self.solver_factory.get_solver(QuestionType.STRATEGY_SELECTION)
        solution = solver.solve({"problem": problem_name})
        correct = solution["recommended_strategy"]
        explanation = solution["explanation"]

        text_lines = [
            "Considerati urmatoarea problema de la cursul de Inteligenta Artificiala:",
            "",
            instance["description"],
            "",
            "Dintre strategiile studiate la curs (BFS, DFS, Backtracking, A*, cautare locala, etc.),",
            "care este cea mai potrivita strategie de rezolvare pentru aceasta instanta (si in general pentru acest tip de problema) si de ce?",
        ]
        text = "\n".join(text_lines)

        return Question(
            id=self._next_id(),
            title=f"Strategia potrivita pentru problema {problem_name}",
            text=text,
            q_type=QuestionType.STRATEGY_SELECTION,
            topic="Probleme clasice / cautare",
            difficulty=difficulty,
            correct_answer=correct,
            explanation=explanation,
            meta={"problem": problem_name, "instance": instance},
        )


class NashQuestionGenerator(QuestionGenerator):
    def generate(self, difficulty: str = "medium") -> Question:
        row_strategies = ["Sus", "Jos"]
        col_strategies = ["Stanga", "Dreapta"]

        payoffs: List[List[tuple[int, int]]] = []
        for _ in range(2):
            row = []
            for _ in range(2):
                row.append((random.randint(0, 5), random.randint(0, 5)))
            payoffs.append(row)

        game = NashGame(row_strategies=row_strategies, col_strategies=col_strategies, payoffs=payoffs)

        solver = self.solver_factory.get_solver(QuestionType.NASH_EQUILIBRIUM)
        solution = solver.solve({"game": game})
        equilibria = solution["equilibria"]
        if equilibria:
            i, j = equilibria[0]
            correct = f"( {row_strategies[i]}, {col_strategies[j]} )"
        else:
            correct = "Nu exista echilibru Nash pur."

        explanation = solution["text"]

        text_lines = [
            "Avem urmatorul joc in forma normala (payoff-urile sunt (jucator1, jucator2)):",
            "",
            "\t\tStanga\t\tDreapta",
            f"Sus\t{payoffs[0][0]}\t{payoffs[0][1]}",
            f"Jos\t{payoffs[1][0]}\t{payoffs[1][1]}",
            "",
            "Exista un echilibru Nash pur? Daca da, care este acesta (strategia fiecarui jucator)?",
        ]
        text = "\n".join(text_lines)

        return Question(
            id=self._next_id(),
            title="Echilibru Nash pur intr-un joc 2x2",
            text=text,
            q_type=QuestionType.NASH_EQUILIBRIUM,
            topic="Teoria jocurilor",
            difficulty=difficulty,
            correct_answer=correct,
            explanation=explanation,
            meta={"game": game},
        )


class CspQuestionGenerator(QuestionGenerator):
    def generate(self, difficulty: str = "medium") -> Question:
        colors = ["rosu", "verde", "albastru"]
        variables = {
            "A": CspVariable("A", list(colors)),
            "B": CspVariable("B", list(colors)),
            "C": CspVariable("C", list(colors)),
            "D": CspVariable("D", list(colors)),
        }
        constraints = [
            CspConstraint("A", "B"),
            CspConstraint("A", "C"),
            CspConstraint("B", "C"),
            CspConstraint("C", "D"),
        ]
        partial_assignment = {"A": random.choice(colors)}

        instance = CspInstance(
            variables=variables,
            constraints=constraints,
            partial_assignment=partial_assignment,
        )

        solver = self.solver_factory.get_solver(QuestionType.CSP_COMPLETION)
        solution = solver.solve({"csp": instance})
        assign = solution["solution"] or {}
        correct_items = [f"{k}={v}" for k, v in sorted(assign.items())]
        correct = ", ".join(correct_items)

        explanation = (
            "Se aplica backtracking cu euristica MRV si forward checking. "
            "Pornind de la asignarea partiala A={a}, se aleg pe rand variabilele cu "
            "cele mai putine valori posibile ramase si se verifica consistenta "
            "fata de toate constrangerile binare."
        ).format(a=partial_assignment["A"])

        text_lines = [
            "Consideram urmatorul CSP de colorare a unui graf cu 3 culori: rosu, verde, albastru.",
            "Variabile: A, B, C, D.",
            "Constrangeri: A!=B, A!=C, B!=C, C!=D.",
            f"Asignare partiala: A={partial_assignment['A']}.",
            "",
            "Daca rezolvam problema folosind algoritmul Backtracking cu MRV + Forward Checking,",
            "care vor fi asignarile pentru variabilele ramase B, C si D intr-o solutie completa?",
        ]
        text = "\n".join(text_lines)

        return Question(
            id=self._next_id(),
            title="CSP cu Backtracking + MRV + Forward Checking",
            text=text,
            q_type=QuestionType.CSP_COMPLETION,
            topic="CSP / Backtracking",
            difficulty=difficulty,
            correct_answer=correct,
            explanation=explanation,
            meta={"csp": instance},
        )


class MinimaxQuestionGenerator(QuestionGenerator):
    def generate(self, difficulty: str = "medium") -> Question:
        leaf_values = [random.randint(-5, 10) for _ in range(8)]

        def make_leaf(v: int) -> GameTreeNode:
            return GameTreeNode(value=v, children=[], is_max_player=False)

        leaves = [make_leaf(v) for v in leaf_values]

        level2: List[GameTreeNode] = []
        for i in range(0, len(leaves), 2):
            node = GameTreeNode(
                value=None, children=[leaves[i], leaves[i + 1]], is_max_player=False
            )
            level2.append(node)

        level1: List[GameTreeNode] = []
        for i in range(0, len(level2), 2):
            node = GameTreeNode(
                value=None, children=[level2[i], level2[i + 1]], is_max_player=True
            )
            level1.append(node)

        root = GameTreeNode(value=None, children=level1, is_max_player=True)

        from .solvers import MinimaxAlphaBetaSolver

        solver = MinimaxAlphaBetaSolver()
        res = solver.solve({"root": root})
        root_value = res["root_value"]
        leaf_count = res["leaf_count"]
        correct = f"Valoarea radacinii = {root_value}, frunze evaluate = {leaf_count}"

        explanation = (
            "Se aplica MinMax cu Alpha-Beta in mod depth-first, de la stanga la dreapta. "
            "Alpha si Beta sunt actualizate la fiecare nod, iar unele frunze sunt taiate "
            "cand se constata ca nu pot influenta rezultatul final."
        )

        text_lines = [
            "Avem un arbore de joc cu adancime 3 (radacina MAX, apoi MIN, apoi MAX si frunze).",
            "Valorile din frunze (de la stanga la dreapta) sunt:",
            str(leaf_values),
            "",
            "Daca aplicam algoritmul MinMax cu optimizarea Alpha-Beta,",
            "care va fi valoarea din radacina si cate noduri frunza vor fi evaluate (nevizate de pruning)?",
        ]
        text = "\n".join(text_lines)

        return Question(
            id=self._next_id(),
            title="MinMax cu Alpha-Beta pe un arbore mic",
            text=text,
            q_type=QuestionType.MINIMAX_ALPHA_BETA,
            topic="Jocuri / MinMax",
            difficulty=difficulty,
            correct_answer=correct,
            explanation=explanation,
            meta={"root": root},
        )


class QuestionFactory:
    def __init__(self) -> None:
        self._generators: Dict[QuestionType, QuestionGenerator] = {
            QuestionType.STRATEGY_SELECTION: StrategyQuestionGenerator(),
            QuestionType.NASH_EQUILIBRIUM: NashQuestionGenerator(),
            QuestionType.CSP_COMPLETION: CspQuestionGenerator(),
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxQuestionGenerator(),
        }

    def get_generator(self, q_type: QuestionType) -> QuestionGenerator:
        return self._generators[q_type]
