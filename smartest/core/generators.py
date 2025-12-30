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
    GridWorld,
    MDPState,
    Transition,
    RLParameters,
    Predicate,
    Action,
    PlanningProblem,
    PartialOrderPlan,
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


class ValueIterationGenerator(QuestionGenerator):
    """Generates Value Iteration problems with grid worlds."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        # Generate random grid world (standard 3x4 grid)
        rows, cols = 3, 4
        
        grid = self._create_grid_world(rows, cols, difficulty)
        
        # Solve for one iteration
        solver = self.solver_factory.get_solver(QuestionType.VALUE_ITERATION)
        solution = solver.solve({"grid": grid, "iterations": 1})
        
        # Choose random non-terminal, non-wall state to ask about
        available_states = [(r, c) for r, c in grid.states.keys() 
                           if not grid.states[(r, c)].is_terminal and (r, c) not in grid.walls]
        if not available_states:
            available_states = [(0, 0)]  # Fallback
        
        target_state = random.choice(available_states)
        
        # Format question
        text = self._format_grid_question(grid, solution, target_state)
        
        # Get specific state value after iteration
        correct_value = solution["values"][target_state]
        correct_policy = solution["policy"].get(target_state, "none")
        
        correct_answer = f"V({target_state}) = {correct_value:.2f}, politica: {correct_policy}"
        
        explanation = f"Folosind ecuatia Bellman: V(s) = max_a Σ P(s'|s,a)[R + γV(s')]. Complexitate: {solution['complexity']}"
        
        return Question(
            id=self._next_id(),
            title="Value Iteration pe Grid World",
            text=text,
            q_type=QuestionType.VALUE_ITERATION,
            topic="MDP / Value Iteration",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={"grid": grid, "solution": solution}
        )
    
    def _create_grid_world(self, rows: int, cols: int, difficulty: str) -> GridWorld:
        """Create a grid world MDP with random configuration."""
        # Randomize grid dimensions
        rows = random.choice([2, 3, 4])
        cols = random.choice([3, 4, 5])
        
        # Random living cost
        living_cost = random.choice([-0.04, -0.02, -0.1])
        
        # Initialize all states with living cost
        states = {}
        for r in range(rows):
            for c in range(cols):
                states[(r, c)] = MDPState(row=r, col=c, is_terminal=False, reward=living_cost)
        
        # Randomly place positive terminal state (goal)
        goal_positions = [(0, cols-1), (rows-1, cols-1), (0, 0)]
        goal_pos = random.choice(goal_positions)
        states[goal_pos].is_terminal = True
        states[goal_pos].reward = random.choice([1.0, 2.0, 5.0])
        
        # Randomly place negative terminal state (penalty) - avoid goal position
        penalty_positions = [(r, c) for r in range(rows) for c in range(cols) 
                            if (r, c) != goal_pos and abs(r - goal_pos[0]) + abs(c - goal_pos[1]) <= 2]
        if penalty_positions:
            penalty_pos = random.choice(penalty_positions)
            states[penalty_pos].is_terminal = True
            states[penalty_pos].reward = random.choice([-1.0, -2.0, -5.0])
        
        # Randomly place walls (0-2 walls)
        num_walls = random.randint(0, min(2, rows * cols // 4))
        walls = []
        available_for_walls = [(r, c) for r in range(rows) for c in range(cols) 
                              if not states[(r, c)].is_terminal]
        if available_for_walls:
            walls = random.sample(available_for_walls, min(num_walls, len(available_for_walls)))
        
        # Random transition probabilities
        intended_prob = random.choice([0.7, 0.8, 0.9])
        perp_prob = (1.0 - intended_prob) / 2
        transition_probs = {"intended": intended_prob, "perpendicular": perp_prob}
        
        # Random discount factor
        if difficulty == "easy":
            discount_factor = random.choice([0.8, 0.85])
        elif difficulty == "hard":
            discount_factor = random.choice([0.95, 0.99])
        else:  # medium
            discount_factor = random.choice([0.9, 0.92])
        
        return GridWorld(
            rows=rows,
            cols=cols,
            states=states,
            discount_factor=discount_factor,
            transition_probs=transition_probs,
            walls=walls
        )
    
    def _format_grid_question(self, grid: GridWorld, solution: Dict[str, Any], target_state: tuple[int, int]) -> str:
        """Format the grid world as a question."""
        lines = [
            f"Consideram un Grid World MDP de dimensiune {grid.rows}x{grid.cols}.",
            f"Factorul de discount γ = {grid.discount_factor}.",
            f"Probabilitatea de miscare in directia intentionata: {grid.transition_probs['intended']}.",
            f"Probabilitatea de miscare perpendiculara: {grid.transition_probs['perpendicular']}.",
            "",
            "Recompense:",
        ]
        
        for (r, c), state in sorted(grid.states.items()):
            if (r, c) in grid.walls:
                lines.append(f"  ({r},{c}): PERETE")
            elif state.is_terminal:
                lines.append(f"  ({r},{c}): {state.reward} (TERMINAL)")
            else:
                lines.append(f"  ({r},{c}): {state.reward}")
        
        lines.extend([
            "",
            "Valorile initiale: V(s) = 0 pentru toate starile non-terminale.",
            "",
            "Aplicati un pas al algoritmului Value Iteration si calculati:",
            f"1. Care este valoarea utilitatii pentru starea {target_state} dupa acest pas?",
            f"2. Care este politica recomandata in aceasta stare?"
        ])
        
        return "\n".join(lines)


class PolicyIterationGenerator(QuestionGenerator):
    """Generates Policy Iteration comparison questions."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        text_lines = [
            "Considerati algoritmii Value Iteration si Policy Iteration pentru rezolvarea problemelor MDP.",
            "",
            "Intrebari:",
            "1. Prin ce difera algoritmii Policy Iteration si Value Iteration?",
            "2. Care dintre ei converge de obicei mai rapid (in numar de iteratii)?",
            "3. Care este complexitatea unei iteratii pentru fiecare algoritm?"
        ]
        text = "\n".join(text_lines)
        
        correct_answer = (
            "1. Policy Iteration alterneaza intre Policy Evaluation (calcul V pentru politica curenta) "
            "si Policy Improvement (actualizare politica). Value Iteration face update direct al valorilor. "
            "2. Policy Iteration converge mai rapid in numar de iteratii. "
            "3. Value Iteration: O(|S|^2 * |A|) per iteratie. Policy Iteration: O(|S|^2 * |A| + |S|^3) per iteratie (mai costisitor)."
        )
        
        explanation = (
            "Policy Iteration este mai eficient in numar de iteratii dar fiecare iteratie este mai costisitoare "
            "deoarece trebuie sa evalueze complet politica curenta (rezolvare sistem de ecuatii). "
            "Value Iteration face update-uri simple dar necesita mai multe iteratii pentru convergenta."
        )
        
        return Question(
            id=self._next_id(),
            title="Comparatie Value Iteration vs Policy Iteration",
            text=text,
            q_type=QuestionType.POLICY_ITERATION,
            topic="MDP / Policy Iteration",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={}
        )


class QLearningGenerator(QuestionGenerator):
    """Generates Q-learning problems with transition sequences."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        # Generate transition sequence
        transitions = self._generate_transitions(difficulty)
        
        # Random parameters
        alpha = random.choice([0.1, 0.2, 0.3])
        gamma = random.choice([0.8, 0.9, 0.95])
        epsilon = random.choice([0.1, 0.2, 0.3])
        params = RLParameters(alpha=alpha, gamma=gamma, epsilon=epsilon)
        
        # Solve
        solver = self.solver_factory.get_solver(QuestionType.Q_LEARNING)
        solution = solver.solve({
            "transitions": transitions,
            "parameters": params,
            "initial_q": {}
        })
        
        text = self._format_qlearning_question(transitions, params)
        
        # Format Q-values for answer
        q_values = solution["q_values"]
        correct_lines = ["Valorile Q finale:"]
        for (state, action), value in sorted(q_values.items()):
            correct_lines.append(f"Q({state}, {action}) = {value:.2f}")
        
        # Add policy
        policy = solution["policy"]
        correct_lines.append("\nPolitica extrasă:")
        for state, action in sorted(policy.items()):
            correct_lines.append(f"π({state}) = {action}")
        
        correct_answer = "\n".join(correct_lines)
        
        explanation = (
            "Se aplica formula de update Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]. "
            f"Parametrii: α={params.alpha} (learning rate), γ={params.gamma} (discount factor), ε={params.epsilon} (exploration)."
        )
        
        return Question(
            id=self._next_id(),
            title="Q-learning cu secventa de tranzitii",
            text=text,
            q_type=QuestionType.Q_LEARNING,
            topic="Reinforcement Learning / Q-learning",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={"transitions": transitions, "params": params, "solution": solution}
        )
    
    def _generate_transitions(self, difficulty: str) -> List[Transition]:
        """Generate a random sequence of transitions."""
        # Random number of transitions based on difficulty
        if difficulty == "easy":
            num_transitions = random.randint(2, 3)
        elif difficulty == "hard":
            num_transitions = random.randint(5, 8)
        else:  # medium
            num_transitions = random.randint(3, 5)
        
        transitions = []
        actions = ["up", "down", "left", "right"]
        
        # Start from random initial state
        current_state = (random.randint(0, 2), random.randint(0, 2))
        
        for i in range(num_transitions):
            action = random.choice(actions)
            
            # Generate next state based on action (with some randomness)
            r, c = current_state
            if action == "up":
                next_state = (max(0, r - 1), c)
            elif action == "down":
                next_state = (min(2, r + 1), c)
            elif action == "left":
                next_state = (r, max(0, c - 1))
            else:  # right
                next_state = (r, min(3, c + 1))
            
            # Random reward (mostly negative, sometimes positive)
            if i == num_transitions - 1 and random.random() > 0.5:
                # Last transition might reach goal
                reward = random.choice([1.0, 2.0, 5.0])
            else:
                reward = random.choice([-0.04, -0.02, -0.1, 0.0])
            
            transitions.append(Transition(
                state=current_state,
                action=action,
                next_state=next_state,
                reward=reward
            ))
            
            current_state = next_state
        
        return transitions
    
    def _format_qlearning_question(self, transitions: List[Transition], params: RLParameters) -> str:
        """Format Q-learning question."""
        lines = [
            "Consideram algoritmul Q-learning cu urmatorii parametri:",
            f"- α (learning rate) = {params.alpha}",
            f"- γ (discount factor) = {params.gamma}",
            f"- ε (exploration rate) = {params.epsilon}",
            "",
            "Se observa urmatoarea secventa de tranzitii (s, a, s', r):",
        ]
        
        for i, t in enumerate(transitions, 1):
            lines.append(f"{i}. s={t.state}, a={t.action}, s'={t.next_state}, r={t.reward}")
        
        lines.extend([
            "",
            "Valorile Q initiale sunt toate 0.",
            "Actiunile posibile: up, down, left, right.",
            "",
            "Calculati:",
            "1. Valorile Q finale dupa procesarea tuturor tranzitiilor",
            "2. Politica extrasă (actiunea cu Q maxim pentru fiecare stare)"
        ])
        
        return "\n".join(lines)


class TDLearningGenerator(QuestionGenerator):
    """Generates TD-learning problems."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        # Random number of transitions
        if difficulty == "easy":
            num_transitions = random.randint(2, 3)
        else:
            num_transitions = random.randint(3, 5)
        
        # Generate random transitions
        transitions = []
        actions = ["up", "down", "left", "right"]
        current_state = (random.randint(0, 2), random.randint(0, 2))
        
        for i in range(num_transitions):
            action = random.choice(actions)
            r, c = current_state
            
            # Simple next state logic
            if action == "up":
                next_state = (max(0, r - 1), c)
            elif action == "down":
                next_state = (min(2, r + 1), c)
            elif action == "left":
                next_state = (r, max(0, c - 1))
            else:
                next_state = (r, min(3, c + 1))
            
            # Random reward
            if i == num_transitions - 1:
                reward = random.choice([1.0, 2.0])  # Goal at end
            else:
                reward = random.choice([-0.04, -0.02])
            
            transitions.append(Transition(
                state=current_state,
                action=action,
                next_state=next_state,
                reward=reward
            ))
            current_state = next_state
        
        # Random parameters
        alpha = random.choice([0.1, 0.2, 0.3])
        gamma = random.choice([0.8, 0.9, 0.95])
        params = RLParameters(alpha=alpha, gamma=gamma, epsilon=0.0)
        
        # Initialize V values to 0 for all encountered states
        all_states = set([t.state for t in transitions] + [t.next_state for t in transitions])
        initial_v = {state: 0.0 for state in all_states}
        
        solver = self.solver_factory.get_solver(QuestionType.TD_LEARNING)
        solution = solver.solve({
            "transitions": transitions,
            "parameters": params,
            "initial_v": initial_v
        })
        
        text_lines = [
            "Consideram algoritmul TD-learning (TD(0)) cu parametri:",
            f"- α (learning rate) = {params.alpha}",
            f"- γ (discount factor) = {params.gamma}",
            "",
            "Valori initiale: V(s) = 0 pentru toate starile.",
            "",
            "Se observa urmatoarea secventa de tranzitii:",
        ]
        
        for i, t in enumerate(transitions, 1):
            text_lines.append(f"{i}. s={t.state}, s'={t.next_state}, r={t.reward}")
        
        text_lines.extend([
            "",
            "Aplicati algoritmul TD(0) si calculati valorile finale V(s) pentru toate starile."
        ])
        
        text = "\n".join(text_lines)
        
        values = solution["values"]
        correct_lines = ["Valorile finale:"]
        for state, value in sorted(values.items()):
            correct_lines.append(f"V({state}) = {value:.3f}")
        
        correct_answer = "\n".join(correct_lines)
        
        explanation = (
            "Se aplica formula TD(0): V(s) ← V(s) + α[r + γV(s') - V(s)]. "
            "TD-error = r + γV(s') - V(s) masoara diferenta intre estimarea curenta si noua informatie."
        )
        
        return Question(
            id=self._next_id(),
            title="TD-learning (TD(0)) cu secventa de observatii",
            text=text,
            q_type=QuestionType.TD_LEARNING,
            topic="Reinforcement Learning / TD-learning",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={"transitions": transitions, "params": params, "solution": solution}
        )


class RLParametersGenerator(QuestionGenerator):
    """Generates questions about RL parameters (alpha, gamma, epsilon)."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        text_lines = [
            "In contextul metodei ε-greedy Q-learning, avem trei parametri importanti:",
            "- α (alpha) - learning rate",
            "- γ (gamma) - discount factor",
            "- ε (epsilon) - exploration rate",
            "",
            "Intrebari:",
            "1. Care este rolul fiecarui parametru?",
            "2. Ce se intampla daca setam α = 0?",
            "3. Ce se intampla daca setam γ = 0?",
            "4. Ce se intampla daca setam ε = 0?"
        ]
        text = "\n".join(text_lines)
        
        correct_answer = (
            "1. Roluri:\n"
            "   - α: controleaza cat de mult influenteaza noile informatii valorile Q (viteza de invatare)\n"
            "   - γ: determina importanta recompenselor viitoare (0=miopic, 1=previzionar)\n"
            "   - ε: probabilitatea de explorare (alegere actiune aleatorie vs cea mai buna)\n"
            "2. α=0: Nu se invata nimic nou, valorile Q raman neschimbate\n"
            "3. γ=0: Se considera doar recompensa imediata, ignorand viitorul\n"
            "4. ε=0: Nu se exploreaza, se alege mereu cea mai buna actiune cunoscuta (risc de a ramane in optim local)"
        )
        
        explanation = (
            "Parametrii trebuie balansati: α prea mare -> instabilitate, α prea mic -> invatare lenta. "
            "γ aproape de 1 -> ia in considerare viitorul indepartat. "
            "ε trebuie sa scada in timp (exploration -> exploitation)."
        )
        
        return Question(
            id=self._next_id(),
            title="Parametri RL: alpha, gamma, epsilon",
            text=text,
            q_type=QuestionType.RL_PARAMETERS,
            topic="Reinforcement Learning / Parametri",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={}
        )


class StripsActionGenerator(QuestionGenerator):
    """Generator for STRIPS action definition questions."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        """Generate a STRIPS action definition question."""
        # Define planning domains
        domains = ["shopping", "blocksworld", "container"]
        domain = random.choice(domains)
        
        if domain == "shopping":
            action, action_obj = self._generate_shopping_action()
        elif domain == "blocksworld":
            action, action_obj = self._generate_blocksworld_action()
        else:  # container
            action, action_obj = self._generate_container_action()
        
        # Generate question
        question_text = (
            f"Descrieți operația {action} în limbajul STRIPS pentru domeniul {domain}.\n\n"
            f"Specificați:\n"
            f"1. Precondițiile (ce trebuie să fie adevărat înainte)\n"
            f"2. Add-list (ce devine adevărat după execuție)\n"
            f"3. Delete-list (ce devine fals după execuție)"
        )
        
        # Get solution from solver
        solver = self.solver_factory.get_solver(QuestionType.STRIPS_ACTION_DEFINITION)
        solution = solver.solve({"action": action_obj})
        
        correct_answer = solution["formatted_action"]
        explanation = (
            f"Operația {action} este definită astfel în STRIPS:\n"
            f"{correct_answer}\n\n"
            f"Acest lucru înseamnă că pentru a executa această acțiune, "
            f"precondițiile trebuie să fie satisfăcute în starea curentă, "
            f"iar după execuție, predicatele din add-list devin adevărate "
            f"și cele din delete-list devin false."
        )
        
        return Question(
            id=self._next_id(),
            title=f"Definire Acțiune STRIPS: {action}",
            text=question_text,
            q_type=QuestionType.STRIPS_ACTION_DEFINITION,
            topic="Planning - STRIPS",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={"domain": domain, "action": str(action_obj)}
        )
    
    def _generate_shopping_action(self) -> tuple[str, Action]:
        """Generate a shopping domain action."""
        locations = ["home", "store1", "store2", "office"]
        items = ["milk", "bread", "eggs", "coffee"]
        
        action_type = random.choice(["go", "buy"])
        
        if action_type == "go":
            from_loc = random.choice(locations)
            to_loc = random.choice([l for l in locations if l != from_loc])
            
            action = Action(
                name="Go",
                parameters=[from_loc, to_loc],
                preconditions=[Predicate("At", ["agent", from_loc])],
                add_effects=[Predicate("At", ["agent", to_loc])],
                delete_effects=[Predicate("At", ["agent", from_loc])]
            )
            return f"Go({from_loc}, {to_loc})", action
        else:  # buy
            item = random.choice(items)
            store = random.choice(["store1", "store2"])
            
            action = Action(
                name="Buy",
                parameters=[item, store],
                preconditions=[
                    Predicate("At", ["agent", store]),
                    Predicate("Sells", [store, item])
                ],
                add_effects=[Predicate("Have", [item])],
                delete_effects=[]
            )
            return f"Buy({item}, {store})", action
    
    def _generate_blocksworld_action(self) -> tuple[str, Action]:
        """Generate a blocksworld domain action."""
        blocks = ["A", "B", "C", "D"]
        
        action_type = random.choice(["fromtable", "totable", "stack", "unstack"])
        
        if action_type == "fromtable":
            block = random.choice(blocks)
            action = Action(
                name="FromTable",
                parameters=[block],
                preconditions=[
                    Predicate("OnTable", [block]),
                    Predicate("Clear", [block]),
                    Predicate("HandEmpty", [])
                ],
                add_effects=[Predicate("Holding", [block])],
                delete_effects=[
                    Predicate("OnTable", [block]),
                    Predicate("Clear", [block]),
                    Predicate("HandEmpty", [])
                ]
            )
            return f"FromTable({block})", action
        elif action_type == "totable":
            block = random.choice(blocks)
            action = Action(
                name="ToTable",
                parameters=[block],
                preconditions=[Predicate("Holding", [block])],
                add_effects=[
                    Predicate("OnTable", [block]),
                    Predicate("Clear", [block]),
                    Predicate("HandEmpty", [])
                ],
                delete_effects=[Predicate("Holding", [block])]
            )
            return f"ToTable({block})", action
        elif action_type == "stack":
            block1 = random.choice(blocks)
            block2 = random.choice([b for b in blocks if b != block1])
            action = Action(
                name="Stack",
                parameters=[block1, block2],
                preconditions=[
                    Predicate("Holding", [block1]),
                    Predicate("Clear", [block2])
                ],
                add_effects=[
                    Predicate("On", [block1, block2]),
                    Predicate("Clear", [block1]),
                    Predicate("HandEmpty", [])
                ],
                delete_effects=[
                    Predicate("Holding", [block1]),
                    Predicate("Clear", [block2])
                ]
            )
            return f"Stack({block1}, {block2})", action
        else:  # unstack
            block1 = random.choice(blocks)
            block2 = random.choice([b for b in blocks if b != block1])
            action = Action(
                name="Unstack",
                parameters=[block1, block2],
                preconditions=[
                    Predicate("On", [block1, block2]),
                    Predicate("Clear", [block1]),
                    Predicate("HandEmpty", [])
                ],
                add_effects=[
                    Predicate("Holding", [block1]),
                    Predicate("Clear", [block2])
                ],
                delete_effects=[
                    Predicate("On", [block1, block2]),
                    Predicate("Clear", [block1]),
                    Predicate("HandEmpty", [])
                ]
            )
            return f"Unstack({block1}, {block2})", action
    
    def _generate_container_action(self) -> tuple[str, Action]:
        """Generate a container domain action."""
        containers = ["box1", "box2", "bottle"]
        items = ["ball", "pen", "key"]
        
        action_type = random.choice(["placecap", "removecap", "insert"])
        
        if action_type == "placecap":
            container = random.choice(containers)
            action = Action(
                name="PlaceCap",
                parameters=[container],
                preconditions=[
                    Predicate("Open", [container]),
                    Predicate("HasCap", [container], positive=False)
                ],
                add_effects=[Predicate("HasCap", [container])],
                delete_effects=[Predicate("Open", [container])]
            )
            return f"PlaceCap({container})", action
        elif action_type == "removecap":
            container = random.choice(containers)
            action = Action(
                name="RemoveCap",
                parameters=[container],
                preconditions=[Predicate("HasCap", [container])],
                add_effects=[
                    Predicate("Open", [container]),
                    Predicate("HasCap", [container], positive=False)
                ],
                delete_effects=[Predicate("HasCap", [container])]
            )
            return f"RemoveCap({container})", action
        else:  # insert
            item = random.choice(items)
            container = random.choice(containers)
            action = Action(
                name="Insert",
                parameters=[item, container],
                preconditions=[
                    Predicate("Open", [container]),
                    Predicate("Holding", [item])
                ],
                add_effects=[
                    Predicate("Inside", [item, container]),
                    Predicate("HandEmpty", [])
                ],
                delete_effects=[Predicate("Holding", [item])]
            )
            return f"Insert({item}, {container})", action


class AdlActionGenerator(QuestionGenerator):
    """Generator for ADL action definition questions (with conditional effects)."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        """Generate an ADL action definition question."""
        # Define domains with conditional effects
        domains = ["logistics", "robot_navigation"]
        domain = random.choice(domains)
        
        if domain == "logistics":
            action, action_obj = self._generate_logistics_action()
        else:  # robot_navigation
            action, action_obj = self._generate_robot_action()
        
        # Generate question
        question_text = (
            f"Descrieți operația {action} în limbajul ADL pentru domeniul {domain}.\n\n"
            f"Specificați:\n"
            f"1. Precondițiile\n"
            f"2. Add-list\n"
            f"3. Delete-list\n"
            f"4. Efecte condiționate (CÂND ... ATUNCI ...)"
        )
        
        # Get solution from solver
        solver = self.solver_factory.get_solver(QuestionType.ADL_ACTION_DEFINITION)
        solution = solver.solve({"action": action_obj})
        
        correct_answer = solution["formatted_action"]
        explanation = (
            f"Operația {action} în ADL include efecte condiționate:\n"
            f"{correct_answer}\n\n"
            f"Efectele condiționate se activează doar dacă condiția este satisfăcută."
        )
        
        return Question(
            id=self._next_id(),
            title=f"Definire Acțiune ADL: {action}",
            text=question_text,
            q_type=QuestionType.ADL_ACTION_DEFINITION,
            topic="Planning - ADL",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={"domain": domain, "action": str(action_obj)}
        )
    
    def _generate_logistics_action(self) -> tuple[str, Action]:
        """Generate a logistics domain action with conditional effects."""
        packages = ["package1", "package2"]
        vehicles = ["truck1", "van1"]
        
        action_type = random.choice(["load", "unload"])
        
        if action_type == "load":
            package = random.choice(packages)
            vehicle = random.choice(vehicles)
            
            action = Action(
                name="Load",
                parameters=[package, vehicle],
                preconditions=[
                    Predicate("At", [package, "loc"]),
                    Predicate("At", [vehicle, "loc"])
                ],
                add_effects=[Predicate("In", [package, vehicle])],
                delete_effects=[Predicate("At", [package, "loc"])],
                conditional_effects=[
                    ([Predicate("Heavy", [package])], [Predicate("Slow", [vehicle])])
                ]
            )
            return f"Load({package}, {vehicle})", action
        else:  # unload
            package = random.choice(packages)
            vehicle = random.choice(vehicles)
            
            action = Action(
                name="Unload",
                parameters=[package, vehicle],
                preconditions=[Predicate("In", [package, vehicle])],
                add_effects=[Predicate("At", [package, "loc"])],
                delete_effects=[Predicate("In", [package, vehicle])],
                conditional_effects=[
                    ([Predicate("Slow", [vehicle])], [Predicate("Slow", [vehicle], positive=False)])
                ]
            )
            return f"Unload({package}, {vehicle})", action
    
    def _generate_robot_action(self) -> tuple[str, Action]:
        """Generate a robot navigation action with conditional effects."""
        locations = ["room1", "room2", "corridor"]
        
        from_loc = random.choice(locations)
        to_loc = random.choice([l for l in locations if l != from_loc])
        
        action = Action(
            name="Move",
            parameters=[from_loc, to_loc],
            preconditions=[Predicate("At", ["robot", from_loc])],
            add_effects=[Predicate("At", ["robot", to_loc])],
            delete_effects=[Predicate("At", ["robot", from_loc])],
            conditional_effects=[
                ([Predicate("Dark", [to_loc])], [Predicate("NeedLight", ["robot"])])
            ]
        )
        return f"Move({from_loc}, {to_loc})", action


class PartialOrderPlanGenerator(QuestionGenerator):
    """Generator for partial order planning questions."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        """Generate a POP question."""
        domains = ["shopping", "blocksworld"]
        domain = random.choice(domains)
        
        if domain == "shopping":
            problem = self._generate_shopping_problem()
        else:
            problem = self._generate_blocksworld_problem()
        
        # Format initial state and goals
        initial_str = ", ".join(str(p) for p in problem.initial_state)
        goals_str = ", ".join(str(p) for p in problem.goal_state)
        
        question_text = (
            f"Construiți un plan incomplet folosind algoritmul de planificare cu ordine parțială (POP) "
            f"pentru următoarea problemă din domeniul {domain}:\n\n"
            f"Stare inițială: {initial_str}\n\n"
            f"Obiective: {goals_str}\n\n"
            f"Specificați:\n"
            f"1. Acțiunile planului (minim 3 acțiuni)\n"
            f"2. Ordinea parțială (relațiile de precedență)\n"
            f"3. Linkurile cauzale (cine produce ce pentru cine)"
        )
        
        # Get solution from solver
        solver = self.solver_factory.get_solver(QuestionType.PARTIAL_ORDER_PLAN)
        solution = solver.solve({"problem": problem})
        
        correct_answer = solution["formatted_plan"]
        explanation = (
            f"Planul parțial pentru această problemă:\n"
            f"{correct_answer}\n\n"
            f"Algoritmul POP construiește planuri incrementale, adăugând acțiuni "
            f"pentru a satisface obiectivele și rezolvând amenințările."
        )
        
        return Question(
            id=self._next_id(),
            title=f"Partial Order Planning - {domain}",
            text=question_text,
            q_type=QuestionType.PARTIAL_ORDER_PLAN,
            topic="Planning - POP",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={"domain": domain}
        )
    
    def _generate_shopping_problem(self) -> PlanningProblem:
        """Generate a shopping domain planning problem."""
        items = random.sample(["milk", "bread", "eggs", "coffee"], k=random.randint(1, 2))
        stores = ["store1", "store2"]
        store = random.choice(stores)
        
        initial_state = [
            Predicate("At", ["agent", "home"]),
        ]
        for item in items:
            initial_state.append(Predicate("Sells", [store, item]))
        
        goal_state = [Predicate("Have", [item]) for item in items]
        if random.choice([True, False]):
            goal_state.append(Predicate("At", ["agent", "home"]))
        
        actions = [
            Action(
                name="Go",
                parameters=["home", store],
                preconditions=[Predicate("At", ["agent", "home"])],
                add_effects=[Predicate("At", ["agent", store])],
                delete_effects=[Predicate("At", ["agent", "home"])]
            ),
            Action(
                name="Go",
                parameters=[store, "home"],
                preconditions=[Predicate("At", ["agent", store])],
                add_effects=[Predicate("At", ["agent", "home"])],
                delete_effects=[Predicate("At", ["agent", store])]
            ),
        ]
        
        for item in items:
            actions.append(
                Action(
                    name="Buy",
                    parameters=[item, store],
                    preconditions=[
                        Predicate("At", ["agent", store]),
                        Predicate("Sells", [store, item])
                    ],
                    add_effects=[Predicate("Have", [item])],
                    delete_effects=[]
                )
            )
        
        return PlanningProblem(
            domain_name="shopping",
            objects=["agent", "home", store] + items,
            initial_state=initial_state,
            goal_state=goal_state,
            actions=actions
        )
    
    def _generate_blocksworld_problem(self) -> PlanningProblem:
        """Generate a blocksworld planning problem."""
        blocks = ["A", "B", "C"]
        
        # Random initial configuration
        initial_state = [Predicate("HandEmpty", [])]
        if random.choice([True, False]):
            initial_state.extend([
                Predicate("OnTable", ["A"]),
                Predicate("On", ["B", "A"]),
                Predicate("OnTable", ["C"]),
                Predicate("Clear", ["B"]),
                Predicate("Clear", ["C"])
            ])
        else:
            initial_state.extend([
                Predicate("OnTable", ["A"]),
                Predicate("OnTable", ["B"]),
                Predicate("OnTable", ["C"]),
                Predicate("Clear", ["A"]),
                Predicate("Clear", ["B"]),
                Predicate("Clear", ["C"])
            ])
        
        # Random goal
        goal_state = [
            Predicate("On", ["A", "B"])
        ]
        
        # Define actions
        actions = []
        for block in blocks:
            actions.append(
                Action(
                    name="FromTable",
                    parameters=[block],
                    preconditions=[
                        Predicate("OnTable", [block]),
                        Predicate("Clear", [block]),
                        Predicate("HandEmpty", [])
                    ],
                    add_effects=[Predicate("Holding", [block])],
                    delete_effects=[
                        Predicate("OnTable", [block]),
                        Predicate("Clear", [block]),
                        Predicate("HandEmpty", [])
                    ]
                )
            )
            actions.append(
                Action(
                    name="ToTable",
                    parameters=[block],
                    preconditions=[Predicate("Holding", [block])],
                    add_effects=[
                        Predicate("OnTable", [block]),
                        Predicate("Clear", [block]),
                        Predicate("HandEmpty", [])
                    ],
                    delete_effects=[Predicate("Holding", [block])]
                )
            )
        
        for b1 in blocks:
            for b2 in blocks:
                if b1 != b2:
                    actions.append(
                        Action(
                            name="Stack",
                            parameters=[b1, b2],
                            preconditions=[
                                Predicate("Holding", [b1]),
                                Predicate("Clear", [b2])
                            ],
                            add_effects=[
                                Predicate("On", [b1, b2]),
                                Predicate("Clear", [b1]),
                                Predicate("HandEmpty", [])
                            ],
                            delete_effects=[
                                Predicate("Holding", [b1]),
                                Predicate("Clear", [b2])
                            ]
                        )
                    )
        
        return PlanningProblem(
            domain_name="blocksworld",
            objects=blocks,
            initial_state=initial_state,
            goal_state=goal_state,
            actions=actions
        )


class PlanValidationGenerator(QuestionGenerator):
    """Generator for plan validation questions."""
    
    def generate(self, difficulty: str = "medium") -> Question:
        """Generate a plan validation question."""
        # Create a simple problem
        problem = self._create_simple_problem()
        
        # Create a plan (correct or with errors)
        has_errors = random.choice([True, False])
        plan = self._create_plan(problem, has_errors)
        
        # Format plan
        plan_str = "\n".join(f"{i+1}. {action}" for i, action in enumerate(plan))
        initial_str = ", ".join(str(p) for p in problem.initial_state)
        goals_str = ", ".join(str(p) for p in problem.goal_state)
        
        question_text = (
            f"Verificați corectitudinea următorului plan:\n\n"
            f"Stare inițială: {initial_str}\n"
            f"Obiective: {goals_str}\n\n"
            f"Plan:\n{plan_str}\n\n"
            f"Planul este corect? Dacă nu, identificați erorile."
        )
        
        # Validate plan
        solver = self.solver_factory.get_solver(QuestionType.PLAN_VALIDATION)
        solution = solver.solve({"problem": problem, "plan": plan})
        
        if solution["valid"]:
            correct_answer = "Da, planul este corect. Toate precondițiile sunt satisfăcute și obiectivele sunt atinse."
        else:
            errors_str = "\n".join(f"- {err}" for err in solution["errors"])
            correct_answer = f"Nu, planul are următoarele erori:\n{errors_str}"
        
        explanation = (
            f"Validarea planului:\n"
            f"{correct_answer}\n\n"
            f"Pentru a valida un plan, verificăm:\n"
            f"1. Precondițiile fiecărei acțiuni sunt satisfăcute\n"
            f"2. Obiectivele sunt atinse la sfârșitul planului"
        )
        
        return Question(
            id=self._next_id(),
            title="Validare Plan",
            text=question_text,
            q_type=QuestionType.PLAN_VALIDATION,
            topic="Planning - Validation",
            difficulty=difficulty,
            correct_answer=correct_answer,
            explanation=explanation,
            meta={"has_errors": has_errors}
        )
    
    def _create_simple_problem(self) -> PlanningProblem:
        """Create a simple shopping problem for validation."""
        initial_state = [
            Predicate("At", ["agent", "home"]),
            Predicate("Sells", ["store1", "milk"])
        ]
        
        goal_state = [
            Predicate("Have", ["milk"])
        ]
        
        actions = [
            Action(
                name="Go",
                parameters=["home", "store1"],
                preconditions=[Predicate("At", ["agent", "home"])],
                add_effects=[Predicate("At", ["agent", "store1"])],
                delete_effects=[Predicate("At", ["agent", "home"])]
            ),
            Action(
                name="Buy",
                parameters=["milk", "store1"],
                preconditions=[
                    Predicate("At", ["agent", "store1"]),
                    Predicate("Sells", ["store1", "milk"])
                ],
                add_effects=[Predicate("Have", ["milk"])],
                delete_effects=[]
            )
        ]
        
        return PlanningProblem(
            domain_name="shopping",
            objects=["agent", "home", "store1", "milk"],
            initial_state=initial_state,
            goal_state=goal_state,
            actions=actions
        )
    
    def _create_plan(self, problem: PlanningProblem, with_errors: bool) -> List[Action]:
        """Create a plan (correct or with errors)."""
        if not with_errors:
            # Correct plan
            return [
                problem.actions[0],  # Go(home, store1)
                problem.actions[1],  # Buy(milk, store1)
            ]
        else:
            # Plan with error: trying to buy before going to store
            return [
                problem.actions[1],  # Buy(milk, store1) - ERROR: not at store!
                problem.actions[0],  # Go(home, store1)
            ]


class QuestionFactory:
    def __init__(self) -> None:
        self._generators: Dict[QuestionType, QuestionGenerator] = {
            QuestionType.STRATEGY_SELECTION: StrategyQuestionGenerator(),
            QuestionType.NASH_EQUILIBRIUM: NashQuestionGenerator(),
            QuestionType.CSP_COMPLETION: CspQuestionGenerator(),
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxQuestionGenerator(),
            QuestionType.VALUE_ITERATION: ValueIterationGenerator(),
            QuestionType.POLICY_ITERATION: PolicyIterationGenerator(),
            QuestionType.Q_LEARNING: QLearningGenerator(),
            QuestionType.TD_LEARNING: TDLearningGenerator(),
            QuestionType.RL_PARAMETERS: RLParametersGenerator(),
            QuestionType.STRIPS_ACTION_DEFINITION: StripsActionGenerator(),
            QuestionType.ADL_ACTION_DEFINITION: AdlActionGenerator(),
            QuestionType.PARTIAL_ORDER_PLAN: PartialOrderPlanGenerator(),
            QuestionType.PLAN_VALIDATION: PlanValidationGenerator(),
        }

    def get_generator(self, q_type: QuestionType) -> QuestionGenerator:
        return self._generators[q_type]
