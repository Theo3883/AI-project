from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from .models import (
    QuestionType, 
    NashGame, 
    CspInstance, 
    GameTreeNode,
    GridWorld,
    MDPState,
    Transition,
    RLParameters,
    QTable,
    Predicate,
    Action,
    PlanningProblem,
    PartialOrderPlan
)


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


class ValueIterationSolver(Solver):
    """
    Implements Bellman equation: V(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
    Complexity: O(|S|² |A|) per iteration
    """
    
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        grid: GridWorld = data["grid"]
        iterations: int = data.get("iterations", 1)
        
        # Initialize values to 0
        V = {(r, c): 0.0 for r in range(grid.rows) for c in range(grid.cols)}
        
        # Set terminal states
        for (r, c), state in grid.states.items():
            if state.is_terminal:
                V[(r, c)] = state.reward
        
        # Track which states were updated
        updated_states = []
        
        for _ in range(iterations):
            V_new = V.copy()
            for (r, c) in V:
                if (r, c) in grid.walls or grid.states[(r, c)].is_terminal:
                    continue
                
                # Bellman update
                max_value = float('-inf')
                for action in ["up", "down", "left", "right"]:
                    value = 0.0
                    for (next_state, prob) in grid.get_neighbors((r, c), action):
                        reward = grid.states[next_state].reward
                        value += prob * (reward + grid.discount_factor * V[next_state])
                    max_value = max(max_value, value)
                
                V_new[(r, c)] = max_value
                if V_new[(r, c)] != V[(r, c)]:
                    updated_states.append((r, c))
            
            V = V_new
        
        # Extract policy
        policy = self._extract_policy(grid, V)
        
        return {
            "values": V,
            "policy": policy,
            "updated_states": updated_states,
            "complexity": f"O(|S|^2 * |A|) = O({grid.rows * grid.cols}^2 * 4) per iteration"
        }
    
    def _extract_policy(self, grid: GridWorld, V: Dict) -> Dict[tuple[int, int], str]:
        """Extract optimal policy from value function."""
        policy = {}
        for (r, c) in V:
            if (r, c) in grid.walls or grid.states[(r, c)].is_terminal:
                continue
            
            best_action = None
            best_value = float('-inf')
            
            for action in ["up", "down", "left", "right"]:
                value = 0.0
                for (next_state, prob) in grid.get_neighbors((r, c), action):
                    reward = grid.states[next_state].reward
                    value += prob * (reward + grid.discount_factor * V[next_state])
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            policy[(r, c)] = best_action
        
        return policy


class PolicyIterationSolver(Solver):
    """
    Policy Iteration: alternates between policy evaluation and policy improvement.
    Converges faster than Value Iteration (fewer iterations, but more expensive per iteration).
    """
    
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        grid: GridWorld = data["grid"]
        
        # Initialize random policy
        policy = {}
        for r in range(grid.rows):
            for c in range(grid.cols):
                if (r, c) not in grid.walls and not grid.states[(r, c)].is_terminal:
                    policy[(r, c)] = "up"
        
        iteration_count = 0
        while True:
            # Policy Evaluation
            V = self._policy_evaluation(grid, policy)
            
            # Policy Improvement
            policy_stable = True
            new_policy = {}
            
            for (r, c) in policy:
                if (r, c) in grid.walls or grid.states[(r, c)].is_terminal:
                    continue
                
                old_action = policy[(r, c)]
                
                # Find best action
                best_action = None
                best_value = float('-inf')
                
                for action in ["up", "down", "left", "right"]:
                    value = self._compute_action_value(grid, V, (r, c), action)
                    if value > best_value:
                        best_value = value
                        best_action = action
                
                new_policy[(r, c)] = best_action
                
                if old_action != best_action:
                    policy_stable = False
            
            iteration_count += 1
            policy = new_policy
            
            if policy_stable:
                break
        
        return {
            "values": V,
            "policy": policy,
            "iterations": iteration_count,
            "difference": "Policy Iteration converge mai rapid (mai putine iteratii) dar fiecare iteratie este mai costisitoare decat Value Iteration"
        }
    
    def _policy_evaluation(self, grid: GridWorld, policy: Dict[tuple[int, int], str]) -> Dict[tuple[int, int], float]:
        """Evaluate policy to compute state values."""
        V = {(r, c): 0.0 for r in range(grid.rows) for c in range(grid.cols)}
        
        # Set terminal states
        for (r, c), state in grid.states.items():
            if state.is_terminal:
                V[(r, c)] = state.reward
        
        # Iterative policy evaluation
        threshold = 0.01
        while True:
            delta = 0
            V_new = V.copy()
            
            for (r, c) in V:
                if (r, c) in grid.walls or grid.states[(r, c)].is_terminal:
                    continue
                
                action = policy.get((r, c), "up")
                value = 0.0
                
                for (next_state, prob) in grid.get_neighbors((r, c), action):
                    reward = grid.states[next_state].reward
                    value += prob * (reward + grid.discount_factor * V[next_state])
                
                V_new[(r, c)] = value
                delta = max(delta, abs(V_new[(r, c)] - V[(r, c)]))
            
            V = V_new
            if delta < threshold:
                break
        
        return V
    
    def _compute_action_value(self, grid: GridWorld, V: Dict[tuple[int, int], float], 
                             state: tuple[int, int], action: str) -> float:
        """Compute Q(s,a) = Σ P(s'|s,a)[R + γV(s')]."""
        value = 0.0
        for (next_state, prob) in grid.get_neighbors(state, action):
            reward = grid.states[next_state].reward
            value += prob * (reward + grid.discount_factor * V[next_state])
        return value


class QLearningSolver(Solver):
    """
    Q-learning update rule: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    Model-free temporal difference learning.
    """
    
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transitions: List[Transition] = data["transitions"]
        params: RLParameters = data["parameters"]
        initial_q: Dict = data.get("initial_q", {})
        
        Q = QTable()
        Q.values = initial_q.copy()
        
        for transition in transitions:
            s, a, s_prime, r = transition.state, transition.action, transition.next_state, transition.reward
            
            # Find max Q(s', a')
            max_q_next = max(
                [Q.get(s_prime, action) for action in ["up", "down", "left", "right"]],
                default=0.0
            )
            
            # Q-learning update
            old_q = Q.get(s, a)
            new_q = old_q + params.alpha * (r + params.gamma * max_q_next - old_q)
            Q.set(s, a, new_q)
        
        # Extract policy
        policy = {}
        states = set([t.state for t in transitions] + [t.next_state for t in transitions])
        for state in states:
            policy[state] = Q.get_best_action(state, ["up", "down", "left", "right"])
        
        return {
            "q_values": Q.values,
            "policy": policy,
            "parameters_explanation": {
                "alpha": "Learning rate - controleaza cat de mult influenteaza noile informatii valorile Q. Daca alpha=0, nu se invata nimic nou.",
                "gamma": "Discount factor - determina importanta recompenselor viitoare. Daca gamma=0, se considera doar recompensa imediata.",
                "epsilon": "Exploration rate - probabilitatea de a alege o actiune aleatorie (explorare vs exploatare). Daca epsilon=0, se alege mereu cea mai buna actiune cunoscuta (no exploration)."
            }
        }


class TDLearningSolver(Solver):
    """
    TD(0) update rule: V(s) ← V(s) + α[r + γV(s') - V(s)]
    Temporal difference learning for state values.
    """
    
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transitions: List[Transition] = data["transitions"]
        params: RLParameters = data["parameters"]
        initial_v: Dict = data.get("initial_v", {})
        
        V = initial_v.copy()
        td_errors = []
        
        for transition in transitions:
            s, s_prime, r = transition.state, transition.next_state, transition.reward
            
            if s not in V:
                V[s] = 0.0
            if s_prime not in V:
                V[s_prime] = 0.0
            
            # TD(0) update
            td_error = r + params.gamma * V[s_prime] - V[s]
            V[s] = V[s] + params.alpha * td_error
            td_errors.append(td_error)
        
        return {
            "values": V,
            "td_errors": td_errors
        }


class StripsActionFormatterSolver(Solver):
    """Solver for formatting actions in STRIPS representation."""
    
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats an action in STRIPS language.
        
        Args:
            data: Dictionary containing 'action' (Action object)
        
        Returns:
            Dictionary with formatted STRIPS representation
        """
        action: Action = data["action"]
        
        # Format preconditions
        preconditions_str = ", ".join(str(p) for p in action.preconditions)
        
        # Format add effects
        add_effects_str = ", ".join(str(p) for p in action.add_effects)
        
        # Format delete effects
        delete_effects_str = ", ".join(str(p) for p in action.delete_effects)
        
        formatted = (
            f"Operația {action}:\n"
            f"Precondiții: {preconditions_str if preconditions_str else 'niciuna'}\n"
            f"Add-list: {add_effects_str if add_effects_str else 'niciuna'}\n"
            f"Delete-list: {delete_effects_str if delete_effects_str else 'niciuna'}"
        )
        
        return {
            "formatted_action": formatted,
            "preconditions": action.preconditions,
            "add_effects": action.add_effects,
            "delete_effects": action.delete_effects
        }


class AdlActionFormatterSolver(Solver):
    """Solver for formatting actions in ADL representation (with conditional effects)."""
    
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats an action in ADL language (includes conditional effects).
        
        Args:
            data: Dictionary containing 'action' (Action object)
        
        Returns:
            Dictionary with formatted ADL representation
        """
        action: Action = data["action"]
        
        # Format preconditions
        preconditions_str = ", ".join(str(p) for p in action.preconditions)
        
        # Format add effects
        add_effects_str = ", ".join(str(p) for p in action.add_effects)
        
        # Format delete effects
        delete_effects_str = ", ".join(str(p) for p in action.delete_effects)
        
        formatted = (
            f"Operația {action} (ADL):\n"
            f"Precondiții: {preconditions_str if preconditions_str else 'niciuna'}\n"
            f"Add-list: {add_effects_str if add_effects_str else 'niciuna'}\n"
            f"Delete-list: {delete_effects_str if delete_effects_str else 'niciuna'}"
        )
        
        # Add conditional effects if present
        if action.conditional_effects:
            formatted += "\nEfecte condiționate:"
            for conditions, effects in action.conditional_effects:
                cond_str = ", ".join(str(c) for c in conditions)
                eff_str = ", ".join(str(e) for e in effects)
                formatted += f"\n  CÂND {cond_str} ATUNCI {eff_str}"
        
        return {
            "formatted_action": formatted,
            "preconditions": action.preconditions,
            "add_effects": action.add_effects,
            "delete_effects": action.delete_effects,
            "conditional_effects": action.conditional_effects
        }


class PartialOrderPlanningSolver(Solver):
    """Solver for creating partial order plans (POP)."""
    
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a partial order plan to achieve goals from initial state.
        
        Args:
            data: Dictionary containing 'problem' (PlanningProblem)
        
        Returns:
            Dictionary with partial order plan
        """
        problem: PlanningProblem = data["problem"]
        
        # Initialize plan with Start and Finish actions
        plan_actions = []
        orderings = []
        causal_links = []
        
        # Start action (id=0) produces initial state
        start_action = Action(
            name="Start",
            parameters=[],
            preconditions=[],
            add_effects=problem.initial_state,
            delete_effects=[]
        )
        plan_actions.append((0, start_action))
        
        # Finish action (id=999) requires goal state
        finish_action = Action(
            name="Finish",
            parameters=[],
            preconditions=problem.goal_state,
            add_effects=[],
            delete_effects=[]
        )
        plan_actions.append((999, finish_action))
        
        # Find actions to achieve each goal
        action_id = 1
        for goal in problem.goal_state:
            # Find an action that produces this goal
            producer_action = None
            for action in problem.actions:
                if any(eff.name == goal.name and eff.positive == goal.positive 
                      for eff in action.add_effects):
                    producer_action = action
                    break
            
            if producer_action:
                # Add action to plan
                plan_actions.append((action_id, producer_action))
                
                # Add ordering: Start < action < Finish
                orderings.append((0, action_id))
                orderings.append((action_id, 999))
                
                # Add causal link: action produces goal for Finish
                causal_links.append((action_id, goal, 999))
                
                # Check if action has preconditions that need to be satisfied
                for precond in producer_action.preconditions:
                    # Check if precondition is in initial state
                    if any(p.name == precond.name and p.positive == precond.positive 
                          for p in problem.initial_state):
                        # Add causal link from Start
                        causal_links.append((0, precond, action_id))
                
                action_id += 1
        
        # Create partial order plan
        pop = PartialOrderPlan(
            actions=plan_actions,
            orderings=orderings,
            causal_links=causal_links
        )
        
        return {
            "plan": pop,
            "formatted_plan": str(pop),
            "num_actions": len(plan_actions),
            "num_orderings": len(orderings),
            "num_causal_links": len(causal_links)
        }


class ForwardSearchPlanningSolver(Solver):
    """Solver for validating plans using forward search."""
    
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a plan by applying actions forward from initial state.
        
        Args:
            data: Dictionary containing 'problem' (PlanningProblem) and 'plan' (list of Actions)
        
        Returns:
            Dictionary with validation results
        """
        problem: PlanningProblem = data["problem"]
        plan: List[Action] = data["plan"]
        
        # Start with initial state
        current_state = set(problem.initial_state)
        
        # Track execution
        valid = True
        errors = []
        
        # Apply each action in sequence
        for i, action in enumerate(plan):
            # Check preconditions
            preconditions_satisfied = all(
                any(p.name == s.name and p.parameters == s.parameters and p.positive == s.positive 
                    for s in current_state)
                for p in action.preconditions
            )
            
            if not preconditions_satisfied:
                valid = False
                errors.append(f"Acțiunea {i+1} ({action}) are precondiții nesatisfăcute")
                continue
            
            # Apply effects
            for delete_effect in action.delete_effects:
                # Remove from current state
                current_state = {s for s in current_state 
                               if not (s.name == delete_effect.name and s.parameters == delete_effect.parameters and s.positive == delete_effect.positive)}
            
            for add_effect in action.add_effects:
                current_state.add(add_effect)
        
        # Check if goals are achieved
        goals_achieved = all(
            any(g.name == s.name and g.parameters == s.parameters and g.positive == s.positive 
                for s in current_state)
            for g in problem.goal_state
        )
        
        if not goals_achieved:
            valid = False
            errors.append("Obiectivele nu sunt atinse la sfârșitul planului")
        
        return {
            "valid": valid,
            "errors": errors,
            "final_state": list(current_state),
            "goals_achieved": goals_achieved
        }


class SolverFactory:
    def __init__(self) -> None:
        self._mapping: Dict[QuestionType, type[Solver]] = {
            QuestionType.STRATEGY_SELECTION: StrategySolver,
            QuestionType.NASH_EQUILIBRIUM: NashSolver,
            QuestionType.CSP_COMPLETION: CspSolver,
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxAlphaBetaSolver,
            QuestionType.VALUE_ITERATION: ValueIterationSolver,
            QuestionType.POLICY_ITERATION: PolicyIterationSolver,
            QuestionType.Q_LEARNING: QLearningSolver,
            QuestionType.TD_LEARNING: TDLearningSolver,
            QuestionType.STRIPS_ACTION_DEFINITION: StripsActionFormatterSolver,
            QuestionType.ADL_ACTION_DEFINITION: AdlActionFormatterSolver,
            QuestionType.PARTIAL_ORDER_PLAN: PartialOrderPlanningSolver,
            QuestionType.PLAN_VALIDATION: ForwardSearchPlanningSolver,
        }
    
    def get_solver(self, q_type: QuestionType) -> Solver:
        cls = self._mapping[q_type]
        return cls()

