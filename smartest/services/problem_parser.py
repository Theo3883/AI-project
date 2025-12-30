from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..core.models import (
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
    PlanningProblem
)


@dataclass
class ParsedProblem:
    """Result of parsing a user's question."""
    question_type: QuestionType
    data: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    raw_text: str


class ProblemExtractor(ABC):
    """Base class for extracting problem-specific data."""
    
    @abstractmethod
    def can_extract(self, text: str) -> float:
        """Returns confidence score (0.0 to 1.0) that this extractor can handle the text."""
        pass
    
    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """Extracts problem data from text."""
        pass


class NashGameExtractor(ProblemExtractor):
    """Extracts Nash equilibrium game data from text."""
    
    KEYWORDS = ['nash', 'equilibrium', 'game', 'payoff', 'player', 'strategy', 'strategies']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Check for payoff patterns like (3,2) or (3, 2)
        # BUT: make sure it's not MDP/RL related (which uses (r,c) for states)
        has_payoffs = bool(re.search(r'\(\s*\d+\s*,\s*\d+\s*\)', text))
        
        # If MDP/RL keywords present, reduce confidence significantly
        mdp_rl_keywords = ['mdp', 'markov', 'bellman', 'grid', 'value iteration', 'policy iteration',
                          'q-learning', 'td-learning', 'reinforcement', 'alpha', 'gamma', 'reward']
        has_mdp_rl = any(kw in text_lower for kw in mdp_rl_keywords)
        
        confidence = (keyword_matches / len(self.KEYWORDS)) * 0.7
        if has_payoffs and not has_mdp_rl:
            confidence += 0.3
        elif has_mdp_rl:
            # Penalize if MDP/RL keywords present
            confidence *= 0.2
            
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extracts game matrix from text.
        Supports formats like:
        - "(3,2) (1,4) / (2,3) (4,1)" (rows separated by /)
        - "row 1: (3,2) (1,4), row 2: (2,3) (4,1)"
        - "first row (3,2) and (1,4) second row (2,3) and (4,1)"
        """
        # Find all payoff tuples
        payoff_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        matches = re.findall(payoff_pattern, text)
        
        if not matches:
            raise ValueError("Nu s-au gasit valori de payoff in text. Foloseste formatul (3,2) pentru payoff-uri.")
        
        payoffs_list = [(int(p1), int(p2)) for p1, p2 in matches]
        
        # Determine matrix dimensions (assume square or 2x2 for now)
        num_payoffs = len(payoffs_list)
        if num_payoffs == 4:
            rows, cols = 2, 2
        elif num_payoffs == 6:
            rows, cols = 2, 3
        elif num_payoffs == 9:
            rows, cols = 3, 3
        else:
            # Try to infer from text
            rows = 2
            cols = num_payoffs // rows
        
        # Organize into matrix
        payoffs = []
        for i in range(rows):
            row = []
            for j in range(cols):
                idx = i * cols + j
                if idx < len(payoffs_list):
                    row.append(payoffs_list[idx])
                else:
                    row.append((0, 0))
            payoffs.append(row)
        
        # Extract strategy names if mentioned, otherwise use defaults
        row_strategies = self._extract_strategy_names(text, rows, "row")
        col_strategies = self._extract_strategy_names(text, cols, "col")
        
        game = NashGame(
            row_strategies=row_strategies,
            col_strategies=col_strategies,
            payoffs=payoffs
        )
        
        return {"game": game}
    
    def _extract_strategy_names(self, text: str, count: int, player_type: str) -> List[str]:
        """Try to extract strategy names, or use defaults."""
        text_lower = text.lower()
        
        # Look for common strategy names
        common_names = ['up', 'down', 'left', 'right', 'top', 'bottom', 
                       'sus', 'jos', 'stanga', 'dreapta', 'a', 'b', 'c', 'd']
        
        found_names = [name for name in common_names if name in text_lower]
        
        if len(found_names) >= count:
            return found_names[:count]
        
        # Default names
        if player_type == "row":
            return [f"Row{i+1}" for i in range(count)]
        else:
            return [f"Col{i+1}" for i in range(count)]


class CspExtractor(ProblemExtractor):
    """Extracts CSP (Constraint Satisfaction Problem) data from text."""
    
    KEYWORDS = ['csp', 'constraint', 'variable', 'domain', 'coloring', 'color', 'graph']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Check for variable patterns
        has_variables = bool(re.search(r'\b[A-Z]\b', text))
        
        # Check for constraint patterns like A!=B or A≠B
        has_constraints = bool(re.search(r'[A-Z]\s*(!?=|≠)\s*[A-Z]', text))
        
        confidence = (keyword_matches / len(self.KEYWORDS)) * 0.5
        if has_variables:
            confidence += 0.2
        if has_constraints:
            confidence += 0.3
            
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extracts CSP from text.
        Supports formats like:
        - "variables A, B, C with domains red, blue, green"
        - "constraints: A!=B, B!=C"
        - "A is already red" or "partial assignment A=red"
        """
        # Extract variable names (single capital letters)
        var_pattern = r'\b([A-Z])\b'
        var_matches = re.findall(var_pattern, text)
        variable_names = sorted(set(var_matches))
        
        if not variable_names:
            raise ValueError("Nu s-au gasit variabile. Foloseste litere mari cum ar fi A, B, C pentru variabile.")
        
        # Extract domain/colors
        colors = self._extract_colors(text)
        if not colors:
            colors = ['red', 'green', 'blue']  # Default colors
        
        # Create variables
        variables = {
            name: CspVariable(name, list(colors))
            for name in variable_names
        }
        
        # Extract constraints (A!=B, A≠B, etc.)
        constraints = self._extract_constraints(text, variable_names)
        
        # Extract partial assignment
        partial_assignment = self._extract_partial_assignment(text, variable_names, colors)
        
        csp = CspInstance(
            variables=variables,
            constraints=constraints,
            partial_assignment=partial_assignment
        )
        
        return {"csp": csp}
    
    def _extract_colors(self, text: str) -> List[str]:
        """Extract color names from text."""
        text_lower = text.lower()
        common_colors = ['red', 'green', 'blue', 'yellow', 'rosu', 'verde', 'albastru', 'galben']
        found_colors = [color for color in common_colors if color in text_lower]
        return list(dict.fromkeys(found_colors))  # Remove duplicates, preserve order
    
    def _extract_constraints(self, text: str, variables: List[str]) -> List[CspConstraint]:
        """Extract constraints like A!=B or A≠B."""
        constraints = []
        
        # Pattern for constraints: A!=B or A≠B
        constraint_pattern = r'([A-Z])\s*(!?=|≠)\s*([A-Z])'
        matches = re.findall(constraint_pattern, text)
        
        seen = set()
        for var1, op, var2 in matches:
            if var1 in variables and var2 in variables:
                # Avoid duplicates (A!=B same as B!=A)
                pair = tuple(sorted([var1, var2]))
                if pair not in seen:
                    constraints.append(CspConstraint(var1, var2, "!="))
                    seen.add(pair)
        
        return constraints
    
    def _extract_partial_assignment(self, text: str, variables: List[str], colors: List[str]) -> Dict[str, Any]:
        """Extract partial assignments like 'A is red' or 'A=red'."""
        partial = {}
        
        # Pattern 1: A=red
        pattern1 = r'([A-Z])\s*=\s*(\w+)'
        matches1 = re.findall(pattern1, text)
        for var, value in matches1:
            if var in variables and value.lower() in colors:
                partial[var] = value.lower()
        
        # Pattern 2: A is red, A is already red
        pattern2 = r'([A-Z])\s+is\s+(?:already\s+)?(\w+)'
        matches2 = re.findall(pattern2, text.lower())
        for var_lower, value in matches2:
            var = var_lower.upper()
            if var in variables and value in colors:
                partial[var] = value
        
        return partial


class MinimaxExtractor(ProblemExtractor):
    """Extracts Minimax/Alpha-Beta game tree data from text."""
    
    KEYWORDS = ['minimax', 'alpha', 'beta', 'alpha-beta', 'game tree', 'tree', 'leaf', 'leaves', 'pruning']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Check for number sequences (leaf values)
        has_numbers = bool(re.search(r'\d+', text))
        
        # Check for array/list patterns
        has_array = bool(re.search(r'\[.*\]', text))
        
        confidence = (keyword_matches / len(self.KEYWORDS)) * 0.6
        if has_numbers:
            confidence += 0.2
        if has_array:
            confidence += 0.2
            
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extracts game tree from text.
        Supports formats like:
        - "leaf values: [3, 5, 2, 9, 7, 4, 8, 1]"
        - "leaves: 3, 5, 2, 9, 7, 4, 8, 1"
        - "values 3 5 2 9 7 4 8 1"
        """
        # Try to find array notation first
        array_pattern = r'\[([^\]]+)\]'
        array_match = re.search(array_pattern, text)
        
        if array_match:
            numbers_text = array_match.group(1)
            leaf_values = [int(x.strip()) for x in re.findall(r'-?\d+', numbers_text)]
        else:
            # Find all numbers in text
            leaf_values = [int(x) for x in re.findall(r'-?\d+', text)]
        
        if not leaf_values:
            raise ValueError("Nu s-au gasit valori pentru frunze. Furnizeaza valori precum [3, 5, 2, 9] sau '3, 5, 2, 9'.")
        
        # Ensure we have a power of 2 (binary tree)
        if len(leaf_values) not in [2, 4, 8, 16]:
            raise ValueError(f"Se asteapta 2, 4, 8 sau 16 valori de frunze pentru un arbore binar, s-au primit {len(leaf_values)}.")
        
        # Build tree bottom-up
        root = self._build_tree(leaf_values)
        
        return {"root": root}
    
    def _build_tree(self, leaf_values: List[int]) -> GameTreeNode:
        """Build a game tree from leaf values (bottom-up)."""
        # Create leaf nodes
        current_level = [GameTreeNode(value=v, children=[], is_max_player=False) for v in leaf_values]
        
        # Determine if we start with max or min (alternate levels)
        # For depth 3: root(MAX) -> level1(MIN) -> level2(MAX) -> leaves
        # We work backwards, so leaves are at the deepest level
        is_max = False  # Start with MIN at second-to-last level
        
        # Build tree level by level
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                parent = GameTreeNode(
                    value=None,
                    children=[current_level[i], current_level[i + 1]],
                    is_max_player=is_max
                )
                next_level.append(parent)
            current_level = next_level
            is_max = not is_max  # Alternate
        
        return current_level[0]


class StrategyExtractor(ProblemExtractor):
    """Extracts strategy selection problem data from text."""
    
    KEYWORDS = ['strategy', 'n-queens', 'hanoi', 'knight', 'tour', 'graph coloring', 
                'backtracking', 'dfs', 'bfs', 'problem', 'algorithm', 'approach', 
                'recommend', 'best']
    
    PROBLEM_NAMES = ['n-queens', 'hanoi', 'graph-coloring', 'knights-tour', 'knight']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Check for specific problem names (higher weight)
        has_problem = any(prob in text_lower for prob in self.PROBLEM_NAMES)
        
        # Check for "recommend"/"best" + "strategy"/"algorithm"
        has_recommendation = ('recommend' in text_lower or 'best' in text_lower) and \
                            ('strategy' in text_lower or 'algorithm' in text_lower or 'approach' in text_lower)
        
        confidence = (keyword_matches / len(self.KEYWORDS)) * 0.5
        if has_problem:
            confidence += 0.4
        if has_recommendation:
            confidence += 0.2
            
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extracts strategy problem type from text.
        Supports formats like:
        - "What strategy for n-queens?"
        - "Best approach for Towers of Hanoi"
        - "Solve knights tour problem"
        """
        text_lower = text.lower()
        
        # Identify problem type
        problem = None
        if 'n-queen' in text_lower or 'nqueen' in text_lower or 'queens' in text_lower:
            problem = 'n-queens'
        elif 'hanoi' in text_lower or 'tower' in text_lower:
            problem = 'hanoi'
        elif 'graph' in text_lower and 'color' in text_lower:
            problem = 'graph-coloring'
        elif 'knight' in text_lower and 'tour' in text_lower:
            problem = 'knights-tour'
        
        if not problem:
            raise ValueError("Nu s-a putut identifica tipul problemei. Mentioneaza: n-queens, hanoi, graph-coloring sau knights-tour.")
        
        return {"problem": problem}


class MDPExtractor(ProblemExtractor):
    """Extracts MDP grid world problems from text."""
    
    KEYWORDS = ['mdp', 'markov', 'value iteration', 'policy iteration', 'grid', 'grid world', 
                'bellman', 'utilitate', 'utility', 'discount', 'gamma']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Check for grid dimensions
        has_grid = bool(re.search(r'\d+x\d+', text))
        
        # Check for reward patterns
        has_rewards = bool(re.search(r'reward|recompens', text_lower))
        
        # Check for state patterns
        has_states = bool(re.search(r'\(\s*\d+\s*,\s*\d+\s*\)', text))
        
        confidence = (keyword_matches / len(self.KEYWORDS)) * 0.5
        if has_grid:
            confidence += 0.2
        if has_rewards:
            confidence += 0.15
        if has_states:
            confidence += 0.15
            
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extracts MDP grid world from text.
        Supports formats like:
        - "grid 3x4"
        - "states: (0,0), (0,1), ..."
        - "reward at (0,3) = 1.0"
        - "discount factor gamma = 0.9"
        """
        # Extract grid dimensions
        grid_match = re.search(r'(\d+)\s*x\s*(\d+)', text, re.IGNORECASE)
        if grid_match:
            rows = int(grid_match.group(1))
            cols = int(grid_match.group(2))
        else:
            # Default grid
            rows, cols = 3, 4
        
        # Extract discount factor
        gamma_match = re.search(r'gamma\s*=\s*(0\.\d+)|discount.*?(0\.\d+)', text, re.IGNORECASE)
        if gamma_match:
            gamma = float(gamma_match.group(1) or gamma_match.group(2))
        else:
            gamma = 0.9
        
        # Extract rewards
        reward_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\).*?(-?\d+\.?\d*)'
        reward_matches = re.findall(reward_pattern, text)
        
        states = {}
        for r in range(rows):
            for c in range(cols):
                states[(r, c)] = MDPState(row=r, col=c, is_terminal=False, reward=-0.04)
        
        # Apply extracted rewards
        for r_str, c_str, reward_str in reward_matches:
            r, c = int(r_str), int(c_str)
            reward = float(reward_str)
            if (r, c) in states:
                states[(r, c)].reward = reward
                if abs(reward) >= 1.0:
                    states[(r, c)].is_terminal = True
        
        # Extract walls
        wall_pattern = r'wall.*?\(\s*(\d+)\s*,\s*(\d+)\s*\)|perete.*?\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        wall_matches = re.findall(wall_pattern, text, re.IGNORECASE)
        walls = []
        for match in wall_matches:
            r = int(match[0] or match[2])
            c = int(match[1] or match[3])
            walls.append((r, c))
        
        # Extract transition probabilities
        intended_prob = 0.8
        perp_prob = 0.1
        
        prob_match = re.search(r'intended.*?(0\.\d+)', text, re.IGNORECASE)
        if prob_match:
            intended_prob = float(prob_match.group(1))
        
        perp_match = re.search(r'perpendicular.*?(0\.\d+)', text, re.IGNORECASE)
        if perp_match:
            perp_prob = float(perp_match.group(1))
        
        transition_probs = {"intended": intended_prob, "perpendicular": perp_prob}
        
        grid = GridWorld(
            rows=rows,
            cols=cols,
            states=states,
            discount_factor=gamma,
            transition_probs=transition_probs,
            walls=walls
        )
        
        # Determine number of iterations
        iterations = 1
        iter_match = re.search(r'(\d+)\s*(?:pas|step|iteration)', text, re.IGNORECASE)
        if iter_match:
            iterations = int(iter_match.group(1))
        
        return {"grid": grid, "iterations": iterations}


class RLExtractor(ProblemExtractor):
    """Extracts RL problems (Q-learning, TD-learning) from text."""
    
    KEYWORDS = ['q-learning', 'q learning', 'td-learning', 'td learning', 'temporal difference', 
                'alpha', 'gamma', 'epsilon', 'transition', 'observation', 'learning rate']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Check for transition patterns: (s, a, s', r)
        has_transitions = bool(re.search(r's\s*=|state\s*=|stare\s*=', text_lower))
        
        # Check for parameters
        has_params = bool(re.search(r'alpha\s*=|gamma\s*=|epsilon\s*=', text_lower))
        
        # Check for Q-values
        has_q_values = bool(re.search(r'q\s*\(|q-value|q value', text_lower))
        
        confidence = (keyword_matches / len(self.KEYWORDS)) * 0.5
        if has_transitions:
            confidence += 0.2
        if has_params:
            confidence += 0.2
        if has_q_values:
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extracts RL problem from text.
        Supports formats like:
        - "s=(0,0), a=right, s'=(0,1), r=0"
        - "alpha=0.1, gamma=0.9, epsilon=0.1"
        - "Q((0,0), right) = 0.5"
        """
        # Extract parameters
        alpha = 0.1
        gamma = 0.9
        epsilon = 0.1
        
        alpha_match = re.search(r'alpha\s*=\s*(0\.\d+)', text, re.IGNORECASE)
        if alpha_match:
            alpha = float(alpha_match.group(1))
        
        gamma_match = re.search(r'gamma\s*=\s*(0\.\d+)', text, re.IGNORECASE)
        if gamma_match:
            gamma = float(gamma_match.group(1))
        
        epsilon_match = re.search(r'epsilon\s*=\s*(0\.\d+)', text, re.IGNORECASE)
        if epsilon_match:
            epsilon = float(epsilon_match.group(1))
        
        params = RLParameters(alpha=alpha, gamma=gamma, epsilon=epsilon)
        
        # Extract transitions
        # Pattern: s=(0,0), a=right, s'=(0,1), r=0.5
        transition_pattern = r's\s*=\s*\((\d+),\s*(\d+)\).*?a\s*=\s*(\w+).*?s.*?=\s*\((\d+),\s*(\d+)\).*?r\s*=\s*(-?\d+\.?\d*)'
        transition_matches = re.findall(transition_pattern, text, re.IGNORECASE)
        
        transitions = []
        for match in transition_matches:
            state = (int(match[0]), int(match[1]))
            action = match[2]
            next_state = (int(match[3]), int(match[4]))
            reward = float(match[5])
            transitions.append(Transition(state=state, action=action, next_state=next_state, reward=reward))
        
        # Extract initial Q-values or V-values
        q_value_pattern = r'Q\s*\(\s*\((\d+),\s*(\d+)\)\s*,\s*(\w+)\s*\)\s*=\s*(-?\d+\.?\d*)'
        q_matches = re.findall(q_value_pattern, text, re.IGNORECASE)
        
        initial_q = {}
        for match in q_matches:
            state = (int(match[0]), int(match[1]))
            action = match[2]
            value = float(match[3])
            initial_q[(state, action)] = value
        
        # Extract initial V-values
        v_value_pattern = r'V\s*\(\s*\((\d+),\s*(\d+)\)\s*\)\s*=\s*(-?\d+\.?\d*)'
        v_matches = re.findall(v_value_pattern, text, re.IGNORECASE)
        
        initial_v = {}
        for match in v_matches:
            state = (int(match[0]), int(match[1]))
            value = float(match[2])
            initial_v[state] = value
        
        # Determine problem type
        if 'q-learning' in text.lower() or 'q learning' in text.lower():
            return {
                "transitions": transitions,
                "parameters": params,
                "initial_q": initial_q
            }
        elif 'td' in text.lower() or 'temporal difference' in text.lower():
            return {
                "transitions": transitions,
                "parameters": params,
                "initial_v": initial_v
            }
        else:
            # Default to Q-learning
            return {
                "transitions": transitions,
                "parameters": params,
                "initial_q": initial_q
            }


class StripsAdlExtractor(ProblemExtractor):
    """Extracts STRIPS/ADL action definition problems from text."""
    
    KEYWORDS = ['strips', 'adl', 'operație', 'operatie', 'acțiune', 'actiune', 
                'precondiție', 'preconditie', 'preconditions', 'add-list', 'delete-list',
                'efecte', 'effects', 'go', 'buy', 'fromtable', 'totable', 'placecap', 'removecap', 'insert']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Strong indicators
        has_strips = 'strips' in text_lower
        has_adl = 'adl' in text_lower
        has_action_description = any(phrase in text_lower for phrase in [
            'descrieți operația', 'descrieti operatia', 'describe operation',
            'describe action', 'defineți acțiunea', 'defineti actiunea'
        ])
        has_preconditions = 'precondi' in text_lower
        has_effects = 'efect' in text_lower or 'add' in text_lower or 'delete' in text_lower
        
        confidence = 0.0
        
        if has_strips or has_adl:
            confidence += 0.5
        if has_action_description:
            confidence += 0.3
        if has_preconditions:
            confidence += 0.1
        if has_effects:
            confidence += 0.1
        
        # Boost if multiple keywords present
        if keyword_matches >= 3:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract action definition problem from text.
        Returns domain name and action name if found.
        """
        text_lower = text.lower()
        
        # Detect domain
        domain = "unknown"
        if any(word in text_lower for word in ['shopping', 'cumpărături', 'cumparaturi', 'magazin']):
            domain = "shopping"
        elif any(word in text_lower for word in ['blocksworld', 'blocks', 'cuburi', 'cuburi']):
            domain = "blocksworld"
        elif any(word in text_lower for word in ['container', 'recipient']):
            domain = "container"
        elif any(word in text_lower for word in ['logistics', 'logistică', 'logistica']):
            domain = "logistics"
        
        # Extract action name using patterns like Go(...), Buy(...), etc.
        action_pattern = r'(\w+)\s*\([^)]*\)'
        action_matches = re.findall(action_pattern, text)
        action_name = action_matches[0] if action_matches else "Unknown"
        
        # Check if ADL (conditional effects mentioned)
        is_adl = 'adl' in text_lower or 'condițional' in text_lower or 'conditional' in text_lower or 'când' in text_lower or 'when' in text_lower
        
        return {
            "domain": domain,
            "action_name": action_name,
            "is_adl": is_adl,
            "raw_text": text
        }


class PopExtractor(ProblemExtractor):
    """Extracts Partial Order Planning problems from text."""
    
    KEYWORDS = ['partial order planning', 'pop', 'plan incomplet', 'ordine parțială', 
                'ordine partiala', 'linkuri cauzale', 'causal links', 'ordering constraints',
                'planificare', 'planning']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Strong indicators
        has_pop = 'partial order' in text_lower or 'pop' in text_lower
        has_plan = 'plan' in text_lower
        has_incomplete = 'incomplet' in text_lower or 'incomplete' in text_lower
        has_ordering = 'ordine' in text_lower or 'ordering' in text_lower or 'precedență' in text_lower
        has_causal_links = 'cauzal' in text_lower or 'causal' in text_lower
        has_initial_goal = ('stare inițială' in text_lower or 'initial state' in text_lower) and \
                          ('obiectiv' in text_lower or 'goal' in text_lower)
        
        confidence = 0.0
        
        if has_pop:
            confidence += 0.6
        elif has_plan and has_incomplete:
            confidence += 0.4
        
        if has_ordering:
            confidence += 0.15
        if has_causal_links:
            confidence += 0.15
        if has_initial_goal:
            confidence += 0.1
        
        # Boost if multiple keywords present
        if keyword_matches >= 3:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract POP problem from text.
        Returns initial state, goals, and domain if found.
        """
        text_lower = text.lower()
        
        # Detect domain
        domain = "unknown"
        if any(word in text_lower for word in ['shopping', 'cumpărături', 'cumparaturi']):
            domain = "shopping"
        elif any(word in text_lower for word in ['blocksworld', 'blocks', 'cuburi']):
            domain = "blocksworld"
        
        # Extract predicates (simplified pattern matching)
        # Look for patterns like "At(agent, home)", "Have(milk)", etc.
        predicate_pattern = r'([A-Z][a-zA-Z]*)\s*\([^)]+\)'
        predicates_found = re.findall(predicate_pattern, text)
        
        # Check for initial state markers
        has_initial = 'inițial' in text_lower or 'initial' in text_lower
        
        # Check for goal markers
        has_goal = 'obiectiv' in text_lower or 'goal' in text_lower
        
        return {
            "domain": domain,
            "predicates_found": predicates_found,
            "has_initial_state": has_initial,
            "has_goals": has_goal,
            "raw_text": text
        }


class PlanValidationExtractor(ProblemExtractor):
    """Extracts plan validation problems from text."""
    
    KEYWORDS = ['valid', 'verificați', 'verificati', 'check', 'corect', 'correct',
                'plan', 'acțiuni', 'actiuni', 'actions', 'secvență', 'secventa', 'sequence']
    
    def can_extract(self, text: str) -> float:
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.KEYWORDS if kw in text_lower)
        
        # Strong indicators
        has_validation = any(phrase in text_lower for phrase in [
            'verificați', 'verificati', 'check', 'validate', 'corect', 'correct'
        ])
        has_plan = 'plan' in text_lower
        has_sequence = 'secvență' in text_lower or 'sequence' in text_lower
        
        confidence = 0.0
        
        if has_validation and has_plan:
            confidence += 0.7
        elif has_validation:
            confidence += 0.3
        
        if has_sequence:
            confidence += 0.2
        
        if keyword_matches >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract plan validation problem from text."""
        # Extract action sequence
        # Look for patterns like "1. Go(...)" or "Go(...), Buy(...)"
        action_pattern = r'([A-Z][a-zA-Z]*)\s*\([^)]*\)'
        actions_found = re.findall(action_pattern, text)
        
        return {
            "actions_sequence": actions_found,
            "raw_text": text
        }


class ProblemParser:
    """Main parser that coordinates type detection and extraction."""
    
    def __init__(self) -> None:
        self.extractors: Dict[QuestionType, ProblemExtractor] = {
            QuestionType.NASH_EQUILIBRIUM: NashGameExtractor(),
            QuestionType.CSP_COMPLETION: CspExtractor(),
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxExtractor(),
            QuestionType.STRATEGY_SELECTION: StrategyExtractor(),
            QuestionType.VALUE_ITERATION: MDPExtractor(),
            QuestionType.Q_LEARNING: RLExtractor(),
            QuestionType.STRIPS_ACTION_DEFINITION: StripsAdlExtractor(),
            QuestionType.ADL_ACTION_DEFINITION: StripsAdlExtractor(),  # Same extractor, different handling
            QuestionType.PARTIAL_ORDER_PLAN: PopExtractor(),
            QuestionType.PLAN_VALIDATION: PlanValidationExtractor(),
        }
    
    def parse(self, text: str) -> ParsedProblem:
        """
        Parse user's question text and return structured problem data.
        
        Args:
            text: User's natural language question
            
        Returns:
            ParsedProblem with detected type and extracted data
            
        Raises:
            ValueError: If text cannot be parsed or is ambiguous
        """
        if not text or not text.strip():
            raise ValueError("Te rog furnizeaza o descriere a problemei.")
        
        # Calculate confidence for each extractor
        scores = {}
        for q_type, extractor in self.extractors.items():
            scores[q_type] = extractor.can_extract(text)
        
        text_lower = text.lower()
        
        # Define explicit problem type indicators
        problem_type_indicators = {
            'nash': ['nash', 'echilibru nash'],
            'csp': ['csp', 'constraint satisfaction', 'constrangere', 'satisfacere'],
            'minimax': ['minimax', 'alpha-beta', 'alpha beta'],
            'hanoi': ['hanoi', 'turnuri'],
            'nqueens': ['n-queen', 'nqueen', 'queens'],
            'knights': ['knight', 'cal'],
            'coloring': ['graph color', 'colorare graf'],
            'mdp': ['mdp', 'markov', 'value iteration', 'policy iteration', 'bellman', 'grid world'],
            'rl': ['q-learning', 'q learning', 'td-learning', 'td learning', 'reinforcement learning'],
            'strips': ['strips', 'add-list', 'delete-list'],
            'adl': ['adl', 'conditional effect', 'condițional'],
            'pop': ['partial order', 'pop', 'plan incomplet', 'ordine parțială', 'linkuri cauzale'],
            'plan_validation': ['verificați plan', 'validate plan', 'plan corect'],
        }
        
        # Check for "using/with/cu/folosind" pattern with explicit problem types
        method_connectors = ['folosind', 'using', 'with', 'cu', 'prin', 'via']
        
        for connector in method_connectors:
            if connector in text_lower:
                # Split by the connector and check for different problem types on each side
                parts = text_lower.split(connector)
                if len(parts) >= 2:
                    left_part = parts[0]
                    right_part = ' '.join(parts[1:])
                    
                    # Check which problem types are mentioned in each part
                    left_types = set()
                    right_types = set()
                    
                    for ptype, keywords in problem_type_indicators.items():
                        if any(kw in left_part for kw in keywords):
                            left_types.add(ptype)
                        if any(kw in right_part for kw in keywords):
                            right_types.add(ptype)
                    
                    # If different problem types on each side, it's multiple problems
                    if left_types and right_types and not (left_types & right_types):
                        raise ValueError("Nu stiu sa raspund acum")
        
        # Check for explicit "and" conjunction with multiple problem indicators
        has_and = ' and ' in text_lower or ' si ' in text_lower
        
        if has_and:
            # Split by "and" and check each part for problem type indicators
            parts = re.split(r'\s+(?:and|si)\s+', text_lower, flags=re.IGNORECASE)
            if len(parts) >= 2:
                # Check if different problem keywords appear in different parts
                nash_keywords = ['nash', 'equilibrium', 'payoff', 'game']
                csp_keywords = ['csp', 'constraint', 'variable', 'coloring']
                minimax_keywords = ['minimax', 'alpha', 'beta', 'tree', 'leaf', 'leaves']
                strategy_keywords = ['strategy', 'n-queen', 'hanoi', 'knight', 'tour', 'backtracking']
                mdp_keywords = ['mdp', 'markov', 'value iteration', 'policy iteration', 'bellman', 'grid world']
                rl_keywords = ['q-learning', 'q learning', 'td-learning', 'td learning', 'reinforcement']
                
                keyword_sets = [nash_keywords, csp_keywords, minimax_keywords, strategy_keywords, mdp_keywords, rl_keywords]
                
                # Count how many different problem types are mentioned across parts
                types_found = 0
                for keywords in keyword_sets:
                    found_in_parts = 0
                    for part in parts:
                        if any(kw in part for kw in keywords):
                            found_in_parts += 1
                            break
                    if found_in_parts > 0:
                        types_found += 1
                
                if types_found >= 2:
                    raise ValueError("Nu stiu sa raspund acum")
        
        # Check for explicit mention of multiple distinct problem type names
        # Count how many different problem type categories are explicitly mentioned
        mentioned_categories = set()
        
        # Map to broader categories
        # Note: check for more specific keywords first to avoid confusion
        # (e.g., "alpha-beta" vs "alpha" in RL)
        if any(kw in text_lower for kw in ['nash', 'echilibru']):
            mentioned_categories.add('nash')
        if any(kw in text_lower for kw in ['csp', 'constraint', 'constrangere', 'satisfacere']):
            mentioned_categories.add('csp')
        # Minimax: check for "alpha-beta" or "alpha beta" together, NOT just "alpha" or "beta" alone
        if 'alpha-beta' in text_lower or 'alpha beta' in text_lower or 'minimax' in text_lower or 'pruning' in text_lower:
            mentioned_categories.add('minimax')
        if any(kw in text_lower for kw in ['mdp', 'markov', 'bellman', 'grid world', 'value iteration', 'policy iteration']):
            mentioned_categories.add('mdp')
        if any(kw in text_lower for kw in ['q-learning', 'q learning', 'td-learning', 'td learning', 'reinforcement']):
            mentioned_categories.add('rl')
        if any(kw in text_lower for kw in ['strips', 'add-list', 'delete-list']):
            mentioned_categories.add('strips')
        if any(kw in text_lower for kw in ['adl', 'conditional effect']):
            mentioned_categories.add('adl')
        if any(kw in text_lower for kw in ['partial order', 'pop', 'plan incomplet', 'linkuri cauzale']):
            mentioned_categories.add('pop')
        if any(kw in text_lower for kw in ['verificați plan', 'validate plan']):
            mentioned_categories.add('plan_validation')
        # For strategy problems, only count if they seem to be the main topic
        # (avoid false positives where they're just mentioned in passing)
        strategy_mentions = sum(1 for kw in ['hanoi', 'turnuri', 'n-queen', 'nqueen', 'knight', 'cal'] if kw in text_lower)
        if strategy_mentions > 0:
            mentioned_categories.add('strategy')
        
        # If 2+ distinct problem categories are mentioned, reject
        if len(mentioned_categories) >= 2:
            raise ValueError("Nu stiu sa raspund acum")
        
        # Check if multiple problem types are present (confidence threshold: 0.3)
        high_confidence_types = [
            q_type for q_type, score in scores.items() 
            if score >= 0.3
        ]
        
        # Special case: STRIPS and ADL use same extractor, so if both are high confidence, it's OK
        # Filter them out for the multiple type check
        planning_types = {QuestionType.STRIPS_ACTION_DEFINITION, QuestionType.ADL_ACTION_DEFINITION}
        high_conf_non_planning = [qt for qt in high_confidence_types if qt not in planning_types]
        planning_detected = any(qt in planning_types for qt in high_confidence_types)
        
        # If we have planning + other types, or multiple non-planning types, reject
        if planning_detected and len(high_conf_non_planning) > 0:
            raise ValueError("Nu stiu sa raspund acum")
        if len(high_conf_non_planning) > 1:
            raise ValueError("Nu stiu sa raspund acum")
        
        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score < 0.3:
            raise ValueError(
                "Nu s-a putut determina tipul problemei. Te rog include cuvinte cheie cum ar fi:\n"
                "- Nash/equilibrium/echilibru/joc pentru probleme de echilibru Nash\n"
                "- CSP/constraint/constrangere/variabile pentru probleme CSP\n"
                "- Minimax/alpha-beta/arbore de joc pentru probleme minimax\n"
                "- Strategy/strategie/n-queens/hanoi pentru selectia strategiei\n"
                "- MDP/Markov/Value Iteration/Bellman/grid world pentru probleme MDP\n"
                "- Q-learning/TD-learning/reinforcement learning pentru probleme RL\n"
                "- STRIPS/add-list/delete-list pentru definire acțiuni STRIPS\n"
                "- ADL/conditional effects pentru definire acțiuni ADL\n"
                "- Partial Order Planning/POP/plan incomplet pentru planificare cu ordine parțială\n"
                "- Verificați plan/validate plan pentru validare planuri"
            )
        
        # Extract data using the best extractor
        try:
            extractor = self.extractors[best_type]
            data = extractor.extract(text)
            
            return ParsedProblem(
                question_type=best_type,
                data=data,
                confidence=best_score,
                raw_text=text
            )
        except Exception as e:
            raise ValueError(f"Esec la parsarea problemei de tip {best_type.name}: {str(e)}")

