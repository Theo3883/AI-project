from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..core.models import QuestionType, NashGame, CspInstance, CspVariable, CspConstraint, GameTreeNode


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
        has_payoffs = bool(re.search(r'\(\s*\d+\s*,\s*\d+\s*\)', text))
        
        confidence = (keyword_matches / len(self.KEYWORDS)) * 0.7
        if has_payoffs:
            confidence += 0.3
            
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


class ProblemParser:
    """Main parser that coordinates type detection and extraction."""
    
    def __init__(self) -> None:
        self.extractors: Dict[QuestionType, ProblemExtractor] = {
            QuestionType.NASH_EQUILIBRIUM: NashGameExtractor(),
            QuestionType.CSP_COMPLETION: CspExtractor(),
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxExtractor(),
            QuestionType.STRATEGY_SELECTION: StrategyExtractor(),
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
                
                keyword_sets = [nash_keywords, csp_keywords, minimax_keywords, strategy_keywords]
                
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
        if any(kw in text_lower for kw in ['nash', 'echilibru']):
            mentioned_categories.add('nash')
        if any(kw in text_lower for kw in ['csp', 'constraint', 'constrangere', 'satisfacere']):
            mentioned_categories.add('csp')
        if any(kw in text_lower for kw in ['minimax', 'alpha-beta', 'alpha beta', 'pruning']):
            mentioned_categories.add('minimax')
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
        
        if len(high_confidence_types) > 1:
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
                "- Strategy/strategie/n-queens/hanoi pentru selectia strategiei"
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

