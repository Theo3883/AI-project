from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List


class QuestionType(Enum):
    STRATEGY_SELECTION = auto()
    NASH_EQUILIBRIUM = auto()
    CSP_COMPLETION = auto()
    MINIMAX_ALPHA_BETA = auto()
    # MDP and Reinforcement Learning types
    VALUE_ITERATION = auto()
    POLICY_ITERATION = auto()
    Q_LEARNING = auto()
    TD_LEARNING = auto()
    RL_PARAMETERS = auto()
    MDP_COMPARISON = auto()


@dataclass
class Question:
    id: int
    title: str
    text: str
    q_type: QuestionType
    topic: str
    difficulty: str
    correct_answer: str
    explanation: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    question_id: int
    score: float
    normalized_user_answer: str
    feedback: str
    correct_answer: str
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NashGame:
    row_strategies: List[str]
    col_strategies: List[str]
    payoffs: List[List[tuple[int, int]]]


@dataclass
class CspVariable:
    name: str
    domain: List[Any]


@dataclass
class CspConstraint:
    var1: str
    var2: str
    relation: str = "!="


@dataclass
class CspInstance:
    variables: Dict[str, CspVariable]
    constraints: List[CspConstraint]
    partial_assignment: Dict[str, Any]


@dataclass
class GameTreeNode:
    value: int | None = None
    children: List["GameTreeNode"] = field(default_factory=list)
    is_max_player: bool = True


# MDP and Reinforcement Learning Models

@dataclass
class MDPState:
    """Represents a state in an MDP grid world."""
    row: int
    col: int
    is_terminal: bool = False
    reward: float = 0.0


@dataclass
class GridWorld:
    """Represents an MDP grid world environment."""
    rows: int
    cols: int
    states: Dict[tuple[int, int], MDPState]
    discount_factor: float  # gamma
    transition_probs: Dict[str, float]  # e.g., {"intended": 0.8, "perpendicular": 0.1}
    walls: List[tuple[int, int]] = field(default_factory=list)
    
    def get_neighbors(self, state: tuple[int, int], action: str) -> List[tuple[tuple[int, int], float]]:
        """
        Returns list of (next_state, probability) tuples for a given state and action.
        
        Takes into account stochastic transitions where the agent might end up in
        perpendicular directions with some probability.
        """
        row, col = state
        
        # Define action directions
        action_deltas = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }
        
        # Define perpendicular actions
        perpendicular_actions = {
            "up": ["left", "right"],
            "down": ["left", "right"],
            "left": ["up", "down"],
            "right": ["up", "down"]
        }
        
        if action not in action_deltas:
            return [(state, 1.0)]
        
        neighbors = []
        
        # Intended direction
        intended_prob = self.transition_probs.get("intended", 0.8)
        dr, dc = action_deltas[action]
        next_state = (row + dr, col + dc)
        
        # Check if next state is valid
        if (0 <= next_state[0] < self.rows and 
            0 <= next_state[1] < self.cols and 
            next_state not in self.walls):
            neighbors.append((next_state, intended_prob))
        else:
            # Stay in place if hitting wall or boundary
            neighbors.append((state, intended_prob))
        
        # Perpendicular directions
        perp_prob = self.transition_probs.get("perpendicular", 0.1)
        for perp_action in perpendicular_actions.get(action, []):
            dr, dc = action_deltas[perp_action]
            next_state = (row + dr, col + dc)
            
            if (0 <= next_state[0] < self.rows and 
                0 <= next_state[1] < self.cols and 
                next_state not in self.walls):
                neighbors.append((next_state, perp_prob))
            else:
                # Stay in place if hitting wall or boundary
                neighbors.append((state, perp_prob))
        
        return neighbors


@dataclass
class QTable:
    """Q-value table for Q-learning."""
    values: Dict[tuple[tuple[int, int], str], float] = field(default_factory=dict)  # (state, action) -> Q-value
    
    def get(self, state: tuple[int, int], action: str, default: float = 0.0) -> float:
        """Get Q-value for state-action pair."""
        return self.values.get((state, action), default)
    
    def set(self, state: tuple[int, int], action: str, value: float) -> None:
        """Set Q-value for state-action pair."""
        self.values[(state, action)] = value
    
    def get_best_action(self, state: tuple[int, int], actions: List[str]) -> str:
        """Get action with highest Q-value for given state."""
        best_action = actions[0]
        best_value = self.get(state, best_action)
        
        for action in actions[1:]:
            value = self.get(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action


@dataclass
class Transition:
    """Represents a transition (s, a, s', r) in RL."""
    state: tuple[int, int]
    action: str
    next_state: tuple[int, int]
    reward: float


@dataclass
class RLParameters:
    """Parameters for RL algorithms."""
    alpha: float  # learning rate
    gamma: float  # discount factor
    epsilon: float  # exploration rate
