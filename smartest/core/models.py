from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List


class QuestionType(Enum):
    STRATEGY_SELECTION = auto()
    NASH_EQUILIBRIUM = auto()
    CSP_COMPLETION = auto()
    MINIMAX_ALPHA_BETA = auto()


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
