"""Concrete category strategy implementations."""

from .base import CategoryStrategy
from typing import Dict, Any


class AITypeForSystemStrategy(CategoryStrategy):
    """Strategy for ai_type_for_system category."""
    
    @property
    def name(self) -> str:
        return "ai_type_for_system"
    
    @property
    def instruction(self) -> str:
        return (
            "Create exactly ONE question that asks the learner to classify an agent type. "
            "Provide a CONTEXT with a detailed scenario describing the agent's behavior, decision-making process, and characteristics. "
            "Include enough details so the agent type can be inferred, but do NOT explicitly state the type."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class AlgorithmSelectionStrategy(CategoryStrategy):
    """Strategy for algorithm_selection category."""
    
    @property
    def name(self) -> str:
        return "algorithm_selection"
    
    @property
    def instruction(self) -> str:
        return (
            "Create exactly ONE question asking to choose the most appropriate algorithm "
            "(BFS, DFS, UCS, A*, Greedy Best-First, Hill-Climbing, Simulated Annealing). "
            "Provide a CONTEXT describing a problem scenario with specific characteristics (optimality requirements, heuristic availability, constraints, etc.) "
            "that make one algorithm clearly most appropriate, but do NOT name the algorithm."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class BayesianInferenceStrategy(CategoryStrategy):
    """Strategy for bayesian_inference category."""
    
    @property
    def name(self) -> str:
        return "bayesian_inference"
    
    @property
    def instruction(self) -> str:
        return (
            "Create exactly ONE numeric Bayes question asking for P(H|E). "
            "Provide a CONTEXT with prior P(H), likelihoods P(E|H) and P(E|~H), and all numerical values needed to compute the posterior. "
            "Include the numbers but do NOT state the computed P(H|E)."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class CSPArcConsistencyStrategy(CategoryStrategy):
    """Strategy for csp_arc_consistency category."""
    
    @property
    def name(self) -> str:
        return "csp_arc_consistency"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE small CSP question (≤3 variables) asking for pruned domains after AC-3. "
            "Provide a CONTEXT listing initial domains and binary constraints. Show enough detail to deduce which values get pruned, but do NOT list the final domains."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class GameTheoryNashStrategy(CategoryStrategy):
    """Strategy for game_theory_nash category."""
    
    @property
    def name(self) -> str:
        return "game_theory_nash"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE 2x2 game question asking for Nash equilibria. "
            "Provide a CONTEXT with the complete payoff matrix showing all strategy combinations and payoffs. "
            "Include the matrix so equilibria can be calculated, but do NOT state which strategy pairs are equilibria."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class HeuristicAdmissibilityComboStrategy(CategoryStrategy):
    """Strategy for heuristic_admissibility_combo category."""
    
    @property
    def name(self) -> str:
        return "heuristic_admissibility_combo"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE question asking whether max(h1,h2) or h1+h2 are admissible/consistent. "
            "Provide CONTEXT describing h1 and h2 properties (e.g., both admissible, both consistent, values, monotonicity). "
            "Include enough information to determine admissibility/consistency of the combinations, but do NOT state the conclusions."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class OntologyConceptsStrategy(CategoryStrategy):
    """Strategy for ontology_concepts category."""
    
    @property
    def name(self) -> str:
        return "ontology_concepts"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE ontology question asking for a specific concept or relationship. "
            "Provide CONTEXT with relevant axioms, assertions, and relationships that entail the answer. "
            "Include the necessary facts but do NOT explicitly state the answer."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class PartialOrderPlanningStrategy(CategoryStrategy):
    """Strategy for partial_order_planning category."""
    
    @property
    def name(self) -> str:
        return "partial_order_planning"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE partial-order planning question asking for a plan or ordering. "
            "Provide CONTEXT with initial state, goal state, and action descriptions (preconditions and effects). "
            "Include enough detail to construct a plan, but do NOT provide the plan or ordering."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class PlanningSTRIPSADLStrategy(CategoryStrategy):
    """Strategy for planning_strips_adl category."""
    
    @property
    def name(self) -> str:
        return "planning_strips_adl"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE STRIPS/ADL question asking for encoding or successor state. "
            "Provide CONTEXT with action schema (preconditions, add/delete effects) and initial state. "
            "Include all necessary information, but do NOT state the successor state or encoding."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class QLearningUpdateStrategy(CategoryStrategy):
    """Strategy for q_learning_update category."""
    
    @property
    def name(self) -> str:
        return "q_learning_update"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE Q-learning question asking for the updated Q(s,a). "
            "Provide CONTEXT with transition details: state s, action a, reward r, next state s', learning rate alpha, discount gamma, current Q(s,a), and max_a' Q(s',a'). "
            "Include all numerical values needed for the calculation, but do NOT show the computed update."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class StateRepresentationStrategy(CategoryStrategy):
    """Strategy for state_representation category."""
    
    @property
    def name(self) -> str:
        return "state_representation"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE question asking for a factored state representation or number of states. "
            "Provide CONTEXT describing the domain variables and their possible values. "
            "Include enough detail to determine the state space, but do NOT state the representation or count."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class ValueIterationStepStrategy(CategoryStrategy):
    """Strategy for value_iteration_step category."""
    
    @property
    def name(self) -> str:
        return "value_iteration_step"
    
    @property
    def instruction(self) -> str:
        return (
            "Create ONE value-iteration question asking for V_{k+1} values. "
            "Provide CONTEXT with MDP details: rewards R(s), transition probabilities P(s'|s,a), discount gamma, and current value function V_k(s). "
            "Include all numbers needed to compute V_{k+1}, but do NOT show the updated values."
        )
    
    def build_generation_messages(self, system_prompt: str) -> list:
        task = (
            f"Category: {self.name}\n"
            f"{self.instruction}\n\n"
            "Output format (Markdown):\n"
            "### Question\n"
            "- A single, clearly stated question that can be answered using information from the context.\n"
            "### Context\n"
            "- A well-structured context containing the problem description and all necessary facts.\n"
            "- Use bullet points, tables, or clear sections to organize information (especially for complex answers).\n"
            "- Include all numerical values, relationships, and details needed to answer the question.\n"
            "- Structure information so that components of complex answers can be identified and extracted.\n"
            "- Do NOT explicitly state the answer, but make it possible to deduce or extract all parts of it.\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]


class CategoryStrategyFactory:
    """Factory for creating category strategies."""
    
    _strategies = {
        "ai_type_for_system": AITypeForSystemStrategy,
        "algorithm_selection": AlgorithmSelectionStrategy,
        "bayesian_inference": BayesianInferenceStrategy,
        "csp_arc_consistency": CSPArcConsistencyStrategy,
        "game_theory_nash": GameTheoryNashStrategy,
        "heuristic_admissibility_combo": HeuristicAdmissibilityComboStrategy,
        "ontology_concepts": OntologyConceptsStrategy,
        "partial_order_planning": PartialOrderPlanningStrategy,
        "planning_strips_adl": PlanningSTRIPSADLStrategy,
        "q_learning_update": QLearningUpdateStrategy,
        "state_representation": StateRepresentationStrategy,
        "value_iteration_step": ValueIterationStepStrategy,
    }
    
    @classmethod
    def create(cls, category_name: str) -> CategoryStrategy:
        """
        Create a strategy for the given category.
        
        Args:
            category_name: Name of the category
            
        Returns:
            CategoryStrategy instance
            
        Raises:
            ValueError: If category not found
        """
        strategy_class = cls._strategies.get(category_name)
        if strategy_class is None:
            raise ValueError(f"Unknown category: {category_name}")
        return strategy_class()
    
    @classmethod
    def get_all_categories(cls) -> list[str]:
        """Get list of all available category names."""
        return list(cls._strategies.keys())

