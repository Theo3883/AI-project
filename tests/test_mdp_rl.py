"""
Unit and integration tests for MDP and Reinforcement Learning components.
"""

from smartest.core.models import (
    GridWorld, MDPState, Transition, RLParameters, QuestionType
)
from smartest.core.solvers import (
    ValueIterationSolver, PolicyIterationSolver, 
    QLearningSolver, TDLearningSolver
)
from smartest.core.generators import (
    ValueIterationGenerator, PolicyIterationGenerator,
    QLearningGenerator, TDLearningGenerator, RLParametersGenerator
)
from smartest.services.problem_parser import MDPExtractor, RLExtractor


def test_value_iteration_solver():
    """Test Value Iteration solver with simple grid."""
    # Create a simple 2x2 grid
    states = {
        (0, 0): MDPState(0, 0, False, -0.04),
        (0, 1): MDPState(0, 1, True, 1.0),
        (1, 0): MDPState(1, 0, False, -0.04),
        (1, 1): MDPState(1, 1, True, -1.0),
    }
    
    grid = GridWorld(
        rows=2,
        cols=2,
        states=states,
        discount_factor=0.9,
        transition_probs={"intended": 1.0, "perpendicular": 0.0},
        walls=[]
    )
    
    solver = ValueIterationSolver()
    result = solver.solve({"grid": grid, "iterations": 10})
    
    assert "values" in result
    assert "policy" in result
    assert result["values"][(0, 1)] == 1.0  # Terminal state
    assert result["values"][(1, 1)] == -1.0  # Terminal state
    print("✓ Value Iteration test passed")


def test_policy_iteration_solver():
    """Test Policy Iteration solver."""
    # Create a simple 2x2 grid
    states = {
        (0, 0): MDPState(0, 0, False, -0.04),
        (0, 1): MDPState(0, 1, True, 1.0),
        (1, 0): MDPState(1, 0, False, -0.04),
        (1, 1): MDPState(1, 1, True, -1.0),
    }
    
    grid = GridWorld(
        rows=2,
        cols=2,
        states=states,
        discount_factor=0.9,
        transition_probs={"intended": 1.0, "perpendicular": 0.0},
        walls=[]
    )
    
    solver = PolicyIterationSolver()
    result = solver.solve({"grid": grid})
    
    assert "values" in result
    assert "policy" in result
    assert "iterations" in result
    assert result["iterations"] > 0
    print("✓ Policy Iteration test passed")


def test_qlearning_solver():
    """Test Q-learning solver."""
    transitions = [
        Transition(state=(0, 0), action="right", next_state=(0, 1), reward=0.0),
        Transition(state=(0, 1), action="right", next_state=(0, 2), reward=0.0),
        Transition(state=(0, 2), action="down", next_state=(1, 2), reward=1.0),
    ]
    
    params = RLParameters(alpha=0.1, gamma=0.9, epsilon=0.1)
    
    solver = QLearningSolver()
    result = solver.solve({
        "transitions": transitions,
        "parameters": params,
        "initial_q": {}
    })
    
    assert "q_values" in result
    assert "policy" in result
    assert "parameters_explanation" in result
    assert len(result["q_values"]) > 0
    print("✓ Q-learning test passed")


def test_tdlearning_solver():
    """Test TD-learning solver."""
    transitions = [
        Transition(state=(0, 0), action="right", next_state=(0, 1), reward=-0.04),
        Transition(state=(0, 1), action="right", next_state=(0, 2), reward=-0.04),
        Transition(state=(0, 2), action="right", next_state=(0, 3), reward=1.0),
    ]
    
    params = RLParameters(alpha=0.1, gamma=0.9, epsilon=0.0)
    
    solver = TDLearningSolver()
    result = solver.solve({
        "transitions": transitions,
        "parameters": params,
        "initial_v": {}
    })
    
    assert "values" in result
    assert "td_errors" in result
    assert len(result["values"]) > 0
    assert len(result["td_errors"]) == len(transitions)
    print("✓ TD-learning test passed")


def test_value_iteration_generator():
    """Test Value Iteration question generator."""
    generator = ValueIterationGenerator()
    question = generator.generate("medium")
    
    assert question.q_type == QuestionType.VALUE_ITERATION
    assert question.topic == "MDP / Value Iteration"
    assert "grid" in question.meta
    assert "solution" in question.meta
    assert len(question.text) > 0
    assert len(question.correct_answer) > 0
    print("✓ Value Iteration generator test passed")


def test_qlearning_generator():
    """Test Q-learning question generator."""
    generator = QLearningGenerator()
    question = generator.generate("medium")
    
    assert question.q_type == QuestionType.Q_LEARNING
    assert question.topic == "Reinforcement Learning / Q-learning"
    assert "transitions" in question.meta
    assert "params" in question.meta
    assert len(question.text) > 0
    print("✓ Q-learning generator test passed")


def test_rl_parameters_generator():
    """Test RL parameters question generator."""
    generator = RLParametersGenerator()
    question = generator.generate("medium")
    
    assert question.q_type == QuestionType.RL_PARAMETERS
    assert "alpha" in question.text.lower()
    assert "gamma" in question.text.lower()
    assert "epsilon" in question.text.lower()
    print("✓ RL parameters generator test passed")


def test_mdp_extractor():
    """Test MDP extractor for parsing."""
    extractor = MDPExtractor()
    
    text = "Grid 3x4 cu gamma=0.9, reward la (0,3)=1.0 si (1,3)=-1.0"
    confidence = extractor.can_extract(text)
    
    assert confidence > 0.3
    
    data = extractor.extract(text)
    assert "grid" in data
    assert data["grid"].rows == 3
    assert data["grid"].cols == 4
    assert data["grid"].discount_factor == 0.9
    print("✓ MDP extractor test passed")


def test_rl_extractor():
    """Test RL extractor for parsing."""
    extractor = RLExtractor()
    
    text = "Q-learning cu alpha=0.1, gamma=0.9, epsilon=0.1. Tranzitie: s=(0,0), a=right, s'=(0,1), r=0.5"
    confidence = extractor.can_extract(text)
    
    assert confidence > 0.3
    
    data = extractor.extract(text)
    assert "parameters" in data
    assert data["parameters"].alpha == 0.1
    assert data["parameters"].gamma == 0.9
    assert "transitions" in data
    assert len(data["transitions"]) > 0
    print("✓ RL extractor test passed")


def test_integration_value_iteration():
    """Integration test: generate question, solve, evaluate."""
    # Generate question
    generator = ValueIterationGenerator()
    question = generator.generate("easy")
    
    # Check question structure
    assert question.correct_answer is not None
    assert question.explanation is not None
    assert question.meta["grid"] is not None
    
    # Verify solution exists
    solution = question.meta["solution"]
    assert "values" in solution
    assert "policy" in solution
    
    print("✓ Integration test (Value Iteration) passed")


def test_integration_qlearning():
    """Integration test: generate question, solve, evaluate."""
    # Generate question
    generator = QLearningGenerator()
    question = generator.generate("easy")
    
    # Check question structure
    assert question.correct_answer is not None
    assert "Q(" in question.correct_answer  # Should contain Q-values
    
    # Verify solution exists
    solution = question.meta["solution"]
    assert "q_values" in solution
    assert "policy" in solution
    
    print("✓ Integration test (Q-learning) passed")


def run_all_tests():
    """Run all tests."""
    print("Running MDP and RL tests...\n")
    
    # Solver tests
    print("Testing Solvers:")
    test_value_iteration_solver()
    test_policy_iteration_solver()
    test_qlearning_solver()
    test_tdlearning_solver()
    
    # Generator tests
    print("\nTesting Generators:")
    test_value_iteration_generator()
    test_qlearning_generator()
    test_rl_parameters_generator()
    
    # Extractor tests
    print("\nTesting Extractors:")
    test_mdp_extractor()
    test_rl_extractor()
    
    # Integration tests
    print("\nTesting Integration:")
    test_integration_value_iteration()
    test_integration_qlearning()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    run_all_tests()

