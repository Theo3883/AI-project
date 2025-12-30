"""
Comprehensive tests for STRIPS/ADL and POP planning implementation.
Tests generators, solvers, parsers, evaluators, and end-to-end integration.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smartest.core.models import (
    QuestionType, Predicate, Action, PlanningProblem, PartialOrderPlan
)
from smartest.core.generators import (
    StripsActionGenerator, AdlActionGenerator, 
    PartialOrderPlanGenerator, PlanValidationGenerator
)
from smartest.core.solvers import (
    StripsActionFormatterSolver, AdlActionFormatterSolver,
    PartialOrderPlanningSolver, ForwardSearchPlanningSolver
)
from smartest.services.problem_parser import (
    StripsAdlExtractor, PopExtractor, PlanValidationExtractor, ProblemParser
)
from smartest.core.evaluators import PlanningEvaluator
from smartest.services.text_processing import TextProcessor
from smartest.services.qa_service import QAService


class TestPredicate:
    """Test Predicate model."""
    
    def test_predicate_creation(self):
        pred = Predicate("At", ["agent", "home"], positive=True)
        assert pred.name == "At"
        assert pred.parameters == ["agent", "home"]
        assert pred.positive == True
        print("✓ Predicate creation works")
    
    def test_predicate_string(self):
        pred_pos = Predicate("At", ["agent", "home"], positive=True)
        pred_neg = Predicate("At", ["agent", "home"], positive=False)
        assert str(pred_pos) == "At(agent, home)"
        assert str(pred_neg) == "¬At(agent, home)"
        print("✓ Predicate string representation works")
    
    def test_predicate_equality(self):
        pred1 = Predicate("At", ["agent", "home"], positive=True)
        pred2 = Predicate("At", ["agent", "home"], positive=True)
        pred3 = Predicate("At", ["agent", "store"], positive=True)
        assert pred1 == pred2
        assert pred1 != pred3
        print("✓ Predicate equality works")


class TestAction:
    """Test Action model."""
    
    def test_action_creation(self):
        action = Action(
            name="Go",
            parameters=["home", "store"],
            preconditions=[Predicate("At", ["agent", "home"])],
            add_effects=[Predicate("At", ["agent", "store"])],
            delete_effects=[Predicate("At", ["agent", "home"])]
        )
        assert action.name == "Go"
        assert len(action.preconditions) == 1
        assert len(action.add_effects) == 1
        assert len(action.delete_effects) == 1
        print("✓ Action creation works")
    
    def test_action_string(self):
        action = Action(
            name="Go",
            parameters=["home", "store"],
            preconditions=[],
            add_effects=[],
            delete_effects=[]
        )
        assert str(action) == "Go(home, store)"
        print("✓ Action string representation works")


class TestStripsActionGenerator:
    """Test STRIPS action generator."""
    
    def test_generate_question(self):
        generator = StripsActionGenerator()
        question = generator.generate()
        
        assert question.q_type == QuestionType.STRIPS_ACTION_DEFINITION
        assert "STRIPS" in question.text
        assert len(question.correct_answer) > 0
        assert "Precondiții" in question.correct_answer or "Preconditions" in question.correct_answer
        print(f"✓ STRIPS generator creates valid question: {question.title}")
    
    def test_randomization(self):
        """Test that generator produces different questions."""
        generator = StripsActionGenerator()
        questions = [generator.generate() for _ in range(10)]
        
        # Check that we get different actions or domains
        unique_texts = set(q.text for q in questions)
        assert len(unique_texts) > 3, "Generator should produce varied questions"
        print(f"✓ STRIPS generator randomization: {len(unique_texts)} unique questions out of 10")


class TestAdlActionGenerator:
    """Test ADL action generator."""
    
    def test_generate_question(self):
        generator = AdlActionGenerator()
        question = generator.generate()
        
        assert question.q_type == QuestionType.ADL_ACTION_DEFINITION
        assert "ADL" in question.text
        assert len(question.correct_answer) > 0
        print(f"✓ ADL generator creates valid question: {question.title}")
    
    def test_randomization(self):
        generator = AdlActionGenerator()
        questions = [generator.generate() for _ in range(10)]
        
        unique_texts = set(q.text for q in questions)
        assert len(unique_texts) > 2, "ADL generator should produce varied questions"
        print(f"✓ ADL generator randomization: {len(unique_texts)} unique questions out of 10")


class TestPartialOrderPlanGenerator:
    """Test POP generator."""
    
    def test_generate_question(self):
        generator = PartialOrderPlanGenerator()
        question = generator.generate()
        
        assert question.q_type == QuestionType.PARTIAL_ORDER_PLAN
        assert "partial" in question.text.lower() or "pop" in question.text.lower()
        assert len(question.correct_answer) > 0
        print(f"✓ POP generator creates valid question: {question.title}")
    
    def test_randomization(self):
        generator = PartialOrderPlanGenerator()
        questions = [generator.generate() for _ in range(10)]
        
        unique_texts = set(q.text for q in questions)
        assert len(unique_texts) > 3, "POP generator should produce varied questions"
        print(f"✓ POP generator randomization: {len(unique_texts)} unique questions out of 10")


class TestPlanValidationGenerator:
    """Test plan validation generator."""
    
    def test_generate_question(self):
        generator = PlanValidationGenerator()
        question = generator.generate()
        
        assert question.q_type == QuestionType.PLAN_VALIDATION
        assert "verific" in question.text.lower() or "valid" in question.text.lower()
        assert len(question.correct_answer) > 0
        print(f"✓ Plan validation generator creates valid question: {question.title}")
    
    def test_randomization(self):
        generator = PlanValidationGenerator()
        questions = [generator.generate() for _ in range(10)]
        
        # Should have mix of valid and invalid plans
        has_errors_count = sum(1 for q in questions if q.meta.get("has_errors", False))
        assert 2 <= has_errors_count <= 8, "Should have mix of valid and invalid plans"
        print(f"✓ Plan validation randomization: {has_errors_count} plans with errors out of 10")


class TestStripsActionFormatterSolver:
    """Test STRIPS action formatter solver."""
    
    def test_solve_simple_action(self):
        action = Action(
            name="Go",
            parameters=["home", "store"],
            preconditions=[Predicate("At", ["agent", "home"])],
            add_effects=[Predicate("At", ["agent", "store"])],
            delete_effects=[Predicate("At", ["agent", "home"])]
        )
        
        solver = StripsActionFormatterSolver()
        result = solver.solve({"action": action})
        
        assert "formatted_action" in result
        assert "Go(home, store)" in result["formatted_action"]
        assert "Precondiții" in result["formatted_action"]
        print("✓ STRIPS solver formats action correctly")


class TestAdlActionFormatterSolver:
    """Test ADL action formatter solver."""
    
    def test_solve_action_with_conditional_effects(self):
        action = Action(
            name="Load",
            parameters=["package", "truck"],
            preconditions=[Predicate("At", ["package", "loc"])],
            add_effects=[Predicate("In", ["package", "truck"])],
            delete_effects=[Predicate("At", ["package", "loc"])],
            conditional_effects=[
                ([Predicate("Heavy", ["package"])], [Predicate("Slow", ["truck"])])
            ]
        )
        
        solver = AdlActionFormatterSolver()
        result = solver.solve({"action": action})
        
        assert "formatted_action" in result
        assert "ADL" in result["formatted_action"]
        assert "condiționate" in result["formatted_action"] or "conditional" in result["formatted_action"].lower()
        print("✓ ADL solver formats action with conditional effects correctly")


class TestPartialOrderPlanningSolver:
    """Test POP solver."""
    
    def test_solve_simple_problem(self):
        problem = PlanningProblem(
            domain_name="shopping",
            objects=["agent", "home", "store1", "milk"],
            initial_state=[
                Predicate("At", ["agent", "home"]),
                Predicate("Sells", ["store1", "milk"])
            ],
            goal_state=[
                Predicate("Have", ["milk"])
            ],
            actions=[
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
        )
        
        solver = PartialOrderPlanningSolver()
        result = solver.solve({"problem": problem})
        
        assert "plan" in result
        assert "num_actions" in result
        assert result["num_actions"] >= 2  # At least Start, action(s), Finish
        print(f"✓ POP solver creates plan with {result['num_actions']} actions")


class TestForwardSearchPlanningSolver:
    """Test forward search planning solver."""
    
    def test_validate_correct_plan(self):
        problem = PlanningProblem(
            domain_name="shopping",
            objects=["agent", "home", "store1", "milk"],
            initial_state=[
                Predicate("At", ["agent", "home"]),
                Predicate("Sells", ["store1", "milk"])
            ],
            goal_state=[
                Predicate("Have", ["milk"])
            ],
            actions=[]
        )
        
        plan = [
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
        
        solver = ForwardSearchPlanningSolver()
        result = solver.solve({"problem": problem, "plan": plan})
        
        assert result["valid"] == True
        assert result["goals_achieved"] == True
        assert len(result["errors"]) == 0
        print("✓ Forward search solver validates correct plan")
    
    def test_validate_incorrect_plan(self):
        problem = PlanningProblem(
            domain_name="shopping",
            objects=["agent", "home", "store1", "milk"],
            initial_state=[
                Predicate("At", ["agent", "home"]),
                Predicate("Sells", ["store1", "milk"])
            ],
            goal_state=[
                Predicate("Have", ["milk"])
            ],
            actions=[]
        )
        
        # Incorrect plan: Try to buy before going to store
        plan = [
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
        
        solver = ForwardSearchPlanningSolver()
        result = solver.solve({"problem": problem, "plan": plan})
        
        assert result["valid"] == False
        assert len(result["errors"]) > 0
        print(f"✓ Forward search solver detects invalid plan: {result['errors'][0]}")


class TestStripsAdlExtractor:
    """Test STRIPS/ADL extractor."""
    
    def test_can_extract_strips(self):
        extractor = StripsAdlExtractor()
        text = "Descrieți operația Go(home, store) în limbajul STRIPS. Specificați precondițiile, add-list și delete-list."
        
        confidence = extractor.can_extract(text)
        assert confidence > 0.5, f"Should detect STRIPS problem (got {confidence})"
        print(f"✓ STRIPS extractor confidence: {confidence:.2f}")
    
    def test_extract_strips(self):
        extractor = StripsAdlExtractor()
        text = "Descrieți operația Go(home, store) în limbajul STRIPS pentru domeniul shopping."
        
        data = extractor.extract(text)
        assert "domain" in data
        assert data["domain"] == "shopping"
        print(f"✓ STRIPS extractor extracts domain: {data['domain']}")


class TestPopExtractor:
    """Test POP extractor."""
    
    def test_can_extract_pop(self):
        extractor = PopExtractor()
        text = "Construiți un plan incomplet folosind partial order planning pentru stare inițială At(agent, home) și obiectiv Have(milk)."
        
        confidence = extractor.can_extract(text)
        assert confidence > 0.5, f"Should detect POP problem (got {confidence})"
        print(f"✓ POP extractor confidence: {confidence:.2f}")
    
    def test_extract_pop(self):
        extractor = PopExtractor()
        text = "POP pentru shopping: stare inițială At(agent, home), obiectiv Have(milk)."
        
        data = extractor.extract(text)
        assert "domain" in data
        assert data["domain"] == "shopping"
        print(f"✓ POP extractor extracts domain: {data['domain']}")


class TestPlanValidationExtractor:
    """Test plan validation extractor."""
    
    def test_can_extract_validation(self):
        extractor = PlanValidationExtractor()
        text = "Verificați următorul plan: 1. Go(home, store) 2. Buy(milk, store). Este corect?"
        
        confidence = extractor.can_extract(text)
        assert confidence > 0.5, f"Should detect validation problem (got {confidence})"
        print(f"✓ Validation extractor confidence: {confidence:.2f}")


class TestPlanningEvaluator:
    """Test planning evaluator."""
    
    def test_evaluate_strips_complete(self):
        from smartest.core.models import Question
        
        question = Question(
            id=1,
            title="Test STRIPS",
            text="Describe Go action",
            q_type=QuestionType.STRIPS_ACTION_DEFINITION,
            topic="Planning",
            difficulty="medium",
            correct_answer="Operația Go(home, store):\nPrecondiții: At(agent, home)\nAdd-list: At(agent, store)\nDelete-list: At(agent, home)",
            explanation="",
            meta={}
        )
        
        # Use keywords that evaluator recognizes
        user_answer = "Precondițiile sunt At(agent, home). Add-list include At(agent, store). Delete-list conține At(agent, home)."
        
        evaluator = PlanningEvaluator(TextProcessor())
        result = evaluator.evaluate(question, user_answer)
        
        assert result.score == 100.0, f"Complete answer should get 100% (got {result.score}%)"
        print(f"✓ Planning evaluator scores complete answer: {result.score}%")
    
    def test_evaluate_strips_partial(self):
        from smartest.core.models import Question
        
        question = Question(
            id=1,
            title="Test STRIPS",
            text="Describe Go action",
            q_type=QuestionType.STRIPS_ACTION_DEFINITION,
            topic="Planning",
            difficulty="medium",
            correct_answer="Operația Go(home, store):\nPrecondiții: At(agent, home)\nAdd-list: At(agent, store)\nDelete-list: At(agent, home)",
            explanation="",
            meta={}
        )
        
        user_answer = "Precondițiile sunt At(agent, home). Add-list: At(agent, store)."
        
        evaluator = PlanningEvaluator(TextProcessor())
        result = evaluator.evaluate(question, user_answer)
        
        assert result.score < 100.0, "Partial answer should not get 100%"
        assert result.score >= 65.0, "Partial answer with 2/3 components should get at least 65%"
        print(f"✓ Planning evaluator scores partial answer: {result.score}%")


class TestQAServiceIntegration:
    """Test end-to-end Q&A service integration."""
    
    def test_qa_strips_question(self):
        qa_service = QAService()
        question = "Descrieți operația Go(home, store1) în limbajul STRIPS pentru domeniul shopping. Specificați precondițiile, add-list și delete-list."
        
        response = qa_service.answer_question(question)
        
        assert response.success == True
        assert response.detected_type_enum in [QuestionType.STRIPS_ACTION_DEFINITION, QuestionType.ADL_ACTION_DEFINITION]
        assert len(response.solution) > 0
        print(f"✓ Q&A service handles STRIPS question (type: {response.detected_type})")
    
    def test_qa_pop_question(self):
        qa_service = QAService()
        question = "Construiți un plan incomplet folosind partial order planning pentru shopping: stare inițială At(agent, home), Sells(store1, milk), obiectiv Have(milk)."
        
        response = qa_service.answer_question(question)
        
        assert response.success == True
        assert response.detected_type_enum == QuestionType.PARTIAL_ORDER_PLAN
        assert len(response.solution) > 0
        print(f"✓ Q&A service handles POP question (confidence: {response.confidence:.2f})")


def run_all_tests():
    """Run all test classes."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE PLANNING TESTS (STRIPS/ADL/POP)")
    print("="*60 + "\n")
    
    test_classes = [
        TestPredicate,
        TestAction,
        TestStripsActionGenerator,
        TestAdlActionGenerator,
        TestPartialOrderPlanGenerator,
        TestPlanValidationGenerator,
        TestStripsActionFormatterSolver,
        TestAdlActionFormatterSolver,
        TestPartialOrderPlanningSolver,
        TestForwardSearchPlanningSolver,
        TestStripsAdlExtractor,
        TestPopExtractor,
        TestPlanValidationExtractor,
        TestPlanningEvaluator,
        TestQAServiceIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        
        instance = test_class()
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name} FAILED: {e}")
            except Exception as e:
                print(f"✗ {method_name} ERROR: {e}")
    
    print("\n" + "="*60)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    print("="*60 + "\n")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

