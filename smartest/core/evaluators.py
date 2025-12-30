from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from .models import Question, QuestionType, EvaluationResult
from ..services.text_processing import TextProcessor


class AnswerEvaluator(ABC):
    def __init__(self, text_processor: TextProcessor) -> None:
        self.text_processor = text_processor

    @abstractmethod
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        ...


class StrategyEvaluator(AnswerEvaluator):
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)
        score = self.text_processor.keyword_score(norm_correct, norm_user)

        feedback = (
            f"Raspunsul tau este evaluat la {score:.1f}%. "
            f"Strategia recomandata este: {question.correct_answer}."
        )

        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={},
        )


class NashEvaluator(AnswerEvaluator):
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)
        score = self.text_processor.keyword_score(norm_correct, norm_user)

        feedback = (
            f"Raspunsul tau este evaluat la {score:.1f}%. "
            f"Echilibrul Nash asteptat: {question.correct_answer}."
        )

        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={},
        )


class CspEvaluator(AnswerEvaluator):
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)

        correct_pairs = self.text_processor.extract_assignments(norm_correct)
        user_pairs = self.text_processor.extract_assignments(norm_user)

        total = max(len(correct_pairs), 1)
        matches = 0
        for var, val in correct_pairs.items():
            if var in user_pairs and user_pairs[var] == val:
                matches += 1

        structural_score = 100.0 * matches / total
        text_score = self.text_processor.keyword_score(norm_correct, norm_user)
        score = 0.7 * structural_score + 0.3 * text_score

        feedback = (
            f"Ai {matches} variabile corect din {total}. "
            f"Scor total: {score:.1f}%. Asignarea de referinta este: {question.correct_answer}."
        )

        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={"matches": matches, "total": total},
        )


class MinimaxEvaluator(AnswerEvaluator):
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)

        correct_numbers = self.text_processor.extract_numbers(norm_correct)
        user_numbers = self.text_processor.extract_numbers(norm_user)

        score = 0.0
        if correct_numbers and len(correct_numbers) == len(user_numbers):
            matches = sum(1 for a, b in zip(correct_numbers, user_numbers) if a == b)
            score = 100.0 * matches / len(correct_numbers)
        else:
            score = self.text_processor.keyword_score(norm_correct, norm_user)

        feedback = (
            f"Scor: {score:.1f}%. "
            f"Raspuns de referinta: {question.correct_answer}."
        )

        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={},
        )


class MDPEvaluator(AnswerEvaluator):
    """Evaluates answers for MDP problems (Value/Policy Iteration)."""
    
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        # Check if the correct answer expects both value and policy
        correct_has_value = "V(" in question.correct_answer
        correct_has_policy = "politica" in question.correct_answer.lower() or "policy" in question.correct_answer.lower()
        
        # Check what user provided
        user_has_value = "V(" in user_answer or any(char.isdigit() for char in user_answer)
        user_has_policy = "politica" in user_answer.lower() or "policy" in user_answer.lower() or \
                         any(action in user_answer.lower() for action in ["up", "down", "left", "right", "sus", "jos", "stanga", "dreapta"])
        
        # Extract numerical values from user answer
        user_numbers = self.text_processor.extract_numbers(user_answer)
        correct_numbers = self.text_processor.extract_numbers(question.correct_answer)
        
        # Compare values with tolerance
        value_score = self._compare_with_tolerance(user_numbers, correct_numbers, tolerance=0.05)
        
        # Compare policy
        policy_score = self._compare_policy(user_answer, question.correct_answer) if user_has_policy else 0.0
        
        # Calculate final score based on what was expected and provided
        if correct_has_value and correct_has_policy:
            # Both value and policy expected
            if not user_has_value or not user_has_policy:
                # Penalize if user didn't provide both
                if user_has_value and not user_has_policy:
                    score = value_score * 0.5  # Only 50% for partial answer
                    feedback = f"Scor: {score:.1f}%. Ai furnizat doar valoarea, lipseste politica! Raspuns complet asteptat: {question.correct_answer}"
                elif user_has_policy and not user_has_value:
                    score = policy_score * 0.5  # Only 50% for partial answer
                    feedback = f"Scor: {score:.1f}%. Ai furnizat doar politica, lipseste valoarea! Raspuns complet asteptat: {question.correct_answer}"
                else:
                    score = 0.0
                    feedback = f"Scor: {score:.1f}%. Raspuns incomplet. Raspuns asteptat: {question.correct_answer}"
            else:
                # Both provided, weight them
                score = 0.6 * value_score + 0.4 * policy_score
                feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        elif correct_has_value:
            # Only value expected
            score = value_score
            feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        elif correct_has_policy:
            # Only policy expected
            score = policy_score
            feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        else:
            # Fallback
            score = value_score
            feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        
        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=user_answer,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={}
        )
    
    def _compare_with_tolerance(self, user_nums: list, correct_nums: list, tolerance: float = 0.05) -> float:
        """Compare numbers with floating point tolerance."""
        if not correct_nums:
            return self.text_processor.keyword_score(
                str(correct_nums), str(user_nums)
            )
        
        if len(user_nums) != len(correct_nums):
            return 0.0
        
        matches = 0
        for u, c in zip(user_nums, correct_nums):
            if abs(u - c) <= tolerance * abs(c) if c != 0 else abs(u - c) <= tolerance:
                matches += 1
        
        return 100.0 * matches / len(correct_nums)
    
    def _compare_policy(self, user_answer: str, correct_answer: str) -> float:
        """Compare policy actions."""
        user_lower = user_answer.lower()
        correct_lower = correct_answer.lower()
        
        actions = ["up", "down", "left", "right", "sus", "jos", "stanga", "dreapta"]
        
        user_actions = [a for a in actions if a in user_lower]
        correct_actions = [a for a in actions if a in correct_lower]
        
        if not correct_actions:
            return 100.0
        
        # Check if any user action matches correct action
        # (allowing for Romanian/English equivalents)
        equivalents = {
            "up": "sus", "down": "jos", "left": "stanga", "right": "dreapta",
            "sus": "up", "jos": "down", "stanga": "left", "dreapta": "right"
        }
        
        for user_action in user_actions:
            if user_action in correct_actions:
                return 100.0
            equiv = equivalents.get(user_action)
            if equiv and equiv in correct_actions:
                return 100.0
        
        return 0.0


class RLEvaluator(AnswerEvaluator):
    """Evaluates answers for RL problems (Q-learning, TD-learning)."""
    
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        # Extract numerical values from user answer
        user_numbers = self.text_processor.extract_numbers(user_answer)
        correct_numbers = self.text_processor.extract_numbers(question.correct_answer)
        
        # For RL, we're more lenient with numerical precision
        numerical_score = self._compare_with_tolerance(user_numbers, correct_numbers, tolerance=0.1)
        
        # Also check for policy if present
        policy_score = 0.0
        if "politica" in user_answer.lower() or "policy" in user_answer.lower() or "π" in user_answer:
            policy_score = self._compare_policy(user_answer, question.correct_answer)
        
        # Check for Q-values format
        q_format_score = 0.0
        if "q(" in user_answer.lower():
            q_format_score = 20.0  # Bonus for correct format
        
        # Combine scores
        if policy_score > 0:
            score = 0.5 * numerical_score + 0.4 * policy_score + 0.1 * q_format_score
        else:
            score = 0.9 * numerical_score + 0.1 * q_format_score
        
        feedback = f"Scor: {score:.1f}%. Raspuns asteptat: {question.correct_answer}"
        
        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=user_answer,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={}
        )
    
    def _compare_with_tolerance(self, user_nums: list, correct_nums: list, tolerance: float = 0.1) -> float:
        """Compare numbers with floating point tolerance."""
        if not correct_nums:
            return 50.0  # Partial credit if no numbers expected
        
        if len(user_nums) < len(correct_nums):
            return 0.0
        
        matches = 0
        for i, c in enumerate(correct_nums):
            if i < len(user_nums):
                u = user_nums[i]
                if abs(u - c) <= tolerance * abs(c) if c != 0 else abs(u - c) <= tolerance:
                    matches += 1
        
        return 100.0 * matches / len(correct_nums)
    
    def _compare_policy(self, user_answer: str, correct_answer: str) -> float:
        """Compare policy actions."""
        user_lower = user_answer.lower()
        correct_lower = correct_answer.lower()
        
        actions = ["up", "down", "left", "right", "sus", "jos", "stanga", "dreapta"]
        
        user_actions = [a for a in actions if a in user_lower]
        correct_actions = [a for a in actions if a in correct_lower]
        
        if not correct_actions:
            return 100.0
        
        # Check if actions overlap
        equivalents = {
            "up": "sus", "down": "jos", "left": "stanga", "right": "dreapta",
            "sus": "up", "jos": "down", "stanga": "left", "dreapta": "right"
        }
        
        matches = 0
        for correct_action in correct_actions:
            if correct_action in user_actions:
                matches += 1
            else:
                equiv = equivalents.get(correct_action)
                if equiv and equiv in user_actions:
                    matches += 1
        
        return 100.0 * matches / len(correct_actions) if correct_actions else 100.0


class PlanningEvaluator(AnswerEvaluator):
    """Evaluator for planning problems (STRIPS/ADL/POP/Validation)."""
    
    def evaluate(self, question: Question, user_answer: str) -> EvaluationResult:
        norm_correct = self.text_processor.normalize(question.correct_answer)
        norm_user = self.text_processor.normalize(user_answer)
        
        q_type = question.q_type
        
        if q_type in [QuestionType.STRIPS_ACTION_DEFINITION, QuestionType.ADL_ACTION_DEFINITION]:
            return self._evaluate_action_definition(question, user_answer, norm_correct, norm_user)
        elif q_type == QuestionType.PARTIAL_ORDER_PLAN:
            return self._evaluate_pop(question, user_answer, norm_correct, norm_user)
        elif q_type == QuestionType.PLAN_VALIDATION:
            return self._evaluate_validation(question, user_answer, norm_correct, norm_user)
        else:
            # Fallback
            score = self.text_processor.keyword_score(norm_correct, norm_user)
            return EvaluationResult(
                question_id=question.id,
                score=score,
                normalized_user_answer=norm_user,
                feedback=f"Scor: {score:.1f}%",
                correct_answer=question.correct_answer,
                extra_data={}
            )
    
    def _evaluate_action_definition(self, question: Question, user_answer: str, 
                                    norm_correct: str, norm_user: str) -> EvaluationResult:
        """Evaluate STRIPS/ADL action definitions."""
        score = 0.0
        feedback_parts = []
        
        # Check for preconditions (30%)
        precondition_keywords = ['preconditi', 'precond', 'before', 'inainte', 'precondi']
        has_preconditions = any(kw in norm_user for kw in precondition_keywords)
        
        if has_preconditions:
            score += 30.0
            feedback_parts.append("✓ Precondițiile sunt menționate")
        else:
            feedback_parts.append("✗ Lipsesc precondițiile")
        
        # Check for add-list (35%)
        add_keywords = ['add-list', 'add list', 'add', 'adaug', 'include', 'contine', 'devine adevarat', 'becomes true', 'efecte pozitive']
        has_add_list = any(kw in norm_user for kw in add_keywords)
        
        # Also check if user mentions predicates that should be added
        if has_add_list or self._check_predicates_mentioned(norm_correct, norm_user, 'add'):
            score += 35.0
            feedback_parts.append("✓ Add-list este menționat")
        else:
            feedback_parts.append("✗ Lipsește add-list")
        
        # Check for delete-list (35%)
        delete_keywords = ['delete-list', 'delete list', 'delete', 'delet', 'sterge', 'contine', 'include', 'devine fals', 'becomes false', 'efecte negative']
        has_delete_list = any(kw in norm_user for kw in delete_keywords)
        
        if has_delete_list or self._check_predicates_mentioned(norm_correct, norm_user, 'delete'):
            score += 35.0
            feedback_parts.append("✓ Delete-list este menționat")
        else:
            feedback_parts.append("✗ Lipsește delete-list")
        
        # Bonus for ADL conditional effects
        if question.q_type == QuestionType.ADL_ACTION_DEFINITION:
            conditional_keywords = ['cand', 'when', 'atunci', 'then', 'conditional', 'conditional']
            if any(kw in norm_user for kw in conditional_keywords):
                feedback_parts.append("✓ Efecte condiționate menționate")
        
        feedback = f"Scor: {score:.1f}%\n" + "\n".join(feedback_parts)
        
        if score < 100:
            feedback += f"\n\nRăspuns așteptat:\n{question.correct_answer}"
        
        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={"preconditions": has_preconditions, "add_list": has_add_list, "delete_list": has_delete_list}
        )
    
    def _evaluate_pop(self, question: Question, user_answer: str, 
                     norm_correct: str, norm_user: str) -> EvaluationResult:
        """Evaluate partial order plans."""
        score = 0.0
        feedback_parts = []
        
        # Check for minimum number of actions (20%)
        # Count action mentions (patterns like "Go", "Buy", "FromTable", etc.)
        import re
        action_pattern = r'\b[A-Z][a-z]+\s*\('
        actions_found = re.findall(action_pattern, user_answer)
        num_actions = len(actions_found)
        
        if num_actions >= 3:
            score += 20.0
            feedback_parts.append(f"✓ Minim 3 acțiuni prezente ({num_actions} găsite)")
        else:
            feedback_parts.append(f"✗ Insuficiente acțiuni ({num_actions} găsite, minim 3 necesare)")
        
        # Check for ordering/precedence (30%)
        ordering_keywords = ['<', 'inainte', 'before', 'precedent', 'ordine', 'ordering']
        has_ordering = any(kw in norm_user for kw in ordering_keywords)
        
        if has_ordering:
            score += 30.0
            feedback_parts.append("✓ Ordonare parțială specificată")
        else:
            feedback_parts.append("✗ Lipsește specificarea ordinii parțiale")
        
        # Check for causal links (30%)
        causal_keywords = ['cauzal', 'causal', 'link', 'legatur', 'produce', 'provides']
        has_causal_links = any(kw in norm_user for kw in causal_keywords) or '-->' in user_answer
        
        if has_causal_links:
            score += 30.0
            feedback_parts.append("✓ Linkuri cauzale menționate")
        else:
            feedback_parts.append("✗ Lipsesc linkurile cauzale")
        
        # Check if goals are addressed (20%)
        goal_keywords = ['obiectiv', 'goal', 'final', 'finish']
        has_goals = any(kw in norm_user for kw in goal_keywords)
        
        if has_goals:
            score += 20.0
            feedback_parts.append("✓ Obiectivele sunt abordate")
        else:
            feedback_parts.append("✗ Obiectivele nu sunt menționate")
        
        feedback = f"Scor: {score:.1f}%\n" + "\n".join(feedback_parts)
        
        if score < 80:
            feedback += f"\n\nRăspuns așteptat:\n{question.correct_answer}"
        
        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={"num_actions": num_actions, "has_ordering": has_ordering, "has_causal_links": has_causal_links}
        )
    
    def _evaluate_validation(self, question: Question, user_answer: str, 
                            norm_correct: str, norm_user: str) -> EvaluationResult:
        """Evaluate plan validation answers."""
        # Check if user correctly identified validity
        is_correct_valid = ('da' in norm_user and 'da' in norm_correct) or \
                          ('nu' in norm_user and 'nu' in norm_correct) or \
                          ('yes' in norm_user and 'yes' in norm_correct.lower()) or \
                          ('no' in norm_user and 'no' in norm_correct.lower())
        
        score = 0.0
        feedback_parts = []
        
        if is_correct_valid:
            score += 50.0
            feedback_parts.append("✓ Validitatea planului identificată corect")
        else:
            feedback_parts.append("✗ Validitatea planului identificată incorect")
        
        # Check for error identification (if plan has errors)
        if 'erori' in norm_correct or 'error' in norm_correct.lower():
            error_keywords = ['eroare', 'error', 'preconditi', 'obiectiv', 'goal']
            has_error_description = any(kw in norm_user for kw in error_keywords)
            
            if has_error_description:
                score += 50.0
                feedback_parts.append("✓ Erorile sunt descrise")
            else:
                feedback_parts.append("✗ Erorile nu sunt descrise")
        else:
            # Plan is valid, user should confirm
            if is_correct_valid:
                score += 50.0
        
        feedback = f"Scor: {score:.1f}%\n" + "\n".join(feedback_parts)
        
        if score < 100:
            feedback += f"\n\nRăspuns așteptat:\n{question.correct_answer}"
        
        return EvaluationResult(
            question_id=question.id,
            score=score,
            normalized_user_answer=norm_user,
            feedback=feedback,
            correct_answer=question.correct_answer,
            extra_data={"is_correct_valid": is_correct_valid}
        )
    
    def _check_predicates_mentioned(self, norm_correct: str, norm_user: str, list_type: str) -> bool:
        """Check if specific predicates from correct answer are mentioned in user answer."""
        # Simple heuristic: check if key predicate names appear
        import re
        predicate_pattern = r'\b[A-Z][a-z]+\('
        predicates_in_correct = set(re.findall(predicate_pattern, norm_correct))
        predicates_in_user = set(re.findall(predicate_pattern, norm_user))
        
        # Check overlap
        if predicates_in_correct and predicates_in_user:
            overlap = len(predicates_in_correct & predicates_in_user)
            return overlap >= len(predicates_in_correct) * 0.5  # At least 50% overlap
        
        return False


class EvaluatorFactory:
    def __init__(self, text_processor: TextProcessor) -> None:
        self.text_processor = text_processor
        self._mapping: Dict[QuestionType, type[AnswerEvaluator]] = {
            QuestionType.STRATEGY_SELECTION: StrategyEvaluator,
            QuestionType.NASH_EQUILIBRIUM: NashEvaluator,
            QuestionType.CSP_COMPLETION: CspEvaluator,
            QuestionType.MINIMAX_ALPHA_BETA: MinimaxEvaluator,
            QuestionType.VALUE_ITERATION: MDPEvaluator,
            QuestionType.POLICY_ITERATION: MDPEvaluator,
            QuestionType.Q_LEARNING: RLEvaluator,
            QuestionType.TD_LEARNING: RLEvaluator,
            QuestionType.RL_PARAMETERS: RLEvaluator,
            QuestionType.MDP_COMPARISON: MDPEvaluator,
            QuestionType.STRIPS_ACTION_DEFINITION: PlanningEvaluator,
            QuestionType.ADL_ACTION_DEFINITION: PlanningEvaluator,
            QuestionType.PARTIAL_ORDER_PLAN: PlanningEvaluator,
            QuestionType.PLAN_VALIDATION: PlanningEvaluator,
        }

    def get_evaluator(self, q_type: QuestionType) -> AnswerEvaluator:
        cls = self._mapping[q_type]
        return cls(self.text_processor)
