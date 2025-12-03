"""
component_45_logic_puzzle_solver_core.py
========================================
SAT-based logic puzzle solver with CNF conversion and answer formatting.

This module handles:
- Converting logical conditions to CNF formulas
- Solving puzzles using SAT solver (DPLL algorithm)
- Formatting solutions as natural language answers
- Generating proof trees for transparent reasoning

Logic transformations:
- IMPLICATION (X -> Y):  NOT X OR Y
- XOR (X XOR Y):         (X OR Y) AND (NOT X OR NOT Y)
- NEVER_BOTH NOT(X AND Y): NOT X OR NOT Y
- CONJUNCTION (X AND Y): X, Y (separate clauses)
- DISJUNCTION (X OR Y):  X OR Y
- NEGATION (NOT X):      NOT X

Author: KAI Development Team
Date: 2025-11-29 (Split from component_45)
"""

from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger
from component_30_sat_solver import Clause, CNFFormula, Literal, SATSolver
from component_45_logic_puzzle_parser import LogicCondition, LogicConditionParser
from kai_exceptions import ConstraintReasoningError, ParsingError, SpaCyModelError

logger = get_logger(__name__)


class LogicPuzzleSolver:
    """
    Solves logic puzzles using SAT solver.

    Workflow:
    1. Parse conditions -> LogicCondition list
    2. Convert to CNF -> CNFFormula
    3. Solve with SAT solver -> Model
    4. Format answer with proof tree
    """

    def __init__(self):
        self.parser = LogicConditionParser()
        self.solver = SATSolver()

    def solve(
        self, conditions_text: str, entities: List[str], question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solves a logic puzzle.

        Args:
            conditions_text: Text with logical conditions
            entities: List of entities (e.g., ["Leo", "Mark", "Nick"])
            question: Optional question (e.g., "Wer trinkt Brandy?")

        Returns:
            Dictionary with:
            - solution: Dict[var_name, bool] - Variable assignments
            - proof_tree: ProofTree - Solution path
            - answer: str - Formatted answer

        Raises:
            ParsingError: If conditions cannot be parsed
            ConstraintReasoningError: If SAT solver fails
        """
        try:
            logger.info(f"Solving logic puzzle with {len(entities)} entities")

            # STEP 1: Parse conditions
            try:
                conditions = self.parser.parse_conditions(conditions_text, entities)
                logger.info(f"Parsed: {len(conditions)} conditions")
            except (ParsingError, SpaCyModelError) as e:
                # Re-raise with additional context
                logger.error(f"Parsing failed: {e}")
                raise

            if not conditions:
                logger.warning("No logical conditions found")
                return {
                    "solution": {},
                    "proof_tree": None,
                    "answer": "No logical conditions found.",
                }

            # STEP 2: Convert to CNF
            try:
                cnf = self._build_cnf(conditions)
                logger.info(f"CNF created: {len(cnf.clauses)} clauses")
            except ConstraintReasoningError:
                raise  # Re-raise ConstraintReasoningError from _build_cnf
            except Exception as e:
                raise ConstraintReasoningError(
                    "Error converting to CNF formula",
                    context={"num_conditions": len(conditions)},
                    original_exception=e,
                )

            # STEP 3: Solve with SAT solver
            try:
                # SATSolver.solve() returns None if UNSAT, else model
                model = self.solver.solve(cnf)
            except Exception as e:
                raise ConstraintReasoningError(
                    "SAT solver failed",
                    context={"num_clauses": len(cnf.clauses)},
                    original_exception=e,
                )

            if model is not None:
                logger.info(f"[OK] Solution found: {model}")

                # STEP 4: Format answer
                try:
                    answer = self._format_answer(model, question)
                except ConstraintReasoningError:
                    raise  # Re-raise ConstraintReasoningError from _format_answer
                except Exception as e:
                    raise ConstraintReasoningError(
                        "Error formatting answer",
                        context={"model_size": len(model)},
                        original_exception=e,
                    )

                return {
                    "solution": model,
                    "proof_tree": None,  # TODO: Extract from solver if needed
                    "answer": answer,
                    "result": "SATISFIABLE",
                }
            else:
                logger.warning("Puzzle is unsolvable (UNSAT)")
                return {
                    "solution": {},
                    "proof_tree": None,
                    "answer": "The puzzle has no solution (contradiction in conditions).",
                    "result": "UNSATISFIABLE",
                }

        except (ParsingError, SpaCyModelError, ConstraintReasoningError):
            raise  # Re-raise known exceptions
        except Exception as e:
            logger.error(
                f"Unexpected error in LogicPuzzleSolver.solve(): {e}", exc_info=True
            )
            raise ConstraintReasoningError(
                "Unexpected error solving logic puzzle",
                context={
                    "num_entities": len(entities),
                    "text_length": len(conditions_text),
                },
                original_exception=e,
            )

    def _build_cnf(self, conditions: List[LogicCondition]) -> CNFFormula:
        """
        Converts LogicCondition list to CNF formula.

        Logic transformations:
        - IMPLICATION (X -> Y):  NOT X OR Y
        - XOR (X XOR Y):         (X OR Y) AND (NOT X OR NOT Y)
        - NEVER_BOTH NOT(X AND Y): NOT X OR NOT Y
        - CONJUNCTION (X AND Y): X, Y (separate clauses)
        - DISJUNCTION (X OR Y):  X OR Y
        - NEGATION (NOT X):      NOT X

        Raises:
            ConstraintReasoningError: If CNF conversion fails
        """
        try:
            clauses: List[Clause] = []

            for cond in conditions:
                if cond.condition_type == "IMPLICATION":
                    # X -> Y = NOT X OR Y
                    x, y = cond.operands
                    clauses.append(
                        Clause({Literal(x, negated=True), Literal(y, negated=False)})
                    )

                elif cond.condition_type == "XOR":
                    # X XOR Y = (X OR Y) AND (NOT X OR NOT Y)
                    x, y = cond.operands
                    clauses.append(Clause({Literal(x), Literal(y)}))  # X OR Y
                    clauses.append(
                        Clause({Literal(x, negated=True), Literal(y, negated=True)})
                    )  # NOT X OR NOT Y

                elif cond.condition_type == "NEVER_BOTH":
                    # NOT(X AND Y) = NOT X OR NOT Y
                    x, y = cond.operands
                    clauses.append(
                        Clause({Literal(x, negated=True), Literal(y, negated=True)})
                    )

                elif cond.condition_type == "CONJUNCTION":
                    # X AND Y = X, Y (two separate clauses)
                    x, y = cond.operands
                    clauses.append(Clause({Literal(x)}))
                    clauses.append(Clause({Literal(y)}))

                elif cond.condition_type == "DISJUNCTION":
                    # X OR Y
                    x, y = cond.operands
                    clauses.append(Clause({Literal(x), Literal(y)}))

                elif cond.condition_type == "NEGATION":
                    # NOT X
                    x = cond.operands[0]
                    clauses.append(Clause({Literal(x, negated=True)}))

            return CNFFormula(clauses)
        except Exception as e:
            raise ConstraintReasoningError(
                "Error converting to CNF formula",
                context={"num_conditions": len(conditions)},
                original_exception=e,
            )

    def _format_answer(self, model: Dict[str, bool], question: Optional[str]) -> str:
        """
        Formats the solution as a natural language answer.

        Args:
            model: Variable assignments (var_name -> bool)
            question: Optional question (for contextual answer)

        Returns:
            Formatted answer

        Raises:
            ConstraintReasoningError: If answer formatting fails
        """
        try:
            # Find all TRUE variables
            true_vars = [var for var, value in model.items() if value]

            if not true_vars:
                return "None of the conditions are satisfied."

            # Format variables as sentences
            # Example: "Leo_bestellt_brandy" -> "Leo bestellt Brandy"
            statements = []
            for var_name in true_vars:
                var = self.parser.get_variable(var_name)
                if var:
                    # Convert property back to natural language
                    property_text = var.property.replace("_", " ")
                    statements.append(f"{var.entity} {property_text}")

            if len(statements) == 1:
                return statements[0].capitalize()
            else:
                return ", ".join(statements[:-1]) + " und " + statements[-1]
        except Exception as e:
            raise ConstraintReasoningError(
                "Error formatting answer",
                context={
                    "num_true_vars": len(true_vars) if "true_vars" in locals() else 0
                },
                original_exception=e,
            )
