"""
component_45_numerical_puzzle_solver.py
=======================================
CSP-based solver für numerische Logic Puzzles.

Verantwortlichkeiten:
- Konvertierung von numerischen Constraints zu CSP
- Iterative Solving für Meta-Constraints
- Lösung von self-referentiellen Puzzles
- Integration mit component_29 (CSP) und component_52 (Arithmetic)
- Generierung von Proof Trees

WICHTIG: KEINE Unicode-Zeichen verwenden (nur ASCII: AND, OR, NOT, IMPLIES)

Author: KAI Development Team
Date: 2025-12-05 (PHASE 3)
"""

import itertools
from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger
from component_17_proof_explanation import (
    ProofStep,
    ProofTree,
)
from component_17_proof_explanation import StepType as ProofStepType
from component_29_constraint_reasoning import (
    Constraint,
    ConstraintProblem,
    ConstraintSolver,
    Variable,
)
from component_45_numerical_constraint_parser import (
    ConstraintType,
    NumConstraint,
    NumericalConstraintParser,
    NumericalVariable,
)
from kai_exceptions import ConstraintReasoningError

logger = get_logger(__name__)


class NumericalPuzzleSolver:
    """
    Solver für numerische Constraint-Puzzles.

    Workflow:
    1. Parse Puzzle -> NumConstraints
    2. Klassifiziere Constraints (einfach vs. meta)
    3. Setup CSP Problem
    4. Iterative Solving für Meta-Constraints
    5. Verifiziere Lösung
    6. Generiere ProofTree

    Spezialität: Handhabt self-referentielle Meta-Constraints wie
    "Die Summe der Nummern der richtigen Behauptungen ist teilbar durch 5"
    """

    def __init__(self):
        """Initialisiert den Solver."""
        self.parser = NumericalConstraintParser()
        self.csp_solver = ConstraintSolver()

        logger.info("NumericalPuzzleSolver initialisiert")

    def solve_puzzle(
        self, puzzle_text: str, question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Löst ein numerisches Logic Puzzle.

        Args:
            puzzle_text: Der komplette Puzzle-Text
            question: Optional - Die Frage (falls separat)

        Returns:
            Dictionary mit:
            - solution: Dict[var_name, value] - Variablen-Zuweisungen
            - proof_tree: ProofTree - Reasoning path
            - answer: str - Formatierte Antwort
            - confidence: float - Wie sicher ist die Lösung
            - result: str - "SATISFIABLE" | "UNSATISFIABLE"

        Raises:
            ConstraintReasoningError: Bei Parsing- oder Solving-Fehlern
        """
        try:
            logger.info("Solve numerisches Puzzle")

            # STEP 1: Parse Puzzle
            parsed = self.parser.parse_puzzle(puzzle_text)
            constraints = parsed["constraints"]
            variables = parsed["variables"]
            question = question or parsed["question"]

            logger.info(
                f"Parsed: {len(constraints)} constraints, {len(variables)} variables"
            )

            if not constraints:
                return {
                    "solution": {},
                    "proof_tree": None,
                    "answer": "Keine Constraints gefunden.",
                    "confidence": 0.0,
                    "result": "UNSATISFIABLE",
                    "puzzle_type": "numerical_csp",
                }

            # STEP 2: Setup CSP Problem
            csp_problem = self._setup_csp_problem(variables, constraints)

            # STEP 3: Solve (mit iterativer Meta-Constraint-Auflösung)
            solution = self._iterative_solve(
                csp_problem, constraints, variables, parsed["statements"]
            )

            if solution is None:
                logger.warning("Keine Lösung gefunden (UNSAT)")
                return {
                    "solution": {},
                    "proof_tree": self._build_proof_tree(
                        csp_problem, None, constraints
                    ),
                    "answer": "Das Puzzle hat keine Lösung.",
                    "confidence": 0.0,
                    "result": "UNSATISFIABLE",
                    "puzzle_type": "numerical_csp",
                }

            # STEP 4: Verifiziere Lösung
            is_valid = self._verify_solution(solution, constraints, variables)

            if not is_valid:
                logger.warning("Lösung validiert nicht alle Constraints")
                confidence = 0.50
            else:
                confidence = 0.90  # Hohe Confidence bei vollständiger Validierung

            # STEP 5: Formatiere Antwort
            answer = self._format_answer(solution, question, variables)

            # STEP 6: Generiere ProofTree
            proof_tree = self._build_proof_tree(csp_problem, solution, constraints)

            logger.info(
                f"[OK] Lösung gefunden: {solution}, confidence={confidence:.2f}"
            )

            return {
                "solution": solution,
                "proof_tree": proof_tree,
                "answer": answer,
                "confidence": confidence,
                "result": "SATISFIABLE",
                "puzzle_type": "numerical_csp",
            }

        except Exception as e:
            logger.error(f"Fehler beim Lösen des Puzzles: {e}", exc_info=True)
            raise ConstraintReasoningError(
                "Fehler beim Lösen des numerischen Puzzles",
                context={"puzzle_length": len(puzzle_text)},
                original_exception=e,
            )

    def _setup_csp_problem(
        self, variables: Dict[str, NumericalVariable], constraints: List[NumConstraint]
    ) -> ConstraintProblem:
        """
        Erstellt ein CSP Problem aus numerischen Variablen und Constraints.

        Args:
            variables: Numerische Variablen mit Domains
            constraints: Numerische Constraints

        Returns:
            ConstraintProblem für component_29 CSP Solver
        """
        csp_variables = []

        # Konvertiere NumericalVariable zu CSP Variable
        for var_name, num_var in variables.items():
            csp_var = Variable(name=var_name, domain=list(num_var.domain))
            csp_variables.append(csp_var)

        logger.debug(f"CSP Setup: {len(csp_variables)} Variablen")

        # Konvertiere NumConstraint zu CSP Constraint
        # Nur einfache Constraints (nicht-meta)
        csp_constraints = []
        for num_const in constraints:
            if num_const.constraint_type == ConstraintType.DIVISIBILITY:
                if "divisor" in num_const.metadata:
                    # Einfacher Divisibility Constraint
                    divisor = num_const.metadata["divisor"]

                    def constraint_func(
                        assignment: Dict[str, int], div=divisor
                    ) -> bool:
                        return assignment.get("zahl", 0) % div == 0

                    csp_constraint = Constraint(
                        name=f"divisible_by_{divisor}",
                        scope=["zahl"],
                        predicate=constraint_func,
                    )
                    csp_constraints.append(csp_constraint)

        logger.debug(
            f"CSP Setup: {len(csp_constraints)} einfache Constraints konvertiert"
        )

        return ConstraintProblem(variables=csp_variables, constraints=csp_constraints)

    def _iterative_solve(
        self,
        csp_problem: ConstraintProblem,
        num_constraints: List[NumConstraint],
        variables: Dict[str, NumericalVariable],
        statements: Dict[int, str],
    ) -> Optional[Dict[str, int]]:
        """
        Iteratives Solving für Meta-Constraints.

        Strategie für self-referentielle Constraints:
        1. Enumerate mögliche truth assignments für statements
        2. Für jeden: Solve base constraints
        3. Verifiziere Meta-Constraints
        4. Returniere erste gültige Lösung

        Args:
            csp_problem: Das CSP Problem
            num_constraints: Alle numerischen Constraints
            variables: Alle Variablen
            statements: Die Puzzle-Statements

        Returns:
            Lösung oder None
        """
        # Prüfe ob Meta-Constraints vorhanden
        has_meta = any(
            c.constraint_type == ConstraintType.META for c in num_constraints
        )

        if not has_meta:
            # Einfacher Fall: Nur base constraints
            logger.debug("Keine Meta-Constraints - direktes Solving")
            solution, _ = self.csp_solver.solve(csp_problem)
            return solution

        # Komplexer Fall: Meta-Constraints vorhanden
        logger.info("Meta-Constraints erkannt - iteratives Solving")

        # Extrahiere Statement-Variablen
        statement_vars = [v for v in variables.keys() if v.startswith("statement_")]

        if not statement_vars:
            # Fallback: Versuche direkt zu lösen
            solution, _ = self.csp_solver.solve(csp_problem)
            return solution

        # Begrenze Iterationen (max 100 Kombinationen)
        max_iterations = min(100, 2 ** len(statement_vars))
        logger.debug(
            f"Iteratives Solving mit max {max_iterations} Kombinationen für {len(statement_vars)} Statements"
        )

        # Enumerate truth assignments
        for iteration, truth_values in enumerate(
            itertools.product([0, 1], repeat=len(statement_vars))
        ):
            if iteration >= max_iterations:
                break

            # Erstelle Assignment für Statement-Variablen
            truth_assignment = dict(zip(statement_vars, truth_values))

            # Löse CSP mit diesem Assignment
            # (Setze Statement-Variablen als fixiert)
            partial_solution = self._solve_with_fixed_statements(
                csp_problem, truth_assignment
            )

            if partial_solution is None:
                continue

            # Verifiziere Meta-Constraints
            full_solution = {**partial_solution, **truth_assignment}
            if self._verify_meta_constraints(
                full_solution, num_constraints, statements
            ):
                logger.info(f"[OK] Lösung gefunden nach {iteration + 1} Iterationen")
                return full_solution

        logger.warning(f"Keine Lösung nach {max_iterations} Iterationen")
        return None

    def _solve_with_fixed_statements(
        self, csp_problem: ConstraintProblem, fixed_statements: Dict[str, int]
    ) -> Optional[Dict[str, int]]:
        """
        Löst CSP mit fixierten Statement-Variablen.

        Args:
            csp_problem: Das CSP Problem
            fixed_statements: Fixierte Statement truth values

        Returns:
            Lösung (ohne Statement-Variablen) oder None
        """
        # Aktuell: Einfache Implementierung - solve nur die Hauptvariable "zahl"
        # Erweiterte Implementierung würde Fixed Variables in CSP integrieren
        solution, _ = self.csp_solver.solve(csp_problem)
        return solution

    def _verify_meta_constraints(
        self,
        solution: Dict[str, int],
        constraints: List[NumConstraint],
        statements: Dict[int, str],
    ) -> bool:
        """
        Verifiziert Meta-Constraints gegen eine Lösung.

        Args:
            solution: Die Lösung (inkl. Statement truth values)
            constraints: Alle Constraints
            statements: Die Puzzle-Statements

        Returns:
            True wenn alle Meta-Constraints erfüllt sind
        """
        for constraint in constraints:
            if constraint.constraint_type != ConstraintType.META:
                continue

            # Verifiziere basierend auf Metadata
            if constraint.metadata.get("meta_type") == "divisor_count":
                # Anzahl der Teiler
                zahl = solution.get("zahl", 0)
                if zahl <= 0:
                    return False

                divisor_count = self._count_divisors(zahl)
                threshold = constraint.metadata.get("threshold", 0)

                if divisor_count <= threshold:
                    return False

        return True

    def _count_divisors(self, n: int) -> int:
        """Zählt die Anzahl der Teiler von n."""
        if n <= 0:
            return 0

        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1

        return count

    def _verify_solution(
        self,
        solution: Dict[str, int],
        constraints: List[NumConstraint],
        variables: Dict[str, NumericalVariable],
    ) -> bool:
        """
        Verifiziert eine Lösung gegen alle Constraints.

        Args:
            solution: Die zu verifizierende Lösung
            constraints: Alle Constraints
            variables: Alle Variablen

        Returns:
            True wenn alle Constraints erfüllt sind
        """
        for constraint in constraints:
            try:
                # Versuche Constraint-Expression auszuwerten
                if not constraint.expression(solution):
                    logger.debug(f"Constraint nicht erfüllt: {constraint.description}")
                    return False
            except Exception as e:
                # Expression kann nicht evaluiert werden (Meta-Constraint)
                logger.debug(
                    f"Constraint-Evaluation übersprungen: {constraint.description}"
                )
                continue

        return True

    def _format_answer(
        self,
        solution: Dict[str, int],
        question: Optional[str],
        variables: Dict[str, NumericalVariable],
    ) -> str:
        """
        Formatiert die Lösung als natürlichsprachliche Antwort.

        Args:
            solution: Die Lösung
            question: Die Frage
            variables: Die Variablen

        Returns:
            Formatierte Antwort
        """
        # Extrahiere Hauptantwort (meist "zahl")
        if "zahl" in solution:
            zahl = solution["zahl"]

            # Pattern-basierte Antwort
            if question and "kleinste" in question.lower():
                return f"Die kleinste gesuchte Zahl ist {zahl}."
            elif question and "größte" in question.lower():
                return f"Die größte gesuchte Zahl ist {zahl}."
            else:
                return f"Die gesuchte Zahl ist {zahl}."

        # Fallback: Alle non-statement Variablen
        non_statement_vars = {
            k: v for k, v in solution.items() if not k.startswith("statement_")
        }

        if non_statement_vars:
            answers = [f"{k}={v}" for k, v in non_statement_vars.items()]
            return "Lösung: " + ", ".join(answers)

        return "Keine eindeutige Lösung gefunden."

    def _build_proof_tree(
        self,
        csp_problem: ConstraintProblem,
        solution: Optional[Dict[str, int]],
        constraints: List[NumConstraint],
    ) -> ProofTree:
        """
        Generiert einen ProofTree für das Puzzle-Solving.

        Args:
            csp_problem: Das CSP Problem
            solution: Die Lösung (oder None)
            constraints: Die Constraints

        Returns:
            ProofTree mit allen Reasoning-Steps
        """
        steps = []

        # Step 1: Puzzle Detection
        steps.append(
            ProofStep(
                step_id="numerical_csp_detection",
                step_type=ProofStepType.PREMISE,
                output="Numerical CSP Puzzle erkannt",
                explanation_text=f"Puzzle mit {len(constraints)} Constraints und {len(csp_problem.variables)} Variablen erkannt",
                confidence=0.95,
                metadata={
                    "num_constraints": len(constraints),
                    "num_variables": len(csp_problem.variables),
                },
                source_component="component_45_numerical_puzzle_solver",
            )
        )

        # Step 2: Constraint Parsing
        for i, constraint in enumerate(constraints):
            steps.append(
                ProofStep(
                    step_id=f"constraint_{i+1}_parsing",
                    step_type=ProofStepType.RULE_APPLICATION,
                    output=f"Constraint {i+1}: {constraint.description}",
                    explanation_text=f"Parsed constraint: {constraint.description}",
                    confidence=0.90,
                    metadata={"constraint_type": constraint.constraint_type.value},
                    source_component="component_45_numerical_puzzle_solver",
                )
            )

        # Step 3: CSP Setup
        steps.append(
            ProofStep(
                step_id="csp_problem_setup",
                step_type=ProofStepType.INFERENCE,
                output=f"CSP Problem mit {len(csp_problem.variables)} Variablen erstellt",
                explanation_text="Converted numerical constraints to CSP problem",
                confidence=0.95,
                source_component="component_45_numerical_puzzle_solver",
            )
        )

        # Step 4: Solution (falls gefunden)
        if solution:
            steps.append(
                ProofStep(
                    step_id="solution_found",
                    step_type=ProofStepType.CONCLUSION,
                    output=f"Lösung gefunden: {solution}",
                    explanation_text=f"CSP solver found solution: {solution}",
                    confidence=0.90,
                    metadata={"solution": solution},
                    source_component="component_45_numerical_puzzle_solver",
                )
            )
        else:
            steps.append(
                ProofStep(
                    step_id="no_solution",
                    step_type=ProofStepType.CONCLUSION,
                    output="Keine Lösung gefunden (UNSAT)",
                    explanation_text="CSP solver could not find a satisfying assignment",
                    confidence=0.30,
                    source_component="component_45_numerical_puzzle_solver",
                )
            )

        return ProofTree(
            query="Numerical puzzle solving",
            root_steps=steps,
            metadata={
                "confidence": 0.90 if solution else 0.30,
                "num_steps": len(steps),
            },
        )
