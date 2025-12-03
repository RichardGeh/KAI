"""
component_30_sat_solver.py
==========================
SAT-Solver for propositional logic reasoning - FACADE for Modular Architecture

This file provides backward compatibility for the refactored SAT solver.
The original monolithic implementation has been split into three focused modules:

1. component_30_sat_solver_core.py (~590 lines)
   - Core data structures (Literal, Clause, CNFFormula)
   - DPLL algorithm with unit propagation
   - Basic SAT solving
   - ProofTree integration

2. component_30_cnf_converter.py (~440 lines)
   - PropositionalFormula data structure
   - CNF conversion (implications, De Morgan, distribution)
   - SAT encoding utilities (implication, iff, at-most-one, etc.)
   - Example problem encodings

3. component_30_clause_learner.py (~360 lines)
   - Watched literals optimization
   - Knowledge base consistency checking
   - Conflict analysis
   - Enhanced DPLL solver

This facade delegates all operations to the specialized modules while maintaining
the original API for backward compatibility.

Features (preserved):
- CNF (Conjunctive Normal Form) representation
- DPLL algorithm with watched literals for efficient SAT solving
- Unit propagation and pure literal elimination
- Model finding and unsatisfiable core extraction
- Integration with Neo4j knowledge base for consistency checking
- ProofTree generation for explanations

Author: KAI Development Team
Date: 2025-10-30 | Refactored: 2025-11-28 (Task 12 - Phase 4)
Updated: 2025-12-02 | Added BaseReasoningEngine interface (Phase 5)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

# Import from clause learner module
from component_30_clause_learner import (
    EnhancedDPLLSolver,
    KnowledgeBaseChecker,
    WatchedLiterals,
)

# Import from converter module
from component_30_cnf_converter import (
    CNFConverter,
    PropositionalFormula,
    PropositionalOperator,
    SATEncoder,
    create_knights_and_knaves_problem,
    solve_propositional,
)

# Import all exports from core module
from component_30_sat_solver_core import (
    PROOF_AVAILABLE,
    Clause,
    CNFFormula,
    DPLLSolver,
    Literal,
    ProofStep,
    ProofTree,
    SATResult,
    StepType,
)
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult

logger = logging.getLogger(__name__)


# ============================================================================
# Simplified SAT Solver API (User-Facing Wrapper)
# ============================================================================


class SATSolver(BaseReasoningEngine):
    """
    Simplified SAT-Solver API - User-Friendly Wrapper

    This class provides a simplified, user-friendly API for the DPLL-Solver
    that matches the original specification.

    Features:
    - DPLL algorithm with unit propagation
    - Pure literal elimination
    - ProofTree integration for transparent reasoning
    - BaseReasoningEngine interface for orchestration

    Example:
        >>> solver = SATSolver(enable_proof=True)
        >>> formula = CNFFormula([...])
        >>> model = solver.solve(formula)
        >>> if model:
        >>>     print(f"SAT: {model}")
        >>>     proof = solver.get_proof_tree()
    """

    def __init__(self, enable_proof: bool = True):
        """
        Initialize SAT solver.

        Args:
            enable_proof: Generate ProofTree for proofs
        """
        self.enable_proof = enable_proof
        # Use EnhancedDPLLSolver which has watched literals optimization
        self._solver = EnhancedDPLLSolver(enable_proof=enable_proof)

    def solve(self, formula: CNFFormula) -> Optional[Dict[str, bool]]:
        """
        Find satisfying assignment for CNF formula.

        Args:
            formula: CNF formula

        Returns:
            Dict[str, bool]: Satisfying assignment (SAT)
            None: No solution (UNSAT)
        """
        result, model = self._solver.solve(formula)
        return model if result == SATResult.SATISFIABLE else None

    def dpll(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Optional[Dict[str, bool]]:
        """
        DPLL algorithm with unit propagation (with partial assignment).

        Args:
            formula: CNF formula
            assignment: Partial assignment

        Returns:
            Dict[str, bool]: Satisfying assignment (SAT)
            None: No solution (UNSAT)
        """
        result, model = self._solver.solve(formula, initial_assignment=assignment)
        return model if result == SATResult.SATISFIABLE else None

    def unit_propagate(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Tuple[CNFFormula, Dict[str, bool]]:
        """
        Unit Propagation (Constraint Propagation for SAT).

        Finds unit clauses and propagates their assignments.

        Args:
            formula: CNF formula
            assignment: Current assignment

        Returns:
            (simplified_formula, extended_assignment)
        """
        # Simplify formula with current assignment
        simplified = formula.simplify(assignment)
        extended_assignment = assignment.copy()

        # Iteratively find unit clauses and propagate
        changed = True
        while changed:
            changed = False
            unit_literals = simplified.get_unit_clauses()

            if unit_literals:
                for lit in unit_literals:
                    if lit.variable not in extended_assignment:
                        # New assignment
                        value = not lit.negated
                        extended_assignment[lit.variable] = value
                        changed = True

                # Simplify with new assignments
                simplified = simplified.simplify(extended_assignment)

        return simplified, extended_assignment

    def pure_literal_elimination(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Tuple[CNFFormula, Dict[str, bool]]:
        """
        Eliminate pure literals (variables with only one polarity).

        Args:
            formula: CNF formula
            assignment: Current assignment

        Returns:
            (simplified_formula, extended_assignment)
        """
        simplified = formula.simplify(assignment)
        extended_assignment = assignment.copy()

        # Find pure literals
        pure_literals = simplified.get_pure_literals(extended_assignment)

        for lit in pure_literals:
            value = not lit.negated
            extended_assignment[lit.variable] = value

        # Simplify with new assignments
        if pure_literals:
            simplified = simplified.simplify(extended_assignment)

        return simplified, extended_assignment

    def get_proof_tree(self, query: str = "SAT Solution") -> Optional["ProofTree"]:
        """
        Get ProofTree from last solve() call.

        Args:
            query: Query string for the proof tree

        Returns:
            ProofTree: Proof tree
            None: If proof disabled or component_17 not available
        """
        return self._solver.get_proof_tree(query)

    # ========================================================================
    # BaseReasoningEngine Interface Implementation
    # ========================================================================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Execute SAT reasoning on the given query.

        The query should contain a CNF formula representation or problem encoding.
        Context can specify:
        - 'formula': CNF formula object
        - 'problem_type': Type of SAT problem (e.g., 'knights_and_knaves')
        - 'enable_proof': Whether to generate proof tree

        Args:
            query: Query string describing the SAT problem
            context: Context with formula and parameters

        Returns:
            ReasoningResult with satisfying assignment or UNSAT result
        """
        # Extract formula from context
        formula = context.get("formula")
        if formula is None:
            return ReasoningResult(
                success=False,
                answer="No formula provided in context",
                confidence=0.0,
                strategy_used="sat_solver",
                metadata={"error": "missing_formula"},
            )

        # Solve the formula
        model = self.solve(formula)

        if model is not None:
            # SAT - found satisfying assignment
            answer = f"SAT: {model}"
            proof_tree = self.get_proof_tree(query) if self.enable_proof else None

            return ReasoningResult(
                success=True,
                answer=answer,
                confidence=1.0,  # SAT solving is deterministic
                proof_tree=proof_tree,
                strategy_used="sat_solver_dpll",
                metadata={
                    "result": "SATISFIABLE",
                    "model": model,
                    "num_variables": len(model),
                },
            )
        else:
            # UNSAT - no solution exists
            proof_tree = self.get_proof_tree(query) if self.enable_proof else None

            return ReasoningResult(
                success=True,  # Successfully determined UNSAT
                answer="UNSAT: No satisfying assignment exists",
                confidence=1.0,
                proof_tree=proof_tree,
                strategy_used="sat_solver_dpll",
                metadata={"result": "UNSATISFIABLE"},
            )

    def get_capabilities(self) -> List[str]:
        """
        Return SAT solver capabilities.

        Returns:
            List of capability identifiers
        """
        return [
            "constraint",
            "sat_solving",
            "boolean_satisfiability",
            "propositional_logic",
            "dpll_algorithm",
            "unit_propagation",
            "pure_literal_elimination",
            "cnf_reasoning",
        ]

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for SAT solving.

        SAT solving can be expensive (NP-complete), but modern solvers
        with heuristics are often practical. Cost depends on:
        - Number of variables
        - Number of clauses
        - Structure of the formula

        Args:
            query: Query string (may contain hints about problem size)

        Returns:
            Estimated cost in [0.0, 1.0+] range
        """
        # Simple heuristic: SAT is generally medium to expensive
        # Without formula inspection, assume medium-high cost
        # If context provided more info, we could be more precise
        return 0.7  # Medium-high cost for SAT problems


# ============================================================================
# Convenience Functions
# ============================================================================


def solve_cnf(
    formula: CNFFormula, enable_proof: bool = False
) -> Optional[Dict[str, bool]]:
    """
    Solve CNF formula (convenience function).

    Args:
        formula: CNF formula
        enable_proof: Generate proof tree

    Returns:
        Satisfying assignment or None (UNSAT)
    """
    solver = SATSolver(enable_proof=enable_proof)
    return solver.solve(formula)


# Note: solve_propositional is already exported from component_30_cnf_converter


# ============================================================================
# Example Usage (preserved from original for backward compatibility)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Simple SAT problem
    print("=== Example 1: Simple SAT ===")
    formula1 = CNFFormula(
        [
            Clause({Literal("x"), Literal("y")}),
            Clause({Literal("x", True), Literal("z")}),  # NOT x OR z
            Clause({Literal("y", True), Literal("z", True)}),  # NOT y OR NOT z
        ]
    )
    print(f"Formula: {formula1}")

    solver_basic = DPLLSolver()
    result, model = solver_basic.solve(formula1)
    print(f"Result: {result.value}")
    if model:
        print(f"Model: {model}")

    # Example 2: Knights and Knaves
    print("\n=== Example 2: Knights and Knaves ===")
    puzzle = create_knights_and_knaves_problem()
    result, model = solver_basic.solve(puzzle)
    print(f"Result: {result.value}")
    if model:
        print("Solution:")
        for var, value in sorted(model.items()):
            role = "Knight" if value else "Knave"
            print(f"  {var}: {role}")

    # Example 3: Consistency checking
    print("\n=== Example 3: Consistency Checking ===")
    checker = KnowledgeBaseChecker(solver_basic)

    rules = [
        ([Literal("bird")], Literal("can_fly")),
        ([Literal("penguin")], Literal("bird")),
        ([Literal("penguin")], Literal("can_fly", True)),  # NOT can_fly
    ]

    is_consistent, model = checker.check_rule_consistency(rules)
    print(f"Rules consistent: {is_consistent}")
    if model:
        print(f"Model: {model}")

    # Example 4: Using simplified API
    print("\n=== Example 4: Simplified API ===")
    solver = SATSolver(enable_proof=True)
    model = solver.solve(formula1)
    print(f"Model: {model}")
    if model and solver.enable_proof:
        proof_tree = solver.get_proof_tree()
        if proof_tree:
            print("Proof tree generated successfully")


# ============================================================================
# Public API - Re-export all classes for backward compatibility
# ============================================================================

__all__ = [
    # Core data structures
    "SATResult",
    "Literal",
    "Clause",
    "CNFFormula",
    # Propositional formulas
    "PropositionalOperator",
    "PropositionalFormula",
    # Converters and encoders
    "CNFConverter",
    "SATEncoder",
    # Solvers
    "DPLLSolver",
    "EnhancedDPLLSolver",
    "SATSolver",
    # Utilities
    "WatchedLiterals",
    "KnowledgeBaseChecker",
    # Functions
    "solve_cnf",
    "solve_propositional",
    "create_knights_and_knaves_problem",
    # Proof system
    "PROOF_AVAILABLE",
    "StepType",
    "ProofStep",
    "ProofTree",
]
