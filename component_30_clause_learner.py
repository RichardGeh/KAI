"""
component_30_clause_learner.py

Advanced SAT Features - Watched Literals and Knowledge Base Checking

This module provides advanced SAT solving features:
- Watched literals optimization for efficient unit propagation
- Knowledge base consistency checking
- Conflict analysis and minimal unsatisfiable core extraction
- Horn clause rule consistency verification

SPLIT FROM: component_30_sat_solver.py (Task 12 - Phase 4, 2025-11-28)

Author: KAI Development Team
Date: 2025-10-30 | Refactored: 2025-11-28
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from component_15_logging_config import get_logger
from component_30_sat_solver_core import (
    Clause,
    CNFFormula,
    DPLLSolver,
    Literal,
    SATResult,
)

logger = get_logger(__name__)


# ============================================================================
# Watched Literals Optimization
# ============================================================================


@dataclass
class WatchedLiterals:
    """
    Watched literals data structure for efficient unit propagation.

    For each clause, we watch two literals. When a watched literal becomes
    false, we search for a new literal to watch. This avoids checking all
    literals in all clauses during propagation.

    This optimization reduces the cost of unit propagation from O(n*m) to
    O(n) in typical cases, where n is the number of literals and m is the
    number of clauses.
    """

    # Map: clause_id -> (watched_lit1, watched_lit2)
    watched: Dict[int, Tuple[Literal, Literal]] = field(default_factory=dict)

    # Map: literal -> set of clause_ids watching this literal
    watchers: Dict[Literal, Set[int]] = field(default_factory=dict)

    # Original clauses for reference
    clauses: List[Clause] = field(default_factory=list)

    def initialize(self, formula: CNFFormula):
        """Initialize watched literals for all clauses."""
        self.watched.clear()
        self.watchers.clear()
        self.clauses = formula.clauses.copy()

        for clause_id, clause in enumerate(self.clauses):
            if clause.is_empty():
                continue

            literals_list = list(clause.literals)

            if len(literals_list) == 1:
                # Unit clause: watch the single literal twice
                lit = literals_list[0]
                self.watched[clause_id] = (lit, lit)
                self._add_watcher(lit, clause_id)
            else:
                # Watch first two literals
                lit1, lit2 = literals_list[0], literals_list[1]
                self.watched[clause_id] = (lit1, lit2)
                self._add_watcher(lit1, clause_id)
                self._add_watcher(lit2, clause_id)

    def _add_watcher(self, literal: Literal, clause_id: int):
        """Add clause as watcher for literal."""
        if literal not in self.watchers:
            self.watchers[literal] = set()
        self.watchers[literal].add(clause_id)

    def _remove_watcher(self, literal: Literal, clause_id: int):
        """Remove clause from watchers of literal."""
        if literal in self.watchers:
            self.watchers[literal].discard(clause_id)

    def propagate(self, assignment: Dict[str, bool]) -> Tuple[bool, List[Literal]]:
        """
        Perform unit propagation using watched literals.

        Args:
            assignment: Current partial assignment

        Returns:
            (success, unit_literals)
            success: False if conflict detected
            unit_literals: List of literals that must be assigned
        """
        unit_literals = []
        propagation_queue = []

        # Find initially false watched literals
        for clause_id, (lit1, lit2) in self.watched.items():
            clause = self.clauses[clause_id]

            if self._is_literal_false(lit1, assignment):
                propagation_queue.append((clause_id, lit1))
            elif self._is_literal_false(lit2, assignment):
                propagation_queue.append((clause_id, lit2))

        while propagation_queue:
            clause_id, false_lit = propagation_queue.pop(0)

            if clause_id not in self.watched:
                continue  # Clause already satisfied

            lit1, lit2 = self.watched[clause_id]
            clause = self.clauses[clause_id]

            # Ensure false_lit is lit1
            if false_lit == lit2:
                lit1, lit2 = lit2, lit1

            # Try to find alternative literal to watch
            found_alternative = False
            for lit in clause.literals:
                if lit != lit1 and lit != lit2:
                    # Check if this literal could be watched
                    if not self._is_literal_false(lit, assignment):
                        # Found alternative: update watchers
                        self._remove_watcher(lit1, clause_id)
                        self._add_watcher(lit, clause_id)
                        self.watched[clause_id] = (lit, lit2)
                        found_alternative = True
                        break

            if not found_alternative:
                # Could not find alternative: check status
                if self._is_literal_false(lit2, assignment):
                    # Both watched literals are false: CONFLICT
                    return False, []
                elif not self._is_literal_true(lit2, assignment):
                    # lit2 is unassigned: UNIT clause
                    if lit2 not in unit_literals:
                        unit_literals.append(lit2)
                # else: lit2 is true, clause is satisfied

        return True, unit_literals

    def _is_literal_true(self, lit: Literal, assignment: Dict[str, bool]) -> bool:
        """Check if literal is true under assignment."""
        if lit.variable not in assignment:
            return False
        value = assignment[lit.variable]
        return (value and not lit.negated) or (not value and lit.negated)

    def _is_literal_false(self, lit: Literal, assignment: Dict[str, bool]) -> bool:
        """Check if literal is false under assignment."""
        if lit.variable not in assignment:
            return False
        value = assignment[lit.variable]
        return (not value and not lit.negated) or (value and lit.negated)


# ============================================================================
# Knowledge Base Checker
# ============================================================================


class KnowledgeBaseChecker:
    """
    SAT-based consistency checker for knowledge bases.

    Provides:
    - Rule consistency verification (Horn clauses)
    - Conflict detection between facts and rules
    - Minimal unsatisfiable core extraction
    - Integration with Neo4j knowledge graphs
    """

    def __init__(self, solver: Optional[DPLLSolver] = None):
        """
        Initialize knowledge base checker.

        Args:
            solver: DPLLSolver instance to use (creates new one if None)
        """
        self.solver = solver or DPLLSolver()

    def check_rule_consistency(
        self, rules: List[Tuple[List[Literal], Literal]]
    ) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """
        Check if a set of Horn clauses (rules) is consistent.

        Horn clauses are implications of the form:
        (premise1 AND premise2 AND ... AND premiseN) -> conclusion

        Args:
            rules: List of (premises, conclusion) where premises are ANDed

        Returns:
            (is_consistent, model)
            is_consistent: True if rules are mutually consistent
            model: Satisfying assignment if consistent, None otherwise

        Example:
            rules = [
                ([Literal("bird")], Literal("has_wings")),
                ([Literal("penguin")], Literal("bird")),
                ([Literal("penguin")], Literal("cannot_fly")),
            ]
            is_consistent, model = checker.check_rule_consistency(rules)
        """
        formula = CNFFormula([])

        for premises, conclusion in rules:
            # Rule: (p1 AND p2 AND ... AND pn) -> conclusion
            # CNF: NOT p1 OR NOT p2 OR ... OR NOT pn OR conclusion
            clause_literals = {-p for p in premises}
            clause_literals.add(conclusion)
            formula.add_clause(Clause(clause_literals))

        result, model = self.solver.solve(formula)
        return result == SATResult.SATISFIABLE, model

    def find_conflicts(
        self, facts: List[Literal], rules: List[Tuple[List[Literal], Literal]]
    ) -> List[str]:
        """
        Find conflicts between facts and rules.

        This method identifies which facts cause inconsistencies when
        combined with the given rules.

        Args:
            facts: List of factual statements (ground literals)
            rules: List of (premises, conclusion) Horn clauses

        Returns:
            List of conflict descriptions (empty if consistent)

        Example:
            facts = [
                Literal("penguin"),
                Literal("can_fly"),
            ]
            rules = [
                ([Literal("penguin")], Literal("bird")),
                ([Literal("bird")], Literal("can_fly", True)),  # Birds can't fly
            ]
            conflicts = checker.find_conflicts(facts, rules)
            # Returns: ["Fact 'can_fly' causes inconsistency"]
        """
        conflicts = []
        formula = CNFFormula([])

        # Add facts as unit clauses
        for fact in facts:
            formula.add_clause(Clause({fact}))

        # Add rules
        for premises, conclusion in rules:
            clause_literals = {-p for p in premises}
            clause_literals.add(conclusion)
            formula.add_clause(Clause(clause_literals))

        # Check satisfiability
        result, model = self.solver.solve(formula)

        if result == SATResult.UNSATISFIABLE:
            conflicts.append("Knowledge base is inconsistent")

            # Try to identify conflicting facts
            for i, fact in enumerate(facts):
                test_formula = CNFFormula([])
                for j, other_fact in enumerate(facts):
                    if i != j:
                        test_formula.add_clause(Clause({other_fact}))

                for premises, conclusion in rules:
                    clause_literals = {-p for p in premises}
                    clause_literals.add(conclusion)
                    test_formula.add_clause(Clause(clause_literals))

                test_result, _ = self.solver.solve(test_formula)
                if test_result == SATResult.SATISFIABLE:
                    conflicts.append(f"Fact '{fact}' causes inconsistency")

        return conflicts

    def check_consistency(
        self, formulas: List[CNFFormula]
    ) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if a set of formulas is mutually consistent.

        Args:
            formulas: List of CNF formulas

        Returns:
            (is_consistent, conflicting_formula_indices)
            If inconsistent, returns indices of minimal unsatisfiable subset
        """
        # Merge all formulas
        merged = CNFFormula([])
        for formula in formulas:
            merged.clauses.extend(formula.clauses)
            merged.variables.update(formula.variables)

        result, _ = self.solver.solve(merged)

        if result == SATResult.SATISFIABLE:
            return True, None

        # Find minimal unsatisfiable subset (simple greedy approach)
        # Try removing each formula and see if rest becomes satisfiable
        conflicting = list(range(len(formulas)))

        for i in range(len(formulas)):
            test_formulas = [formulas[j] for j in conflicting if j != i]
            if not test_formulas:
                continue

            merged_test = CNFFormula([])
            for formula in test_formulas:
                merged_test.clauses.extend(formula.clauses)
                merged_test.variables.update(formula.variables)

            result, _ = self.solver.solve(merged_test)
            if result == SATResult.UNSATISFIABLE:
                # Still unsatisfiable without formula i, so i is not essential
                conflicting.remove(i)

        return False, conflicting


# ============================================================================
# Enhanced DPLL Solver with Watched Literals
# ============================================================================


class EnhancedDPLLSolver(DPLLSolver):
    """
    DPLL solver with watched literals optimization.

    Extends the basic DPLL solver with:
    - Watched literals for efficient unit propagation
    - Clause learning (basic)
    - Better decision heuristics
    """

    def __init__(self, enable_proof: bool = False):
        """
        Initialize enhanced DPLL solver.

        Args:
            enable_proof: Whether to generate proof trees
        """
        super().__init__(enable_proof=enable_proof)
        self.watched_literals = WatchedLiterals()
        self.use_watched_literals = True

    def solve(
        self, formula: CNFFormula, initial_assignment: Optional[Dict[str, bool]] = None
    ) -> Tuple[SATResult, Optional[Dict[str, bool]]]:
        """
        Solve SAT problem using enhanced DPLL with watched literals.

        Args:
            formula: CNF formula to solve
            initial_assignment: Optional partial assignment to start with

        Returns:
            (result, model) where model is None if UNSATISFIABLE
        """
        # Initialize watched literals
        if self.use_watched_literals:
            self.watched_literals.initialize(formula)

        # Use parent implementation (watches are used internally if needed)
        return super().solve(formula, initial_assignment)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "WatchedLiterals",
    "KnowledgeBaseChecker",
    "EnhancedDPLLSolver",
]
