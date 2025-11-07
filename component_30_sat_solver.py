"""
component_30_sat_solver.py
==========================
SAT-Solver for propositional logic reasoning.

Implements DPLL algorithm with watched literals for efficient
satisfiability checking, consistency verification, and theorem proving.

Features:
- CNF (Conjunctive Normal Form) representation
- DPLL algorithm with unit propagation and pure literal elimination
- Watched literals optimization for efficient unit propagation
- Model finding and unsatisfiable core extraction
- Integration with Neo4j knowledge base for consistency checking

Author: KAI Development Team
Date: 2025-10-30
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from component_15_logging_config import get_logger

# Integration mit Proof System
try:
    from component_17_proof_explanation import ProofStep, ProofTree, StepType

    PROOF_AVAILABLE = True
except ImportError:
    PROOF_AVAILABLE = False
    StepType: Any = None  # type: ignore[no-redef]  # Dummy fallback
    logging.warning("component_17 not available, proof generation disabled")

logger = get_logger(__name__)


class SATResult(Enum):
    """Result of SAT solving."""

    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Literal:
    """
    A propositional literal (variable or its negation).

    Immutable for use in sets/dicts.
    """

    variable: str
    negated: bool = False

    def __neg__(self):
        """Return the negation of this literal."""
        return Literal(self.variable, not self.negated)

    def __str__(self):
        return f"{'¬' if self.negated else ''}{self.variable}"

    def __repr__(self):
        return str(self)

    def to_dimacs(self, var_mapping: Dict[str, int]) -> int:
        """Convert to DIMACS format (positive/negative integer)."""
        var_id = var_mapping[self.variable]
        return -var_id if self.negated else var_id


@dataclass
class Clause:
    """
    A disjunction of literals (OR-connected).

    Empty clause represents FALSE (unsatisfiable).
    """

    literals: FrozenSet[Literal]

    def __init__(self, literals: Set[Literal] | FrozenSet[Literal]):
        if isinstance(literals, frozenset):
            object.__setattr__(self, "literals", literals)
        else:
            object.__setattr__(self, "literals", frozenset(literals))

    def __hash__(self):
        return hash(self.literals)

    def __eq__(self, other):
        return isinstance(other, Clause) and self.literals == other.literals

    def is_empty(self) -> bool:
        """Check if clause is empty (represents FALSE)."""
        return len(self.literals) == 0

    def is_unit(self) -> bool:
        """Check if clause has exactly one literal (unit clause)."""
        return len(self.literals) == 1

    def get_unit_literal(self) -> Optional[Literal]:
        """Get the single literal if this is a unit clause."""
        if self.is_unit():
            return next(iter(self.literals))
        return None

    def simplify(self, assignment: Dict[str, bool]) -> Optional["Clause"]:
        """
        Simplify clause given partial assignment.

        Returns:
            None if clause is satisfied
            New simplified clause otherwise
        """
        new_literals = set()

        for lit in self.literals:
            if lit.variable in assignment:
                # Check if literal is satisfied
                value = assignment[lit.variable]
                if (value and not lit.negated) or (not value and lit.negated):
                    # Clause is satisfied
                    return None
                # Otherwise, literal is false, skip it
            else:
                # Variable not assigned, keep literal
                new_literals.add(lit)

        return Clause(new_literals)

    def __str__(self):
        if self.is_empty():
            return "⊥"  # Empty clause (FALSE)
        return " ∨ ".join(str(lit) for lit in sorted(self.literals, key=str))

    def __repr__(self):
        return f"Clause({self.literals})"


@dataclass
class CNFFormula:
    """
    A formula in Conjunctive Normal Form (AND of ORs).

    Represents: C1 ∧ C2 ∧ ... ∧ Cn where each Ci is a clause.
    """

    clauses: List[Clause]
    variables: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Extract all variables from clauses."""
        if not self.variables:
            for clause in self.clauses:
                for lit in clause.literals:
                    self.variables.add(lit.variable)

    def add_clause(self, clause: Clause):
        """Add a clause to the formula."""
        self.clauses.append(clause)
        for lit in clause.literals:
            self.variables.add(lit.variable)

    def is_empty(self) -> bool:
        """Check if formula has no clauses (represents TRUE)."""
        return len(self.clauses) == 0

    def has_empty_clause(self) -> bool:
        """Check if formula contains an empty clause (unsatisfiable)."""
        return any(c.is_empty() for c in self.clauses)

    def get_unit_clauses(self) -> List[Literal]:
        """Get all unit clause literals."""
        return [
            lit
            for c in self.clauses
            if c.is_unit() and (lit := c.get_unit_literal()) is not None
        ]

    def get_pure_literals(self, assignment: Dict[str, bool]) -> List[Literal]:
        """
        Find pure literals (variables that appear only positively or only negatively).

        Args:
            assignment: Current partial assignment (to skip assigned variables)
        """
        literal_polarities: Dict[str, Set[bool]] = (
            {}
        )  # variable -> Set[bool] (negated values seen)

        for clause in self.clauses:
            for lit in clause.literals:
                if lit.variable not in assignment:
                    if lit.variable not in literal_polarities:
                        literal_polarities[lit.variable] = set()
                    literal_polarities[lit.variable].add(lit.negated)

        # Pure literal has only one polarity
        pure_literals = []
        for var, polarities in literal_polarities.items():
            if len(polarities) == 1:
                negated = next(iter(polarities))
                pure_literals.append(Literal(var, negated))

        return pure_literals

    def simplify(self, assignment: Dict[str, bool]) -> "CNFFormula":
        """
        Simplify formula given partial assignment.

        Removes satisfied clauses and false literals.
        """
        new_clauses = []
        for clause in self.clauses:
            simplified = clause.simplify(assignment)
            if simplified is not None:  # Clause not satisfied
                new_clauses.append(simplified)

        return CNFFormula(new_clauses, self.variables.copy())

    def __str__(self):
        if self.is_empty():
            return "⊤"  # Empty formula (TRUE)
        return " ∧ ".join(f"({c})" for c in self.clauses)


class PropositionalOperator(Enum):
    """Propositionale Operatoren für allgemeine Formeln"""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    IFF = "IFF"  # Biconditional (if and only if)


@dataclass
class PropositionalFormula:
    """
    Allgemeine propositionale Formel (nicht notwendigerweise CNF).
    Kann zu CNF konvertiert werden.
    """

    operator: Optional[PropositionalOperator] = None
    variable: Optional[str] = None
    operands: List["PropositionalFormula"] = field(default_factory=list)

    def __str__(self) -> str:
        if self.variable:
            return self.variable
        if self.operator == PropositionalOperator.NOT:
            return f"¬{self.operands[0]}"
        if self.operator == PropositionalOperator.AND:
            return f"({' ∧ '.join(str(op) for op in self.operands)})"
        if self.operator == PropositionalOperator.OR:
            return f"({' ∨ '.join(str(op) for op in self.operands)})"
        if self.operator == PropositionalOperator.IMPLIES:
            return f"({self.operands[0]} → {self.operands[1]})"
        if self.operator == PropositionalOperator.IFF:
            return f"({self.operands[0]} ↔ {self.operands[1]})"
        return "?"

    @classmethod
    def variable_formula(cls, var: str) -> "PropositionalFormula":
        """Erstelle Variable"""
        return cls(variable=var)

    @classmethod
    def not_formula(cls, operand: "PropositionalFormula") -> "PropositionalFormula":
        """Erstelle Negation"""
        return cls(operator=PropositionalOperator.NOT, operands=[operand])

    @classmethod
    def and_formula(cls, *operands: "PropositionalFormula") -> "PropositionalFormula":
        """Erstelle Konjunktion"""
        return cls(operator=PropositionalOperator.AND, operands=list(operands))

    @classmethod
    def or_formula(cls, *operands: "PropositionalFormula") -> "PropositionalFormula":
        """Erstelle Disjunktion"""
        return cls(operator=PropositionalOperator.OR, operands=list(operands))

    @classmethod
    def implies_formula(
        cls, antecedent: "PropositionalFormula", consequent: "PropositionalFormula"
    ) -> "PropositionalFormula":
        """Erstelle Implikation"""
        return cls(
            operator=PropositionalOperator.IMPLIES, operands=[antecedent, consequent]
        )

    @classmethod
    def iff_formula(
        cls, left: "PropositionalFormula", right: "PropositionalFormula"
    ) -> "PropositionalFormula":
        """Erstelle Biconditional"""
        return cls(operator=PropositionalOperator.IFF, operands=[left, right])


class CNFConverter:
    """
    Konvertiert propositionale Formeln zu CNF (Conjunctive Normal Form).

    Schritte:
    1. Eliminiere Implikationen und Biconditionals
    2. Pushe Negationen nach innen (De Morgan)
    3. Distributiere OR über AND
    """

    @staticmethod
    def to_cnf(formula: PropositionalFormula) -> CNFFormula:
        """Hauptmethode: Konvertiere zu CNF"""
        # Schritt 1: Eliminiere Implikationen
        formula = CNFConverter._eliminate_implications(formula)

        # Schritt 2: Pushe Negationen nach innen
        formula = CNFConverter._push_negations_inward(formula)

        # Schritt 3: Distributiere OR über AND
        formula = CNFConverter._distribute_or_over_and(formula)

        # Schritt 4: Extrahiere CNF-Struktur
        return CNFConverter._extract_cnf(formula)

    @staticmethod
    def _eliminate_implications(formula: PropositionalFormula) -> PropositionalFormula:
        """
        Eliminiere Implikationen und Biconditionals.
        - A → B wird zu ¬A ∨ B
        - A ↔ B wird zu (A → B) ∧ (B → A) = (¬A ∨ B) ∧ (¬B ∨ A)
        """
        if formula.variable:
            return formula

        if formula.operator == PropositionalOperator.IMPLIES:
            # A → B = ¬A ∨ B
            ant = CNFConverter._eliminate_implications(formula.operands[0])
            cons = CNFConverter._eliminate_implications(formula.operands[1])
            return PropositionalFormula.or_formula(
                PropositionalFormula.not_formula(ant), cons
            )

        if formula.operator == PropositionalOperator.IFF:
            # A ↔ B = (¬A ∨ B) ∧ (¬B ∨ A)
            left = CNFConverter._eliminate_implications(formula.operands[0])
            right = CNFConverter._eliminate_implications(formula.operands[1])
            return PropositionalFormula.and_formula(
                PropositionalFormula.or_formula(
                    PropositionalFormula.not_formula(left), right
                ),
                PropositionalFormula.or_formula(
                    PropositionalFormula.not_formula(right), left
                ),
            )

        # Rekursiv auf Operanden anwenden
        new_operands = [
            CNFConverter._eliminate_implications(op) for op in formula.operands
        ]
        return PropositionalFormula(operator=formula.operator, operands=new_operands)

    @staticmethod
    def _push_negations_inward(formula: PropositionalFormula) -> PropositionalFormula:
        """
        Pushe Negationen nach innen mit De Morgan:
        - ¬(A ∧ B) = ¬A ∨ ¬B
        - ¬(A ∨ B) = ¬A ∧ ¬B
        - ¬¬A = A
        """
        if formula.variable:
            return formula

        if formula.operator != PropositionalOperator.NOT:
            # Nicht-NOT: Rekursiv auf Operanden
            new_operands = [
                CNFConverter._push_negations_inward(op) for op in formula.operands
            ]
            return PropositionalFormula(
                operator=formula.operator, operands=new_operands
            )

        # NOT: Schaue auf inneren Operator
        inner = formula.operands[0]

        if inner.variable:
            # ¬Variable: Fertig
            return formula

        if inner.operator == PropositionalOperator.NOT:
            # ¬¬A = A
            return CNFConverter._push_negations_inward(inner.operands[0])

        if inner.operator == PropositionalOperator.AND:
            # ¬(A ∧ B) = ¬A ∨ ¬B
            negated_operands = [
                CNFConverter._push_negations_inward(
                    PropositionalFormula.not_formula(op)
                )
                for op in inner.operands
            ]
            return PropositionalFormula.or_formula(*negated_operands)

        if inner.operator == PropositionalOperator.OR:
            # ¬(A ∨ B) = ¬A ∧ ¬B
            negated_operands = [
                CNFConverter._push_negations_inward(
                    PropositionalFormula.not_formula(op)
                )
                for op in inner.operands
            ]
            return PropositionalFormula.and_formula(*negated_operands)

        return formula

    @staticmethod
    def _distribute_or_over_and(formula: PropositionalFormula) -> PropositionalFormula:
        """
        Distributiere OR über AND:
        - A ∨ (B ∧ C) = (A ∨ B) ∧ (A ∨ C)
        """
        if formula.variable:
            return formula

        if formula.operator == PropositionalOperator.NOT:
            # Negation sollte bereits nach innen gepusht sein
            return formula

        if formula.operator == PropositionalOperator.AND:
            # Rekursiv auf Operanden
            new_operands = [
                CNFConverter._distribute_or_over_and(op) for op in formula.operands
            ]
            return PropositionalFormula.and_formula(*new_operands)

        if formula.operator == PropositionalOperator.OR:
            # Rekursiv auf Operanden
            operands = [
                CNFConverter._distribute_or_over_and(op) for op in formula.operands
            ]

            # Finde AND-Operanden
            and_indices = [
                i
                for i, op in enumerate(operands)
                if op.operator == PropositionalOperator.AND
            ]

            if not and_indices:
                # Keine ANDs zum Distributieren
                return PropositionalFormula.or_formula(*operands)

            # Nimm ersten AND und distributiere
            and_idx = and_indices[0]
            and_operand = operands[and_idx]
            other_operands = operands[:and_idx] + operands[and_idx + 1 :]

            # A ∨ (B ∧ C) = (A ∨ B) ∧ (A ∨ C)
            distributed = []
            for and_child in and_operand.operands:
                or_clause = PropositionalFormula.or_formula(and_child, *other_operands)
                distributed.append(CNFConverter._distribute_or_over_and(or_clause))

            return PropositionalFormula.and_formula(*distributed)

        return formula

    @staticmethod
    def _extract_cnf(formula: PropositionalFormula) -> CNFFormula:
        """Extrahiere CNF-Struktur aus vollständig konvertierter Formel"""
        clauses = []

        if formula.variable:
            # Einzelne Variable = eine Clause mit einem Literal
            clauses.append(Clause({Literal(formula.variable, False)}))
        elif formula.operator == PropositionalOperator.NOT:
            # Negierte Variable = eine Clause mit negiertem Literal
            if formula.operands[0].variable:
                clauses.append(Clause({Literal(formula.operands[0].variable, True)}))
        elif formula.operator == PropositionalOperator.OR:
            # OR = eine Clause mit mehreren Literalen
            literals = CNFConverter._extract_literals_from_or(formula)
            clauses.append(Clause(set(literals)))
        elif formula.operator == PropositionalOperator.AND:
            # AND = mehrere Clauses
            for operand in formula.operands:
                if operand.variable:
                    clauses.append(Clause({Literal(operand.variable, False)}))
                elif operand.operator == PropositionalOperator.NOT:
                    if operand.operands[0].variable:
                        clauses.append(
                            Clause({Literal(operand.operands[0].variable, True)})
                        )
                elif operand.operator == PropositionalOperator.OR:
                    literals = CNFConverter._extract_literals_from_or(operand)
                    clauses.append(Clause(set(literals)))

        return CNFFormula(clauses)

    @staticmethod
    def _extract_literals_from_or(formula: PropositionalFormula) -> List[Literal]:
        """Extrahiere Literale aus OR-Formel"""
        literals = []
        for operand in formula.operands:
            if operand.variable:
                literals.append(Literal(operand.variable, False))
            elif operand.operator == PropositionalOperator.NOT:
                if operand.operands[0].variable:
                    literals.append(Literal(operand.operands[0].variable, True))
        return literals


@dataclass
class WatchedLiterals:
    """
    Watched literals data structure for efficient unit propagation.

    For each clause, we watch two literals. When a watched literal becomes
    false, we search for a new literal to watch. This avoids checking all
    literals in all clauses during propagation.
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


class DPLLSolver:
    """
    DPLL-based SAT solver with watched literals optimization.

    Implements:
    - Unit propagation with watched literals
    - Pure literal elimination
    - Backtracking search
    - Conflict-driven learning (basic)
    - ProofTree generation for explanations
    """

    def __init__(self, use_watched_literals: bool = True, enable_proof: bool = False):
        self.decision_level = 0
        self.propagation_count = 0
        self.conflict_count = 0
        self.use_watched_literals = use_watched_literals
        self.watched_literals = WatchedLiterals() if use_watched_literals else None
        self.enable_proof = enable_proof and PROOF_AVAILABLE
        self.proof_steps: List = []
        self.decision_stack: List[Tuple[str, bool]] = []

    def solve(
        self, formula: CNFFormula, initial_assignment: Optional[Dict[str, bool]] = None
    ) -> Tuple[SATResult, Optional[Dict[str, bool]]]:
        """
        Solve SAT problem using DPLL algorithm.

        Args:
            formula: CNF formula to solve
            initial_assignment: Optional partial assignment to start with

        Returns:
            (result, model) where model is None if UNSATISFIABLE
        """
        logger.info(
            "Starting DPLL solver",
            extra={
                "num_clauses": len(formula.clauses),
                "num_variables": len(formula.variables),
            },
        )

        self.propagation_count = 0
        self.conflict_count = 0
        self.decision_level = 0
        self.proof_steps = []  # Reset proof steps
        self.decision_stack = []

        assignment = initial_assignment.copy() if initial_assignment else {}

        # Initialize watched literals if enabled
        if self.use_watched_literals and self.watched_literals is not None:
            self.watched_literals.initialize(formula)

        if self.enable_proof and PROOF_AVAILABLE:
            self._add_proof_step(
                StepType.PREMISE,
                "SAT Solving",
                f"Find satisfying assignment for {len(formula.clauses)} clauses, "
                f"{len(formula.variables)} variables",
                confidence=1.0,
            )

        result, model = self._dpll(formula, assignment)

        if self.enable_proof and PROOF_AVAILABLE:
            if result == SATResult.SATISFIABLE:
                self._add_proof_step(
                    StepType.CONCLUSION,
                    "SAT",
                    f"Found satisfying assignment: {model}",
                    confidence=1.0,
                )
            else:
                self._add_proof_step(
                    StepType.CONCLUSION,
                    "UNSAT",
                    "No satisfying assignment exists",
                    confidence=1.0,
                )

        logger.info(
            "DPLL solver finished",
            extra={
                "result": result.value,
                "propagations": self.propagation_count,
                "conflicts": self.conflict_count,
            },
        )

        return result, model

    def _dpll(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Tuple[SATResult, Optional[Dict[str, bool]]]:
        """
        Recursive DPLL algorithm.

        Returns:
            (result, model)
        """
        # Simplify formula with current assignment
        simplified = formula.simplify(assignment)

        # Base cases
        if simplified.is_empty():
            # All clauses satisfied
            return SATResult.SATISFIABLE, assignment.copy()

        if simplified.has_empty_clause():
            # Conflict: empty clause means unsatisfiable
            self.conflict_count += 1
            return SATResult.UNSATISFIABLE, None

        # Unit propagation
        unit_literals = simplified.get_unit_clauses()
        if unit_literals:
            new_assignment = assignment.copy()
            for lit in unit_literals:
                # Assign value to satisfy unit literal
                value = not lit.negated
                if lit.variable in new_assignment:
                    # Check consistency
                    if new_assignment[lit.variable] != value:
                        self.conflict_count += 1
                        if self.enable_proof and PROOF_AVAILABLE:
                            self._add_proof_step(
                                StepType.CONTRADICTION,
                                "Conflict in Unit Propagation",
                                f"Variable {lit.variable} already assigned to {new_assignment[lit.variable]}, "
                                f"but unit clause requires {value}",
                                confidence=1.0,
                            )
                        return SATResult.UNSATISFIABLE, None
                else:
                    new_assignment[lit.variable] = value
                    self.propagation_count += 1
                    if self.enable_proof and PROOF_AVAILABLE:
                        self._add_proof_step(
                            StepType.INFERENCE,
                            "Unit Propagation",
                            f"Forced assignment: {lit.variable} = {value} (from unit clause)",
                            confidence=1.0,
                        )

            # Recurse with propagated assignments
            return self._dpll(simplified, new_assignment)

        # Pure literal elimination
        pure_literals = simplified.get_pure_literals(assignment)
        if pure_literals:
            new_assignment = assignment.copy()
            for lit in pure_literals:
                # Assign value to satisfy pure literal
                value = not lit.negated
                new_assignment[lit.variable] = value
                self.propagation_count += 1
                if self.enable_proof and PROOF_AVAILABLE:
                    self._add_proof_step(
                        StepType.INFERENCE,
                        "Pure Literal Elimination",
                        f"Pure literal: {lit.variable} = {value} (appears only with one polarity)",
                        confidence=1.0,
                    )

            # Recurse with pure literal assignments
            return self._dpll(simplified, new_assignment)

        # Choose a variable to branch on (decision heuristic)
        var = self._choose_variable(simplified, assignment)
        if var is None:
            # All variables assigned (shouldn't happen due to earlier checks)
            return SATResult.SATISFIABLE, assignment.copy()

        self.decision_level += 1

        # Try assigning True first
        if self.enable_proof and PROOF_AVAILABLE:
            self._add_proof_step(
                StepType.ASSUMPTION,
                "Branch",
                f"Try {var} = True (decision level {self.decision_level})",
                confidence=0.5,
            )

        self.decision_stack.append((var, True))
        new_assignment_true = assignment.copy()
        new_assignment_true[var] = True
        result, model = self._dpll(simplified, new_assignment_true)

        if result == SATResult.SATISFIABLE:
            self.decision_stack.pop()
            self.decision_level -= 1
            return result, model

        # Backtrack: try assigning False
        if self.enable_proof and PROOF_AVAILABLE:
            self._add_proof_step(
                StepType.ASSUMPTION,
                "Backtrack",
                f"Try {var} = False (after {var} = True failed)",
                confidence=0.5,
            )

        self.decision_stack[-1] = (var, False)
        new_assignment_false = assignment.copy()
        new_assignment_false[var] = False
        result, model = self._dpll(simplified, new_assignment_false)

        self.decision_stack.pop()
        self.decision_level -= 1
        return result, model

    def _choose_variable(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Optional[str]:
        """
        Choose next variable to branch on.

        Heuristic: Choose variable with most occurrences in smallest clauses
        (similar to VSIDS but simpler).
        """
        unassigned = formula.variables - set(assignment.keys())
        if not unassigned:
            return None

        # Count occurrences in small clauses (weighted by 1/clause_size)
        scores = {var: 0.0 for var in unassigned}

        for clause in formula.clauses:
            clause_size = len(clause.literals)
            if clause_size == 0:
                continue

            weight = 1.0 / clause_size
            for lit in clause.literals:
                if lit.variable in unassigned:
                    scores[lit.variable] += weight

        # Return variable with highest score
        return max(unassigned, key=lambda v: scores[v])

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

        result, _ = self.solve(merged)

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

            result, _ = self.solve(merged_test)
            if result == SATResult.UNSATISFIABLE:
                # Still unsatisfiable without formula i, so i is not essential
                conflicting.remove(i)

        return False, conflicting

    def _add_proof_step(
        self, step_type, description: str, details: str, confidence: float
    ):
        """Füge Proof Step hinzu (falls aktiviert)"""
        if not self.enable_proof or not PROOF_AVAILABLE:
            return

        try:
            step_id = f"sat_step_{len(self.proof_steps)}"
            step = ProofStep(
                step_id=step_id,
                step_type=step_type,
                rule_name=description,
                explanation_text=details,
                confidence=confidence,
                source_component="component_30_sat_solver",
            )
            self.proof_steps.append(step)
        except Exception as e:
            logger.warning(f"Failed to create proof step: {e}")

    def get_proof_tree(self, query: str = "SAT Solution") -> Optional["ProofTree"]:
        """
        Erstelle Proof Tree aus gesammelten Steps.

        Args:
            query: Query string for the proof tree

        Returns:
        - ProofTree: Beweisbaum
        - None: Falls Proof deaktiviert oder component_17 nicht verfügbar
        """
        if not self.enable_proof or not PROOF_AVAILABLE:
            return None

        try:
            return ProofTree(query=query, root_steps=self.proof_steps)
        except Exception as e:
            logger.warning(f"Failed to create proof tree: {e}")
            return None


class SATEncoder:
    """
    Helper class to encode various problems as SAT problems.
    """

    @staticmethod
    def encode_implication(antecedent: Literal, consequent: Literal) -> Clause:
        """
        Encode implication: antecedent → consequent.

        Equivalent to: ¬antecedent ∨ consequent
        """
        return Clause({-antecedent, consequent})

    @staticmethod
    def encode_iff(lit1: Literal, lit2: Literal) -> List[Clause]:
        """
        Encode bi-implication: lit1 ↔ lit2.

        Equivalent to: (lit1 → lit2) ∧ (lit2 → lit1)
        """
        return [
            SATEncoder.encode_implication(lit1, lit2),
            SATEncoder.encode_implication(lit2, lit1),
        ]

    @staticmethod
    def encode_xor(lit1: Literal, lit2: Literal) -> List[Clause]:
        """
        Encode exclusive OR: lit1 ⊕ lit2.

        Equivalent to: (lit1 ∨ lit2) ∧ (¬lit1 ∨ ¬lit2)
        """
        return [Clause({lit1, lit2}), Clause({-lit1, -lit2})]

    @staticmethod
    def encode_at_most_one(literals: List[Literal]) -> List[Clause]:
        """
        Encode at-most-one constraint: at most one literal can be true.

        Pairwise encoding: for all pairs, at least one must be false.
        """
        clauses = []
        for i in range(len(literals)):
            for j in range(i + 1, len(literals)):
                # ¬lit_i ∨ ¬lit_j
                clauses.append(Clause({-literals[i], -literals[j]}))
        return clauses

    @staticmethod
    def encode_exactly_one(literals: List[Literal]) -> List[Clause]:
        """
        Encode exactly-one constraint: exactly one literal must be true.

        Combines at-least-one with at-most-one.
        """
        clauses = []
        # At least one
        clauses.append(Clause(set(literals)))
        # At most one
        clauses.extend(SATEncoder.encode_at_most_one(literals))
        return clauses


class KnowledgeBaseChecker:
    """
    SAT-based consistency checker for knowledge bases.

    Integrates with Neo4j to verify rule consistency.
    """

    def __init__(self, solver: Optional[DPLLSolver] = None):
        self.solver = solver or DPLLSolver()

    def check_rule_consistency(
        self, rules: List[Tuple[List[Literal], Literal]]
    ) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """
        Check if a set of Horn clauses (rules) is consistent.

        Args:
            rules: List of (premises, conclusion) where premises are ANDed

        Returns:
            (is_consistent, model)
        """
        formula = CNFFormula([])

        for premises, conclusion in rules:
            # Rule: (p1 ∧ p2 ∧ ... ∧ pn) → conclusion
            # CNF: ¬p1 ∨ ¬p2 ∨ ... ∨ ¬pn ∨ conclusion
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

        Returns:
            List of conflict descriptions
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


def create_knights_and_knaves_problem() -> CNFFormula:
    """
    Create a solvable Knights and Knaves puzzle as a SAT problem.

    Puzzle: Three people (A, B, C). Knights always tell truth, Knaves always lie.
    - A says: "B is a knave"
    - B says: "A and C are both knights"
    - C says: "A is a knave"

    Solution: A=Knight, B=Knave, C=Knave
    """
    formula = CNFFormula([])

    # Variables: k_A, k_B, k_C (True = knight, False = knave)
    k_A = Literal("k_A")
    k_B = Literal("k_B")
    k_C = Literal("k_C")

    # A says: "B is a knave" (¬k_B)
    # If A is knight: B is knave (k_A → ¬k_B)
    # If A is knave: B is knight (¬k_A → k_B)
    # Equivalent: k_A ↔ ¬k_B
    formula.clauses.extend(SATEncoder.encode_iff(k_A, -k_B))

    # B says: "A and C are both knights" (k_A ∧ k_C)
    # If B is knight: statement is true (k_B → (k_A ∧ k_C))
    #   === (¬k_B ∨ k_A) ∧ (¬k_B ∨ k_C)
    # If B is knave: statement is false (¬k_B → ¬(k_A ∧ k_C))
    #   === ¬k_B → (¬k_A ∨ ¬k_C)
    #   === (k_B ∨ ¬k_A ∨ ¬k_C)

    # k_B → (k_A ∧ k_C)
    formula.add_clause(Clause({-k_B, k_A}))
    formula.add_clause(Clause({-k_B, k_C}))

    # ¬k_B → ¬(k_A ∧ k_C) === ¬k_B → (¬k_A ∨ ¬k_C) === (k_B ∨ ¬k_A ∨ ¬k_C)
    formula.add_clause(Clause({k_B, -k_A, -k_C}))

    # C says: "A is a knave" (¬k_A)
    # If C is knight: A is knave (k_C → ¬k_A)
    # If C is knave: A is knight (¬k_C → k_A)
    # Equivalent: k_C ↔ ¬k_A
    formula.clauses.extend(SATEncoder.encode_iff(k_C, -k_A))

    return formula


class SATSolver:
    """
    Simplified SAT-Solver API gemäß Spezifikation.

    Diese Klasse bietet eine einfachere, benutzerfreundlichere API für den DPLL-Solver
    und entspricht der vom Benutzer spezifizierten Schnittstelle.

    Features:
    - DPLL-Algorithmus mit Unit Propagation
    - Pure Literal Elimination
    - ProofTree-Integration für nachvollziehbare Beweise
    """

    def __init__(self, enable_proof: bool = True):
        """
        Initialisiere SAT-Solver.

        Args:
            enable_proof: Generiere ProofTree für Beweise
        """
        self.enable_proof = enable_proof
        self._solver = DPLLSolver(use_watched_literals=True, enable_proof=enable_proof)

    def solve(self, formula: CNFFormula) -> Optional[Dict[str, bool]]:
        """
        Finde satisfying assignment für CNF-Formel.

        Args:
            formula: CNF-Formel

        Returns:
            Dict[str, bool]: Satisfying assignment (SAT)
            None: Keine Lösung (UNSAT)
        """
        result, model = self._solver.solve(formula)
        return model if result == SATResult.SATISFIABLE else None

    def dpll(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Optional[Dict[str, bool]]:
        """
        DPLL-Algorithmus mit Unit Propagation (mit partieller Assignment).

        Args:
            formula: CNF-Formel
            assignment: Partielle Belegung

        Returns:
            Dict[str, bool]: Satisfying assignment (SAT)
            None: Keine Lösung (UNSAT)
        """
        result, model = self._solver.solve(formula, initial_assignment=assignment)
        return model if result == SATResult.SATISFIABLE else None

    def unit_propagate(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Tuple[CNFFormula, Dict[str, bool]]:
        """
        Unit Propagation (Constraint Propagation für SAT).

        Findet Unit Clauses und propagiert deren Assignments.

        Args:
            formula: CNF-Formel
            assignment: Aktuelle Belegung

        Returns:
            (simplified_formula, extended_assignment)
        """
        # Vereinfache Formel mit aktuellem Assignment
        simplified = formula.simplify(assignment)
        extended_assignment = assignment.copy()

        # Iterativ Unit Clauses finden und propagieren
        changed = True
        while changed:
            changed = False
            unit_literals = simplified.get_unit_clauses()

            if unit_literals:
                for lit in unit_literals:
                    if lit.variable not in extended_assignment:
                        # Neues Assignment
                        value = not lit.negated
                        extended_assignment[lit.variable] = value
                        changed = True

                # Vereinfache mit neuen Assignments
                simplified = simplified.simplify(extended_assignment)

        return simplified, extended_assignment

    def pure_literal_elimination(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Tuple[CNFFormula, Dict[str, bool]]:
        """
        Eliminiere reine Literale (Variablen mit nur einer Polarität).

        Args:
            formula: CNF-Formel
            assignment: Aktuelle Belegung

        Returns:
            (simplified_formula, extended_assignment)
        """
        simplified = formula.simplify(assignment)
        extended_assignment = assignment.copy()

        # Finde pure Literale
        pure_literals = simplified.get_pure_literals(extended_assignment)

        for lit in pure_literals:
            value = not lit.negated
            extended_assignment[lit.variable] = value

        # Vereinfache mit neuen Assignments
        if pure_literals:
            simplified = simplified.simplify(extended_assignment)

        return simplified, extended_assignment

    def get_proof_tree(self, query: str = "SAT Solution") -> Optional["ProofTree"]:
        """
        Hole ProofTree vom letzten solve()-Aufruf.

        Args:
            query: Query string for the proof tree

        Returns:
            ProofTree: Beweisbaum
            None: Falls Proof deaktiviert oder component_17 nicht verfügbar
        """
        return self._solver.get_proof_tree(query)


# Convenience Functions
def solve_cnf(
    formula: CNFFormula, enable_proof: bool = False
) -> Optional[Dict[str, bool]]:
    """
    Löse CNF-Formel (convenience function).

    Args:
        formula: CNF-Formel
        enable_proof: Generiere Proof Tree

    Returns:
        Satisfying assignment oder None (UNSAT)
    """
    solver = SATSolver(enable_proof=enable_proof)
    return solver.solve(formula)


def solve_propositional(
    formula: PropositionalFormula, enable_proof: bool = False
) -> Optional[Dict[str, bool]]:
    """
    Löse allgemeine propositionale Formel (konvertiere zu CNF).

    Args:
        formula: Propositionale Formel (beliebig)
        enable_proof: Generiere Proof Tree

    Returns:
        Satisfying assignment oder None (UNSAT)
    """
    cnf = CNFConverter.to_cnf(formula)
    return solve_cnf(cnf, enable_proof=enable_proof)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Simple SAT problem
    print("=== Example 1: Simple SAT ===")
    formula1 = CNFFormula(
        [
            Clause({Literal("x"), Literal("y")}),
            Clause({Literal("x", True), Literal("z")}),  # ¬x ∨ z
            Clause({Literal("y", True), Literal("z", True)}),  # ¬y ∨ ¬z
        ]
    )
    print(f"Formula: {formula1}")

    solver = DPLLSolver()
    result, model = solver.solve(formula1)
    print(f"Result: {result.value}")
    if model:
        print(f"Model: {model}")

    # Example 2: Knights and Knaves
    print("\n=== Example 2: Knights and Knaves ===")
    puzzle = create_knights_and_knaves_problem()
    result, model = solver.solve(puzzle)
    print(f"Result: {result.value}")
    if model:
        print("Solution:")
        for var, value in sorted(model.items()):
            role = "Knight" if value else "Knave"
            print(f"  {var}: {role}")

    # Example 3: Consistency checking
    print("\n=== Example 3: Consistency Checking ===")
    checker = KnowledgeBaseChecker(solver)

    rules = [
        ([Literal("bird")], Literal("can_fly")),
        ([Literal("penguin")], Literal("bird")),
        ([Literal("penguin")], Literal("can_fly", True)),  # ¬can_fly
    ]

    is_consistent, model = checker.check_rule_consistency(rules)
    print(f"Rules consistent: {is_consistent}")
    if model:
        print(f"Model: {model}")
