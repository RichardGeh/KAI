"""
component_30_cnf_converter.py

CNF Conversion and SAT Encoding Utilities

This module provides tools for converting propositional formulas to CNF
and encoding common constraints as SAT problems:
- PropositionalFormula data structure for general formulas
- CNF conversion (elimination of implications, De Morgan, distribution)
- SAT encoders for common constraints (implication, iff, at-most-one, etc.)
- Example problem encodings (Knights and Knaves, etc.)

SPLIT FROM: component_30_sat_solver.py (Task 12 - Phase 4, 2025-11-28)

Author: KAI Development Team
Date: 2025-10-30 | Refactored: 2025-11-28
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from component_15_logging_config import get_logger
from component_30_sat_solver_core import Clause, CNFFormula, Literal

logger = get_logger(__name__)


# ============================================================================
# Propositional Formula Representation
# ============================================================================


class PropositionalOperator(Enum):
    """Propositional operators for general formulas"""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    IFF = "IFF"  # Biconditional (if and only if)


@dataclass
class PropositionalFormula:
    """
    General propositional formula (not necessarily CNF).
    Can be converted to CNF.
    """

    operator: PropositionalOperator | None = None
    variable: str | None = None
    operands: List["PropositionalFormula"] = field(default_factory=list)

    def __str__(self) -> str:
        if self.variable:
            return self.variable
        if self.operator == PropositionalOperator.NOT:
            return f"NOT {self.operands[0]}"
        if self.operator == PropositionalOperator.AND:
            return f"({' AND '.join(str(op) for op in self.operands)})"
        if self.operator == PropositionalOperator.OR:
            return f"({' OR '.join(str(op) for op in self.operands)})"
        if self.operator == PropositionalOperator.IMPLIES:
            return f"({self.operands[0]} -> {self.operands[1]})"
        if self.operator == PropositionalOperator.IFF:
            return f"({self.operands[0]} <-> {self.operands[1]})"
        return "?"

    @classmethod
    def variable_formula(cls, var: str) -> "PropositionalFormula":
        """Create variable"""
        return cls(variable=var)

    @classmethod
    def not_formula(cls, operand: "PropositionalFormula") -> "PropositionalFormula":
        """Create negation"""
        return cls(operator=PropositionalOperator.NOT, operands=[operand])

    @classmethod
    def and_formula(cls, *operands: "PropositionalFormula") -> "PropositionalFormula":
        """Create conjunction"""
        return cls(operator=PropositionalOperator.AND, operands=list(operands))

    @classmethod
    def or_formula(cls, *operands: "PropositionalFormula") -> "PropositionalFormula":
        """Create disjunction"""
        return cls(operator=PropositionalOperator.OR, operands=list(operands))

    @classmethod
    def implies_formula(
        cls, antecedent: "PropositionalFormula", consequent: "PropositionalFormula"
    ) -> "PropositionalFormula":
        """Create implication"""
        return cls(
            operator=PropositionalOperator.IMPLIES, operands=[antecedent, consequent]
        )

    @classmethod
    def iff_formula(
        cls, left: "PropositionalFormula", right: "PropositionalFormula"
    ) -> "PropositionalFormula":
        """Create biconditional"""
        return cls(operator=PropositionalOperator.IFF, operands=[left, right])


# ============================================================================
# CNF Converter
# ============================================================================


class CNFConverter:
    """
    Convert propositional formulas to CNF (Conjunctive Normal Form).

    Steps:
    1. Eliminate implications and biconditionals
    2. Push negations inward (De Morgan)
    3. Distribute OR over AND
    """

    @staticmethod
    def to_cnf(formula: PropositionalFormula) -> CNFFormula:
        """Main method: Convert to CNF"""
        # Step 1: Eliminate implications
        formula = CNFConverter._eliminate_implications(formula)

        # Step 2: Push negations inward
        formula = CNFConverter._push_negations_inward(formula)

        # Step 3: Distribute OR over AND
        formula = CNFConverter._distribute_or_over_and(formula)

        # Step 4: Extract CNF structure
        return CNFConverter._extract_cnf(formula)

    @staticmethod
    def _eliminate_implications(formula: PropositionalFormula) -> PropositionalFormula:
        """
        Eliminate implications and biconditionals.
        - A -> B becomes NOT A OR B
        - A <-> B becomes (A -> B) AND (B -> A) = (NOT A OR B) AND (NOT B OR A)
        """
        if formula.variable:
            return formula

        if formula.operator == PropositionalOperator.IMPLIES:
            # A -> B = NOT A OR B
            ant = CNFConverter._eliminate_implications(formula.operands[0])
            cons = CNFConverter._eliminate_implications(formula.operands[1])
            return PropositionalFormula.or_formula(
                PropositionalFormula.not_formula(ant), cons
            )

        if formula.operator == PropositionalOperator.IFF:
            # A <-> B = (NOT A OR B) AND (NOT B OR A)
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

        # Recursively apply to operands
        new_operands = [
            CNFConverter._eliminate_implications(op) for op in formula.operands
        ]
        return PropositionalFormula(operator=formula.operator, operands=new_operands)

    @staticmethod
    def _push_negations_inward(formula: PropositionalFormula) -> PropositionalFormula:
        """
        Push negations inward with De Morgan:
        - NOT (A AND B) = NOT A OR NOT B
        - NOT (A OR B) = NOT A AND NOT B
        - NOT NOT A = A
        """
        if formula.variable:
            return formula

        if formula.operator != PropositionalOperator.NOT:
            # Not-NOT: Recursively on operands
            new_operands = [
                CNFConverter._push_negations_inward(op) for op in formula.operands
            ]
            return PropositionalFormula(
                operator=formula.operator, operands=new_operands
            )

        # NOT: Look at inner operator
        inner = formula.operands[0]

        if inner.variable:
            # NOT Variable: Done
            return formula

        if inner.operator == PropositionalOperator.NOT:
            # NOT NOT A = A
            return CNFConverter._push_negations_inward(inner.operands[0])

        if inner.operator == PropositionalOperator.AND:
            # NOT (A AND B) = NOT A OR NOT B
            negated_operands = [
                CNFConverter._push_negations_inward(
                    PropositionalFormula.not_formula(op)
                )
                for op in inner.operands
            ]
            return PropositionalFormula.or_formula(*negated_operands)

        if inner.operator == PropositionalOperator.OR:
            # NOT (A OR B) = NOT A AND NOT B
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
        Distribute OR over AND:
        - A OR (B AND C) = (A OR B) AND (A OR C)
        """
        if formula.variable:
            return formula

        if formula.operator == PropositionalOperator.NOT:
            # Negation should already be pushed inward
            return formula

        if formula.operator == PropositionalOperator.AND:
            # Recursively on operands
            new_operands = [
                CNFConverter._distribute_or_over_and(op) for op in formula.operands
            ]
            return PropositionalFormula.and_formula(*new_operands)

        if formula.operator == PropositionalOperator.OR:
            # Recursively on operands
            operands = [
                CNFConverter._distribute_or_over_and(op) for op in formula.operands
            ]

            # Find AND operands
            and_indices = [
                i
                for i, op in enumerate(operands)
                if op.operator == PropositionalOperator.AND
            ]

            if not and_indices:
                # No ANDs to distribute
                return PropositionalFormula.or_formula(*operands)

            # Take first AND and distribute
            and_idx = and_indices[0]
            and_operand = operands[and_idx]
            other_operands = operands[:and_idx] + operands[and_idx + 1 :]

            # A OR (B AND C) = (A OR B) AND (A OR C)
            distributed = []
            for and_child in and_operand.operands:
                or_clause = PropositionalFormula.or_formula(and_child, *other_operands)
                distributed.append(CNFConverter._distribute_or_over_and(or_clause))

            return PropositionalFormula.and_formula(*distributed)

        return formula

    @staticmethod
    def _extract_cnf(formula: PropositionalFormula) -> CNFFormula:
        """Extract CNF structure from fully converted formula"""
        clauses = []

        if formula.variable:
            # Single variable = one clause with one literal
            clauses.append(Clause({Literal(formula.variable, False)}))
        elif formula.operator == PropositionalOperator.NOT:
            # Negated variable = one clause with negated literal
            if formula.operands[0].variable:
                clauses.append(Clause({Literal(formula.operands[0].variable, True)}))
        elif formula.operator == PropositionalOperator.OR:
            # OR = one clause with multiple literals
            literals = CNFConverter._extract_literals_from_or(formula)
            clauses.append(Clause(set(literals)))
        elif formula.operator == PropositionalOperator.AND:
            # AND = multiple clauses
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
        """Extract literals from OR formula"""
        literals = []
        for operand in formula.operands:
            if operand.variable:
                literals.append(Literal(operand.variable, False))
            elif operand.operator == PropositionalOperator.NOT:
                if operand.operands[0].variable:
                    literals.append(Literal(operand.operands[0].variable, True))
        return literals


# ============================================================================
# SAT Encoder (Common Constraints)
# ============================================================================


class SATEncoder:
    """
    Helper class to encode various constraints as SAT problems.
    """

    @staticmethod
    def encode_implication(antecedent: Literal, consequent: Literal) -> Clause:
        """
        Encode implication: antecedent -> consequent.

        Equivalent to: NOT antecedent OR consequent
        """
        return Clause({-antecedent, consequent})

    @staticmethod
    def encode_iff(lit1: Literal, lit2: Literal) -> List[Clause]:
        """
        Encode bi-implication: lit1 <-> lit2.

        Equivalent to: (lit1 -> lit2) AND (lit2 -> lit1)
        """
        return [
            SATEncoder.encode_implication(lit1, lit2),
            SATEncoder.encode_implication(lit2, lit1),
        ]

    @staticmethod
    def encode_xor(lit1: Literal, lit2: Literal) -> List[Clause]:
        """
        Encode exclusive OR: lit1 XOR lit2.

        Equivalent to: (lit1 OR lit2) AND (NOT lit1 OR NOT lit2)
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
                # NOT lit_i OR NOT lit_j
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


# ============================================================================
# Example Problem Encodings
# ============================================================================


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

    # A says: "B is a knave" (NOT k_B)
    # If A is knight: B is knave (k_A -> NOT k_B)
    # If A is knave: B is knight (NOT k_A -> k_B)
    # Equivalent: k_A <-> NOT k_B
    formula.clauses.extend(SATEncoder.encode_iff(k_A, -k_B))

    # B says: "A and C are both knights" (k_A AND k_C)
    # If B is knight: statement is true (k_B -> (k_A AND k_C))
    #   === (NOT k_B OR k_A) AND (NOT k_B OR k_C)
    # If B is knave: statement is false (NOT k_B -> NOT (k_A AND k_C))
    #   === NOT k_B -> (NOT k_A OR NOT k_C)
    #   === (k_B OR NOT k_A OR NOT k_C)

    # k_B -> (k_A AND k_C)
    formula.add_clause(Clause({-k_B, k_A}))
    formula.add_clause(Clause({-k_B, k_C}))

    # NOT k_B -> NOT (k_A AND k_C) === NOT k_B -> (NOT k_A OR NOT k_C) === (k_B OR NOT k_A OR NOT k_C)
    formula.add_clause(Clause({k_B, -k_A, -k_C}))

    # C says: "A is a knave" (NOT k_A)
    # If C is knight: A is knave (k_C -> NOT k_A)
    # If C is knave: A is knight (NOT k_C -> k_A)
    # Equivalent: k_C <-> NOT k_A
    formula.clauses.extend(SATEncoder.encode_iff(k_C, -k_A))

    return formula


# ============================================================================
# Convenience Functions
# ============================================================================


def solve_propositional(formula: PropositionalFormula, enable_proof: bool = False):
    """
    Solve general propositional formula (convert to CNF).

    Args:
        formula: Propositional formula (arbitrary)
        enable_proof: Generate proof tree

    Returns:
        Satisfying assignment or None (UNSAT)
    """
    from component_30_sat_solver_core import DPLLSolver, SATResult

    cnf = CNFConverter.to_cnf(formula)
    solver = DPLLSolver(enable_proof=enable_proof)
    result, model = solver.solve(cnf)
    return model if result == SATResult.SATISFIABLE else None


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "PropositionalOperator",
    "PropositionalFormula",
    "CNFConverter",
    "SATEncoder",
    "create_knights_and_knaves_problem",
    "solve_propositional",
]
