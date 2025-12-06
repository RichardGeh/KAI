"""
component_45_numerical_constraint_parser.py
==========================================
Parser für numerische Constraints in deutschen Logic Puzzles.

Verantwortlichkeiten:
- Parsen von Teilbarkeits-Constraints ("teilbar durch X")
- Parsen von arithmetischen Constraints ("Summe der", "Differenz")
- Parsen von Meta-Constraints ("Anzahl der richtigen Behauptungen")
- Extraktion von numerischen Variablen und Domains
- Konvertierung zu formalen Constraint-Strukturen

WICHTIG: KEINE Unicode-Zeichen verwenden (nur ASCII: AND, OR, NOT, IMPLIES)

Author: KAI Development Team
Date: 2025-12-05 (PHASE 2)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class ConstraintType(Enum):
    """Typen von numerischen Constraints."""

    DIVISIBILITY = "divisibility"  # Teilbarkeit: "teilbar durch X"
    ARITHMETIC = "arithmetic"  # Arithmetik: "Summe", "Differenz", "Produkt"
    META = "meta"  # Meta-Constraints: "Anzahl der richtigen"
    LOGICAL = "logical"  # Logische Bedingungen: "aufeinanderfolgend"
    COMPARISON = "comparison"  # Vergleiche: "groesser als", "kleiner als"


@dataclass
class NumConstraint:
    """
    Repräsentiert einen numerischen Constraint.

    Attributes:
        constraint_type: Art des Constraints (DIVISIBILITY, ARITHMETIC, etc.)
        variables: Beteiligte Variablen
        expression: Callable zum Testen des Constraints
        statement_id: ID der Aussage (für Meta-Constraints)
        description: Beschreibung in natürlicher Sprache
        metadata: Zusätzliche Metadaten
    """

    constraint_type: ConstraintType
    variables: List[str]
    expression: Callable[[Dict[str, int]], bool]
    statement_id: Optional[int] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NumericalVariable:
    """
    Repräsentiert eine numerische Variable.

    Attributes:
        name: Name der Variable (z.B. "zahl", "X", "statement_1")
        domain: Mögliche Werte (Set von Integers)
        is_meta: Ob es eine Meta-Variable ist (bezieht sich auf andere Variablen)
        description: Beschreibung in natürlicher Sprache
    """

    name: str
    domain: Set[int] = field(
        default_factory=lambda: set(range(1, 101))
    )  # Default: 1-100
    is_meta: bool = False
    description: str = ""


class NumericalConstraintParser:
    """
    Parser für numerische Constraints in deutschen Logic Puzzles.

    Workflow:
    1. Segmentiere Text in numbered statements
    2. Parse jeden Statement für Constraints
    3. Extrahiere Variablen und infere Domains
    4. Konvertiere zu formalen NumConstraint Objekten
    """

    def __init__(self):
        """Initialisiert den Parser."""
        # Patterns für Constraint-Typen
        self.patterns = {
            "divisibility": [
                (
                    r"teilbar\s+durch\s+(?:die\s+)?(\w+)",
                    lambda var, divisor: f"{var} teilbar durch {divisor}",
                ),
                (
                    r"vielfaches\s+von\s+(\w+)",
                    lambda var, mult: f"{var} Vielfaches von {mult}",
                ),
            ],
            "arithmetic": [
                (
                    r"summe\s+der\s+(\w+(?:\s+\w+)*)",
                    lambda vars_str: f"Summe({vars_str})",
                ),
                (
                    r"differenz\s+zwischen\s+(?:den\s+)?(\w+)\s+und\s+(\w+)",
                    lambda a, b: f"Differenz({a}, {b})",
                ),
                (
                    r"produkt\s+(?:der\s+)?(\w+(?:\s+\w+)*)",
                    lambda vars_str: f"Produkt({vars_str})",
                ),
            ],
            "comparison": [
                (
                    r"(?:ist\s+)?gr(?:oe|ö)sser\s+als\s+(\d+)",
                    lambda var, num: f"{var} > {num}",
                ),
                (
                    r"(?:ist\s+)?kleiner\s+als\s+(\d+)",
                    lambda var, num: f"{var} < {num}",
                ),
            ],
            "meta": [
                (
                    r"anzahl\s+der\s+(?:richtigen?|falschen?)\s+behauptungen?",
                    lambda: "Anzahl(richtige/falsche Behauptungen)",
                ),
                (
                    r"nummern?\s+der\s+(?:richtigen?|falschen?)",
                    lambda: "Nummern(richtige/falsche)",
                ),
            ],
        }

        logger.info("NumericalConstraintParser initialisiert")

    def parse_puzzle(self, text: str) -> Dict[str, Any]:
        """
        Parst ein numerisches Logic Puzzle.

        Args:
            text: Der Puzzle-Text (mit numbered statements)

        Returns:
            Dictionary mit:
            - constraints: List[NumConstraint] - Alle Constraints
            - variables: Dict[str, NumericalVariable] - Alle Variablen
            - statements: List[str] - Alle Statements
            - question: Optional[str] - Die Frage
        """
        logger.info("Parse numerisches Puzzle")

        # Schritt 1: Extrahiere Statements
        statements = self._extract_statements(text)
        logger.debug(f"Extrahiert: {len(statements)} Statements")

        # Schritt 2: Extrahiere Frage (letzter Satz ohne Nummer)
        question = self._extract_question(text)
        if question:
            logger.debug(f"Frage gefunden: {question[:50]}...")

        # Schritt 3: Parse Constraints aus Statements
        constraints = []
        for stmt_id, stmt_text in statements.items():
            stmt_constraints = self._parse_constraint(stmt_id, stmt_text)
            constraints.extend(stmt_constraints)

        logger.info(f"Parsed: {len(constraints)} Constraints")

        # Schritt 4: Extrahiere Variablen
        variables = self._extract_variables(text, statements)
        logger.debug(f"Extrahiert: {len(variables)} Variablen")

        # Schritt 5: Infere Domains
        variables = self._infer_domains(variables, constraints)

        return {
            "constraints": constraints,
            "variables": variables,
            "statements": statements,
            "question": question,
        }

    def _extract_statements(self, text: str) -> Dict[int, str]:
        """
        Extrahiert numbered statements aus Text.

        Format: "1. Text", "2. Text", etc.

        Returns:
            Dict[statement_id, statement_text]
        """
        statements = {}
        # ROBUST PATTERN: Works for both clean and mangled (joined) text
        # (\d+)[.)]\s+ = capture statement number, dot/paren, whitespace
        # ([^.?!]+[.!]) = capture text until sentence-ending punctuation (period or exclamation)
        #
        # Why this works:
        # - Handles "1. Text." at line start
        # - Handles "1. Text. 2. Text." on same line (after orchestrator joins segments)
        # - Stops at period/exclamation (won't include "Welche...?" question)
        # - Simple and robust
        pattern = r"(\d+)[.)]\s+([^.?!]+[.!])"

        matches = re.findall(pattern, text)

        for match in matches:
            stmt_id = int(match[0])
            stmt_text = match[1].strip()
            statements[stmt_id] = stmt_text

        return statements

    def _extract_question(self, text: str) -> Optional[str]:
        """
        Extrahiert die Frage aus dem Text.

        Annahme: Frage ist der letzte Satz und endet mit "?"
        """
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.endswith("?"):
                # Entferne numbered statement prefix falls vorhanden
                line = re.sub(r"^\d+[.)]\s+", "", line)
                return line

        return None

    def _parse_constraint(self, stmt_id: int, stmt_text: str) -> List[NumConstraint]:
        """
        Parst Constraints aus einem Statement.

        Args:
            stmt_id: ID des Statements
            stmt_text: Text des Statements

        Returns:
            Liste von NumConstraint Objekten
        """
        constraints = []
        stmt_lower = stmt_text.lower()

        # DIVISIBILITY Constraints
        if "teilbar" in stmt_lower:
            constraint = self._parse_divisibility_constraint(stmt_id, stmt_text)
            if constraint:
                constraints.append(constraint)

        # ARITHMETIC Constraints
        if "summe" in stmt_lower or "differenz" in stmt_lower:
            constraint = self._parse_arithmetic_constraint(stmt_id, stmt_text)
            if constraint:
                constraints.append(constraint)

        # COMPARISON Constraints
        if "größer" in stmt_lower or "kleiner" in stmt_lower or "grösser" in stmt_lower:
            constraint = self._parse_comparison_constraint(stmt_id, stmt_text)
            if constraint:
                constraints.append(constraint)

        # META Constraints
        if "anzahl" in stmt_lower or "nummern" in stmt_lower or "richtig" in stmt_lower:
            constraint = self._parse_meta_constraint(stmt_id, stmt_text)
            if constraint:
                constraints.append(constraint)

        return constraints

    def _parse_divisibility_constraint(
        self, stmt_id: int, stmt_text: str
    ) -> Optional[NumConstraint]:
        """Parst Teilbarkeits-Constraint."""
        # Pattern: Handle both "teilbar durch X" AND "durch X teilbar" (reversed)
        # Try "teilbar durch X" first (more common)
        pattern1 = r"teilbar\s+durch\s+(?:die\s+)?(.+?)(?:\.|$|,)"
        match = re.search(pattern1, stmt_text, re.IGNORECASE)

        # If no match, try reversed order: "durch X teilbar"
        if not match:
            pattern2 = r"durch\s+(\d+)\s+teilbar"
            match = re.search(pattern2, stmt_text, re.IGNORECASE)

        if not match:
            return None

        divisor_expr = match.group(1).strip()

        # Einfacher Fall: Zahl als Divisor
        if divisor_expr.isdigit():
            divisor = int(divisor_expr)

            def expr(vals: Dict[str, int]) -> bool:
                return vals.get("zahl", 0) % divisor == 0

            return NumConstraint(
                constraint_type=ConstraintType.DIVISIBILITY,
                variables=["zahl"],
                expression=expr,
                statement_id=stmt_id,
                description=f"Zahl teilbar durch {divisor}",
                metadata={"divisor": divisor},
            )

        # Komplexer Fall: Arithmetischer Ausdruck als Divisor
        else:
            # Markiere als Meta-Constraint (wird später aufgelöst)
            def expr(vals: Dict[str, int]) -> bool:
                # Placeholder - wird vom Solver evaluiert
                return True

            return NumConstraint(
                constraint_type=ConstraintType.META,
                variables=["zahl"],
                expression=expr,
                statement_id=stmt_id,
                description=f"Zahl teilbar durch {divisor_expr}",
                metadata={"divisor_expr": divisor_expr, "requires_evaluation": True},
            )

    def _parse_arithmetic_constraint(
        self, stmt_id: int, stmt_text: str
    ) -> Optional[NumConstraint]:
        """Parst arithmetischen Constraint."""
        # Pattern 1: "Summe der X ist teilbar durch Y"
        summe_pattern1 = r"summe\s+der\s+(.+?)\s+ist\s+teilbar\s+durch\s+(\d+)"
        match = re.search(summe_pattern1, stmt_text, re.IGNORECASE)

        if match:
            operands_str = match.group(1).strip()
            divisor = int(match.group(2))

            def expr(vals: Dict[str, int]) -> bool:
                # Placeholder - Meta-Constraint
                return True

            return NumConstraint(
                constraint_type=ConstraintType.META,
                variables=["zahl"],
                expression=expr,
                statement_id=stmt_id,
                description=f"Summe({operands_str}) teilbar durch {divisor}",
                metadata={
                    "operation": "sum",
                    "operands": operands_str,
                    "divisor": divisor,
                    "requires_evaluation": True,
                },
            )

        # Pattern 2: "Summe der X beträgt Y" (equals constraint)
        summe_pattern2 = r"summe\s+der\s+(.+?)\s+betr(?:ä|ae)gt\s+(\d+)"
        match = re.search(summe_pattern2, stmt_text, re.IGNORECASE)

        if match:
            operands_str = match.group(1).strip()
            target_sum = int(match.group(2))

            def expr(vals: Dict[str, int]) -> bool:
                # Placeholder - Meta-Constraint
                return True

            return NumConstraint(
                constraint_type=ConstraintType.META,
                variables=["zahl"],
                expression=expr,
                statement_id=stmt_id,
                description=f"Summe({operands_str}) = {target_sum}",
                metadata={
                    "operation": "sum",
                    "operands": operands_str,
                    "target_value": target_sum,
                    "requires_evaluation": True,
                },
            )

        return None

    def _parse_comparison_constraint(
        self, stmt_id: int, stmt_text: str
    ) -> Optional[NumConstraint]:
        """Parst Vergleichs-Constraint."""
        # Pattern 1: "groesser als NUMBER" oder "grösser als NUMBER" oder "größer als NUMBER"
        # Handles: oe/ö and ss/ß (Swiss vs German spelling)
        greater_pattern1 = r"gr(?:oe|ö)(?:ss|ß)er\s+als\s+(\d+)"
        match = re.search(greater_pattern1, stmt_text, re.IGNORECASE)

        if match:
            threshold = int(match.group(1))

            def expr(vals: Dict[str, int]) -> bool:
                # Meta-Constraint - benötigt Kontext
                return True

            return NumConstraint(
                constraint_type=ConstraintType.META,
                variables=["zahl"],
                expression=expr,
                statement_id=stmt_id,
                description=f"Wert > {threshold}",
                metadata={
                    "comparison": ">",
                    "threshold": threshold,
                    "requires_evaluation": True,
                },
            )

        # Pattern 2: "X ist größer als Y" (comparing two variables/expressions)
        # E.g., "erste Ziffer ist größer als letzte Ziffer"
        greater_pattern2 = r"(.+?)\s+ist\s+gr(?:oe|ö)(?:ss|ß)er\s+als\s+(?:die\s+)?(.+)"
        match = re.search(greater_pattern2, stmt_text, re.IGNORECASE)

        if match:
            left_expr = match.group(1).strip()
            right_expr = match.group(2).strip()

            def expr(vals: Dict[str, int]) -> bool:
                # Meta-Constraint - benötigt Kontext
                return True

            return NumConstraint(
                constraint_type=ConstraintType.META,
                variables=["zahl"],
                expression=expr,
                statement_id=stmt_id,
                description=f"{left_expr} > {right_expr}",
                metadata={
                    "comparison": ">",
                    "left": left_expr,
                    "right": right_expr,
                    "requires_evaluation": True,
                },
            )

        return None

    def _parse_meta_constraint(
        self, stmt_id: int, stmt_text: str
    ) -> Optional[NumConstraint]:
        """Parst Meta-Constraint (bezieht sich auf andere Constraints)."""
        # Pattern: "Anzahl der Teiler"
        if "anzahl der teiler" in stmt_text.lower():

            def expr(vals: Dict[str, int]) -> bool:
                # Placeholder
                return True

            return NumConstraint(
                constraint_type=ConstraintType.META,
                variables=["zahl"],
                expression=expr,
                statement_id=stmt_id,
                description="Anzahl der Teiler",
                metadata={"meta_type": "divisor_count", "requires_evaluation": True},
            )

        return None

    def _extract_variables(
        self, text: str, statements: Dict[int, str]
    ) -> Dict[str, NumericalVariable]:
        """
        Extrahiert numerische Variablen aus Text.

        Returns:
            Dict[var_name, NumericalVariable]
        """
        variables = {}

        # Heuristik 1: "gesuchte Zahl"
        if re.search(r"\bgesuchte\s+zahl\b", text, re.IGNORECASE):
            variables["zahl"] = NumericalVariable(
                name="zahl",
                domain=set(range(1, 101)),  # Default: 1-100
                is_meta=False,
                description="Die gesuchte Zahl",
            )

        # Heuristik 2: Statement truth values (für Meta-Constraints)
        for stmt_id in statements.keys():
            variables[f"statement_{stmt_id}"] = NumericalVariable(
                name=f"statement_{stmt_id}",
                domain={0, 1},  # Boolean: 0=false, 1=true
                is_meta=True,
                description=f"Wahrheitswert von Aussage {stmt_id}",
            )

        return variables

    def _infer_domains(
        self, variables: Dict[str, NumericalVariable], constraints: List[NumConstraint]
    ) -> Dict[str, NumericalVariable]:
        """
        Inferiert/narrowt Domains basierend auf Constraints.

        Args:
            variables: Aktuelle Variablen mit Domains
            constraints: Liste von Constraints

        Returns:
            Aktualisierte Variablen mit verengten Domains
        """
        # Für einfache Constraints: Enge Domain ein
        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.DIVISIBILITY:
                divisor = constraint.metadata.get("divisor")
                if divisor and "zahl" in variables:
                    # Domain: Nur Vielfache von divisor
                    old_domain = variables["zahl"].domain
                    new_domain = {x for x in old_domain if x % divisor == 0}
                    variables["zahl"].domain = new_domain
                    logger.debug(
                        f"Domain verengt auf Vielfache von {divisor}: {len(new_domain)} Werte"
                    )

        return variables
