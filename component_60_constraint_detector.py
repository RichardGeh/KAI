"""
Constraint Detection Module - Generisches Erkennen von Constraint-Problemen

Erkennt wenn eine Eingabe ein logisches Constraint-Problem darstellt
(z.B. Logik-Raetsel mit WENN-DANN-Regeln) und bereitet die Daten
fuer den Constraint-Solver auf.

WICHTIG: Generische Loesung, funktioniert fuer alle Arten von Logik-Problemen,
nicht nur fuer spezifische Raetsel-Typen.

KEINE UNICODE-ZEICHEN: Nutzt ASCII-Text (AND, OR, NOT, IMPLIES, XOR)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from component_15_logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConstraintDetectorConfig:
    """
    Immutable configuration for ConstraintDetector.

    Using frozen dataclass ensures thread safety by preventing modifications
    after initialization.
    """

    min_conditional_rules: int = 2
    confidence_threshold: float = 0.65

    # Confidence calculation weights
    max_variable_bonus: float = 0.25
    variable_bonus_per_item: float = 0.04
    max_constraint_bonus: float = 0.25
    constraint_bonus_per_item: float = 0.04

    def __post_init__(self):
        """Validate configuration values."""
        if self.min_conditional_rules < 1:
            raise ValueError("min_conditional_rules must be >= 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be [0,1]")
        if not 0.0 <= self.max_variable_bonus <= 1.0:
            raise ValueError("max_variable_bonus must be [0,1]")
        if not 0.0 <= self.max_constraint_bonus <= 1.0:
            raise ValueError("max_constraint_bonus must be [0,1]")


@dataclass
class LogicalVariable:
    """
    Repraesentiert eine Variable in einem Constraint-Problem.

    Beispiele:
    - Person: Leo, Mark, Nick
    - Schalter: S1, S2, S3
    - Zustand: oben, unten, an, aus
    """

    name: str
    domain: Set[str] = field(default_factory=set)  # Moegliche Werte
    value: Optional[str] = None  # Aktueller Wert (wenn bekannt)

    def __hash__(self):
        return hash(self.name)


@dataclass
class LogicalConstraint:
    """
    Repraesentiert eine logische Bedingung.

    Constraint-Typen (ASCII only, kein Unicode):
    - IMPLIES: A IMPLIES B (Wenn A dann B)
    - AND: A AND B AND C (A und B und C)
    - OR: A OR B (A oder B)
    - XOR: A XOR B (A oder B, aber nicht beide)
    - NOT: NOT A (Nicht A)
    - EQUAL: A EQUAL B (A gleich B)
    """

    constraint_type: str  # IMPLIES, AND, OR, XOR, NOT, EQUAL
    variables: List[str]  # Beteiligte Variablen
    conditions: Dict[str, any] = field(default_factory=dict)  # Bedingungen
    metadata: Dict[str, any] = field(default_factory=dict)  # Zusatzinfo


@dataclass
class ConstraintProblem:
    """
    Repraesentiert ein vollstaendiges Constraint-Problem.

    Wird verwendet wenn KAI erkennt dass mehrere CONDITIONAL-Regeln
    ein zusammenhaengendes logisches Problem beschreiben.
    """

    variables: Dict[str, LogicalVariable] = field(default_factory=dict)
    constraints: List[LogicalConstraint] = field(default_factory=list)
    context: str = ""  # Original-Kontext
    confidence: float = 0.0  # Wie sicher sind wir dass es ein Constraint-Problem ist


class ConstraintDetector:
    """
    Erkennt ob eine Eingabe/Kontext ein Constraint-Problem darstellt.

    Funktionsweise:
    1. Zaehle CONDITIONAL-Regeln im Kontext
    2. Extrahiere Variablen aus den Regeln
    3. Wenn >= Schwellenwert: Aktiviere Constraint-Reasoning

    GENERISCH: Funktioniert fuer beliebige Logik-Probleme:
    - Personen-Raetsel (wer trinkt was)
    - Schalter-Raetsel (welche Stellung)
    - Zuordnungs-Probleme (wer wohnt wo)
    - etc.

    Thread-safe: Uses immutable frozen config to prevent race conditions.
    """

    def __init__(
        self,
        min_conditional_rules: int = 2,
        confidence_threshold: float = 0.65,
        config: Optional[ConstraintDetectorConfig] = None,
    ):
        """
        Args:
            min_conditional_rules: Mindestanzahl CONDITIONAL-Regeln (default: 2)
            confidence_threshold: Minimale Confidence fuer Erkennung (default: 0.65)
            config: Optional ConstraintDetectorConfig (overrides individual params)
        """
        # Use provided config or create from parameters
        if config is not None:
            self.config = config
        else:
            self.config = ConstraintDetectorConfig(
                min_conditional_rules=min_conditional_rules,
                confidence_threshold=confidence_threshold,
            )

        # Pattern fuer logische Operatoren (deutsch + englisch)
        # WICHTIG: "dann" ist oft optional in natuerlicher Sprache!
        # Pattern mit Komma-Trennung (genauer)
        self.logical_patterns = {
            "IMPLIES": [
                # Mit Komma: "Wenn X, dann Y" oder "Wenn X, Y"
                r"\bwenn\s+(.+?)\s*,\s*(?:dann\s+|so\s+)?(.+?)(?:\.|$)",
                # Ohne Komma aber mit "dann": "Wenn X dann Y"
                r"\bwenn\s+(.+?)\s+(?:dann|so)\s+(.+?)(?:\.|$)",
                # Falls-Pattern
                r"\bfalls\s+(.+?)\s*,\s*(?:dann\s+)?(.+?)(?:\.|$)",
                r"\bfalls\s+(.+?)\s+dann\s+(.+?)(?:\.|$)",
                # Sofern-Pattern
                r"\bsofern\s+(.+?)\s*,\s*(?:dann\s+)?(.+?)(?:\.|$)",
                # Nur-wenn-Pattern: "X nur wenn Y" (umgekehrte Implikation!)
                r"(.+?)\s+nur\s+(?:dann\s+)?wenn\s+(.+?)(?:\.|$)",
                # Sobald-Pattern: "Sobald X, Y"
                r"\bsobald\s+(.+?)\s*,\s*(?:dann\s+)?(.+?)(?:\.|$)",
                # English
                r"if\s+(.+?)\s*,\s*(?:then\s+)?(.+?)(?:\.|$)",
                r"if\s+(.+?)\s+then\s+(.+?)(?:\.|$)",
                r"(.+?)\s+only\s+if\s+(.+?)(?:\.|$)",
            ],
            "AND": [
                r"(.+?)\s+und\s+(.+)",
                r"(.+?)\s+sowie\s+(.+)",
                r"sowohl\s+(.+?)\s+als\s+auch\s+(.+)",
                r"beides\s*:\s*(.+?)\s+und\s+(.+)",
                r"(.+?)\s+and\s+(.+)",
                r"both\s+(.+?)\s+and\s+(.+)",
            ],
            "OR": [
                r"(.+?)\s+oder\s+(.+)",
                r"mindestens\s+(?:ein|eine)(?:r|s)?\s+von\s+(.+?)\s+(?:und|oder)\s+(.+)",
                r"(.+?)\s+oder\s+(.+?)\s+oder\s+beides",
                r"(.+?)\s+or\s+(.+)",
            ],
            "XOR": [
                # Klassisches XOR: "X oder Y, aber nie beide"
                r"(.+?)\s+oder\s+(.+?)\s*,?\s*aber\s+(?:nie|nicht)\s+beide",
                # Entweder-oder
                r"entweder\s+(.+?)\s+oder\s+(.+)",
                # Nur einer von X und Y
                r"nur\s+(?:ein|eine)(?:r|s)?\s+von\s+(.+?)\s+(?:und|oder)\s+(.+)",
                r"genau\s+(?:ein|eine)(?:r|s)?\s+(?:von\s+)?(.+?)\s+(?:und|oder)\s+(.+)",
                # Nicht beide gleichzeitig
                r"(?:nicht|nie)\s+(?:beide|beides)\s+(?:gleichzeitig|zugleich)\s*:\s*(.+?)\s+(?:und|oder)\s+(.+)",
                # English
                r"either\s+(.+?)\s+or\s+(.+?)\s+but\s+not\s+both",
                r"exactly\s+one\s+of\s+(.+?)\s+(?:and|or)\s+(.+)",
            ],
            "NOT": [
                r"nicht\s+(.+)",
                r"kein(?:e|s|er|en)?\s+(.+)",
                r"niemals\s+(.+)",
                r"nie\s+(.+)",
                r"not\s+(.+)",
                r"never\s+(.+)",
            ],
        }

        logger.info(
            f"ConstraintDetector initialisiert | "
            f"min_rules={self.config.min_conditional_rules}, "
            f"threshold={self.config.confidence_threshold}"
        )

    def detect_constraint_problem(
        self, text: str, learned_rules: Optional[List[Dict]] = None
    ) -> Optional[ConstraintProblem]:
        """
        Prueft ob Text/Kontext ein Constraint-Problem darstellt.

        Args:
            text: Der zu analysierende Text
            learned_rules: Optional Liste von bereits gelernten CONDITIONAL-Regeln

        Returns:
            ConstraintProblem wenn erkannt, sonst None
        """
        # Zaehle CONDITIONAL-Pattern im Text
        conditional_count = self._count_conditional_patterns(text)

        logger.debug(
            f"Constraint-Detection | conditional_patterns={conditional_count}, "
            f"threshold={self.config.min_conditional_rules}"
        )

        # Pruefe ob genug CONDITIONAL-Regeln vorhanden
        if conditional_count < self.config.min_conditional_rules:
            logger.debug("Zu wenig CONDITIONAL-Pattern fuer Constraint-Problem")
            return None

        # Extrahiere Variablen und Constraints
        variables = self._extract_variables(text)
        constraints = self._extract_constraints(text)

        if not variables or not constraints:
            logger.debug("Keine Variablen oder Constraints extrahiert")
            return None

        # Berechne Confidence
        confidence = self._calculate_confidence(
            conditional_count, len(variables), len(constraints)
        )

        if confidence < self.config.confidence_threshold:
            logger.debug(
                f"Confidence zu niedrig | {confidence:.2f} < {self.config.confidence_threshold}"
            )
            return None

        # Erstelle ConstraintProblem
        problem = ConstraintProblem(
            variables=variables,
            constraints=constraints,
            context=text,
            confidence=confidence,
        )

        logger.info(
            f"[Constraint-Problem erkannt] | "
            f"variables={len(variables)}, constraints={len(constraints)}, "
            f"confidence={confidence:.2f}"
        )

        return problem

    def _count_conditional_patterns(self, text: str) -> int:
        """Zaehlt WENN-DANN Pattern im Text."""
        text_lower = text.lower()
        count = 0

        for pattern in self.logical_patterns["IMPLIES"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            count += len(matches)
            if matches:
                logger.debug(f"Pattern matched: {pattern} -> {len(matches)} matches")

        logger.debug(f"Total CONDITIONAL patterns found: {count}")
        return count

    def _extract_variables(self, text: str) -> Dict[str, LogicalVariable]:
        """
        Extrahiert Variablen aus dem Text.

        Heuristiken:
        - Grossgeschriebene Woerter (Namen: Leo, Mark, Nick)
        - Nummern mit Kontext (Schalter 1, S2, Position 3)
        - Haeufig wiederholte Nomen
        """
        variables = {}
        text_lower = text.lower()

        # Heuristik 1: Namen (Grossbuchstaben am Wortanfang)
        # Matcht sowohl "Leo", "Mark" als auch einzelne Buchstaben "A", "B", "C"
        name_pattern = r"\b([A-Z](?:[a-z]+)?)\b"
        names = set(re.findall(name_pattern, text))

        for name in names:
            # Filter haeufige deutsche Woerter (die, der, wenn, etc.)
            if name.lower() not in [
                "wenn",
                "dann",
                "falls",
                "aber",
                "oder",
                "und",
                "ist",
            ]:
                # Mind. 2x erwähnt (oder einzelner Buchstabe mit mind. 1x)
                min_count = 1 if len(name) == 1 else 2
                if text_lower.count(name.lower()) >= min_count:
                    variables[name] = LogicalVariable(
                        name=name, domain={"aktiv", "inaktiv"}  # Default-Domain
                    )

        # Heuristik 2: Nummern/Schalter
        switch_pattern = r"(?:schalter|switch|s)\s*(\d+)|(\d+)\s*(?:oben|unten|an|aus)"
        switches = re.findall(switch_pattern, text_lower)

        for match in switches:
            num = match[0] or match[1]
            if num:
                var_name = f"S{num}"
                if var_name not in variables:
                    variables[var_name] = LogicalVariable(
                        name=var_name, domain={"oben", "unten", "an", "aus", "1", "0"}
                    )

        return variables

    def _extract_constraints(self, text: str) -> List[LogicalConstraint]:
        """
        Extrahiert logische Constraints aus dem Text.

        Nutzt die logical_patterns um IMPLIES, AND, OR, XOR, NOT zu finden.
        """
        constraints = []
        text_lower = text.lower()

        # Extrahiere IMPLIES (Wenn...dann)
        for pattern in self.logical_patterns["IMPLIES"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                premise = match[0].strip()
                conclusion = match[1].strip()

                constraint = LogicalConstraint(
                    constraint_type="IMPLIES",
                    variables=[premise, conclusion],
                    conditions={"premise": premise, "conclusion": conclusion},
                    metadata={"pattern": "conditional"},
                )
                constraints.append(constraint)

        # Extrahiere XOR (oder...aber nicht beide)
        for pattern in self.logical_patterns["XOR"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                a = match[0].strip()
                b = match[1].strip()

                constraint = LogicalConstraint(
                    constraint_type="XOR",
                    variables=[a, b],
                    conditions={"a": a, "b": b},
                    metadata={"pattern": "exclusive_or"},
                )
                constraints.append(constraint)

        # Extrahiere AND (und/sowie/sowohl...als auch)
        for pattern in self.logical_patterns["AND"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                a = match[0].strip()
                b = match[1].strip()

                constraint = LogicalConstraint(
                    constraint_type="AND",
                    variables=[a, b],
                    conditions={"a": a, "b": b},
                    metadata={"pattern": "conjunction"},
                )
                constraints.append(constraint)

        # Extrahiere OR (oder/mindestens einer)
        for pattern in self.logical_patterns["OR"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                a = match[0].strip()
                b = match[1].strip()

                constraint = LogicalConstraint(
                    constraint_type="OR",
                    variables=[a, b],
                    conditions={"a": a, "b": b},
                    metadata={"pattern": "disjunction"},
                )
                constraints.append(constraint)

        # Extrahiere NOT (nicht/kein/niemals)
        for pattern in self.logical_patterns["NOT"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # NOT hat nur 1 capture group
                negated = match.strip() if isinstance(match, str) else match[0].strip()

                constraint = LogicalConstraint(
                    constraint_type="NOT",
                    variables=[negated],
                    conditions={"negated": negated},
                    metadata={"pattern": "negation"},
                )
                constraints.append(constraint)

        return constraints

    def _calculate_confidence(
        self, conditional_count: int, variable_count: int, constraint_count: int
    ) -> float:
        """
        Berechnet Confidence dass es sich um ein Constraint-Problem handelt.

        Faktoren:
        - Anzahl CONDITIONAL-Pattern (je mehr, desto hoeher)
        - Anzahl Variablen (mindestens 2 noetig)
        - Anzahl Constraints (mindestens 2 noetig)

        Uses config values for weights to make tuning easier and self-documenting.
        """
        if variable_count < 2 or constraint_count < 2:
            return 0.0

        # Base confidence basierend auf CONDITIONAL-Pattern
        # Wenn conditional_count >= min_rules: base_conf >= 1.0
        base_conf = min(
            1.0, conditional_count / max(1, self.config.min_conditional_rules)
        )

        # Bonus fuer mehr Variablen/Constraints (values from config)
        variable_bonus = min(
            self.config.max_variable_bonus,
            variable_count * self.config.variable_bonus_per_item,
        )
        constraint_bonus = min(
            self.config.max_constraint_bonus,
            constraint_count * self.config.constraint_bonus_per_item,
        )

        confidence = base_conf + variable_bonus + constraint_bonus

        return min(1.0, confidence)

    def is_constraint_query(self, query: str, problem: ConstraintProblem) -> bool:
        """
        Prueft ob eine Frage sich auf ein Constraint-Problem bezieht.

        Args:
            query: Die Frage (z.B. "Wer trinkt Brandy?")
            problem: Das erkannte Constraint-Problem

        Returns:
            True wenn die Frage eine der Variablen referenziert
        """
        query_lower = query.lower()

        # Pruefe ob Variablen-Namen in Frage vorkommen
        for var_name in problem.variables.keys():
            if var_name.lower() in query_lower:
                return True

        # Pruefe ob Frage nach Loesung fragt
        solution_keywords = [
            "wer",
            "welche",
            "was",
            "wie viele",
            "which",
            "who",
            "what",
        ]
        return any(keyword in query_lower for keyword in solution_keywords)

    def detect_numerical_constraints(self, text: str) -> Dict[str, Any]:
        """
        Erkennt numerische Constraints in Text (für Zahlen-Rätsel).

        Patterns für:
        - Teilbarkeit: "teilbar durch X", "Vielfaches von X"
        - Arithmetik: "Summe der", "Differenz", "Produkt", "Quotient"
        - Meta-Constraints: "Anzahl der richtigen", "erste/letzte richtige"
        - Boolean: "richtig", "falsch", "wahr"

        Args:
            text: Der zu analysierende Text

        Returns:
            Dictionary mit:
            - has_numerical_constraints: bool
            - constraint_types: List[str] - Erkannte Constraint-Typen
            - numerical_variables: List[str] - Erkannte numerische Variablen
            - meta_constraints: bool - Ob Meta-Constraints vorhanden sind
            - confidence: float - Wie sicher ist die Erkennung
        """
        text_lower = text.lower()

        # Pattern-Kategorien
        divisibility_patterns = [
            r"\bteilbar\s+durch\s+(\d+|die\s+\w+)\b",
            r"\bvielfaches\s+von\s+(\d+|\w+)\b",
            r"\brest\s+(\d+)\s+bei\s+teilung\b",
        ]

        arithmetic_patterns = [
            r"\bsumme\s+der\s+(\w+)\b",
            r"\bdifferenz\s+(?:der\s+)?(\w+)\s+und\s+(\w+)\b",
            r"\bprodukt\s+(?:der\s+)?(\w+)\b",
            r"\bquotient\s+(?:der\s+)?(\w+)\b",
        ]

        meta_patterns = [
            r"\banzahl\s+der\s+(?:richtigen?|falschen?)\s+behauptungen?\b",
            r"\b(?:erste|letzte|n-te)\s+(?:richtige|falsche)\s+behauptung\b",
            r"\bnummern?\s+der\s+(?:richtigen?|falschen?)\b",
            r"\bteiler\b",  # "Anzahl der Teiler"
        ]

        boolean_patterns = [
            r"\b(?:ist\s+)?richtig\b",
            r"\b(?:ist\s+)?falsch\b",
            r"\b(?:ist\s+)?wahr\b",
            r"\bbehauptung(?:en)?\b",
        ]

        # Zähle Matches pro Kategorie
        divisibility_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE)) for p in divisibility_patterns
        )
        arithmetic_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE)) for p in arithmetic_patterns
        )
        meta_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE)) for p in meta_patterns
        )
        boolean_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE)) for p in boolean_patterns
        )

        # Sammle Constraint-Typen
        constraint_types = []
        if divisibility_count > 0:
            constraint_types.append("DIVISIBILITY")
        if arithmetic_count > 0:
            constraint_types.append("ARITHMETIC")
        if meta_count > 0:
            constraint_types.append("META")
        if boolean_count > 0:
            constraint_types.append("BOOLEAN")

        # Extrahiere numerische Variablen
        numerical_variables = self._extract_numerical_variables(text)

        # Hat numerische Constraints?
        # META and BOOLEAN constraints are inherently numerical (truth values, counts)
        is_meta_or_boolean = "META" in constraint_types or "BOOLEAN" in constraint_types
        has_numerical = (
            is_meta_or_boolean  # META/BOOLEAN are always numerical
            or len(constraint_types) >= 2
            or (len(constraint_types) >= 1 and len(numerical_variables) >= 1)
        )

        # Berechne Confidence
        total_matches = (
            divisibility_count + arithmetic_count + meta_count + boolean_count
        )
        confidence = min(
            1.0, 0.5 + (total_matches * 0.1) + (len(constraint_types) * 0.1)
        )

        result = {
            "has_numerical_constraints": has_numerical,
            "constraint_types": constraint_types,
            "numerical_variables": numerical_variables,
            "meta_constraints": meta_count > 0,
            "confidence": confidence,
            "constraint_counts": {
                "divisibility": divisibility_count,
                "arithmetic": arithmetic_count,
                "meta": meta_count,
                "boolean": boolean_count,
            },
        }

        if has_numerical:
            logger.info(
                f"[Numerische Constraints erkannt] | "
                f"types={constraint_types}, variables={len(numerical_variables)}, "
                f"meta={meta_count > 0}, confidence={confidence:.2f}"
            )

        return result

    def _extract_numerical_variables(self, text: str) -> List[str]:
        """
        Extrahiert numerische Variablen aus Text.

        Heuristiken:
        - "gesuchte Zahl", "die Zahl"
        - "X", "Y", "Z" als Variable
        - "Nummer 1", "Nummer 2", etc.
        - Numbered statements (1., 2., 3., ...)
        """
        variables = []
        text_lower = text.lower()

        # Heuristik 1: "gesuchte Zahl", "die Zahl"
        if re.search(r"\b(?:gesuchte|die)\s+zahl\b", text_lower):
            variables.append("zahl")

        # Heuristik 2: Einzelne Großbuchstaben als Variablen
        single_vars = re.findall(r"\b([X-Z])\b", text)
        variables.extend(single_vars)

        # Heuristik 3: "Nummer X"
        nummern = re.findall(r"\bnummer\s+(\d+)\b", text_lower)
        variables.extend([f"nummer_{n}" for n in nummern])

        # Heuristik 4: Numbered statements (extract statement IDs)
        statements = re.findall(r"\b(\d+)\.\s", text)
        if len(statements) >= 3:  # Mindestens 3 numbered statements
            variables.extend([f"statement_{s}" for s in statements[:10]])  # Max 10

        return list(set(variables))  # Remove duplicates
