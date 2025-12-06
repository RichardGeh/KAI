"""
Tests für Numerical Constraint Parser (PHASE 2)

Testet das Parsen von numerischen Constraints aus deutschen Logic Puzzles:
- Teilbarkeits-Constraints
- Arithmetische Constraints
- Meta-Constraints
- Vergleichs-Constraints
- Variable Extraction
- Domain Inference
"""

import pytest

from component_45_numerical_constraint_parser import (
    ConstraintType,
    NumericalConstraintParser,
)


@pytest.fixture
def parser():
    """Fixture für NumericalConstraintParser."""
    return NumericalConstraintParser()


class TestStatementExtraction:
    """Tests für _extract_statements()."""

    def test_extract_simple_statements(self, parser):
        """Test: Einfache numbered statements werden extrahiert."""
        text = """
        1. Die Zahl ist teilbar durch 3.
        2. Die Zahl ist teilbar durch 7.
        3. Die Summe ist 10.
        """

        statements = parser._extract_statements(text)

        assert len(statements) == 3
        assert 1 in statements
        assert 2 in statements
        assert 3 in statements
        assert "teilbar durch 3" in statements[1]

    def test_extract_multiline_statements(self, parser):
        """Test: Mehrzeilige Statements werden korrekt extrahiert."""
        text = """
        1. Die gesuchte Zahl ist teilbar durch die Differenz
           zwischen den Nummern.
        2. Die Summe ist 10.
        """

        statements = parser._extract_statements(text)

        assert len(statements) == 2
        assert "Differenz" in statements[1]

    def test_extract_with_parentheses(self, parser):
        """Test: Statements mit Klammern-Format (1) werden extrahiert."""
        text = """
        1) Die Zahl ist teilbar durch 3.
        2) Die Zahl ist größer als 10.
        """

        statements = parser._extract_statements(text)

        assert len(statements) == 2
        assert "teilbar durch 3" in statements[1]

    def test_extract_statements_with_trailing_question(self, parser):
        """Test: Statements werden korrekt extrahiert mit unnummeriertem Trailing Text."""
        text = """Gesucht ist eine dreistellige Zahl mit folgenden Eigenschaften:
1. Die Zahl ist durch 3 teilbar.
2. Die Summe der Ziffern beträgt 15.
3. Die erste Ziffer ist größer als die letzte Ziffer.
Welche Zahl wird gesucht?"""

        statements = parser._extract_statements(text)

        assert len(statements) == 3, f"Expected 3 statements, got {len(statements)}"
        assert 1 in statements
        assert 2 in statements
        assert 3 in statements
        assert "teilbar" in statements[1].lower()
        assert "summe" in statements[2].lower()
        assert "größer" in statements[3].lower()
        # Ensure question text is NOT in statement 3
        assert "welche zahl" not in statements[3].lower()


class TestQuestionExtraction:
    """Tests für _extract_question()."""

    def test_extract_question_simple(self, parser):
        """Test: Einfache Frage wird extrahiert."""
        text = """
        1. Die Zahl ist teilbar durch 3.
        Was ist die kleinste gesuchte Zahl?
        """

        question = parser._extract_question(text)

        assert question is not None
        assert "kleinste gesuchte Zahl" in question
        assert question.endswith("?")

    def test_extract_question_with_number(self, parser):
        """Test: Numbered question wird extrahiert (ohne Nummer)."""
        text = """
        1. Die Zahl ist teilbar durch 3.
        2. Was ist die Zahl?
        """

        question = parser._extract_question(text)

        assert question is not None
        assert not question.startswith("2.")
        assert "Was ist die Zahl" in question


class TestDivisibilityConstraints:
    """Tests für Teilbarkeits-Constraints."""

    def test_parse_simple_divisibility(self, parser):
        """Test: Einfache Teilbarkeit wird geparst."""
        stmt_text = "Die Zahl ist teilbar durch 3."

        constraint = parser._parse_divisibility_constraint(1, stmt_text)

        assert constraint is not None
        assert constraint.constraint_type == ConstraintType.DIVISIBILITY
        assert "zahl" in constraint.variables
        assert constraint.metadata["divisor"] == 3

    def test_parse_multiple_divisors(self, parser):
        """Test: Teilbarkeit durch mehrere Zahlen."""
        text = """
        1. Die Zahl ist teilbar durch 3.
        2. Die Zahl ist teilbar durch 7.
        """

        result = parser.parse_puzzle(text)

        # Mindestens 1 DIVISIBILITY Constraint (andere könnten META sein)
        div_constraints = [
            c
            for c in result["constraints"]
            if c.constraint_type == ConstraintType.DIVISIBILITY
        ]
        assert len(div_constraints) >= 1

    def test_parse_complex_divisor_expression(self, parser):
        """Test: Komplexer Divisor-Ausdruck wird als META markiert."""
        stmt_text = "Die Zahl ist teilbar durch die Differenz zwischen X und Y."

        constraint = parser._parse_divisibility_constraint(1, stmt_text)

        assert constraint is not None
        assert constraint.constraint_type == ConstraintType.META
        assert constraint.metadata["requires_evaluation"] is True


class TestArithmeticConstraints:
    """Tests für arithmetische Constraints."""

    def test_parse_sum_divisibility(self, parser):
        """Test: 'Summe der X ist teilbar durch Y' wird geparst."""
        stmt_text = "Die Summe der Nummern ist teilbar durch 5."

        constraint = parser._parse_arithmetic_constraint(1, stmt_text)

        assert constraint is not None
        assert constraint.constraint_type == ConstraintType.META
        assert constraint.metadata["operation"] == "sum"
        assert constraint.metadata["divisor"] == 5

    def test_parse_sum_without_divisibility(self, parser):
        """Test: Summe ohne Teilbarkeit wird geparst."""
        stmt_text = "Die Summe der Zahlen ist 10."

        # Dieser Fall wird noch nicht geparst (Return None)
        constraint = parser._parse_arithmetic_constraint(1, stmt_text)

        # Erwarte None weil Pattern nicht matched
        assert constraint is None


class TestComparisonConstraints:
    """Tests für Vergleichs-Constraints."""

    def test_parse_greater_than(self, parser):
        """Test: 'größer als X' wird geparst."""
        stmt_text = "Die Anzahl ist größer als 10."

        constraint = parser._parse_comparison_constraint(1, stmt_text)

        assert constraint is not None
        assert constraint.constraint_type == ConstraintType.META
        assert constraint.metadata["comparison"] == ">"
        assert constraint.metadata["threshold"] == 10

    def test_parse_greater_than_alternative_spelling(self, parser):
        """Test: 'grösser als' (mit ö) wird geparst."""
        stmt_text = "Die Anzahl ist grösser als 10."

        constraint = parser._parse_comparison_constraint(1, stmt_text)

        assert constraint is not None
        assert constraint.metadata["threshold"] == 10


class TestMetaConstraints:
    """Tests für Meta-Constraints."""

    def test_parse_divisor_count(self, parser):
        """Test: 'Anzahl der Teiler' wird geparst."""
        stmt_text = "Die Anzahl der Teiler der Zahl ist größer als 10."

        constraint = parser._parse_meta_constraint(1, stmt_text)

        assert constraint is not None
        assert constraint.constraint_type == ConstraintType.META
        assert constraint.metadata["meta_type"] == "divisor_count"


class TestVariableExtraction:
    """Tests für Variable Extraction."""

    def test_extract_gesuchte_zahl(self, parser):
        """Test: 'gesuchte Zahl' Variable wird extrahiert."""
        text = "Die gesuchte Zahl ist teilbar durch 3."

        result = parser.parse_puzzle(text)

        assert "zahl" in result["variables"]
        assert result["variables"]["zahl"].name == "zahl"
        assert not result["variables"]["zahl"].is_meta

    def test_extract_statement_variables(self, parser):
        """Test: Statement truth value Variablen werden extrahiert."""
        text = """
        1. Die Zahl ist teilbar durch 3.
        2. Die Zahl ist teilbar durch 7.
        """

        result = parser.parse_puzzle(text)

        assert "statement_1" in result["variables"]
        assert "statement_2" in result["variables"]
        assert result["variables"]["statement_1"].is_meta is True
        assert result["variables"]["statement_1"].domain == {0, 1}

    def test_variable_descriptions(self, parser):
        """Test: Variablen haben Beschreibungen."""
        text = "Die gesuchte Zahl ist teilbar durch 3."

        result = parser.parse_puzzle(text)

        assert result["variables"]["zahl"].description == "Die gesuchte Zahl"


class TestDomainInference:
    """Tests für Domain Inference."""

    def test_domain_narrowing_divisibility(self, parser):
        """Test: Domain wird auf Vielfache eingeschränkt."""
        text = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        """

        result = parser.parse_puzzle(text)

        # Prüfe dass Domain nur Vielfache von 3 enthält
        zahl_domain = result["variables"]["zahl"].domain
        assert all(x % 3 == 0 for x in zahl_domain)
        assert 3 in zahl_domain
        assert 6 in zahl_domain
        assert 5 not in zahl_domain

    def test_domain_default_range(self, parser):
        """Test: Default Domain ist 1-100."""
        text = "Die gesuchte Zahl ist interessant."

        result = parser.parse_puzzle(text)

        if "zahl" in result["variables"]:
            zahl_domain = result["variables"]["zahl"].domain
            assert 1 in zahl_domain
            assert 100 in zahl_domain or len(zahl_domain) <= 100


class TestFullPuzzleParsing:
    """Tests für vollständiges Puzzle-Parsing."""

    def test_parse_simple_puzzle(self, parser):
        """Test: Einfaches Puzzle wird vollständig geparst."""
        text = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        2. Die gesuchte Zahl ist teilbar durch 7.
        Was ist die kleinste gesuchte Zahl?
        """

        result = parser.parse_puzzle(text)

        assert len(result["statements"]) == 2
        assert len(result["constraints"]) >= 1  # Mindestens 1 Constraint
        assert "zahl" in result["variables"]
        assert result["question"] is not None

    def test_parse_complex_puzzle(self, parser):
        """Test: Komplexes Puzzle mit Meta-Constraints wird geparst."""
        text = """
        1. Die gesuchte Zahl ist teilbar durch die Differenz.
        2. Die Summe der Nummern ist teilbar durch 5.
        3. Die Anzahl der Teiler ist größer als 10.
        Was ist die kleinste gesuchte Zahl?
        """

        result = parser.parse_puzzle(text)

        assert len(result["statements"]) == 3
        assert len(result["constraints"]) >= 2
        # Prüfe auf Meta-Constraints
        meta_constraints = [
            c for c in result["constraints"] if c.constraint_type == ConstraintType.META
        ]
        assert len(meta_constraints) >= 1

    def test_constraint_statement_ids(self, parser):
        """Test: Constraints haben korrekte statement_ids."""
        text = """
        1. Die Zahl ist teilbar durch 3.
        2. Die Zahl ist teilbar durch 7.
        """

        result = parser.parse_puzzle(text)

        for constraint in result["constraints"]:
            assert constraint.statement_id in [1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
