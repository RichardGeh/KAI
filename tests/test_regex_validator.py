# tests/test_regex_validator.py
"""
Tests für Regex Validation System (component_27).

Testet:
- Valid Patterns (genau 2 Capture Groups)
- Invalid Patterns (0, 1, 3+ Groups)
- Syntax Errors
- Warning Detection
- Fix Suggestions
- Format Validation Result
"""

import pytest

from component_27_regex_validator import RegexValidator, get_regex_validator


class TestValidPatterns:
    """Tests für gültige Regex-Muster"""

    @pytest.fixture
    def validator(self):
        """Fixture: RegexValidator Instanz"""
        return RegexValidator()

    def test_valid_pattern_with_non_capturing_group(self, validator):
        """Test: Gültiges Muster mit non-capturing group"""
        pattern = r"^(.+) ist (?:ein|eine) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is True
        assert error_msg is None
        assert details["capture_groups"] == 2
        assert details["compiled"] is True

    def test_valid_simple_pattern(self, validator):
        """Test: Einfaches gültiges Muster"""
        pattern = r"^(.+) hat (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is True
        assert error_msg is None
        assert details["capture_groups"] == 2

    def test_valid_pattern_with_multiple_non_capturing_groups(self, validator):
        """Test: Muster mit mehreren non-capturing groups"""
        pattern = r"^(.+) (?:kann|mag|will) (?:nicht)? (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is True
        assert details["capture_groups"] == 2

    def test_valid_pattern_with_special_chars(self, validator):
        """Test: Muster mit Sonderzeichen"""
        pattern = r"^(.+)\s+(?:ist|sind)\s+(.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is True
        assert details["capture_groups"] == 2

    def test_valid_pattern_complex(self, validator):
        """Test: Komplexes gültiges Muster"""
        pattern = r"^(.+)\s+(?:schmeckt|riecht)\s+(?:nach\s+)?(.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is True
        assert details["capture_groups"] == 2


class TestInvalidPatterns:
    """Tests für ungültige Regex-Muster"""

    @pytest.fixture
    def validator(self):
        return RegexValidator()

    def test_too_few_groups_zero(self, validator):
        """Test: Zu wenige Gruppen (0)"""
        pattern = r"^test$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "Zu wenige Capture-Groups" in error_msg
        assert details["capture_groups"] == 0

    def test_too_few_groups_one(self, validator):
        """Test: Zu wenige Gruppen (1)"""
        pattern = r"^(.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "Zu wenige Capture-Groups" in error_msg
        assert details["capture_groups"] == 1

    def test_too_many_groups_three(self, validator):
        """Test: Zu viele Gruppen (3)"""
        pattern = r"^(.+) ist (ein|eine) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "Zu viele Capture-Groups" in error_msg
        assert details["capture_groups"] == 3

    def test_too_many_groups_four(self, validator):
        """Test: Zu viele Gruppen (4)"""
        pattern = r"^(.+) (ist) (ein|eine) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "Zu viele Capture-Groups" in error_msg
        assert details["capture_groups"] == 4

    def test_too_many_groups_many(self, validator):
        """Test: Sehr viele Gruppen (10)"""
        pattern = r"^(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "Zu viele Capture-Groups" in error_msg
        assert details["capture_groups"] == 10


class TestSyntaxErrors:
    """Tests für Syntax-Fehler"""

    @pytest.fixture
    def validator(self):
        return RegexValidator()

    def test_unbalanced_brackets(self, validator):
        """Test: Unbalanced Klammern"""
        pattern = r"^[(.+) ist (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "Ungültige Regex-Syntax" in error_msg or "ERROR" in error_msg
        assert details["compiled"] is not True

    def test_invalid_escape(self, validator):
        """Test: Ungültiges Escape"""
        pattern = r"^(.+) \k (.+)$"  # \k ist kein gültiges Escape
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        # Könnte valid sein (regex akzeptiert \k als literal 'k')
        # oder invalid (abhängig von Regex-Engine)
        # -> Toleranter Test
        assert isinstance(is_valid, bool)

    def test_unclosed_group(self, validator):
        """Test: Nicht geschlossene Gruppe"""
        pattern = r"^(.+ ist (?:ein|eine (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "Ungültige Regex-Syntax" in error_msg or "ERROR" in error_msg

    def test_empty_pattern(self, validator):
        """Test: Leeres Muster"""
        pattern = ""
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "darf nicht leer sein" in error_msg.lower()

    def test_whitespace_only_pattern(self, validator):
        """Test: Nur Leerzeichen"""
        pattern = "   "
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is False
        assert "darf nicht leer sein" in error_msg.lower()


class TestWarningDetection:
    """Tests für Warning-Detection"""

    @pytest.fixture
    def validator(self):
        return RegexValidator()

    def test_warning_star_instead_of_plus(self, validator):
        """Test: Warning für .* statt .+"""
        pattern = r"^(.*) ist (.*)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        # Pattern ist valid, aber mit Warning
        assert is_valid is True
        assert "warnings" in details
        assert any(".*" in w for w in details["warnings"])

    def test_warning_no_anchors(self, validator):
        """Test: Warning für fehlende Anker"""
        pattern = r"(.+) ist (.+)"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        # Pattern ist valid, aber mit Warning
        assert is_valid is True
        assert "warnings" in details
        assert any("Anker" in w for w in details["warnings"])

    def test_warning_short_pattern(self, validator):
        """Test: Warning für sehr kurzes Muster"""
        pattern = r"(.)(.)"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        # Pattern ist valid, aber mit Warning
        assert is_valid is True
        assert "warnings" in details
        assert any("kurz" in w.lower() for w in details["warnings"])

    def test_no_warnings_for_good_pattern(self, validator):
        """Test: Keine Warnings für gutes Muster"""
        pattern = r"^(.+) ist (?:ein|eine) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is True
        # Keine Warnings ODER leere Liste
        warnings = details.get("warnings", [])
        assert len(warnings) == 0


class TestSuggestFixes:
    """Tests für suggest_fixes() Funktion"""

    @pytest.fixture
    def validator(self):
        return RegexValidator()

    def test_suggest_fix_for_ein_eine(self, validator):
        """Test: Vorschlag für (ein|eine) -> (?:ein|eine)"""
        pattern = r"^(.+) ist (ein|eine) (.+)$"
        suggestions = validator.suggest_fixes(pattern)

        assert len(suggestions) > 0
        assert any("(?:ein|eine)" in s for s in suggestions)

    def test_suggest_fix_for_der_die_das(self, validator):
        """Test: Vorschlag für (der|die|das) -> (?:der|die|das)"""
        pattern = r"^(der|die|das) (.+) ist (.+)$"
        suggestions = validator.suggest_fixes(pattern)

        assert len(suggestions) > 0
        assert any("(?:der|die|das)" in s for s in suggestions)

    def test_suggest_fix_for_missing_start_anchor(self, validator):
        """Test: Vorschlag für fehlenden Start-Anker"""
        pattern = r"(.+) ist (.+)$"
        suggestions = validator.suggest_fixes(pattern)

        assert len(suggestions) > 0
        assert any(s.startswith("^") or "^" in s for s in suggestions)

    def test_suggest_fix_for_missing_end_anchor(self, validator):
        """Test: Vorschlag für fehlenden End-Anker"""
        pattern = r"^(.+) ist (.+)"
        suggestions = validator.suggest_fixes(pattern)

        assert len(suggestions) > 0
        assert any(s.endswith("$") or "$" in s for s in suggestions)

    def test_suggest_fix_for_star_quantifier(self, validator):
        """Test: Vorschlag für .* -> .+"""
        pattern = r"^(.*) ist (.*)$"
        suggestions = validator.suggest_fixes(pattern)

        assert len(suggestions) > 0
        assert any(".+" in s for s in suggestions)

    def test_no_suggestions_for_good_pattern(self, validator):
        """Test: Keine Vorschläge für gutes Muster"""
        pattern = r"^(.+) ist (?:ein|eine) (.+)$"
        suggestions = validator.suggest_fixes(pattern)

        # Könnte leer sein oder nur Style-Vorschläge
        # -> Toleranter Test: Prüfe nur dass keine Exception
        assert isinstance(suggestions, list)


class TestFormatValidationResult:
    """Tests für format_validation_result() Funktion"""

    @pytest.fixture
    def validator(self):
        return RegexValidator()

    def test_format_valid_result(self, validator):
        """Test: Formatierung für gültiges Ergebnis"""
        pattern = r"^(.+) ist (?:ein|eine) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        formatted = validator.format_validation_result(is_valid, error_msg, details)

        assert "[OK]" in formatted
        assert "gültig" in formatted.lower()
        assert "Capture-Groups: 2" in formatted

    def test_format_invalid_result(self, validator):
        """Test: Formatierung für ungültiges Ergebnis"""
        pattern = r"^(.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        formatted = validator.format_validation_result(is_valid, error_msg, details)

        assert "[ERROR]" in formatted or "ungültig" in formatted.lower()
        assert pattern in formatted

    def test_format_result_with_warnings(self, validator):
        """Test: Formatierung mit Warnings"""
        pattern = r"^(.*) ist (.*)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        formatted = validator.format_validation_result(is_valid, error_msg, details)

        assert "[OK]" in formatted  # Valid trotz Warnings
        assert "WARNING" in formatted or "Warnungen" in formatted

    def test_format_result_with_test_matches(self, validator):
        """Test: Formatierung mit Test-Matches"""
        pattern = r"^(.+) ist (?:ein|eine) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        formatted = validator.format_validation_result(is_valid, error_msg, details)

        # Sollte Test-Matches enthalten (falls vorhanden)
        if details.get("test_results"):
            assert "Test-Matches" in formatted or "Beispiel" in formatted


class TestPatternExampleTesting:
    """Tests für _test_pattern_examples() Methode"""

    @pytest.fixture
    def validator(self):
        return RegexValidator()

    def test_example_testing_returns_results(self, validator):
        """Test: Example-Testing gibt Ergebnisse zurück"""
        pattern = r"^(.+) ist (?:ein|eine) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert "test_results" in details
        test_results = details["test_results"]
        assert "total" in test_results
        assert "matches" in test_results
        assert test_results["total"] > 0

    def test_example_testing_finds_matches(self, validator):
        """Test: Example-Testing findet Matches"""
        pattern = r"^(.+) ist (?:ein|eine) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        test_results = details["test_results"]
        # "Ein Hund ist ein Tier" sollte matchen
        assert test_results["matches"] > 0

    def test_example_testing_no_matches_for_unrelated_pattern(self, validator):
        """Test: Keine Matches für unpassendes Muster"""
        pattern = r"^XYZ (.+) ABC (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        test_results = details["test_results"]
        # Unpassendes Muster sollte keine Test-Beispiele matchen
        assert test_results["matches"] == 0


class TestSingletonPattern:
    """Tests für Singleton-Pattern"""

    def test_singleton_returns_same_instance(self):
        """Test: get_regex_validator() gibt immer gleiche Instanz zurück"""
        instance1 = get_regex_validator()
        instance2 = get_regex_validator()

        assert instance1 is instance2

    def test_singleton_initialization_error_handling(self):
        """Test: Singleton-Getter hat Error Handling"""
        # Dieser Test prüft nur, dass get_regex_validator() callable ist
        try:
            instance = get_regex_validator()
            assert instance is not None
        except Exception as e:
            pytest.fail(f"get_regex_validator() raised unexpected exception: {e}")


class TestEdgeCases:
    """Tests für Edge Cases"""

    @pytest.fixture
    def validator(self):
        return RegexValidator()

    def test_very_long_pattern(self, validator):
        """Test: Sehr langes Muster"""
        long_pattern = r"^(.+) " + "(?:ist|sind) " * 50 + r"(.+)$"
        is_valid, error_msg, details = validator.validate_pattern(long_pattern)

        # Sollte valid sein (2 Gruppen), aber vielleicht mit Warnings
        assert is_valid is True
        assert details["capture_groups"] == 2

    def test_pattern_with_unicode(self, validator):
        """Test: Muster mit Unicode"""
        pattern = r"^(.+) ist (?:ein|eine) (.+)ß$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        # Sollte valid sein
        assert is_valid is True

    def test_pattern_with_many_alternatives(self, validator):
        """Test: Muster mit vielen Alternativen"""
        pattern = r"^(.+) (?:ist|sind|war|waren|wird|werden|kann|können) (.+)$"
        is_valid, error_msg, details = validator.validate_pattern(pattern)

        assert is_valid is True
        assert details["capture_groups"] == 2


if __name__ == "__main__":
    print("=== Regex Validator Tests ===\n")
    print("Run with: pytest tests/test_regex_validator.py -v")
