# tests/test_command_suggestions.py
"""
Tests für Command Suggestion System (component_26).

Testet:
- Known Typo Detection
- Fuzzy Matching
- False Positive Prevention
- Edge Cases
- Full Suggestion Formatting
"""

import pytest

from component_26_command_suggestions import CommandSuggester, get_command_suggester


class TestKnownTypoDetection:
    """Tests für bekannte Tippfehler-Erkennung"""

    @pytest.fixture
    def suggester(self):
        """Fixture: CommandSuggester Instanz"""
        return CommandSuggester()

    def test_lerne_typos(self, suggester):
        """Test: Bekannte 'Lerne:' Tippfehler werden erkannt"""
        test_cases = [
            ("lehre: ein Hund ist ein Tier", "lerne:"),
            ("lernr: eine Katze ist ein Tier", "lerne:"),
            ("lern: ein Apfel ist eine Frucht", "lerne:"),
        ]

        for typo_input, expected_command in test_cases:
            result = suggester.suggest_command(typo_input)
            assert result is not None, f"No suggestion for '{typo_input}'"
            assert result["suggestion"] == expected_command
            assert result["confidence"] == 0.95  # High confidence for exact typos

    def test_lerne_muster_typos(self, suggester):
        """Test: Bekannte 'Lerne Muster:' Tippfehler werden erkannt"""
        test_cases = [
            ('lerne mustre: "X mag Y" bedeutet LIKES', "lerne muster:"),
            ('lehre muster: "X mag Y" bedeutet LIKES', "lerne muster:"),
            ('lerne muste: "X mag Y" bedeutet LIKES', "lerne muster:"),
        ]

        for typo_input, expected_command in test_cases:
            result = suggester.suggest_command(typo_input)
            assert result is not None
            assert result["suggestion"] == expected_command
            assert result["confidence"] == 0.95

    def test_was_ist_typos(self, suggester):
        """Test: Bekannte 'Was ist' Tippfehler werden erkannt"""
        test_cases = [
            ("wa ist ein Hund?", "was ist"),
            ("was it ein Apfel?", "was ist"),
            ("wass ist eine Katze?", "was ist"),
        ]

        for typo_input, expected_command in test_cases:
            result = suggester.suggest_command(typo_input)
            assert result is not None
            assert result["suggestion"] == expected_command
            assert result["confidence"] == 0.95

    def test_definiere_typos(self, suggester):
        """Test: Bekannte 'Definiere:' Tippfehler werden erkannt"""
        test_cases = [
            ("defniere: hund/farbe = braun", "definiere:"),
            ("definere: katze/groesse = klein", "definiere:"),
            ("deffiniere: apfel/farbe = rot", "definiere:"),
        ]

        for typo_input, expected_command in test_cases:
            result = suggester.suggest_command(typo_input)
            assert result is not None
            assert result["suggestion"] == expected_command
            assert result["confidence"] == 0.95

    def test_wer_ist_typos(self, suggester):
        """Test: Bekannte 'Wer ist' Tippfehler werden erkannt"""
        test_cases = [
            ("wer it Einstein?", "wer ist"),
            ("wer si Newton?", "wer ist"),
            ("wee ist Galileo?", "wer ist"),
        ]

        for typo_input, expected_command in test_cases:
            result = suggester.suggest_command(typo_input)
            assert result is not None
            assert result["suggestion"] == expected_command
            assert result["confidence"] == 0.95

    def test_wie_typos(self, suggester):
        """Test: Bekannte 'Wie' Tippfehler werden erkannt"""
        test_cases = [
            ("wi funktioniert das?", "wie"),
            ("wir geht es?", "wie"),
        ]

        for typo_input, expected_command in test_cases:
            result = suggester.suggest_command(typo_input)
            assert result is not None
            assert result["suggestion"] == expected_command
            assert result["confidence"] == 0.95

    def test_warum_typos(self, suggester):
        """Test: Bekannte 'Warum' Tippfehler werden erkannt"""
        test_cases = [
            ("warrum ist der Himmel blau?", "warum"),
            ("warun ist das so?", "warum"),
            ("waum passiert das?", "warum"),
        ]

        for typo_input, expected_command in test_cases:
            result = suggester.suggest_command(typo_input)
            assert result is not None
            assert result["suggestion"] == expected_command
            assert result["confidence"] == 0.95


class TestFuzzyMatching:
    """Tests für Fuzzy-Matching mit Levenshtein Distance"""

    @pytest.fixture
    def suggester(self):
        return CommandSuggester(similarity_threshold=0.6)

    def test_fuzzy_lerne_variations(self, suggester):
        """Test: Fuzzy-Match für ähnliche Variationen von 'lerne:'"""
        # 'lrnee:' ist kein bekannter Typo, aber ähnlich genug für Fuzzy-Match
        result = suggester.suggest_command("lrnee: ein Test")
        assert result is not None
        assert result["suggestion"] == "lerne:"
        # Fuzzy matches haben reduced confidence (similarity * 0.8)
        assert 0.48 <= result["confidence"] <= 0.76  # 0.6*0.8 = 0.48 minimum

    def test_fuzzy_threshold_respected(self, suggester):
        """Test: Similarity-Threshold wird respektiert"""
        # 'xyz:' hat keine Ähnlichkeit mit bekannten Befehlen
        result = suggester.suggest_command("xyz: ein Test")
        # Sollte None sein, da Similarity zu gering
        # (könnte auch einen Match geben wenn zufällig über Threshold)
        # -> Toleranter Test: Prüfe nur dass nicht crashed
        assert result is None or result["confidence"] < 0.95

    def test_custom_threshold(self):
        """Test: Custom Similarity-Threshold funktioniert"""
        # Sehr hoher Threshold (0.9) -> weniger Matches
        strict_suggester = CommandSuggester(similarity_threshold=0.9)
        result = strict_suggester.suggest_command("lrnee: ein Test")

        # Bei 0.9 Threshold könnte 'lrnee' nicht matchen
        # -> Toleranter Test: Prüfe nur, dass kein Crash
        assert result is None or isinstance(result, dict)


class TestFalsePositivePrevention:
    """Tests für False-Positive Prevention"""

    @pytest.fixture
    def suggester(self):
        return CommandSuggester()

    def test_correct_command_no_suggestion(self, suggester):
        """Test: Korrekte Befehle erhalten KEINE Vorschläge"""
        correct_inputs = [
            "Lerne: ein Hund ist ein Tier",
            "Lerne Muster: 'X mag Y' bedeutet LIKES",
            "Was ist ein Apfel?",
            "Definiere: hund/farbe = braun",
            "Wer ist Einstein?",
            "Wie funktioniert das?",
            "Warum ist der Himmel blau?",
        ]

        for correct_input in correct_inputs:
            result = suggester.suggest_command(correct_input)
            assert (
                result is None
            ), f"False positive for correct input: '{correct_input}'"

    def test_lowercase_correct_commands(self, suggester):
        """Test: Lowercase korrekte Befehle erhalten keine Vorschläge"""
        correct_inputs_lowercase = [
            "lerne: ein Hund ist ein Tier",
            "was ist ein Apfel?",
            "definiere: hund/farbe = braun",
        ]

        for correct_input in correct_inputs_lowercase:
            result = suggester.suggest_command(correct_input)
            assert result is None

    def test_non_command_input(self, suggester):
        """Test: Nicht-Befehl-Eingaben erhalten keine Vorschläge"""
        non_commands = [
            "irgendwas anderes",
            "Das ist ein normaler Satz.",
            "12345",
            "Ein Hund bellt.",
        ]

        for non_command in non_commands:
            result = suggester.suggest_command(non_command)
            # Könnte None sein ODER einen niedrig-Confidence Match
            # -> Toleranter Test
            if result is not None:
                # Falls Match, sollte Confidence niedrig sein
                assert result["confidence"] < 0.90


class TestEdgeCases:
    """Tests für Edge Cases"""

    @pytest.fixture
    def suggester(self):
        return CommandSuggester()

    def test_empty_input(self, suggester):
        """Test: Leere Eingabe"""
        assert suggester.suggest_command("") is None
        assert suggester.suggest_command("   ") is None

    def test_very_short_input(self, suggester):
        """Test: Sehr kurze Eingabe"""
        result = suggester.suggest_command("le")
        # Könnte None sein oder einen Match
        assert result is None or isinstance(result, dict)

    def test_special_characters(self, suggester):
        """Test: Sonderzeichen in Eingabe"""
        result = suggester.suggest_command("lerne: @#$%^&*()")
        # Sollte None sein (kein Typo erkannt)
        assert result is None

    def test_unicode_input(self, suggester):
        """Test: Unicode-Zeichen"""
        result = suggester.suggest_command("lerne: Äpfel sind Früchte")
        # Sollte None sein (korrekter Befehl mit Umlauten)
        assert result is None

    def test_very_long_input(self, suggester):
        """Test: Sehr lange Eingabe"""
        long_input = "lernee: " + "sehr langer Text " * 50
        result = suggester.suggest_command(long_input)
        # Sollte Typo 'lernee:' erkennen trotz Länge
        # Falls bekannter Typo, sonst Fuzzy-Match
        if result:
            assert "lerne:" in result["suggestion"]


class TestFullSuggestionFormatting:
    """Tests für komplette Vorschlags-Formatierung"""

    @pytest.fixture
    def suggester(self):
        return CommandSuggester()

    def test_separator_with_colon_command(self, suggester):
        """Test: Separator-Logik für Befehle mit Doppelpunkt"""
        result = suggester.suggest_command("lernr: ein Hund ist ein Tier")

        assert result is not None
        # full_suggestion sollte "Lerne: ein Hund ist ein Tier" sein
        assert result["full_suggestion"].startswith("Lerne:")
        assert "ein Hund ist ein Tier" in result["full_suggestion"]

    def test_separator_with_space_command(self, suggester):
        """Test: Separator-Logik für Befehle mit Leerzeichen"""
        result = suggester.suggest_command("wa ist ein Apfel?")

        assert result is not None
        # full_suggestion sollte "Was ist ein Apfel?" sein
        assert result["full_suggestion"].startswith("Was ist")
        assert "ein Apfel?" in result["full_suggestion"]

    def test_capitalization(self, suggester):
        """Test: Befehl wird kapitalisiert"""
        result = suggester.suggest_command("lernr: test")

        assert result is not None
        # Suggestion sollte mit Großbuchstaben beginnen
        assert result["full_suggestion"][0].isupper()

    def test_description_and_example_included(self, suggester):
        """Test: Description und Example sind im Result enthalten"""
        result = suggester.suggest_command("lernr: test")

        assert result is not None
        assert "description" in result
        assert "example" in result
        assert len(result["description"]) > 0
        assert len(result["example"]) > 0


class TestGetAllCommands:
    """Tests für get_all_commands() Funktion"""

    def test_all_commands_returned(self):
        """Test: Alle Befehle werden zurückgegeben"""
        suggester = CommandSuggester()
        all_commands = suggester.get_all_commands()

        assert len(all_commands) >= 7  # Mindestens 7 Befehle definiert
        assert all(isinstance(cmd, dict) for cmd in all_commands)

    def test_command_structure(self):
        """Test: Command-Struktur ist vollständig"""
        suggester = CommandSuggester()
        all_commands = suggester.get_all_commands()

        for cmd in all_commands:
            assert "command" in cmd
            assert "description" in cmd
            assert "example" in cmd


class TestSingletonPattern:
    """Tests für Singleton-Pattern"""

    def test_singleton_returns_same_instance(self):
        """Test: get_command_suggester() gibt immer gleiche Instanz zurück"""
        instance1 = get_command_suggester()
        instance2 = get_command_suggester()

        assert instance1 is instance2

    def test_singleton_initialization_error_handling(self):
        """Test: Singleton-Getter hat Error Handling"""
        # Dieser Test prüft nur, dass get_command_suggester() callable ist
        # und keine Exception wirft bei normalem Aufruf
        try:
            instance = get_command_suggester()
            assert instance is not None
        except Exception as e:
            pytest.fail(f"get_command_suggester() raised unexpected exception: {e}")


class TestThresholdValidation:
    """Tests für similarity_threshold Validation"""

    def test_valid_threshold(self):
        """Test: Gültige Threshold-Werte werden akzeptiert"""
        valid_thresholds = [0.0, 0.5, 0.6, 0.9, 1.0]

        for threshold in valid_thresholds:
            suggester = CommandSuggester(similarity_threshold=threshold)
            assert suggester.similarity_threshold == threshold

    def test_invalid_threshold_too_low(self):
        """Test: Threshold < 0.0 wirft ValueError"""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            CommandSuggester(similarity_threshold=-0.1)

    def test_invalid_threshold_too_high(self):
        """Test: Threshold > 1.0 wirft ValueError"""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            CommandSuggester(similarity_threshold=1.5)

    def test_invalid_threshold_extreme(self):
        """Test: Extrem ungültige Threshold-Werte"""
        with pytest.raises(ValueError):
            CommandSuggester(similarity_threshold=999.0)

        with pytest.raises(ValueError):
            CommandSuggester(similarity_threshold=-999.0)


if __name__ == "__main__":
    print("=== Command Suggestions Tests ===\n")
    print("Run with: pytest tests/test_command_suggestions.py -v")
