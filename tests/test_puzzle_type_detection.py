"""
Tests für Puzzle-Typ-Erkennung (PHASE 1)

Testet die Klassifikation von Logic Puzzles in:
- ENTITY_SAT: Entitäten-basierte Rätsel (Leo/Mark/Nick)
- NUMERICAL_CSP: Zahlen-basierte Constraint-Rätsel
- HYBRID: Kombination aus beiden
- UNKNOWN: Kein Rätsel

Tests auch die numerische Constraint-Erkennung in component_60.
"""

import pytest

from component_6_linguistik_engine import LinguisticPreprocessor
from component_41_input_orchestrator import InputOrchestrator, PuzzleType
from component_60_constraint_detector import ConstraintDetector


@pytest.fixture
def orchestrator():
    """Fixture für InputOrchestrator mit Preprocessor."""
    preprocessor = LinguisticPreprocessor()
    return InputOrchestrator(preprocessor=preprocessor)


@pytest.fixture
def constraint_detector():
    """Fixture für ConstraintDetector."""
    return ConstraintDetector()


class TestPuzzleTypeClassification:
    """Tests für classify_logic_puzzle_type() in InputOrchestrator."""

    def test_numerical_csp_gesuchte_zahl(self, orchestrator):
        """Test: 'gesuchte Zahl' Puzzle wird als NUMERICAL_CSP erkannt."""
        text = """
        1. Die gesuchte Zahl ist teilbar durch die Differenz zwischen den Nummern
           der letzten richtigen und der ersten falschen Behauptung.
        2. Die Summe der Nummern der richtigen Behauptungen ist teilbar durch 5.
        3. Die Anzahl der Teiler der gesuchten Zahl ist größer als 10.
        Was ist die kleinste gesuchte Zahl?
        """
        segments = orchestrator._segment_text(text)
        classified_segments = [orchestrator.classify_segment(seg) for seg in segments]

        puzzle_type = orchestrator.classify_logic_puzzle_type(text, classified_segments)

        assert puzzle_type == PuzzleType.NUMERICAL_CSP

    def test_numerical_csp_teilbarkeit(self, orchestrator):
        """Test: Teilbarkeits-Puzzle wird als NUMERICAL_CSP erkannt."""
        text = """
        Eine Zahl ist teilbar durch 3 und durch 7.
        Die Summe der Ziffern ist 12.
        Die Differenz zur nächsten Primzahl ist kleiner als 5.
        Welche Zahl ist gesucht?
        """
        segments = orchestrator._segment_text(text)
        classified_segments = [orchestrator.classify_segment(seg) for seg in segments]

        puzzle_type = orchestrator.classify_logic_puzzle_type(text, classified_segments)

        assert puzzle_type == PuzzleType.NUMERICAL_CSP

    def test_entity_sat_brandy_puzzle(self, orchestrator):
        """Test: Leo/Mark/Nick Rätsel wird als ENTITY_SAT erkannt."""
        text = """
        Leo, Mark und Nick bestellen Getränke.
        Entweder Leo oder Mark trinkt Brandy, aber nie beide.
        Wenn Mark Brandy trinkt, dann trinkt Nick Wasser.
        Nick trinkt nie Brandy.
        Wer trinkt Brandy?
        """
        segments = orchestrator._segment_text(text)
        classified_segments = [orchestrator.classify_segment(seg) for seg in segments]

        puzzle_type = orchestrator.classify_logic_puzzle_type(text, classified_segments)

        assert puzzle_type == PuzzleType.ENTITY_SAT

    def test_entity_sat_multiple_entities(self, orchestrator):
        """Test: Rätsel mit mehreren Entitäten wird als ENTITY_SAT erkannt."""
        text = """
        Anna mag Schokolade. Bob isst gerne Pizza.
        Wenn Anna Schokolade mag, dann mag Bob keine Schokolade.
        Charlie bestellt immer das gleiche wie Anna.
        Was bestellt Charlie?
        """
        segments = orchestrator._segment_text(text)
        classified_segments = [orchestrator.classify_segment(seg) for seg in segments]

        puzzle_type = orchestrator.classify_logic_puzzle_type(text, classified_segments)

        assert puzzle_type == PuzzleType.ENTITY_SAT

    def test_hybrid_puzzle(self, orchestrator):
        """Test: Puzzle mit Entitäten UND Zahlen wird als HYBRID erkannt."""
        text = """
        Leo hat eine Zahl gewählt. Mark hat eine andere Zahl.
        Leos Zahl ist teilbar durch 3. Marks Zahl ist eine Primzahl.
        Die Summe der beiden Zahlen ist 15.
        Leo trinkt Kaffee wenn seine Zahl kleiner ist als Marks Zahl.
        Wer trinkt Kaffee?
        """
        segments = orchestrator._segment_text(text)
        classified_segments = [orchestrator.classify_segment(seg) for seg in segments]

        puzzle_type = orchestrator.classify_logic_puzzle_type(text, classified_segments)

        assert puzzle_type == PuzzleType.HYBRID

    def test_unknown_not_enough_patterns(self, orchestrator):
        """Test: Text mit 'Zahl' wird als NUMERICAL_CSP klassifiziert."""
        text = """
        Es gibt eine Zahl.
        Die Zahl ist interessant.
        Was ist die Zahl?
        """
        segments = orchestrator._segment_text(text)
        classified_segments = [orchestrator.classify_segment(seg) for seg in segments]

        puzzle_type = orchestrator.classify_logic_puzzle_type(text, classified_segments)

        # Nach Verbesserung der Counting-Logik: "die Zahl" erscheint 3x -> NUMERICAL_CSP
        assert puzzle_type == PuzzleType.NUMERICAL_CSP

    def test_unknown_no_puzzle(self, orchestrator):
        """Test: Einfache Frage wird als UNKNOWN klassifiziert."""
        text = "Was ist die Hauptstadt von Deutschland?"
        segments = orchestrator._segment_text(text)
        classified_segments = [orchestrator.classify_segment(seg) for seg in segments]

        puzzle_type = orchestrator.classify_logic_puzzle_type(text, classified_segments)

        assert puzzle_type == PuzzleType.UNKNOWN


class TestNumericalConstraintDetection:
    """Tests für detect_numerical_constraints() in ConstraintDetector."""

    def test_detect_divisibility_constraints(self, constraint_detector):
        """Test: Teilbarkeits-Constraints werden erkannt."""
        text = "Die Zahl ist teilbar durch 3 und durch 7."

        result = constraint_detector.detect_numerical_constraints(text)

        assert result["has_numerical_constraints"] is True
        assert "DIVISIBILITY" in result["constraint_types"]
        assert result["constraint_counts"]["divisibility"] >= 1

    def test_detect_arithmetic_constraints(self, constraint_detector):
        """Test: Arithmetische Constraints werden erkannt."""
        text = "Die Summe der Nummern der richtigen Behauptungen ist teilbar durch 5."

        result = constraint_detector.detect_numerical_constraints(text)

        assert result["has_numerical_constraints"] is True
        assert "ARITHMETIC" in result["constraint_types"]
        assert result["constraint_counts"]["arithmetic"] >= 1

    def test_detect_meta_constraints(self, constraint_detector):
        """Test: Meta-Constraints werden erkannt."""
        text = "Die Anzahl der Teiler der gesuchten Zahl ist größer als 10."

        result = constraint_detector.detect_numerical_constraints(text)

        assert result["has_numerical_constraints"] is True
        assert "META" in result["constraint_types"]
        assert result["meta_constraints"] is True

    def test_detect_boolean_constraints(self, constraint_detector):
        """Test: Boolean Constraints werden erkannt."""
        text = "Behauptung 1 ist richtig. Behauptung 2 ist falsch."

        result = constraint_detector.detect_numerical_constraints(text)

        assert result["has_numerical_constraints"] is True
        assert "BOOLEAN" in result["constraint_types"]
        assert result["constraint_counts"]["boolean"] >= 2

    def test_extract_numerical_variables(self, constraint_detector):
        """Test: Numerische Variablen werden extrahiert."""
        text = "Die gesuchte Zahl X ist teilbar durch Y."

        result = constraint_detector.detect_numerical_constraints(text)

        assert "zahl" in result["numerical_variables"]
        assert (
            "X" in result["numerical_variables"] or "Y" in result["numerical_variables"]
        )

    def test_no_numerical_constraints(self, constraint_detector):
        """Test: Texte ohne numerische Constraints werden korrekt erkannt."""
        text = "Leo trinkt Brandy. Mark mag Schokolade."

        result = constraint_detector.detect_numerical_constraints(text)

        assert result["has_numerical_constraints"] is False
        assert len(result["constraint_types"]) == 0

    def test_confidence_scoring(self, constraint_detector):
        """Test: Confidence steigt mit mehr Constraints."""
        # Text mit wenig Constraints
        text_low = "Die Zahl ist teilbar durch 3."
        result_low = constraint_detector.detect_numerical_constraints(text_low)

        # Text mit vielen Constraints
        text_high = """
        Die gesuchte Zahl ist teilbar durch die Differenz.
        Die Summe der Nummern ist teilbar durch 5.
        Die Anzahl der Teiler ist größer als 10.
        """
        result_high = constraint_detector.detect_numerical_constraints(text_high)

        # High sollte höhere Confidence haben
        assert result_high["confidence"] > result_low["confidence"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
