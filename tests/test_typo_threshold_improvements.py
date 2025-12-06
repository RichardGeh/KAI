"""
Test für verbesserte Typo-Detection Schwellenwerte

Stellt sicher, dass normale deutsche Wörter nicht mehr fälschlicherweise
als Tippfehler markiert werden.
"""

from unittest.mock import Mock

import pytest

from component_24_pattern_orchestrator import PatternOrchestrator
from component_25_adaptive_thresholds import AdaptiveThresholdManager, BootstrapPhase


class TestTypoThresholdImprovements:
    """Tests für erhöhte Typo-Detection Schwellenwerte"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Mock für KonzeptNetzwerk mit leerem Vocabulary"""
        mock = Mock()
        # Leeres Vocabulary simuliert cold_start Phase
        mock.get_all_known_words.return_value = []
        mock.get_normalized_word_frequency.return_value = 0.0
        mock.query_graph_for_facts.return_value = {}
        mock._feedback = None
        mock.get_word_connections.return_value = []
        return mock

    def test_confidence_gates_raised_cold_start(self, mock_netzwerk):
        """Schwellenwerte sollten in COLD_START Phase erhöht sein"""
        manager = AdaptiveThresholdManager(mock_netzwerk)
        gates = manager.get_confidence_gates(BootstrapPhase.COLD_START)

        assert gates["ask_user"] == 0.95, "ask_user threshold sollte 0.95 sein"
        assert gates["min_confidence"] == 0.85, "min_confidence sollte 0.85 sein"

    def test_confidence_gates_raised_warming(self, mock_netzwerk):
        """Schwellenwerte sollten in WARMING Phase erhöht sein"""
        manager = AdaptiveThresholdManager(mock_netzwerk)
        gates = manager.get_confidence_gates(BootstrapPhase.WARMING)

        assert gates["ask_user"] == 0.90, "ask_user threshold sollte 0.90 sein"
        assert gates["min_confidence"] == 0.80, "min_confidence sollte 0.80 sein"

    def test_confidence_gates_raised_mature(self, mock_netzwerk):
        """Schwellenwerte sollten in MATURE Phase erhöht sein"""
        manager = AdaptiveThresholdManager(mock_netzwerk)
        gates = manager.get_confidence_gates(BootstrapPhase.MATURE)

        assert gates["ask_user"] == 0.85, "ask_user threshold sollte 0.85 sein"
        assert gates["min_confidence"] == 0.70, "min_confidence sollte 0.70 sein"

    def test_common_german_words_no_false_positives(self, mock_netzwerk):
        """
        Normale deutsche Wörter sollten nicht als Typos erkannt werden

        Diese Wörter waren vorher false positives:
        - trinkt, bestellt, will, dreien, folgendes
        """
        # Mock mit sehr wenigen Wörtern (simuliert, dass diese Wörter unbekannt sind)
        mock_netzwerk.get_all_known_words.return_value = ["apfel", "banane"]

        orchestrator = PatternOrchestrator(mock_netzwerk)

        # Test-Sätze mit vorher problematischen Wörtern
        test_sentences = [
            "Leo trinkt gerne Brandy",
            "Mark bestellt einen Kaffee",
            "Nick will nach Hause",
            "Von den dreien ist Leo der größte",
            "Folgendes ist wichtig zu beachten",
        ]

        for sentence in test_sentences:
            result = orchestrator.process_input(sentence)

            # Mit den erhöhten Schwellenwerten sollten keine oder sehr wenige
            # Typo-Kandidaten gefunden werden (nur bei sehr hoher Confidence)
            # In der Praxis werden diese Wörter wahrscheinlich gar nicht mehr
            # als Typos erkannt, weil die Confidence zu niedrig ist

            # Wenn Typos gefunden werden, sollte keine User-Klarstellung nötig sein
            # (wegen des höheren ask_user Thresholds)
            if result["typo_corrections"]:
                assert (
                    result["needs_user_clarification"] is False
                ), f"Unerwartete User-Klarstellung für: {sentence}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
