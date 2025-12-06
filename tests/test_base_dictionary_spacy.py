"""
Tests for spaCy base dictionary integration in PatternOrchestrator.

This test suite verifies that the base dictionary is loaded correctly from spaCy
and that it effectively prevents false positive typo detections for common German words.
"""

import pytest

from component_1_netzwerk import KonzeptNetzwerk
from component_24_pattern_orchestrator import PatternOrchestrator


class TestBaseDictionarySpacy:
    """Test spaCy base dictionary integration."""

    @pytest.fixture
    def netzwerk(self):
        """Create a fresh KonzeptNetzwerk instance."""
        netz = KonzeptNetzwerk()
        yield netz
        netz.close()

    @pytest.fixture
    def orchestrator(self, netzwerk):
        """Create PatternOrchestrator with base dictionary."""
        return PatternOrchestrator(netzwerk)

    def test_base_dictionary_loads(self, orchestrator):
        """Test that base dictionary loads with significant word count."""
        assert (
            len(orchestrator.base_dictionary) > 100000
        ), f"Expected >100k words, got {len(orchestrator.base_dictionary)}"

    def test_brandy_puzzle_words_recognized(self, orchestrator):
        """Test that words from Brandy puzzle are in base dictionary."""
        brandy_words = [
            "trinkt",  # drinks
            "bestellt",  # orders
            "will",  # wants
            "dreien",  # three (dative plural)
            "folgendes",  # following
            "eine",  # a/an
        ]

        for word in brandy_words:
            assert (
                word.lower() in orchestrator.base_dictionary
            ), f"Word '{word}' should be in base dictionary"

    def test_common_german_verbs_recognized(self, orchestrator):
        """Test that common German verbs are recognized."""
        verbs = [
            "sein",
            "haben",
            "werden",
            "können",
            "müssen",
            "gehen",
            "kommen",
            "sehen",
            "machen",
            "sagen",
            "trinken",
            "essen",
            "bestellen",
            "wollen",
            "denken",
        ]

        for verb in verbs:
            assert (
                verb.lower() in orchestrator.base_dictionary
            ), f"Verb '{verb}' should be in base dictionary"

    def test_common_german_nouns_recognized(self, orchestrator):
        """Test that common German nouns are recognized."""
        nouns = [
            "haus",
            "mann",
            "frau",
            "kind",
            "tag",
            "person",
            "welt",
            "zeit",
            "hand",
            "auge",
        ]

        for noun in nouns:
            assert (
                noun.lower() in orchestrator.base_dictionary
            ), f"Noun '{noun}' should be in base dictionary"

    def test_common_german_adjectives_recognized(self, orchestrator):
        """Test that common German adjectives are recognized."""
        adjectives = [
            "gut",
            "groß",
            "klein",
            "neu",
            "alt",
            "schön",
            "schnell",
            "langsam",
            "hoch",
            "tief",
        ]

        for adj in adjectives:
            assert (
                adj.lower() in orchestrator.base_dictionary
            ), f"Adjective '{adj}' should be in base dictionary"

    def test_no_false_positives_brandy_puzzle(self, orchestrator):
        """Test that Brandy puzzle generates no false positive typo detections."""
        text = "Trinkt eine Person Whiskey und bestellt eine andere Person Vodka?"
        result = orchestrator.process_input(text)

        # Should have zero or very few typo corrections (vodka/whiskey might trigger)
        assert len(result["typo_corrections"]) <= 2, (
            f"Expected <=2 typo corrections, got {len(result['typo_corrections'])}: "
            f"{result['typo_corrections']}"
        )

        # Should not flag common German words
        flagged_words = [c["original"] for c in result["typo_corrections"]]
        common_words = ["trinkt", "eine", "person", "und", "bestellt", "andere"]

        for word in common_words:
            assert word.lower() not in [
                w.lower() for w in flagged_words
            ], f"Common word '{word}' should not be flagged as typo"

    def test_no_false_positives_complex_sentence(self, orchestrator):
        """Test that complex German sentences don't trigger false positives."""
        text = "Die Person trinkt Kaffee während die andere Person Tee bestellt."
        result = orchestrator.process_input(text)

        # Should have zero typo corrections for this valid German sentence
        assert (
            len(result["typo_corrections"]) == 0
        ), f"Expected no typo corrections for valid German, got: {result['typo_corrections']}"

    def test_base_dict_checked_before_netzwerk(self, orchestrator):
        """Test that base dictionary is checked BEFORE Neo4j vocabulary."""
        # Use a common German word that's in base dictionary
        # Even if it's not in Neo4j, it should still be recognized
        test_words = ["gestern", "morgen", "heute", "vielleicht"]

        for word in test_words:
            # Verify it's in base dictionary
            assert (
                word in orchestrator.base_dictionary
            ), f"Word '{word}' should be in base dictionary"

            # Should NOT be flagged as typo (base dict catches it)
            result = orchestrator.process_input(word)
            assert (
                len(result["typo_corrections"]) == 0
            ), f"Word '{word}' in base dictionary should not be flagged as typo"

    def test_capitalized_words_still_work(self, orchestrator):
        """Test that capitalized words are still recognized."""
        text = "Trinkt eine Person Whiskey?"
        result = orchestrator.process_input(text)

        # "Trinkt" and "Person" are capitalized but should be recognized
        flagged = [c["original"] for c in result["typo_corrections"]]
        assert "Trinkt" not in flagged
        assert "eine" not in flagged
        assert "Person" not in flagged

    def test_mixed_case_lookup(self, orchestrator):
        """Test that case-insensitive lookup works correctly."""
        # Base dictionary stores lowercase
        assert "trinkt" in orchestrator.base_dictionary
        assert "TRINKT" not in orchestrator.base_dictionary  # Not stored uppercase

        # But process_input uses .lower() for comparison
        result = orchestrator.process_input("TRINKT")
        assert (
            len(result["typo_corrections"]) == 0
        ), "Uppercase word should still match lowercase base dictionary"

    def test_performance_acceptable(self, orchestrator):
        """Test that base dictionary lookup doesn't significantly slow down processing."""
        import time

        # Process a moderately long text
        text = " ".join(["Ein Mann trinkt Bier"] * 50)  # 150 words

        start = time.time()
        orchestrator.process_input(text)
        duration = time.time() - start

        # Should complete in <1 second (set lookups are O(1))
        assert (
            duration < 1.0
        ), f"Processing {len(text.split())} words took {duration:.2f}s (expected <1s)"

    def test_base_dict_survives_initialization(self, netzwerk):
        """Test that base dictionary is loaded during initialization."""
        # Create new orchestrator
        orch = PatternOrchestrator(netzwerk)

        # Base dictionary should be populated immediately
        assert (
            len(orch.base_dictionary) > 100000
        ), "Base dictionary should be loaded during __init__"

    def test_graceful_degradation_if_spacy_missing(self, monkeypatch):
        """Test that system works even if spaCy model is not available."""
        # Mock spacy.load to raise OSError
        import spacy

        original_load = spacy.load

        def mock_load(model_name):
            raise OSError("Model not found")

        monkeypatch.setattr(spacy, "load", mock_load)

        # Create new orchestrator
        netz = KonzeptNetzwerk()
        try:
            orch = PatternOrchestrator(netz)

            # Should have empty base dictionary but not crash
            assert (
                len(orch.base_dictionary) == 0
            ), "Base dictionary should be empty if spaCy model missing"

            # System should still work (just with more false positives)
            result = orch.process_input("Ein Test")
            assert result is not None

        finally:
            netz.close()
            monkeypatch.setattr(spacy, "load", original_load)
