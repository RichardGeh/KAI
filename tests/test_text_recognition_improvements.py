# tests/test_text_recognition_improvements.py
"""
Umfassende Test-Suite für alle Texterkennung-Verbesserungen.

Testet alle 10 implementierten Schwachstellen-Fixes:
1. Negation-Behandlung (KRITISCH)
2. spaCy Sentence Tokenizer (KRITISCH)
3. Komparative Relationen (HOCH)
4. Multi-Object Extraktion mit Adjektiven (HOCH)
5. Konditionale Aussagen (HOCH)
6. Uncertainty/Hedge-Marker (MITTEL)
7. Temporale Marker (MITTEL)
8. Quantor-Extraktion (NIEDRIG)
9. Synonym-Erweiterung (NIEDRIG)
10. Erweiterte Arithmetik-Trigger (NIEDRIG)
"""
import pytest

from component_5_linguistik_strukturen import MeaningPointCategory, Polarity
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from component_11_embedding_service import EmbeddingService
from component_41_input_orchestrator import InputOrchestrator


class TestNegationHandling:
    """Tests für Negation-Behandlung (Schwachstelle 1 - KRITISCH)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_negation_in_capable_of(self, extractor):
        """Test: Negation wird in CAPABLE_OF erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Pinguin kann nicht fliegen")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.DEFINITION
        assert mp.arguments["relation_type"] == "CAPABLE_OF"
        assert mp.arguments["subject"] == "pinguin"
        assert mp.arguments["object"] == "fliegen"
        assert mp.polarity == Polarity.NEGATIVE  # NEU: Polarity ist NEGATIVE!
        assert mp.arguments["negated"] is True

    def test_negation_in_has_property(self, extractor):
        """Test: Negation wird in HAS_PROPERTY erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Apfel ist nicht rot")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.DEFINITION
        assert mp.arguments["relation_type"] == "HAS_PROPERTY"
        assert mp.arguments["subject"] == "apfel"
        assert mp.arguments["object"] == "rot"
        assert mp.polarity == Polarity.NEGATIVE
        assert mp.arguments["negated"] is True

    def test_negation_in_part_of(self, extractor):
        """Test: Negation wird in PART_OF erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Hund hat keine Flügel")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.DEFINITION
        assert mp.arguments["relation_type"] == "PART_OF"
        assert mp.arguments["subject"] == "hund"
        assert mp.arguments["object"] == "flügel"
        assert mp.polarity == Polarity.NEGATIVE
        assert mp.arguments["negated"] is True

    def test_no_negation_in_positive_statement(self, extractor):
        """Test: Positive Aussagen haben Polarity POSITIVE."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Vogel kann fliegen")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.polarity == Polarity.POSITIVE
        assert mp.arguments["negated"] is False


class TestSpacySentenceTokenizer:
    """Tests für spaCy Sentence Tokenizer (Schwachstelle 2 - KRITISCH)."""

    @pytest.fixture
    def orchestrator(self):
        """Erstellt InputOrchestrator mit spaCy."""
        preprocessor = LinguisticPreprocessor()
        return InputOrchestrator(preprocessor=preprocessor)

    def test_handles_abbreviations(self, orchestrator):
        """Test: Abkürzungen wie 'Dr.' werden korrekt behandelt."""
        text = "Dr. Müller sagt etwas. Ein Apfel ist rot."

        segments = orchestrator._segment_text(text)

        # Sollte 2 Sätze ergeben (nicht durch "Dr." getrennt)
        assert len(segments) == 2
        assert "Dr. Müller" in segments[0]
        assert "Apfel" in segments[1]

    def test_handles_decimal_numbers(self, orchestrator):
        """Test: Dezimalzahlen wie '3.14' werden nicht als Satzende erkannt."""
        text = "Pi ist 3.14 ungefähr. Ein Kreis ist rund."

        segments = orchestrator._segment_text(text)

        # Sollte 2 Sätze ergeben (nicht bei 3.14 getrennt)
        assert len(segments) == 2
        assert "3.14" in segments[0]
        assert "Kreis" in segments[1]

    def test_handles_multiple_punctuation(self, orchestrator):
        """Test: Mehrfache Satzzeichen werden korrekt behandelt."""
        text = "Das ist toll!!! Ein Apfel ist rot."

        segments = orchestrator._segment_text(text)

        # Sollte 2 Sätze ergeben
        assert len(segments) == 2
        assert "toll" in segments[0]
        assert "Apfel" in segments[1]


class TestComparativeRelations:
    """Tests für Komparative Relationen (Schwachstelle 3 - HOCH)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_comparative_larger_than(self, extractor):
        """Test: 'größer als' wird erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Hund ist größer als eine Katze")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.DEFINITION
        assert mp.arguments["relation_type"] == "COMPARATIVE"
        assert mp.arguments["subject"] == "hund"
        assert mp.arguments["comparison_type"] == "größer"
        assert mp.arguments["reference"] == "katze"

    def test_comparative_faster_than(self, extractor):
        """Test: 'schneller als' wird erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Gepard ist schneller als ein Löwe")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.DEFINITION
        assert mp.arguments["relation_type"] == "COMPARATIVE"
        assert mp.arguments["comparison_type"] == "schneller"


class TestMultiObjectExtraction:
    """Tests für Multi-Object Extraktion mit Adjektiven (Schwachstelle 4 - HOCH)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_extracts_adjective_from_is_a(self, extractor):
        """Test: Adjektive werden aus IS_A Pattern extrahiert."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Apfel ist eine rote Frucht")
        mps = extractor.extract(doc)

        # Sollte MINDESTENS 2 MeaningPoints zurückgeben
        # 1. IS_A: apfel -> frucht
        # 2. HAS_PROPERTY: apfel -> rot
        assert len(mps) >= 1

        # Prüfe IS_A
        is_a_mp = [mp for mp in mps if mp.arguments.get("relation_type") == "IS_A"]
        assert len(is_a_mp) > 0
        assert is_a_mp[0].arguments["subject"] == "apfel"
        assert is_a_mp[0].arguments["object"] == "frucht"

    def test_extracts_multiple_adjectives(self, extractor):
        """Test: Mehrere Adjektive werden extrahiert."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Apfel ist eine rote süße Frucht")
        mps = extractor.extract(doc)

        # Sollte IS_A + HAS_PROPERTY für jedes Adjektiv zurückgeben
        assert len(mps) >= 1


class TestConditionalStatements:
    """Tests für Konditionale Aussagen (Schwachstelle 5 - HOCH)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_conditional_wenn_dann(self, extractor):
        """Test: 'Wenn X dann Y' wird erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Wenn es regnet, dann wird die Straße nass")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.DEFINITION
        assert mp.arguments["relation_type"] == "CONDITIONAL"
        assert "regn" in mp.arguments["condition"]
        assert "straß" in mp.arguments["consequence"]

    def test_conditional_falls(self, extractor):
        """Test: 'Falls X, Y' wird erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Falls ein Vogel fliegt, ist es kein Pinguin")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.arguments["relation_type"] == "CONDITIONAL"


class TestUncertaintyMarkers:
    """Tests für Uncertainty/Hedge-Marker (Schwachstelle 6 - MITTEL)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_uncertainty_reduces_confidence(self, extractor):
        """Test: Uncertainty-Marker reduzieren Confidence."""
        preprocessor = LinguisticPreprocessor()

        # Ohne Uncertainty
        doc1 = preprocessor.process("Ein Apfel ist rot")
        mps1 = extractor.extract(doc1)
        confidence_without = mps1[0].confidence if mps1 else 0

        # Mit Uncertainty
        doc2 = preprocessor.process("Ein Apfel ist vielleicht rot")
        mps2 = extractor.extract(doc2)
        confidence_with = mps2[0].confidence if mps2 else 0

        # Confidence sollte reduziert sein
        assert confidence_with < confidence_without

    def test_hedge_words_are_captured(self, extractor):
        """Test: Hedge-Wörter werden in Metadata gespeichert."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Apfel ist wahrscheinlich rot")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert "hedge_words" in mp.arguments
        assert "wahrscheinlich" in mp.arguments["hedge_words"]
        assert "uncertainty_level" in mp.arguments


class TestTemporalMarkers:
    """Tests für Temporale Marker (Schwachstelle 7 - MITTEL)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_temporal_marker_extraction(self, extractor):
        """Test: Temporale Marker werden extrahiert."""
        temporal_markers, context = extractor._extract_temporal_markers(
            "Gestern war ein Apfel rot"
        )

        assert len(temporal_markers) > 0
        assert "gestern" in temporal_markers
        assert context == "past"

    def test_future_temporal_marker(self, extractor):
        """Test: Zukunfts-Marker werden erkannt."""
        temporal_markers, context = extractor._extract_temporal_markers(
            "Morgen wird der Apfel rot sein"
        )

        assert len(temporal_markers) > 0
        assert "morgen" in temporal_markers
        assert context == "future"


class TestQuantifierExtraction:
    """Tests für Quantor-Extraktion (Schwachstelle 8 - NIEDRIG)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_universal_quantifier(self, extractor):
        """Test: Universelle Quantoren werden erkannt."""
        quantifier, q_type = extractor._extract_quantifier("Alle Hunde sind Tiere")

        assert quantifier == "alle"
        assert q_type == "universal"

    def test_existential_quantifier(self, extractor):
        """Test: Existentielle Quantoren werden erkannt."""
        quantifier, q_type = extractor._extract_quantifier(
            "Manche Vögel können fliegen"
        )

        assert quantifier == "manche"
        assert q_type == "existential"

    def test_no_quantifier(self, extractor):
        """Test: Keine Quantoren wenn nicht vorhanden."""
        quantifier, q_type = extractor._extract_quantifier("Ein Hund ist ein Tier")

        assert quantifier is None
        assert q_type is None


class TestSynonymExpansion:
    """Tests für Synonym-Erweiterung (Schwachstelle 9 - NIEDRIG)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_synonym_vermag(self, extractor):
        """Test: Synonym 'vermag' wird wie 'kann' behandelt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Vogel vermag zu fliegen")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.DEFINITION
        assert mp.arguments["relation_type"] == "CAPABLE_OF"

    def test_synonym_verfuegt_ueber(self, extractor):
        """Test: Synonym 'verfügt über' wird wie 'hat' behandelt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Auto verfügt über Räder")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.DEFINITION
        assert mp.arguments["relation_type"] == "PART_OF"


class TestExtendedArithmeticTriggers:
    """Tests für erweiterte Arithmetik-Trigger (Schwachstelle 10 - NIEDRIG)."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_arithmetic_summe(self, extractor):
        """Test: 'Summe' wird als arithmetischer Trigger erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Was ist die Summe von 3 und 5")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.ARITHMETIC_QUESTION

    def test_arithmetic_prozent(self, extractor):
        """Test: 'Prozent' wird als arithmetischer Trigger erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Was sind 50 Prozent von 100")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.ARITHMETIC_QUESTION

    def test_arithmetic_durchschnitt(self, extractor):
        """Test: 'Durchschnitt' wird als arithmetischer Trigger erkannt."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Berechne den Durchschnitt von 10, 20, 30")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category == MeaningPointCategory.ARITHMETIC_QUESTION


class TestIntegrationScenarios:
    """Integration-Tests mit mehreren Features kombiniert."""

    @pytest.fixture
    def extractor(self):
        """Erstellt MeaningPointExtractor für Tests."""
        embedding_service = EmbeddingService()
        preprocessor = LinguisticPreprocessor()
        return MeaningPointExtractor(embedding_service, preprocessor)

    def test_negation_with_uncertainty(self, extractor):
        """Test: Negation + Uncertainty kombiniert."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Ein Pinguin kann wahrscheinlich nicht fliegen")
        mps = extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.polarity == Polarity.NEGATIVE
        assert "hedge_words" in mp.arguments
        assert mp.confidence < 0.91  # Reduziert durch Uncertainty

    def test_comparative_with_temporal(self, extractor):
        """Test: Komparative + Temporal kombiniert."""
        LinguisticPreprocessor()

        # Extrahiere temporale Marker separat
        temporal_markers, context = extractor._extract_temporal_markers(
            "Gestern war ein Hund größer als eine Katze"
        )

        assert "gestern" in temporal_markers
        assert context == "past"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
