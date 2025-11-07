# component_20_pattern_recognition_sequence.py
"""
Wortfolgen-Mustererkennung für Satzvorhersage.

Features:
- N-Gram Probability aus CONNECTION Edges
- Bigram/Trigram Prediction
- Bootstrap (nur wenn genug Daten vorhanden)
"""

from typing import Any, Dict, List

from component_15_logging_config import get_logger
from component_25_adaptive_thresholds import AdaptiveThresholdManager
from kai_config import get_config

logger = get_logger(__name__)
config = get_config()


class SequencePredictor:
    """
    Sagt nächstes Wort in Sequenz vorher basierend auf N-Grammen.
    """

    def __init__(self, netzwerk):
        """
        Args:
            netzwerk: KonzeptNetzwerk Instanz
        """
        self.netzwerk = netzwerk
        self.adaptive_manager = AdaptiveThresholdManager(netzwerk)

        # Verwende adaptive Threshold (fallback auf Config/Default)
        self.min_sequence_count = self.adaptive_manager.get_sequence_threshold()

    def predict_next_word(
        self, context: List[str], max_predictions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Sagt nächstes Wort vorher basierend auf Kontext.

        Args:
            context: Liste der vorherigen Wörter (z.B. ["im", "großen"])
            max_predictions: Max. Anzahl Vorhersagen

        Returns:
            Liste von Predictions mit Confidence
        """
        if not context or len(context) == 0:
            return []

        # Nutze letztes Wort für Bigram-Prediction
        last_word = context[-1].lower()

        # Hole alle Connections von diesem Wort
        connections = self.netzwerk.get_word_connections(last_word, direction="before")

        if not connections:
            logger.debug(f"Keine Connections für '{last_word}' gefunden")
            return []

        # Filtere nach min_sequence_count
        valid_connections = [
            c for c in connections if c["count"] >= self.min_sequence_count
        ]

        if not valid_connections:
            logger.debug(
                f"Keine Connections mit count >= {self.min_sequence_count}",
                extra={"word": last_word},
            )
            return []

        # Berechne Probabilities (bereits in confidence-Field)
        predictions = []
        for conn in valid_connections:
            # Confidence ist bereits normalisiert (count / total_count)
            predictions.append(
                {
                    "word": conn["connected_word"],
                    "confidence": conn["confidence"],
                    "count": conn["count"],
                    "distance": conn["distance"],
                    "reason": f"bigram_{last_word}",
                }
            )

        # Sortiere nach Confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        # Limitiere Ergebnisse
        return predictions[:max_predictions]

    def predict_completion(
        self, partial_sentence: str, max_predictions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Vervollständigt einen angefangenen Satz.

        Args:
            partial_sentence: Angefangener Satz (z.B. "Das Haus ist")
            max_predictions: Max. Anzahl Vorhersagen

        Returns:
            Liste von Vervollständigungen
        """
        words = partial_sentence.strip().split()

        if len(words) == 0:
            return []

        # Nutze letzte 2-3 Wörter als Kontext
        context = words[-3:] if len(words) >= 3 else words

        return self.predict_next_word(context, max_predictions)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def predict_next(netzwerk, context_words: List[str]) -> List[Dict[str, Any]]:
    """
    Standalone-Funktion für Wort-Vorhersage.

    Args:
        netzwerk: KonzeptNetzwerk Instanz
        context_words: Kontext-Wörter

    Returns:
        Predictions
    """
    predictor = SequencePredictor(netzwerk)
    return predictor.predict_next_word(context_words)


if __name__ == "__main__":
    # Test-Code
    print("=== Sequence Prediction Test ===\n")
    print("Modul erfolgreich geladen.")
    print("Für Tests: pytest tests/test_pattern_recognition_sequence.py")
