# component_7_4_linguistic_features.py
"""
Linguistische Feature-Extraktion für Meaning Extraction.
Extrahiert temporale Marker, Quantoren, Unsicherheits-Marker und Negation.

WICHTIG: KEINE Unicode-Zeichen verwenden, die Windows cp1252 Encoding-Probleme verursachen.
Verboten: OK FEHLER -> x / != <= >= etc.
Erlaubt: [OK] [FEHLER] -> * / != <= >= AND OR NOT
"""
import re

from component_5_linguistik_strukturen import Polarity
from component_15_logging_config import get_logger

logger = get_logger(__name__)


class LinguisticFeatureExtractor:
    """
    Extrahiert linguistische Features aus Text:
    - Temporale Marker (Vergangenheit/Gegenwart/Zukunft)
    - Quantoren (alle, manche, einige, keine)
    - Unsicherheits-Marker (Hedges: vielleicht, wahrscheinlich)
    - Negation (nicht, kein) und Polarität
    """

    def extract_temporal_markers(self, text: str) -> tuple[list[str], str | None]:
        """
        Extrahiert temporale Marker aus Text.

        Args:
            text: Text der auf temporale Marker geprüft werden soll

        Returns:
            Tuple von (temporal_markers, temporal_context)
            - temporal_markers: Liste der gefundenen Zeitmarker
            - temporal_context: "past", "present", "future" oder None
        """
        temporal_markers_map = {
            # Vergangenheit
            "gestern": "past",
            "vorgestern": "past",
            "letzte woche": "past",
            "letzten monat": "past",
            "letztes jahr": "past",
            "frueher": "past",
            "damals": "past",
            "einst": "past",
            "vorher": "past",
            "zuvor": "past",
            # Gegenwart
            "heute": "present",
            "jetzt": "present",
            "gerade": "present",
            "aktuell": "present",
            "momentan": "present",
            "derzeit": "present",
            # Zukunft
            "morgen": "future",
            "uebermorgen": "future",
            "naechste woche": "future",
            "naechsten monat": "future",
            "naechstes jahr": "future",
            "spaeter": "future",
            "bald": "future",
            "demnaechst": "future",
            "kuenftig": "future",
            "zukuenftig": "future",
        }

        text_lower = text.lower()
        found_markers = []
        temporal_context = None

        for marker, context in temporal_markers_map.items():
            if marker in text_lower:
                found_markers.append(marker)
                # Setze Context auf den ersten gefundenen Marker
                if temporal_context is None:
                    temporal_context = context

        if found_markers:
            logger.debug(
                f"Temporal-Marker erkannt: {found_markers} -> Context={temporal_context}"
            )

        return found_markers, temporal_context

    def extract_quantifier(self, text: str) -> tuple[str | None, str | None]:
        """
        Extrahiert Quantoren aus Text.

        Args:
            text: Text der auf Quantoren geprüft werden soll

        Returns:
            Tuple von (quantifier, quantifier_type)
            - quantifier: Der gefundene Quantor oder None
            - quantifier_type: "universal", "existential", "majority", "minority", "none" oder None
        """
        quantifiers_map = {
            "alle": "universal",
            "jeder": "universal",
            "jede": "universal",
            "jedes": "universal",
            "saemtliche": "universal",
            "manche": "existential",
            "einige": "existential",
            "mehrere": "existential",
            "viele": "majority",
            "die meisten": "majority",
            "wenige": "minority",
            "kaum": "minority",
            "keine": "none",
            "kein": "none",
        }

        text_lower = text.lower()

        for quantifier, q_type in quantifiers_map.items():
            if (
                re.search(rf"\b{quantifier}\b", text_lower)
                or quantifier in text_lower.split()
            ):
                logger.debug(f"Quantor erkannt: '{quantifier}' -> Type={q_type}")
                return quantifier, q_type

        return None, None

    def detect_uncertainty_markers(self, text: str) -> tuple[float, list[str], str]:
        """
        Erkennt Unsicherheits-Marker (Hedges) und passt Confidence an.

        Args:
            text: Text der auf Unsicherheits-Marker geprüft werden soll

        Returns:
            Tuple von (confidence_multiplier, hedge_words, uncertainty_level)
            - confidence_multiplier: Faktor zur Reduktion der Confidence (0.0-1.0)
            - hedge_words: Liste der gefundenen Hedge-Wörter
            - uncertainty_level: "high", "medium", "low", "none"
        """
        hedge_words_map = {
            "vielleicht": 0.5,  # Sehr unsicher
            "moeglicherweise": 0.5,
            "eventuell": 0.55,
            "vermutlich": 0.65,  # Unsicher
            "wahrscheinlich": 0.7,
            "anscheinend": 0.7,
            "scheinbar": 0.65,
            "meistens": 0.8,  # Etwas unsicher
            "normalerweise": 0.8,
            "ueblicherweise": 0.8,
            "oft": 0.75,
            "haeufig": 0.75,
            "manchmal": 0.6,
            "selten": 0.6,
        }

        text_lower = text.lower()
        found_hedges = []
        confidence_multiplier = 1.0

        for hedge, multiplier in hedge_words_map.items():
            if re.search(rf"\b{hedge}\b", text_lower) or hedge in text_lower.split():
                found_hedges.append(hedge)
                # Nimm den niedrigsten Multiplier wenn mehrere Hedges vorhanden
                confidence_multiplier = min(confidence_multiplier, multiplier)

        # Bestimme Uncertainty Level
        if confidence_multiplier <= 0.6:
            uncertainty_level = "high"
        elif confidence_multiplier <= 0.75:
            uncertainty_level = "medium"
        elif confidence_multiplier < 1.0:
            uncertainty_level = "low"
        else:
            uncertainty_level = "none"

        if found_hedges:
            logger.debug(
                f"Uncertainty-Marker erkannt: {found_hedges} -> Multiplier={confidence_multiplier:.2f}, Level={uncertainty_level}"
            )

        return confidence_multiplier, found_hedges, uncertainty_level

    def extract_negation(self, text: str) -> tuple[str, Polarity]:
        """
        Extrahiert Negation aus Text und bestimmt Polarität.

        Args:
            text: Text der auf Negation geprüft werden soll

        Returns:
            Tuple von (bereinigter_text, polarity)
            - bereinigter_text: Text ohne Negations-Marker
            - polarity: POSITIVE oder NEGATIVE
        """
        negation_markers = [
            "nicht",
            "kein",
            "keine",
            "keinen",
            "keinem",
            "keiner",
            "niemals",
            "nie",
        ]

        text_lower = text.lower()
        has_negation = any(
            re.search(rf"\b{marker}\b", text_lower) or marker in text_lower.split()
            for marker in negation_markers
        )

        # Entferne Negation aus Text für saubere Entity-Extraktion
        cleaned_text = text
        if has_negation:
            for marker in negation_markers:
                # Entferne Negations-Marker mit Whitespace-Handling
                cleaned_text = re.sub(
                    rf"\b{marker}\b\s*", "", cleaned_text, flags=re.IGNORECASE
                )

        polarity = Polarity.NEGATIVE if has_negation else Polarity.POSITIVE

        if has_negation:
            logger.debug(
                f"Negation erkannt: '{text}' -> Polarity={polarity.name}, bereinigt='{cleaned_text.strip()}'"
            )

        return cleaned_text.strip(), polarity
