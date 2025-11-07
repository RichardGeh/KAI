# component_22_pattern_recognition_implicit.py
"""
Implizite Implikations-Erkennung (Lightweight Version).

Erkennt implizite Fakten aus expliziten Aussagen:
- "Haus ist groß" -> impliziert "Haus hat Größe"
- Nutzt vorhandene IS_A, HAS_PROPERTY Relations
"""

from typing import Any, Dict, List

from component_15_logging_config import get_logger
from kai_config import get_config

logger = get_logger(__name__)
config = get_config()

# Einfache Implikations-Regeln
PROPERTY_IMPLICATIONS = {
    "groß": "größe",
    "klein": "größe",
    "rot": "farbe",
    "blau": "farbe",
    "grün": "farbe",
    "schnell": "geschwindigkeit",
    "langsam": "geschwindigkeit",
}


class ImplicationDetector:
    """Erkennt implizite Fakten"""

    def __init__(self, netzwerk):
        self.netzwerk = netzwerk
        self.min_confidence_threshold = config.get(
            "implication_auto_add_threshold", 0.75
        )

    def detect_property_implications(
        self, subject: str, property_value: str
    ) -> List[Dict[str, Any]]:
        """
        Erkennt Property-Implikationen.

        Args:
            subject: z.B. "Haus"
            property_value: z.B. "groß"

        Returns:
            Liste von implizierten Fakten
        """
        implications = []

        # Prüfe ob Property in bekannten Implikationen
        property_lower = property_value.lower()
        if property_lower in PROPERTY_IMPLICATIONS:
            implied_property = PROPERTY_IMPLICATIONS[property_lower]

            implications.append(
                {
                    "subject": subject,
                    "relation": "HAS_PROPERTY",
                    "object": implied_property,
                    "confidence": 0.85,  # Hohe Confidence für bekannte Muster
                    "source": "property_implication",
                    "reasoning": f"'{subject} ist {property_value}' impliziert '{subject} hat {implied_property}'",
                }
            )

        return implications


def detect_implications(netzwerk, subject: str, relation: str, obj: str) -> List[Dict]:
    """Standalone-Funktion"""
    detector = ImplicationDetector(netzwerk)

    if relation == "HAS_PROPERTY":
        return detector.detect_property_implications(subject, obj)

    return []


if __name__ == "__main__":
    print("=== Implication Detection (Lightweight) ===")
    print("Modul geladen.")
