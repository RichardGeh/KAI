# component_7_meaning_point_factory.py
"""
Shared factory function for creating MeaningPoint objects.

WICHTIG: KEINE Unicode-Zeichen verwenden, die Windows cp1252 Encoding-Probleme verursachen.
Verboten: OK FEHLER -> x / != <= >= etc.
Erlaubt: [OK] [FEHLER] -> * / != <= >= AND OR NOT

This module eliminates code duplication across all component_7 modules by providing
a single, centralized factory function for MeaningPoint creation.
"""

import uuid

from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
)
from component_15_logging_config import get_logger

logger = get_logger(__name__)


def create_meaning_point(**kwargs) -> MeaningPoint:
    """
    Factory function for creating MeaningPoint objects with sensible defaults.

    Args:
        **kwargs: Any MeaningPoint attributes (override defaults)

    Returns:
        Fully initialized MeaningPoint object

    Raises:
        Exception: If MeaningPoint creation fails (essential object)

    Example:
        mp = create_meaning_point(
            category=MeaningPointCategory.QUESTION,
            cue="was_ist",
            text_span="Was ist ein Apfel?",
            confidence=0.90,
            arguments={"question_word": "was", "topic": "apfel"}
        )
    """
    try:
        # Sensible defaults
        defaults = {
            "id": f"mp-{uuid.uuid4().hex[:6]}",
            "modality": Modality.DECLARATIVE,
            "polarity": Polarity.POSITIVE,
            "confidence": 0.7,  # Conservative default, often overridden
            "arguments": {},
            "span_offsets": [],
            "source_rules": [],
        }

        # Category-specific modality
        category = kwargs.get("category")
        if category == MeaningPointCategory.QUESTION:
            defaults["modality"] = Modality.INTERROGATIVE
        elif category == MeaningPointCategory.COMMAND:
            defaults["modality"] = Modality.IMPERATIVE
        elif category == MeaningPointCategory.ARITHMETIC_QUESTION:
            # Arithmetic questions are also interrogative
            defaults["modality"] = Modality.INTERROGATIVE

        # Merge with provided parameters (kwargs override defaults)
        defaults.update(kwargs)

        return MeaningPoint(**defaults)

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des MeaningPoints: {e}", exc_info=True)
        # Rethrow, da ein MeaningPoint essentiell ist
        raise
