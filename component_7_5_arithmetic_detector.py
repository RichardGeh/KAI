# component_7_5_arithmetic_detector.py
"""
Arithmetische Frage-Erkennung.
Erkennt Fragen mit arithmetischen Operationen und mathematischen Konzepten.
"""
import uuid

from spacy.tokens import Doc

from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
)
from component_15_logging_config import get_logger

logger = get_logger(__name__)


class ArithmeticQuestionDetector:
    """
    Erkennt arithmetische Fragen automatisch.

    Unterstützte Muster:
    - "Was ist 3 + 5?" -> 0.95
    - "Wie viel ist 7 mal 8?" -> 0.93
    - "Wieviel sind 10 durch 2?" -> 0.92
    - "Berechne 15 minus 6" -> 0.90
    - "Was ist drei plus fünf?" -> 0.95 (Zahlwörter)
    """

    def detect(self, text: str, doc: Doc) -> list[MeaningPoint] | None:
        """
        Erkennt arithmetische Fragen.

        Args:
            text: Der zu analysierende Text
            doc: spaCy Doc für linguistische Analyse

        Returns:
            Liste mit einem ARITHMETIC_QUESTION MeaningPoint oder None
        """
        try:
            text_lower = text.lower().strip()

            # Arithmetische Trigger-Wörter (ERWEITERT!)
            # NEU: Auch Konzepte wie Summe, Differenz, Durchschnitt, Prozent
            arithmetic_operators = [
                "plus",
                "minus",
                "mal",
                "geteilt",
                "durch",
                "+",
                "-",
                "*",
                "/",
                "multipliziert",
                "addiert",
                "subtrahiert",
                "dividiert",
                # NEU: Erweiterte Operatoren
                "summe",
                "differenz",
                "produkt",
                "quotient",
                "durchschnitt",
                "mittelwert",
                "prozent",
                "%",
                "hoch",
                "quadrat",
                "wurzel",
            ]

            # Frage-Trigger (ERWEITERT!)
            question_triggers = [
                "was ist",
                "wie viel",
                "wieviel",
                "wie viele",
                "berechne",
                "rechne",
                "errechne",
                "berechnen",
                # NEU: Erweiterte Trigger
                "ermittle",
                "bestimme",
                "wie hoch",
                "wie gross",
            ]

            # Prüfe auf arithmetische Operatoren
            has_arithmetic = any(op in text_lower for op in arithmetic_operators)

            # Prüfe auf Frage-Trigger
            has_question = any(trigger in text_lower for trigger in question_triggers)

            # Wenn beides vorhanden: Hohe Confidence
            if has_arithmetic and has_question:
                confidence = 0.95
            elif has_arithmetic and text.endswith("?"):
                confidence = 0.90
            elif has_arithmetic:
                confidence = 0.80
            else:
                # Keine arithmetische Frage erkannt
                return None

            logger.debug(
                f"Arithmetische Frage erkannt: '{text}' (confidence={confidence})"
            )

            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.ARITHMETIC_QUESTION,
                    cue="auto_detect_arithmetic",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "query_text": text,
                        "auto_detected": True,
                    },
                )
            ]

        except Exception as e:
            logger.error(f"Fehler bei Arithmetik-Erkennung: {e}", exc_info=True)
            return None

    def _create_meaning_point(self, **kwargs) -> MeaningPoint:
        """
        Factory-Methode zum Erstellen von MeaningPoint-Objekten mit sinnvollen Defaults.

        Args:
            **kwargs: Beliebige MeaningPoint-Attribute (überschreiben Defaults)

        Returns:
            Ein vollständig initialisiertes MeaningPoint-Objekt
        """
        try:
            # Sinnvolle Defaults
            defaults = {
                "id": f"mp-{uuid.uuid4().hex[:6]}",
                "modality": Modality.DECLARATIVE,
                "polarity": Polarity.POSITIVE,
                "confidence": 0.7,  # Konservativ, wird oft überschrieben
                "arguments": {},
                "span_offsets": [],
                "source_rules": [],
            }

            # Kategorie-spezifische Modality
            category = kwargs.get("category")
            if category == MeaningPointCategory.QUESTION:
                defaults["modality"] = Modality.INTERROGATIVE
            elif category == MeaningPointCategory.COMMAND:
                defaults["modality"] = Modality.IMPERATIVE
            elif category == MeaningPointCategory.ARITHMETIC_QUESTION:
                defaults["modality"] = Modality.INTERROGATIVE

            # Merge mit übergebenen Parametern
            defaults.update(kwargs)

            return MeaningPoint(**defaults)

        except Exception as e:
            logger.error(f"Fehler beim Erstellen des MeaningPoints: {e}", exc_info=True)
            # Rethrow, da ein MeaningPoint essentiell ist
            raise
