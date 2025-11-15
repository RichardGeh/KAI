# component_7_1_command_parser.py
"""
Explizite Befehls-Erkennung via Regex-Pattern.
Erkennt eindeutige Befehle mit confidence=1.0.
"""
import re
import uuid

from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
)
from component_15_logging_config import get_logger

logger = get_logger(__name__)


class CommandParser:
    """
    Erkennt explizite, unmissverständliche Befehle via Regex.
    Diese haben immer confidence=1.0 (definiere:) oder 0.95 (Datei-Befehle),
    da sie eindeutig sind.
    """

    def parse(self, text: str) -> list[MeaningPoint] | None:
        """
        Erkennt explizite Befehle via Regex.

        Args:
            text: Der zu parsende Text

        Returns:
            Liste mit einem MeaningPoint (bei Match) oder None
        """
        try:
            # BEFEHL: definiere:
            define_match = re.match(
                r"^\s*definiere:\s*(\S+)\s*/\s*(.*?)\s*=\s*(.*)$", text, re.IGNORECASE
            )
            if define_match:
                topic, key_path_str, value = define_match.groups()
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue="definiere:",
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=1.0,  # Expliziter Befehl = maximale Confidence
                        arguments={
                            "command": "definiere",
                            "topic": topic.lower(),
                            "key_path": [
                                p.strip().lower() for p in key_path_str.split("/")
                            ],
                            "value": value.strip(),
                        },
                    )
                ]

            # BEFEHL: lerne muster:
            learn_pattern_match = re.match(
                r'^\s*lerne muster:\s*"(.*)"(?:\s*bedeutet\s*(\S+))?\s*$',
                text,
                re.IGNORECASE,
            )
            if learn_pattern_match:
                example_sentence, relation_type = learn_pattern_match.groups()
                if relation_type is None:
                    relation_type = "IS_A"

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue="lerne muster:",
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=1.0,  # Expliziter Befehl = maximale Confidence
                        arguments={
                            "command": "learn_pattern",
                            "example_sentence": example_sentence.strip(),
                            "relation_type": relation_type.strip().upper(),
                        },
                    )
                ]

            # BEFEHL: ingestiere text:
            ingest_match = re.match(
                r'^\s*ingestiere text:\s*"(.*)"\s*$', text, re.IGNORECASE | re.DOTALL
            )
            if ingest_match:
                text_to_ingest = ingest_match.group(1)
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue="ingestiere text:",
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=1.0,  # Expliziter Befehl = maximale Confidence
                        arguments={
                            "command": "ingest_text",
                            "text_to_ingest": text_to_ingest.strip(),
                        },
                    )
                ]

            # BEFEHL: lerne: (einfache Lernform für Wörter/Sätze)
            learn_simple_match = re.match(r"^\s*lerne:\s*(.+)\s*$", text, re.IGNORECASE)
            if learn_simple_match:
                text_to_learn = learn_simple_match.group(1).strip()
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue="lerne:",
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=1.0,  # Expliziter Befehl = maximale Confidence
                        arguments={
                            "command": "learn_simple",
                            "text_to_learn": text_to_learn,
                        },
                    )
                ]

            # BEFEHL: Datei-Commands (lese datei:, ingestiere dokument:, verarbeite pdf:, lade datei:)
            file_command_match = re.match(
                r"^\s*(?:lese datei|ingestiere dokument|verarbeite pdf|lade datei):\s*(.+)\s*$",
                text,
                re.IGNORECASE,
            )
            if file_command_match:
                file_path = file_command_match.group(1).strip()

                # Erkenne den spezifischen Command-Typ aus dem Text
                command_type = None
                command_cue = None
                if re.match(r"^\s*lese datei:", text, re.IGNORECASE):
                    command_type = "read_file"
                    command_cue = "lese datei:"
                elif re.match(r"^\s*ingestiere dokument:", text, re.IGNORECASE):
                    command_type = "ingest_document"
                    command_cue = "ingestiere dokument:"
                elif re.match(r"^\s*verarbeite pdf:", text, re.IGNORECASE):
                    command_type = "process_pdf"
                    command_cue = "verarbeite pdf:"
                elif re.match(r"^\s*lade datei:", text, re.IGNORECASE):
                    command_type = "load_file"
                    command_cue = "lade datei:"

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue=command_cue,
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=0.95,  # Sehr hohe Confidence für klare Datei-Commands
                        arguments={
                            "command": command_type,
                            "file_path": file_path,
                        },
                    )
                ]

            return None

        except Exception as e:
            logger.error(f"Fehler beim Parsen expliziter Befehle: {e}", exc_info=True)
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

            # Merge mit übergebenen Parametern
            defaults.update(kwargs)

            return MeaningPoint(**defaults)

        except Exception as e:
            logger.error(f"Fehler beim Erstellen des MeaningPoints: {e}", exc_info=True)
            # Rethrow, da ein MeaningPoint essentiell ist
            raise
