# component_7_meaning_extractor.py (Refactored Orchestrator)
"""
Haupt-Orchestrator für Meaning Extraction.

Koordiniert die zweiphasige Extraktion:
1. Phase: Explizite Befehle (regex-basiert) mit confidence=1.0
2. Phase: Vektor-basierte Prototypen-Erkennung mit distance-basierter confidence

Delegiert an spezialisierte Module:
- component_7_1_command_parser: Explizite Befehle
- component_7_2_definition_detector: Deklarative Aussagen
- component_7_3_question_heuristics: Frage-Heuristiken
- component_7_4_linguistic_features: Linguistische Features
- component_7_5_arithmetic_detector: Arithmetische Fragen
"""
import logging
import uuid
from typing import Any

from spacy.tokens import Doc

from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
)
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_1_command_parser import CommandParser
from component_7_2_definition_detector import DefinitionDetector
from component_7_3_question_heuristics import QuestionHeuristicsExtractor
from component_7_5_arithmetic_detector import ArithmeticQuestionDetector
from component_11_embedding_service import EmbeddingService, ModelNotLoadedError
from component_15_logging_config import get_logger
from component_utils_text_normalization import TextNormalizer

logger = get_logger(__name__)

# Threshold for determining if a vector match is novel or known
# Matches with distance > this value are considered too dissimilar
NOVELTY_THRESHOLD = 5.0


class MeaningPointExtractor:
    """
    Extrahiert MeaningPoints aus Text durch ein zweiphasiges Verfahren:
    1. Phase: Explizite Befehle (regex-basiert) mit confidence=1.0
    2. Phase: Vektor-basierte Prototypen-Erkennung mit distance-basierter confidence
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        preprocessor: LinguisticPreprocessor,
        prototyping_engine=None,
    ):
        """
        Args:
            embedding_service: Service für Vektor-Embeddings
            preprocessor: Linguistischer Vorverarbeitungs-Service
            prototyping_engine: Optional - für Prototypen-Matching (Phase 2)
        """
        self.embedding_service = embedding_service
        self.preprocessor = preprocessor
        self.prototyping_engine = prototyping_engine

        # Initialisiere zentralen TextNormalizer mit spaCy-Integration
        self.text_normalizer = TextNormalizer(preprocessor=preprocessor)

        # Initialisiere spezialisierte Sub-Module
        self.command_parser = CommandParser()
        self.definition_detector = DefinitionDetector(preprocessor)
        self.question_heuristics = QuestionHeuristicsExtractor(preprocessor)
        self.arithmetic_detector = ArithmeticQuestionDetector()

        logger.info(
            "MeaningPointExtractor initialisiert mit zweiphasiger Erkennung und 5 spezialisierten Modulen"
        )

    def extract(self, doc: Doc) -> list[MeaningPoint]:
        """
        Haupt-Orchestrierungs-Methode für die Meaning-Extraktion.

        Flow:
        1. Prüfe auf explizite Befehle (regex) -> confidence=1.0
        2. Falls kein Befehl: Arithmetische Fragen -> confidence=0.80-0.95
        3. Falls keine Arithmetik: Auto-Erkennung von Definitionen -> confidence=0.78-0.93
        4. Falls keine Definition: Vektor-basierte Erkennung -> distance-basierte confidence
        5. Falls kein Vektor-Match: Fallback auf Heuristiken
        6. Falls nichts gefunden: UNKNOWN MeaningPoint mit confidence=0.0

        Args:
            doc: spaCy Doc-Objekt mit vorverarbeitetem Text

        Returns:
            Liste mit genau einem MeaningPoint (niemals leer)
        """
        try:
            # Input-Validierung
            if not doc or not doc.text.strip():
                logger.debug("Leere Eingabe erhalten, keine Extraktion moeglich")
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.UNKNOWN,
                        cue="empty_input",
                        text_span="",
                        confidence=0.0,
                        arguments={},
                    )
                ]

            text = doc.text.strip()

            # Conditional logging für DEBUG
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Starte Meaning-Extraktion",
                    extra={"text_preview": text[:50], "text_length": len(text)},
                )

            # PHASE 1: Explizite Befehle (höchste Priorität, confidence=1.0)
            command_mps = self.command_parser.parse(text)
            if command_mps:
                logger.info(
                    "Expliziter Befehl erkannt",
                    extra={
                        "cue": command_mps[0].cue,
                        "category": command_mps[0].category.name,
                    },
                )
                return command_mps

            # PHASE 1.5: Auto-Erkennung von arithmetischen Fragen
            # Muss VOR Definitions-Erkennung erfolgen (wegen "sind" in "10 durch 2 sind...")
            arithmetic_mps = self.arithmetic_detector.detect(text, doc)
            if arithmetic_mps:
                logger.info(
                    "Arithmetische Frage erkannt",
                    extra={
                        "cue": arithmetic_mps[0].cue,
                        "category": arithmetic_mps[0].category.name,
                        "confidence": arithmetic_mps[0].confidence,
                    },
                )
                return arithmetic_mps

            # PHASE 1.6: Auto-Erkennung von Definitionen (deklarative Aussagen)
            # Dies muss vor dem Vektor-Matching erfolgen, da es spezifischer ist
            definition_mps = self.definition_detector.detect(text, doc)
            if definition_mps:
                logger.info(
                    "Definition erkannt",
                    extra={
                        "cue": definition_mps[0].cue,
                        "category": definition_mps[0].category.name,
                    },
                )
                return definition_mps

            # PHASE 2: Vektor-basierte Prototypen-Erkennung
            if self.prototyping_engine and self.embedding_service.is_available():
                vector_mps = self._extract_with_vector_matching(text, doc)
                if vector_mps:
                    logger.info(
                        "Vektor-Match gefunden",
                        extra={
                            "category": vector_mps[0].category.name,
                            "confidence": vector_mps[0].confidence,
                        },
                    )
                    return vector_mps

            # PHASE 3: Fallback auf Heuristiken (für Rückwärtskompatibilität)
            heuristic_mps = self.question_heuristics.extract(doc)
            if heuristic_mps:
                logger.info(
                    "Heuristik-Match gefunden",
                    extra={
                        "cue": heuristic_mps[0].cue,
                        "category": heuristic_mps[0].category.name,
                    },
                )
                return heuristic_mps

            # PHASE 4: Nichts gefunden -> Gib UNKNOWN mit confidence=0.0 zurück
            logger.warning(
                "Keine Bedeutung extrahiert",
                extra={"text_preview": text[:50], "text_length": len(text)},
            )
            unknown_mp = self._create_meaning_point(
                category=MeaningPointCategory.UNKNOWN,
                cue="no_match_found",
                text_span=text,
                confidence=0.0,  # Maximale Unsicherheit
                arguments={"original_text": text},
            )
            return [unknown_mp]

        except Exception as e:
            logger.error(
                "Fehler bei Meaning-Extraktion",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            # Auch bei Fehler: UNKNOWN zurückgeben statt leere Liste
            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.UNKNOWN,
                    cue="extraction_error",
                    text_span=doc.text if doc else "",
                    confidence=0.0,
                    arguments={"error": str(e)},
                )
            ]

    def _extract_with_vector_matching(self, text: str, doc: Doc) -> list[MeaningPoint]:
        """
        Vektor-basierte Meaning-Extraktion via Prototypen-Matching.

        Args:
            text: Der Eingabetext
            doc: spaCy Doc für zusätzliche linguistische Features

        Returns:
            Liste mit einem MeaningPoint oder leere Liste
        """
        try:
            # Erzeuge Embedding-Vektor für den Eingabesatz
            vector = self.embedding_service.get_embedding(text)

            # Conditional logging für DEBUG (Performance-kritisch)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Embedding fuer Meaning-Extraktion erzeugt",
                    extra={"dimensions": len(vector), "text_preview": text[:30]},
                )

            # Finde besten Match unter allen Prototypen
            match_result = self.prototyping_engine.find_best_match(vector)

            if not match_result:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Kein Prototyp-Match gefunden")
                return []

            prototype, distance = match_result

            # Conditional logging für DEBUG
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Prototyp-Match gefunden",
                    extra={"prototype_id": prototype["id"][:8], "distance": distance},
                )

            # Berechne Confidence aus Distanz (näher = höhere Confidence)
            # confidence = max(0, 1 - (distance / NOVELTY_THRESHOLD))
            # Sicherstellen, dass Confidence zwischen 0 und 1 bleibt
            confidence = max(0.0, min(1.0, 1.0 - (distance / NOVELTY_THRESHOLD)))

            # Nur Matches mit Confidence > 0.3 akzeptieren (empirischer Schwellwert)
            if confidence < 0.3:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Confidence zu niedrig, verwerfe Match",
                        extra={"confidence": confidence, "threshold": 0.3},
                    )
                return []

            # Hole Kategorie aus Prototyp
            category_str = prototype.get("category", "UNKNOWN")
            try:
                category = MeaningPointCategory[category_str.upper()]
            except KeyError:
                logger.warning(
                    "Unbekannte Prototyp-Kategorie",
                    extra={
                        "category": category_str,
                        "prototype_id": prototype["id"][:8],
                    },
                )
                category = MeaningPointCategory.UNKNOWN

            # Extrahiere Thema/Argumente aus dem Satz
            arguments = self._extract_arguments_from_text(text, doc, category)

            # Erstelle MeaningPoint mit berechneter Confidence
            mp = self._create_meaning_point(
                category=category,
                cue=f"vector_match_{prototype['id'][:8]}",
                text_span=text,
                confidence=confidence,
                arguments=arguments,
                source_rules=[f"prototype:{prototype['id']}"],
            )

            logger.info(
                "Vektor-Match erstellt",
                extra={
                    "category": category.name,
                    "confidence": confidence,
                    "distance": distance,
                    "prototype_id": prototype["id"][:8],
                },
            )
            return [mp]

        except ModelNotLoadedError as e:
            logger.warning(f"Embedding-Service nicht verfuegbar: {e}")
            return []
        except Exception as e:
            logger.error(f"Fehler bei Vektor-Matching: {e}", exc_info=True)
            return []

    def _extract_arguments_from_text(
        self, text: str, doc: Doc, category: MeaningPointCategory
    ) -> dict[str, Any]:
        """
        Extrahiert Argumente aus dem Text basierend auf der erkannten Kategorie.

        Args:
            text: Der Eingabetext
            doc: spaCy Doc
            category: Die erkannte Kategorie

        Returns:
            Dictionary mit extrahierten Argumenten
        """
        arguments = {}

        try:
            # Kategorie-spezifische Extraktion
            if category == MeaningPointCategory.QUESTION:
                # Vollständige W-Wort-Liste (konsistent mit Heuristiken)
                wh_words = [
                    "was",
                    "wer",
                    "wie",
                    "wo",
                    "wann",
                    "warum",
                    "welche",
                    "welcher",
                    "welches",
                    "wozu",
                    "wieso",
                    "weshalb",
                ]
                words = text.lower().split()

                for wh in wh_words:
                    if wh in words:
                        arguments["question_word"] = wh
                        # Thema ist oft nach dem Fragewort und "ist"
                        if "ist" in words:
                            ist_idx = words.index("ist")
                            if ist_idx + 1 < len(words):
                                topic_words = words[ist_idx + 1 :]
                                topic = " ".join(topic_words).rstrip("?")
                                arguments["topic"] = self.text_normalizer.clean_entity(
                                    topic
                                )
                        break

            elif category == MeaningPointCategory.COMMAND:
                # Bei Befehlen das Hauptverb extrahieren
                for token in doc:
                    if token.pos_ == "VERB":
                        arguments["action"] = token.lemma_
                        break

            elif category == MeaningPointCategory.DEFINITION:
                # Bei Definitionen Subjekt und Prädikat
                for token in doc:
                    if token.dep_ == "sb":  # Subjekt
                        arguments["subject"] = token.text.lower()
                    elif token.dep_ == "pd":  # Prädikativ
                        arguments["predicate"] = token.text.lower()

            # Fallback: Speichere den gesamten Text
            if not arguments:
                arguments["text"] = text

        except Exception as e:
            logger.warning(f"Fehler bei Argument-Extraktion: {e}")
            arguments["text"] = text

        return arguments

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
