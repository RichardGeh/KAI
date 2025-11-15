# component_7_2_definition_detector.py
"""
Deklarative Aussagen-Erkennung (Definitionen).
Erkennt automatisch deklarative Aussagen und wandelt sie in DEFINITION MeaningPoints um.

Unterstützte Relationstypen:
- IS_A: "X ist ein/eine Y"
- HAS_PROPERTY: "X ist Y" (Adjektiv)
- CAPABLE_OF: "X kann Y"
- PART_OF: "X hat Y"
- LOCATED_IN: "X liegt in Y"
- CONDITIONAL: "Wenn X, dann Y"
- COMPARATIVE: "X ist größer als Y"
"""
import re
import uuid

from spacy.tokens import Doc

from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
)
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_4_linguistic_features import LinguisticFeatureExtractor
from component_15_logging_config import get_logger
from component_utils_text_normalization import TextNormalizer

logger = get_logger(__name__)


class DefinitionDetector:
    """
    Erkennt deklarative Aussagen automatisch und wandelt sie in DEFINITION MeaningPoints um.
    Entfernt die Notwendigkeit für explizite "Ingestiere Text:"-Befehle.
    """

    def __init__(self, preprocessor: LinguisticPreprocessor):
        """
        Args:
            preprocessor: Linguistischer Vorverarbeitungs-Service
        """
        self.preprocessor = preprocessor
        self.text_normalizer = TextNormalizer(preprocessor=preprocessor)
        self.linguistic_features = LinguisticFeatureExtractor()

    def detect(self, text: str, doc: Doc) -> list[MeaningPoint] | None:
        """
        Erkennt deklarative Aussagen automatisch.

        Args:
            text: Der zu analysierende Text
            doc: spaCy Doc für linguistische Analyse

        Returns:
            Liste mit MeaningPoint(s) oder None
        """
        try:
            text_lower = text.lower().strip()

            # Filter 1: Ignoriere Fragen (enthalten Fragewörter ODER enden mit ?)
            # FIX: Prüfe ALLE Wörter, nicht nur das erste (für "Hallo Kai, was ist...")
            question_words = [
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

            # Prüfe ob Text mit Fragezeichen endet (klares Indiz für Frage)
            if text.rstrip().endswith("?"):
                logger.debug("Frage erkannt (Fragezeichen) -> Keine Definition")
                return None

            # Prüfe ob IRGENDEIN Wort ein Fragewort ist (nicht nur das erste)
            words = text_lower.split()
            for word in words[:5]:  # Prüfe erste 5 Wörter (für "Hallo Kai, was...")
                if word in question_words:
                    logger.debug(
                        f"Frage erkannt (Fragewort '{word}') -> Keine Definition"
                    )
                    return None

            # Filter 2: BEHANDLE Konditionale (NICHT ignorieren!)
            # "Wenn X, dann Y" / "Falls X, (dann) Y" werden jetzt verarbeitet
            if mp := self._detect_conditional(text, text_lower):
                return mp

            # Filter 3: BEHANDLE Komparative (NICHT ignorieren!)
            # "größer als", "schneller als", "besser als" werden jetzt verarbeitet
            if mp := self._detect_comparative(text, text_lower):
                return mp

            # Pattern 1: IS_A - "X ist ein/eine Y" (mit Adjektiv-Extraktion!)
            if mp := self._detect_is_a(text, text_lower, doc):
                return mp

            # Pattern 2: IS_A (Plural ohne Artikel) - "X sind Y"
            if mp := self._detect_is_a_plural(text, text_lower, doc):
                return mp

            # Pattern 3: HAS_PROPERTY - "X ist Y" (ohne Artikel -> Eigenschaft)
            if mp := self._detect_has_property(text, text_lower):
                return mp

            # Pattern 4: CAPABLE_OF - "X kann Y"
            if mp := self._detect_capable_of(text, text_lower):
                return mp

            # Pattern 5: PART_OF - "X hat Y" / "X gehört zu Y"
            if mp := self._detect_part_of(text, text_lower):
                return mp

            # Pattern 6 & 7: LOCATED_IN - "X liegt in Y" / "X lebt in Y"
            if mp := self._detect_located_in(text, text_lower):
                return mp

            # Keine deklarative Aussage erkannt
            return None

        except Exception as e:
            logger.error(f"Fehler bei Deklaration-Erkennung: {e}", exc_info=True)
            return None

    def _detect_conditional(
        self, text: str, text_lower: str
    ) -> list[MeaningPoint] | None:
        """Erkennt konditionale Aussagen: "Wenn X, dann Y"."""
        conditional_patterns = [
            # Pattern mit explizitem "dann" - matcht "Wenn X, dann Y" oder "Wenn X dann Y"
            (
                r"^\s*(?:wenn|falls|sofern)\s+(.+?)\s*,?\s*dann\s+(.+?)\s*\.?\s*$",
                "conditional_with_dann",
            ),
            # Pattern ohne "dann" - matcht "Wenn X, Y"
            (
                r"^\s*(?:wenn|falls|sofern)\s+(.+?)\s*,\s*(.+?)\s*\.?\s*$",
                "conditional_simple",
            ),
            # Reversed: "Y, wenn X"
            (
                r"^\s*(.+?),\s+wenn\s+(.+?)\s*\.?\s*$",
                "conditional_reversed",
            ),
        ]

        for pattern, pattern_type in conditional_patterns:
            conditional_match = re.match(pattern, text_lower, re.IGNORECASE)
            if conditional_match:
                if pattern_type in ["conditional_with_dann", "conditional_simple"]:
                    condition_raw = conditional_match.group(1).strip()
                    consequence_raw = conditional_match.group(2).strip()
                else:  # reversed: "Y, wenn X"
                    consequence_raw = conditional_match.group(1).strip()
                    condition_raw = conditional_match.group(2).strip()

                condition = self.text_normalizer.clean_entity(condition_raw)
                consequence = self.text_normalizer.clean_entity(consequence_raw)

                logger.debug(
                    f"CONDITIONAL erkannt: Wenn '{condition}' dann '{consequence}'"
                )

                # Mittlere Confidence für Konditionale (komplexe Logik)
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.DEFINITION,
                        cue="auto_detect_conditional",
                        text_span=text,
                        confidence=0.82,  # Mittlere-Hohe Confidence
                        arguments={
                            "relation_type": "CONDITIONAL",
                            "condition": condition,
                            "consequence": consequence,
                            "condition_raw": condition_raw,
                            "consequence_raw": consequence_raw,
                            "auto_detected": True,
                        },
                    )
                ]

        return None

    def _detect_comparative(
        self, text: str, text_lower: str
    ) -> list[MeaningPoint] | None:
        """Erkennt komparative Aussagen: "X ist größer als Y"."""
        comparative_match = re.match(
            r"^\s*(.+?)\s+ist\s+(groesser|kleiner|schneller|langsamer|besser|schlechter|hoeher|tiefer|laenger|kuerzer|wertvoller|billiger|mehr|weniger|staerker|schwaecher|heller|dunkler|aelter|juenger)\s+als\s+(.+?)\s*\.?\s*$",
            text_lower,
            re.IGNORECASE,
        )
        if comparative_match:
            subject_raw = comparative_match.group(1).strip()
            comparison_type = comparative_match.group(2).strip()
            reference_raw = comparative_match.group(3).strip()

            subject = self.text_normalizer.clean_entity(subject_raw)
            reference = self.text_normalizer.clean_entity(reference_raw)

            logger.debug(
                f"COMPARATIVE erkannt: '{subject}' ist {comparison_type} als '{reference}'"
            )

            # Hohe Confidence für eindeutige Komparative
            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.DEFINITION,
                    cue="auto_detect_comparative",
                    text_span=text,
                    confidence=0.88,  # Hohe Confidence
                    arguments={
                        "subject": subject,
                        "relation_type": "COMPARATIVE",
                        "comparison_type": comparison_type,
                        "reference": reference,
                        "auto_detected": True,
                    },
                )
            ]

        return None

    def _detect_is_a(
        self, text: str, text_lower: str, doc: Doc
    ) -> list[MeaningPoint] | None:
        """Erkennt IS_A Relationen: "X ist ein/eine Y"."""
        is_a_match = re.match(
            r"^\s*(.+?)\s+ist\s+(?:ein|eine|der|die|das)\s+(.+?)\s*\.?\s*$",
            text_lower,
            re.IGNORECASE,
        )
        if is_a_match:
            subject_raw = is_a_match.group(1).strip()
            object_raw = is_a_match.group(2).strip()

            # Bereinige Entities
            subject = self.text_normalizer.clean_entity(subject_raw)
            object_entity_full = self.text_normalizer.clean_entity(object_raw)

            # NEU: Extrahiere Adjektive und erstelle zusätzliche HAS_PROPERTY Relationen
            adjectives = self._extract_adjectives_from_noun_phrase(text, doc)

            # Strip adjectives from object to get only the noun
            object_entity = object_entity_full
            if adjectives:
                # Remove all adjectives from the object string
                for adj in adjectives:
                    object_entity = object_entity.replace(adj, "").strip()

            logger.debug(f"IS_A erkannt: '{subject}' ist ein '{object_entity}'")

            # Haupt-IS_A MeaningPoint
            meaning_points = [
                self._create_meaning_point(
                    category=MeaningPointCategory.DEFINITION,
                    cue="auto_detect_is_a",
                    text_span=text,
                    confidence=0.92,  # Sehr hohe Confidence für klare IS_A Muster
                    arguments={
                        "subject": subject,
                        "relation_type": "IS_A",
                        "object": object_entity,
                        "auto_detected": True,
                    },
                )
            ]
            if adjectives:
                logger.debug(
                    f"Multi-Object Extraktion: {len(adjectives)} Adjektive gefunden fuer '{subject}'"
                )
                for adj in adjectives:
                    meaning_points.append(
                        self._create_meaning_point(
                            category=MeaningPointCategory.DEFINITION,
                            cue="auto_detect_has_property_from_is_a",
                            text_span=text,
                            confidence=0.75,  # Etwas niedrigere Confidence für abgeleitete Properties
                            arguments={
                                "subject": subject,
                                "relation_type": "HAS_PROPERTY",
                                "object": adj,
                                "auto_detected": True,
                                "derived_from": "is_a_pattern",
                            },
                        )
                    )

            return meaning_points

        return None

    def _detect_is_a_plural(
        self, text: str, text_lower: str, doc: Doc
    ) -> list[MeaningPoint] | None:
        """Erkennt IS_A (Plural) Relationen: "X sind Y"."""
        is_a_plural_match = re.match(
            r"^\s*(.+?)\s+sind\s+(?!ein|eine|der|die|das)(.+?)\s*\.?\s*$",
            text_lower,
            re.IGNORECASE,
        )
        if is_a_plural_match:
            subject_raw = is_a_plural_match.group(1).strip()
            object_raw = is_a_plural_match.group(2).strip()

            # Prüfe ob das Objekt wahrscheinlich ein Nomen ist (und nicht Adjektiv)
            # Heuristik: Nomen sind länger als 3 Zeichen und enden nicht auf typische Adjektiv-Endungen
            adjective_endings = ["bar", "lich", "ig", "isch", "los", "voll", "sam"]
            object_clean = self.text_normalizer.clean_entity(object_raw)

            is_likely_noun = len(object_clean) > 2 and not any(
                object_clean.endswith(ending) for ending in adjective_endings
            )

            # Zusätzliche Heuristik: Prüfe mit spaCy ob das Objekt ein Nomen ist
            try:
                # Nutze spaCy für POS-Tagging
                object_tokens = [
                    token for token in doc if token.text.lower() in object_raw.lower()
                ]
                if object_tokens:
                    # Wenn eines der Tokens ein NOUN ist, ist es wahrscheinlich IS_A
                    has_noun = any(token.pos_ == "NOUN" for token in object_tokens)
                    if has_noun:
                        is_likely_noun = True
            except Exception:
                pass  # Fallback auf Heuristik oben

            if is_likely_noun:
                subject = self.text_normalizer.clean_entity(subject_raw)
                object_entity = self.text_normalizer.clean_entity(object_raw)

                logger.debug(
                    f"IS_A (Plural) erkannt: '{subject}' sind '{object_entity}'"
                )

                # PHASE 3 (Schritt 3): Hohe Confidence für Plural IS_A ohne Artikel
                # confidence >= 0.85 -> Auto-Save
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.DEFINITION,
                        cue="auto_detect_is_a_plural",
                        text_span=text,
                        confidence=0.87,  # Hohe Confidence (>= 0.85), auto-save
                        arguments={
                            "subject": subject,
                            "relation_type": "IS_A",
                            "object": object_entity,
                            "auto_detected": True,
                        },
                    )
                ]
            # Sonst: Falle durch zu HAS_PROPERTY

        return None

    def _detect_has_property(
        self, text: str, text_lower: str
    ) -> list[MeaningPoint] | None:
        """Erkennt HAS_PROPERTY Relationen: "X ist Y" (ohne Artikel)."""
        has_property_match = re.match(
            r"^\s*(.+?)\s+(?:ist|sind)\s+(?!ein|eine|der|die|das)(.+?)\s*\.?\s*$",
            text_lower,
            re.IGNORECASE,
        )
        if has_property_match:
            subject_raw = has_property_match.group(1).strip()
            property_raw = has_property_match.group(2).strip()

            # Prüfe, ob es eine Eigenschaft ist (kein weiteres Nomen mit Artikel)
            # Einfache Heuristik: Eigenschaft hat kein "in", "von", "aus"
            if not any(
                prep in property_raw.split()
                for prep in ["in", "von", "aus", "bei", "zu"]
            ):
                subject = self.text_normalizer.clean_entity(subject_raw)

                # Extrahiere Negation aus der Eigenschaft
                property_cleaned, polarity = self.linguistic_features.extract_negation(
                    property_raw
                )
                property_value = self.text_normalizer.clean_entity(property_cleaned)

                # NEU: Prüfe auf Uncertainty-Marker
                (
                    confidence_multiplier,
                    hedge_words,
                    uncertainty_level,
                ) = self.linguistic_features.detect_uncertainty_markers(text)
                base_confidence = 0.78  # Basis-Confidence für HAS_PROPERTY
                adjusted_confidence = base_confidence * confidence_multiplier

                logger.debug(
                    f"HAS_PROPERTY erkannt: '{subject}' ist '{property_value}' (Polarity={polarity.name}, Confidence={adjusted_confidence:.2f})"
                )

                # PHASE 3 (Schritt 3): Mittlere Confidence für Eigenschaften (mehrdeutig)
                # 0.70 <= confidence < 0.85 -> Confirmation erforderlich
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.DEFINITION,
                        cue="auto_detect_has_property",
                        text_span=text,
                        polarity=polarity,  # NEU: Polarity wird gesetzt!
                        confidence=adjusted_confidence,  # NEU: Confidence angepasst durch Uncertainty!
                        arguments={
                            "subject": subject,
                            "relation_type": "HAS_PROPERTY",
                            "object": property_value,
                            "auto_detected": True,
                            "negated": polarity == Polarity.NEGATIVE,
                            "hedge_words": hedge_words,  # NEU: Uncertainty-Marker
                            "uncertainty_level": uncertainty_level,  # NEU: Uncertainty Level
                        },
                    )
                ]

        return None

    def _detect_capable_of(
        self, text: str, text_lower: str
    ) -> list[MeaningPoint] | None:
        """Erkennt CAPABLE_OF Relationen: "X kann Y"."""
        capable_of_match = re.match(
            r"^\s*(.+?)\s+(?:kann|koennen|können|vermag|vermoegen|vermögen|ist faehig zu|ist fähig zu|sind faehig zu|sind fähig zu|ist in der lage zu|sind in der lage zu)\s+(.+?)\s*\.?\s*$",
            text_lower,
            re.IGNORECASE,
        )
        if capable_of_match:
            subject_raw = capable_of_match.group(1).strip()
            ability_raw = capable_of_match.group(2).strip()

            subject = self.text_normalizer.clean_entity(subject_raw)

            # Extrahiere Negation aus der Fähigkeit
            ability_cleaned, polarity = self.linguistic_features.extract_negation(
                ability_raw
            )

            # Detect uncertainty markers (hedge words)
            confidence_multiplier, hedge_words, uncertainty_level = (
                self.linguistic_features.detect_uncertainty_markers(ability_cleaned)
            )

            ability = self.text_normalizer.clean_entity(ability_cleaned)

            # Adjust confidence based on uncertainty
            base_confidence = 0.91
            adjusted_confidence = base_confidence * confidence_multiplier

            logger.debug(
                f"CAPABLE_OF erkannt: '{subject}' kann '{ability}' (Polarity={polarity.name})"
            )

            # PHASE 3 (Schritt 3): Sehr hohe Confidence für "kann"-Konstruktionen
            # confidence >= 0.85 -> Auto-Save
            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.DEFINITION,
                    cue="auto_detect_capable_of",
                    text_span=text,
                    polarity=polarity,  # NEU: Polarity wird gesetzt!
                    confidence=adjusted_confidence,  # NEU: Confidence angepasst durch Uncertainty!
                    arguments={
                        "subject": subject,
                        "relation_type": "CAPABLE_OF",
                        "object": ability,
                        "auto_detected": True,
                        "negated": polarity
                        == Polarity.NEGATIVE,  # NEU: Flag für Negation
                        "hedge_words": hedge_words,  # NEU: Uncertainty-Marker
                        "uncertainty_level": uncertainty_level,  # NEU: Uncertainty Level
                    },
                )
            ]

        return None

    def _detect_part_of(self, text: str, text_lower: str) -> list[MeaningPoint] | None:
        """Erkennt PART_OF Relationen: "X hat Y" / "X gehört zu Y"."""
        part_of_match = re.match(
            r"^\s*(.+?)\s+(?:hat|haben|gehoert zu|gehört zu|gehoeren zu|gehören zu|besitzt|besitzen|verfuegt ueber|verfügt über|verfuegen ueber|verfügen über)\s+(.+?)\s*\.?\s*$",
            text_lower,
            re.IGNORECASE,
        )
        if part_of_match:
            subject_raw = part_of_match.group(1).strip()
            object_raw = part_of_match.group(2).strip()

            subject = self.text_normalizer.clean_entity(subject_raw)

            # Extrahiere Negation aus dem Objekt
            object_cleaned, polarity = self.linguistic_features.extract_negation(
                object_raw
            )
            part = self.text_normalizer.clean_entity(object_cleaned)

            logger.debug(
                f"PART_OF erkannt: '{subject}' hat/gehoert zu '{part}' (Polarity={polarity.name})"
            )

            # PHASE 3 (Schritt 3): Hohe Confidence für PART_OF Relationen
            # confidence >= 0.85 -> Auto-Save
            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.DEFINITION,
                    cue="auto_detect_part_of",
                    text_span=text,
                    polarity=polarity,  # NEU: Polarity wird gesetzt!
                    confidence=0.88,  # Hohe Confidence -> Auto-Save
                    arguments={
                        "subject": subject,
                        "relation_type": "PART_OF",
                        "object": part,
                        "auto_detected": True,
                        "negated": polarity
                        == Polarity.NEGATIVE,  # NEU: Flag für Negation
                    },
                )
            ]

        return None

    def _detect_located_in(
        self, text: str, text_lower: str
    ) -> list[MeaningPoint] | None:
        """Erkennt LOCATED_IN Relationen: "X liegt in Y" / "X lebt in Y"."""
        # Pattern 6: LOCATED_IN - "X liegt in Y" / "X ist in Y" / "X befindet sich in Y"
        located_in_match = re.match(
            r"^\s*(.+?)\s+(?:liegt in|ist in|befindet sich in|befindet sich im|liegt im|ist im)\s+(.+?)\s*\.?\s*$",
            text_lower,
            re.IGNORECASE,
        )
        if located_in_match:
            subject_raw = located_in_match.group(1).strip()
            location_raw = located_in_match.group(2).strip()

            subject = self.text_normalizer.clean_entity(subject_raw)
            location = self.text_normalizer.clean_entity(location_raw)

            logger.debug(f"LOCATED_IN erkannt: '{subject}' liegt in '{location}'")

            # PHASE 3 (Schritt 3): Sehr hohe Confidence für klare Lokations-Muster
            # confidence >= 0.85 -> Auto-Save
            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.DEFINITION,
                    cue="auto_detect_located_in",
                    text_span=text,
                    confidence=0.93,  # Sehr hohe Confidence -> Auto-Save
                    arguments={
                        "subject": subject,
                        "relation_type": "LOCATED_IN",
                        "object": location,
                        "auto_detected": True,
                    },
                )
            ]

        # Pattern 7: LOCATED_IN - "X lebt in Y" / "X wohnt in Y" / "X leben in Y"
        lives_in_match = re.match(
            r"^\s*(.+?)\s+(?:lebt|leben|wohnt|wohnen)\s+(?:in|im)\s+(.+?)\s*\.?\s*$",
            text_lower,
            re.IGNORECASE,
        )
        if lives_in_match:
            subject_raw = lives_in_match.group(1).strip()
            location_raw = lives_in_match.group(2).strip()

            subject = self.text_normalizer.clean_entity(subject_raw)
            location = self.text_normalizer.clean_entity(location_raw)

            logger.debug(
                f"LOCATED_IN (leben/wohnen) erkannt: '{subject}' lebt in '{location}'"
            )

            # PHASE 3 (Schritt 3): Hohe Confidence für "leben in/wohnen in" Muster
            # confidence >= 0.85 -> Auto-Save
            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.DEFINITION,
                    cue="auto_detect_lives_in",
                    text_span=text,
                    confidence=0.89,  # Hohe Confidence -> Auto-Save
                    arguments={
                        "subject": subject,
                        "relation_type": "LOCATED_IN",
                        "object": location,
                        "auto_detected": True,
                    },
                )
            ]

        return None

    def _extract_adjectives_from_noun_phrase(self, text: str, doc: Doc) -> list[str]:
        """
        Extrahiert Adjektive aus einer Nomen-Phrase mit spaCy.

        Args:
            text: Der Text
            doc: spaCy Doc

        Returns:
            Liste von Adjektiven (lowercase, lemmatisiert)
        """
        adjectives = []
        try:
            for token in doc:
                if token.pos_ == "ADJ":
                    # Lemmatisiere Adjektiv und füge hinzu
                    adj_lemma = token.lemma_.lower()
                    if adj_lemma and len(adj_lemma) > 1:
                        adjectives.append(adj_lemma)
                        logger.debug(
                            f"Adjektiv extrahiert: '{token.text}' -> '{adj_lemma}'"
                        )
        except Exception as e:
            logger.debug(f"Fehler bei Adjektiv-Extraktion: {e}")

        return adjectives

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
