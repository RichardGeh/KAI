# component_utils_text_normalization.py
"""
Zentrale Text-Normalisierungs-Utilities für KAI

Verantwortlichkeiten:
- Artikel-Entfernung (der, die, das, ein, eine)
- Plural-zu-Singular-Normalisierung mit spaCy-Integration
- Satzzeichen- und Whitespace-Bereinigung
- Konsistente Entity-Normalisierung für Graph-Speicherung

Diese Funktionen werden von mehreren Komponenten genutzt:
- component_7_meaning_extractor.py (Intent-Extraktion)
- kai_response_formatter.py (Antwort-Formatierung)

Design:
- Dependency Injection für LinguisticPreprocessor (optional)
- Fallback auf regelbasierte Normalisierung wenn kein spaCy verfügbar
- Konservative Strategie: Bei Unsicherheit Wort unverändert lassen
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Zentrale Klasse für Text-Normalisierung mit optionaler spaCy-Integration.

    Diese Klasse kann mit oder ohne LinguisticPreprocessor verwendet werden.
    - MIT Preprocessor: Nutzt spaCy Lemmatization für präzise Plural-Normalisierung
    - OHNE Preprocessor: Nutzt konservative regelbasierte Heuristiken
    """

    def __init__(self, preprocessor=None):
        """
        Initialisiert den TextNormalizer.

        Args:
            preprocessor: Optional - LinguisticPreprocessor für spaCy-Integration
                         Wenn None, wird nur regelbasierte Normalisierung verwendet
        """
        self.preprocessor = preprocessor
        if preprocessor:
            logger.debug("TextNormalizer mit spaCy-Integration initialisiert")
        else:
            logger.debug(
                "TextNormalizer ohne spaCy-Integration initialisiert (nur Regeln)"
            )

    def clean_entity(self, entity_text: str) -> str:
        """
        Entfernt führende Artikel, bereinigt den Text und normalisiert Plurale zu Singularen.

        Beispiele:
        - "der Apfel" -> "apfel"
        - "die Katzen" -> "katze"
        - "ein schöner Hund" -> "schöner hund"
        - "  Computer  " -> "computer"
        - "zur Schule" -> "schule"
        - "des Hauses" -> "hauses"

        Args:
            entity_text: Der zu bereinigende Text

        Returns:
            Bereinigter und normalisierter Text (lowercase, ohne Artikel, Singular)
            Leerer String wenn nur ein Artikel/Funktionswort
        """
        if not entity_text:
            return ""

        lower_text = entity_text.lower().strip()

        # SCHRITT 1: Entferne Artikel und Kontraktionen am Anfang
        # Wichtig: Kontraktionen VOR einfachen Artikeln prüfen (längere zuerst)
        articles_and_contractions = [
            "zur ",
            "zum ",
            "vom ",
            "beim ",
            "im ",
            "am ",
            "ans ",
            "ins ",  # Kontraktionen
            "ein ",
            "eine ",
            "einer ",
            "einen ",
            "einem ",
            "eines ",  # Indefinite Artikel
            "der ",
            "die ",
            "das ",
            "den ",
            "dem ",
            "des ",  # Definite Artikel
        ]
        for article in articles_and_contractions:
            if lower_text.startswith(article):
                lower_text = lower_text[len(article) :].strip()
                break

        # SCHRITT 1b: Wenn nur Artikel/Funktionswort übrig, verwerfe es
        # Blacklist von Funktionswörtern die nicht im Graph gespeichert werden sollen
        function_words = {
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einer",
            "einen",
            "einem",
            "eines",
            "zur",
            "zum",
            "vom",
            "beim",
            "im",
            "am",
            "ans",
            "ins",
            "zu",
            "von",
            "bei",
            "in",
            "an",
            "auf",
            "aus",
            "und",
            "oder",
            "aber",
            "denn",
            "sondern",
            "ist",
            "sind",
            "war",
            "waren",
            "hat",
            "haben",
        }
        if lower_text in function_words:
            logger.debug(f"Funktionswort verworfen: '{entity_text}' -> ''")
            return ""

        # SCHRITT 2: Entferne Satzzeichen am Ende
        lower_text = lower_text.rstrip(".,!?;:")

        # SCHRITT 3: Entferne mehrfache Leerzeichen
        lower_text = re.sub(r"\s+", " ", lower_text)

        # SCHRITT 4: Normalisiere Plural zu Singular
        # Bei multi-word Phrasen: Normalisiere jedes Wort einzeln
        words = lower_text.split()
        if len(words) > 1:
            # Multi-word phrase: normalisiere jedes Wort separat
            normalized_words = [
                self.normalize_plural_to_singular(word) for word in words
            ]
            lower_text = " ".join(normalized_words)
        else:
            # Single word: normalisiere direkt
            lower_text = self.normalize_plural_to_singular(lower_text)

        return lower_text.strip()

    def normalize_plural_to_singular(self, word: str) -> str:
        """
        Konvertiert deutsche Plurale zu Singularen.

        Strategie:
        1. PRIMÄR: spaCy Lemmatization (wenn Preprocessor verfügbar) - hohe Präzision
        2. FALLBACK: Konservative regelbasierte Heuristiken für eindeutige Muster

        Verhindert Fehltransformationen wie:
        - "säugetier" -> "säugeti" (Regel zu aggressiv)
        - "computer" -> "comput" (Fremdwort falsch behandelt)
        - "Luke" -> "Luk" (Name falsch behandelt)
        - "des" -> "der" (Artikel-Normalisierung unerwünscht)

        Args:
            word: Das zu normalisierende Wort

        Returns:
            Normalisiertes Wort (Singular) oder Original bei Unsicherheit
        """
        if not word or len(word) < 3:
            return word

        # WICHTIG: Funktionswörter/Artikel NICHT lemmatisieren
        # Diese sollten bereits durch clean_entity verworfen sein, aber als Sicherheit:
        function_words = {
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einer",
            "einen",
            "einem",
            "eines",
            "zur",
            "zum",
            "vom",
            "beim",
            "im",
            "am",
            "ans",
            "ins",
        }
        if word.lower() in function_words:
            return word  # Unverändert zurückgeben, nicht lemmatisieren

        # STRATEGIE 1: spaCy Lemmatization (bevorzugt, wenn verfügbar)
        if self.preprocessor and hasattr(self.preprocessor, "nlp"):
            try:
                doc = self.preprocessor.nlp(word)
                if len(doc) > 0:
                    token = doc[0]
                    lemma = token.lemma_

                    # WICHTIG: Artikel/Funktionswörter nicht lemmatisieren
                    # spaCy taggt diese als DET (Determiner) oder ADP (Adposition)
                    if token.pos_ in ["DET", "ADP", "CONJ", "CCONJ"]:
                        logger.debug(
                            f"Überspringe Lemmatisierung für Funktionswort: '{word}' (POS: {token.pos_})"
                        )
                        return word

                    # WICHTIG: Verben nicht als Plurale behandeln
                    # Infinitive enden oft auf "-en" (fliegen, laufen, schwimmen)
                    # Diese sollten NICHT durch regelbasierte Plural-Normalisierung laufen
                    if token.pos_ in ["VERB", "AUX"]:
                        # Verwende Lemma wenn verfügbar, sonst Original
                        result_lemma = (
                            lemma.lower() if lemma and len(lemma) >= 2 else word
                        )

                        # Sanity check: Wenn Lemma länger als Original oder sieht falsch aus, nutze Original
                        # (spaCy kann bei falschen POS-Tags falsche Lemmas erzeugen)
                        if (
                            len(result_lemma) > len(word)
                            or result_lemma.endswith("en")
                            and not word.endswith("en")
                        ):
                            logger.debug(
                                f"Verb erkannt, aber Lemma sieht falsch aus: '{word}' -> '{result_lemma}' (nutze Original)"
                            )
                            return word

                        logger.debug(
                            f"Verb erkannt: '{word}' -> '{result_lemma}' (POS: {token.pos_}, keine Plural-Normalisierung)"
                        )
                        return result_lemma

                    # Nur verwenden wenn das Lemma sich unterscheidet und sinnvoll aussieht
                    if lemma and lemma != word and len(lemma) >= 2:
                        logger.debug(f"spaCy lemmatization: '{word}' -> '{lemma}'")
                        return lemma.lower()
            except Exception as e:
                logger.debug(f"spaCy lemmatization fehlgeschlagen für '{word}': {e}")
                # Falle durch zu regelbasierter Normalisierung

        # STRATEGIE 2: Regelbasierte Heuristiken (Fallback oder primär wenn kein spaCy)
        return self._rule_based_singular_normalization(word)

    def _rule_based_singular_normalization(self, word: str) -> str:
        """
        KONSERVATIVE regelbasierte Plural-Normalisierung für deutsche Wörter.

        Strategie: Nur EINDEUTIGE, hochspezifische Plural-Endungen werden behandelt.
        Bei Unsicherheit wird das Wort UNVERÄNDERT zurückgegeben.

        Args:
            word: Das zu normalisierende Wort

        Returns:
            Normalisiertes Wort (Singular) oder Original bei Unsicherheit
        """
        if not word or len(word) < 4:
            return word

        # STUFE 1: Hochspezifische, zusammengesetzte Endungen (100% Confidence)
        # Diese sind eindeutig Plurale und können sicher transformiert werden
        high_confidence_plural_rules = [
            ("ionen", "ion"),  # Aktionen -> Aktion
            ("ungen", "ung"),  # Meldungen -> Meldung
            ("heiten", "heit"),  # Freiheiten -> Freiheit
            ("keiten", "keit"),  # Möglichkeiten -> Möglichkeit
            ("schaften", "schaft"),  # Eigenschaften -> Eigenschaft
            ("tümer", "tum"),  # Reichtümer -> Reichtum
            ("ismen", "ismus"),  # Organismen -> Organismus
        ]

        # Prüfe nur die hochspezifischen Regeln (100% sichere Plurale)
        for plural_ending, singular_ending in high_confidence_plural_rules:
            if word.endswith(plural_ending):
                base = word[: -len(plural_ending)]
                result = base + singular_ending
                logger.debug(
                    f"Regelbasiert: '{word}' -> '{result}' (Regel: {plural_ending}->{singular_ending})"
                )
                return result

        # STUFE 2: Spezielle Behandlung für "-zen" -> "-ze" (sehr sicheres Muster)
        if word.endswith("zen") and len(word) > 4:
            # Nur bei längeren Wörtern: "katzen" -> "katze", aber nicht "zen" -> "ze"
            result = word[:-2] + "e"
            logger.debug(f"Regelbasiert: '{word}' -> '{result}' (zen->ze Regel)")
            return result

        # STUFE 3: Konservative "-en" Regel
        # Nur bei längeren Wörtern (>= 5 Zeichen) und wenn vorletzte Stelle ein Konsonant ist
        if word.endswith("en") and len(word) >= 5:
            # Prüfe ob vorletzte Stelle kein Vokal ist (verhindert Fehler bei Namen wie "Luke", "Anne")
            char_before_en = word[-3]
            vowels = "aeiouyäöü"
            if char_before_en.lower() not in vowels:
                # Sehr wahrscheinlich ein deutscher Plural: "hunden" -> "hund", "tischen" -> "tisch"
                base = word[:-2]
                # Zusätzliche Prüfung: Base sollte mindestens 3 Zeichen haben
                if len(base) >= 3:
                    logger.debug(
                        f"Regelbasiert: '{word}' -> '{base}' (en-Regel mit Konsonant)"
                    )
                    return base

        # KEINE sichere Regel gefunden -> Wort UNVERÄNDERT zurückgeben
        # Dies verhindert Fehler wie:
        # - "computer" -> "comput" (falsch!)
        # - "Luke" -> "Luk" (falsch!)
        # - "katze" -> "katz" (falsch!)
        logger.debug(
            f"Regelbasiert: '{word}' -> '{word}' (keine sichere Regel, unverändert)"
        )
        return word


# ============================================================================
# CONVENIENCE FUNCTIONS (Singleton-Pattern für einfache Nutzung)
# ============================================================================

_default_normalizer: Optional[TextNormalizer] = None


def get_default_normalizer(preprocessor=None) -> TextNormalizer:
    """
    Gibt eine Singleton-Instanz des TextNormalizers zurück.

    Args:
        preprocessor: Optional - LinguisticPreprocessor für spaCy-Integration

    Returns:
        Singleton-Instanz des TextNormalizers
    """
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = TextNormalizer(preprocessor=preprocessor)
    return _default_normalizer


def clean_entity(entity_text: str, preprocessor=None) -> str:
    """
    Convenience-Funktion für Entity-Normalisierung.

    Args:
        entity_text: Der zu bereinigende Text
        preprocessor: Optional - LinguisticPreprocessor

    Returns:
        Bereinigter Text
    """
    normalizer = get_default_normalizer(preprocessor)
    return normalizer.clean_entity(entity_text)


def normalize_plural_to_singular(word: str, preprocessor=None) -> str:
    """
    Convenience-Funktion für Plural-Normalisierung.

    Args:
        word: Das zu normalisierende Wort
        preprocessor: Optional - LinguisticPreprocessor

    Returns:
        Normalisiertes Wort (Singular)
    """
    normalizer = get_default_normalizer(preprocessor)
    return normalizer.normalize_plural_to_singular(word)
