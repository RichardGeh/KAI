# component_6_linguistik_engine.py
import yaml
import spacy
from pathlib import Path
from typing import Dict, List, Optional

# Import exception utilities for user-friendly error messages
from kai_exceptions import SpaCyModelError, get_user_friendly_message
from component_15_logging_config import get_logger

logger = get_logger(__name__)

# WICHTIGER HINWEIS:
# Für diese Komponente müssen spaCy und ein Sprachmodell installiert sein.
# Führen Sie aus:
# pip install spacy PyYAML
# python -m spacy download de_core_news_sm


class ResourceManager:
    """Lädt und verwaltet externe Ressourcen wie Lexika aus YAML-Dateien."""

    def __init__(self, lexika_pfad: str = "lexika"):
        self.lexika_pfad = Path(lexika_pfad)
        self.lexika: Dict[str, List[str]] = {}

    def load(self):
        """Lädt alle .yml-Dateien aus dem angegebenen Verzeichnis."""
        logger.info(f"[ResourceManager] Lade Lexika aus '{self.lexika_pfad}'...")
        if not self.lexika_pfad.is_dir():
            logger.warning(
                f"[ResourceManager] WARNUNG: Verzeichnis '{self.lexika_pfad}' nicht gefunden."
            )
            return

        for file_path in self.lexika_pfad.glob("*.yml"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    key = file_path.stem  # Dateiname ohne Endung
                    if isinstance(data, list):
                        # Wir speichern alles als Kleinbuchstaben für den einfachen Abgleich
                        self.lexika[key] = [str(item).lower() for item in data]
                        logger.info(
                            f"  - Lexikon '{key}' mit {len(self.lexika[key])} Einträgen geladen."
                        )
                    else:
                        logger.warning(
                            f"  - WARNUNG: Datei '{file_path.name}' enthält keine Liste."
                        )
            except yaml.YAMLError as e:
                # Spezifischer Fehler beim YAML-Parsen
                logger.error(
                    f"YAML-Fehler beim Laden von '{file_path.name}': {e}", exc_info=True
                )
            except IOError as e:
                # Spezifischer Fehler beim Datei-Lesen
                logger.error(
                    f"I/O-Fehler beim Laden von '{file_path.name}': {e}", exc_info=True
                )
            except Exception as e:
                # Unerwarteter Fehler
                logger.error(
                    f"Unerwarteter Fehler beim Laden von '{file_path.name}': {e}",
                    exc_info=True,
                )
        logger.info("[ResourceManager] Laden der Lexika abgeschlossen.")


class LinguisticPreprocessor:
    """
    Bereitet einen Rohtext für die Analyse vor, indem es Tokenisierung,
    Lemmatisierung und POS-Tagging mit spaCy durchführt.
    """

    def __init__(self):
        self.nlp = None
        try:
            # Lädt das deutsche Sprachmodell
            self.nlp = spacy.load("de_core_news_sm")
            logger.info(
                "[LinguisticPreprocessor] spaCy-Modell 'de_core_news_sm' erfolgreich geladen."
            )
        except OSError as e:
            # Spezifischer Fehler: Modell nicht gefunden
            spacy_error = SpaCyModelError(
                "spaCy-Modell 'de_core_news_sm' nicht gefunden. "
                "Bitte installieren Sie es mit: python -m spacy download de_core_news_sm",
                model_name="de_core_news_sm",
                original_exception=e,
            )
            logger.warning(f"[LinguisticPreprocessor] {spacy_error}")
            user_msg = get_user_friendly_message(spacy_error)
            logger.info(f"[LinguisticPreprocessor] User-Message: {user_msg}")

            # Graceful Degradation: Fallback auf blankes Modell
            self.nlp = spacy.blank("de")
            logger.warning(
                "[LinguisticPreprocessor] Fallback auf blankes 'de'-Modell (POS/LEMMA unzuverlässig)."
            )
        except Exception as e:
            # Unerwarteter Fehler beim Laden des Modells
            logger.error(
                f"[LinguisticPreprocessor] Unerwarteter Fehler beim Laden des spaCy-Modells: {e}",
                exc_info=True,
            )
            # Graceful Degradation: Fallback auf blankes Modell
            self.nlp = spacy.blank("de")
            logger.warning("[LinguisticPreprocessor] Fallback auf blankes 'de'-Modell.")

    def process(self, text: str) -> Optional[spacy.tokens.Doc]:
        """Verarbeitet einen Text und gibt ein spaCy Doc-Objekt zurück."""
        if not self.nlp:
            return None
        # Führt die gesamte NLP-Pipeline von spaCy auf dem Text aus
        return self.nlp(text)
