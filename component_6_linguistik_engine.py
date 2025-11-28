# component_6_linguistik_engine.py
"""
Linguistische Vorverarbeitung mit spaCy + Lexikon-Management.

WICHTIG: KEINE Unicode-Zeichen verwenden, die Windows cp1252 Encoding-Probleme verursachen.
Verboten: OK FEHLER -> x / != <= >= etc.
Erlaubt: [OK] [FEHLER] -> * / != <= >= AND OR NOT

HINWEIS:
Für diese Komponente müssen spaCy und ein Sprachmodell installiert sein.
Führen Sie aus:
  pip install spacy PyYAML
  python -m spacy download de_core_news_sm
"""
from pathlib import Path
from typing import Dict, List, Optional

import spacy
import yaml

from component_15_logging_config import get_logger

# Import exception utilities for user-friendly error messages
from kai_exceptions import SpaCyModelError, get_user_friendly_message

logger = get_logger(__name__)


class ResourceManager:
    """Lädt und verwaltet externe Ressourcen wie Lexika aus YAML-Dateien."""

    def __init__(self, lexika_pfad: str = "lexika"):
        """
        Initialisiert ResourceManager.

        Args:
            lexika_pfad: Pfad zum Verzeichnis mit YAML-Lexikon-Dateien.
                         Standard: "lexika" (relativ zum Arbeitsverzeichnis)
        """
        self.lexika_pfad = Path(lexika_pfad)
        self.lexika: Dict[str, List[str]] = {}

    def load(self) -> None:
        """Lädt alle .yml-Dateien aus dem angegebenen Verzeichnis."""
        logger.info(f"[ResourceManager] Lade Lexika aus '{self.lexika_pfad}'...")
        if not self.lexika_pfad.is_dir():
            logger.warning(
                f"[ResourceManager] WARNUNG: Verzeichnis '{self.lexika_pfad}' nicht gefunden."
            )
            return

        failed_files = []
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
                        failed_files.append((file_path.name, "Keine Liste"))
            except yaml.YAMLError as e:
                # Expected error - YAML syntax issue
                logger.error(f"YAML-Fehler beim Laden von '{file_path.name}': {e}")
                failed_files.append((file_path.name, "YAML-Fehler"))
            except IOError as e:
                # Expected error - file read issue
                logger.error(f"I/O-Fehler beim Laden von '{file_path.name}': {e}")
                failed_files.append((file_path.name, "I/O-Fehler"))
            except Exception as e:
                # Unexpected error - log with stack trace
                logger.error(
                    f"Unerwarteter Fehler beim Laden von '{file_path.name}': {e}",
                    exc_info=True,
                )
                failed_files.append((file_path.name, "Unerwarteter Fehler"))

        if failed_files:
            logger.warning(
                f"[ResourceManager] {len(failed_files)} Dateien konnten nicht geladen werden: "
                f"{', '.join(f[0] for f in failed_files)}"
            )
        logger.info(
            f"[ResourceManager] Laden der Lexika abgeschlossen. "
            f"{len(self.lexika)} Lexika geladen."
        )

    def get_lexikon(self, key: str) -> Optional[List[str]]:
        """
        Gibt Lexikon zurück oder None wenn nicht gefunden.

        Args:
            key: Lexikon-Schlüssel (Dateiname ohne .yml)

        Returns:
            Liste der Lexikon-Einträge oder None
        """
        return self.lexika.get(key)

    def contains_word(self, lexikon_key: str, word: str) -> bool:
        """
        Prüft ob Wort in Lexikon existiert (case-insensitive).

        Args:
            lexikon_key: Lexikon-Schlüssel
            word: Zu prüfendes Wort

        Returns:
            True wenn Wort im Lexikon vorhanden, sonst False
        """
        lexikon = self.lexika.get(lexikon_key)
        if lexikon is None:
            return False
        return word.lower() in lexikon

    def get_all_lexikon_keys(self) -> List[str]:
        """
        Gibt alle geladenen Lexikon-Schlüssel zurück.

        Returns:
            Liste aller Lexikon-Schlüssel
        """
        return list(self.lexika.keys())


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
                context={"model_name": "de_core_news_sm"},
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
        """
        Verarbeitet einen Text und gibt ein spaCy Doc-Objekt zurück.

        Args:
            text: Zu verarbeitender Text

        Returns:
            spaCy Doc-Objekt oder None bei Fehler
        """
        if not self.nlp:
            logger.warning("[LinguisticPreprocessor] NLP-Modell nicht verfügbar")
            return None

        if not isinstance(text, str):
            logger.error(f"[LinguisticPreprocessor] Ungültiger Input-Typ: {type(text)}")
            return None

        # Optional: Warn for very long texts
        if len(text) > 1_000_000:  # 1MB
            logger.warning(
                f"[LinguisticPreprocessor] Sehr langer Text ({len(text)} Zeichen), "
                "Verarbeitung könnte langsam sein."
            )

        try:
            # Führt die gesamte NLP-Pipeline von spaCy auf dem Text aus
            return self.nlp(text)
        except Exception as e:
            logger.error(
                f"[LinguisticPreprocessor] Fehler bei Text-Verarbeitung: {e}",
                exc_info=True,
            )
            return None
