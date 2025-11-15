# component_24_pattern_orchestrator.py
"""
Orchestriert alle Pattern Recognition Features.

Koordiniert:
- Buchstaben-Ebene (Tippfehler)
- Wortfolgen (Predictions)
- Implikationen (Implizite Fakten)
"""

from typing import Any, Dict, List

from component_15_logging_config import get_logger
from component_19_pattern_recognition_char import TypoCandidateFinder
from component_20_pattern_recognition_sequence import SequencePredictor
from component_22_pattern_recognition_implicit import ImplicationDetector
from component_25_adaptive_thresholds import AdaptiveThresholdManager
from kai_config import get_config

logger = get_logger(__name__)
config = get_config()


class PatternOrchestrator:
    """Zentrale Koordination aller Pattern Recognition Features"""

    def __init__(self, netzwerk):
        self.netzwerk = netzwerk
        self.typo_finder = TypoCandidateFinder(netzwerk)
        self.sequence_predictor = SequencePredictor(netzwerk)
        self.implication_detector = ImplicationDetector(netzwerk)
        self.adaptive_manager = AdaptiveThresholdManager(netzwerk)

        # Session-Whitelist für bestätigte Nicht-Typos (vermeidet Loops)
        self.typo_whitelist = set()

        # Hole phase-abhängige Confidence-Gates
        gates = self.adaptive_manager.get_confidence_gates()
        self.typo_auto_correct_threshold = gates["auto_correct"]
        self.typo_ask_user_threshold = gates["ask_user"]

        # Log aktuellen System-Status
        stats = self.adaptive_manager.get_system_stats()
        logger.info(
            "PatternOrchestrator initialisiert mit adaptiven Thresholds",
            extra={
                "phase": stats["phase"],
                "vocab_size": stats["vocab_size"],
                "typo_threshold": stats["typo_threshold"],
                "seq_threshold": stats["sequence_threshold"],
                "auto_correct_gate": self.typo_auto_correct_threshold,
                "ask_user_gate": self.typo_ask_user_threshold,
            },
        )

    def process_input(self, text: str) -> Dict[str, Any]:
        """
        Verarbeitet User-Input durch alle Pattern Recognition Stufen.

        Returns:
            Dict mit corrections, predictions, implications
        """
        result: Dict[str, Any] = {
            "original_text": text,
            "corrected_text": text,
            "typo_corrections": [],
            "next_word_predictions": [],
            "implications": [],
            "needs_user_clarification": False,
        }

        # Early Exit: Überspringe Pattern Recognition für explizite Commands UND Fragen
        # Diese sollten direkt zum MeaningExtractor gehen ohne Typo-Detection
        import re

        command_prefixes = [
            r"^\s*definiere:",
            r"^\s*lerne muster:",
            r"^\s*ingestiere text:",
            r"^\s*lerne:",
            r"^\s*(?:lese datei|ingestiere dokument|verarbeite pdf|lade datei):",
        ]

        # WICHTIG: Auch Fragen überspringen!
        # Bei "Was ist X?" will der Benutzer eine Antwort, keine Typo-Rückfrage
        question_patterns = [
            r"^\s*(?:was|wer|wie|wo|wann|warum|wieso|weshalb|wozu)\s+",
            r"^\s*frage:\s*",
        ]

        for pattern in command_prefixes + question_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                logger.debug(
                    "Command/Frage erkannt, überspringe Pattern Recognition",
                    extra={"text_preview": text[:50], "pattern": pattern},
                )
                # Gib Input unverändert zurück
                return result

        # 1. Tippfehler-Korrektur
        words = text.split()
        corrected_words = []

        # WICHTIG: Blacklist häufiger deutscher Funktionswörter
        # Diese sollten NIEMALS als Tippfehler korrigiert werden
        function_words_blacklist = {
            # Artikel
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
            # Präpositionen
            "in",
            "an",
            "auf",
            "aus",
            "bei",
            "mit",
            "nach",
            "von",
            "zu",
            "vor",
            "über",
            "unter",
            # Pronomen
            "ich",
            "du",
            "er",
            "sie",
            "es",
            "wir",
            "ihr",
            "mein",
            "dein",
            "sein",
            "ihr",
            "unser",
            "euer",
            "dieser",
            "diese",
            "dieses",
            "jener",
            "jene",
            "jenes",
            # Konjunktionen
            "und",
            "oder",
            "aber",
            "denn",
            "sondern",
            "wenn",
            "als",
            "wie",
            "dass",
            "weil",
            # Häufige Verben
            "ist",
            "sind",
            "war",
            "waren",
            "hat",
            "haben",
            "wird",
            "werden",
            "kann",
            "können",
            # Fragewörter
            "was",
            "wer",
            "wie",
            "wo",
            "wann",
            "warum",
            "wozu",
            "wieso",
            "weshalb",
        }

        for word in words:
            # Entferne Satzzeichen UND Anführungszeichen
            clean_word = word.strip(".,!?;:'\"")
            if len(clean_word) < 3:
                corrected_words.append(word)
                continue

            # WICHTIG: Überspringe Funktionswörter (sollten nie als Typo behandelt werden)
            if clean_word.lower() in function_words_blacklist:
                corrected_words.append(word)
                continue

            # Prüfe Session-Whitelist (bestätigte Nicht-Typos)
            if clean_word.lower() in self.typo_whitelist:
                logger.debug(f"'{clean_word}' in Typo-Whitelist, überspringe")
                corrected_words.append(word)
                continue

            # WICHTIG: Überspringe kapitalisierte Wörter (Eigennamen, Fremdwörter)
            # Heuristik: Wort beginnt mit Großbuchstabe und ist nicht am Satzanfang
            is_capitalized = clean_word[0].isupper() if clean_word else False
            is_all_caps = clean_word.isupper() if clean_word else False

            # Überspringe wenn kapitalisiert (aber nicht ALL CAPS = Akronym)
            if is_capitalized and not is_all_caps:
                logger.debug(
                    f"'{clean_word}' ist kapitalisiert (vermutlich Eigenname), überspringe Typo-Erkennung"
                )
                corrected_words.append(word)
                continue

            # Prüfe ob bekannt
            known = self.netzwerk.get_all_known_words()
            if clean_word.lower() in [w.lower() for w in known]:
                corrected_words.append(word)
                continue

            # Suche Tippfehler-Kandidaten
            candidates = self.typo_finder.find_candidates(clean_word, max_candidates=3)

            if candidates and len(candidates) > 0:
                best = candidates[0]

                # FIX 2024-11: Auto-Korrektur komplett deaktiviert - zu viele falsche Matches
                # Problem: "zur" -> "tür", "gelegt" -> "belebt", "sofern" -> "eltern" mit 1.00 Confidence
                # Lösung: IMMER ask_user, NIEMALS auto_correct

                # DEAKTIVIERT: Auto-Korrektur
                # if best["confidence"] >= self.typo_auto_correct_threshold:
                #     corrected_words.append(best["word"])
                #     result["typo_corrections"].append(...)

                # FORCIERT: Immer ask_user (wenn Confidence hoch genug)
                if best["confidence"] >= self.typo_ask_user_threshold:
                    # Rückfrage (IMMER, auch bei hoher Confidence)
                    corrected_words.append(word)  # Nutze Original
                    result["typo_corrections"].append(
                        {
                            "original": word,
                            "candidates": candidates,
                            "decision": "ask_user",
                        }
                    )
                    result["needs_user_clarification"] = True
                else:
                    # Confidence zu niedrig -> ignoriere
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        result["corrected_text"] = " ".join(corrected_words)

        # 2. Wortfolgen-Vorhersage (nur wenn kein Tippfehler)
        if not result["needs_user_clarification"]:
            predictions = self.sequence_predictor.predict_completion(
                result["corrected_text"]
            )
            result["next_word_predictions"] = predictions

        return result

    def add_to_typo_whitelist(self, word: str):
        """
        Fügt ein Wort zur Typo-Whitelist hinzu.

        Sollte aufgerufen werden wenn Benutzer bestätigt, dass ein Wort korrekt ist.
        Verhindert wiederholte Typo-Rückfragen für dasselbe Wort in dieser Session.

        Args:
            word: Das Wort, das nicht als Typo behandelt werden soll
        """
        self.typo_whitelist.add(word.lower())
        logger.info(f"'{word}' zur Typo-Whitelist hinzugefügt (Session)")

    def detect_implications_for_fact(
        self, subject: str, relation: str, obj: str
    ) -> List[Dict]:
        """Erkennt Implikationen für einen Fakt"""
        if relation == "HAS_PROPERTY":
            return self.implication_detector.detect_property_implications(subject, obj)
        return []


if __name__ == "__main__":
    print("=== Pattern Orchestrator ===")
    print("Modul geladen.")
