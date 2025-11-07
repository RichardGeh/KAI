# component_26_command_suggestions.py
"""
Befehlsvorschläge bei Tippfehlern.

Erkennt häufige Tippfehler in KAI-Befehlen und schlägt Korrekturen vor.
"""

from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class CommandSuggester:
    """
    Schlägt Korrekturen für falsch geschriebene Befehle vor.

    Fokussiert auf die wichtigsten manuell genutzten Befehle:
    - "Lerne:" - einfaches Lernen
    - "Lerne Muster:" - Pattern-Learning
    - "Was ist..." - Fragen
    - "Definiere:" - Attribut-Definitionen
    """

    # Häufig genutzte Befehle (nach Priorität sortiert)
    KNOWN_COMMANDS = {
        "lerne:": {
            "pattern": r"^\s*lerne:\s*",
            "description": "Lerne: <Text>",
            "example": "Lerne: Ein Hund ist ein Tier",
            "typos": ["lehre:", "lerne", "lerne;", "lernr:", "lern:", "lehrne:"],
        },
        "lerne muster:": {
            "pattern": r"^\s*lerne muster:\s*",
            "description": 'Lerne Muster: "<Beispielsatz>" bedeutet <RELATION>',
            "example": 'Lerne Muster: "X schmeckt Y" bedeutet HAS_TASTE',
            "typos": [
                "lerne mustre:",
                "lehre muster:",
                "lerne muste:",
                "lerne master:",
                "lerne muster",
                "lerne musster:",
            ],
        },
        "was ist": {
            "pattern": r"^\s*was ist\s+",
            "description": "Was ist <Konzept>?",
            "example": "Was ist ein Hund?",
            "typos": ["wa ist", "was it", "was si", "was isr", "wass ist", "was idt"],
        },
        "definiere:": {
            "pattern": r"^\s*definiere:\s*",
            "description": "Definiere: <Wort>/<Pfad> = <Wert>",
            "example": "Definiere: hund/farbe = braun",
            "typos": [
                "defniere:",
                "definere:",
                "definire:",
                "deffiniere:",
                "definiere",
                "definere:",
            ],
        },
        "wer ist": {
            "pattern": r"^\s*wer ist\s+",
            "description": "Wer ist <Person>?",
            "example": "Wer ist Einstein?",
            "typos": ["wer it", "wer si", "wer isr", "wee ist", "wer idt"],
        },
        "wie": {
            "pattern": r"^\s*wie\s+",
            "description": "Wie <Frage>?",
            "example": "Wie funktioniert das?",
            "typos": ["wi ", "wir ", "wie?"],
        },
        "warum": {
            "pattern": r"^\s*warum\s+",
            "description": "Warum <Frage>?",
            "example": "Warum ist der Himmel blau?",
            "typos": ["warrum", "warun", "waum", "warumd"],
        },
    }

    def __init__(self, similarity_threshold: float = 0.6):
        """
        Args:
            similarity_threshold: Minimale Ähnlichkeit (0.0-1.0) für Vorschläge
        """
        self.similarity_threshold = similarity_threshold
        logger.info(
            "CommandSuggester initialisiert", extra={"threshold": similarity_threshold}
        )

    def suggest_command(self, user_input: str) -> Optional[Dict[str, any]]:
        """
        Prüft ob die Benutzereingabe ein falsch geschriebener Befehl ist.

        Args:
            user_input: Die Benutzereingabe

        Returns:
            Dict mit Vorschlag oder None wenn kein Tippfehler erkannt
            {
                "original": "lehrne: etwas",
                "suggestion": "lerne:",
                "confidence": 0.85,
                "full_suggestion": "Lerne: etwas",
                "description": "Lerne: <Text>",
                "example": "Lerne: Ein Hund ist ein Tier"
            }
        """
        if not user_input or not user_input.strip():
            return None

        user_input_lower = user_input.lower().strip()

        # WICHTIG: Prüfe zuerst, ob die Eingabe bereits mit einem korrekten Befehl beginnt
        # Verhindert False-Positives wie "Was ist X?" → "Was ist ist X?"
        for correct_cmd in self.KNOWN_COMMANDS.keys():
            if user_input_lower.startswith(correct_cmd):
                # Eingabe ist bereits korrekt, kein Vorschlag nötig
                logger.debug(
                    "Eingabe beginnt bereits mit korrektem Befehl, kein Vorschlag nötig",
                    extra={"input": user_input_lower[:20], "correct_cmd": correct_cmd},
                )
                return None

        # Extrahiere potentiellen Befehl (erste 20 Zeichen)
        potential_command = user_input_lower[:20]

        # 1. Prüfe gegen bekannte Tippfehler (schnell, hohe Confidence)
        typo_match = self._check_known_typos(potential_command)
        if typo_match:
            correct_command, confidence, typo_length = typo_match

            # Rekonstruiere korrigierten vollen Befehl (verwende Länge des TYPOS, nicht des korrekten Befehls)
            original_rest = user_input[typo_length:].lstrip()

            # Befehl mit ":" (z.B. "lerne:") braucht Leerzeichen nach dem Doppelpunkt
            if correct_command.endswith(":"):
                separator = " " if original_rest else ""
            # Befehl mit Leerzeichen (z.B. "was ist") ist bereits mit Leerzeichen
            elif correct_command.endswith(" "):
                separator = ""
            # Kein Leerzeichen am Ende -> füge eins hinzu
            else:
                separator = " " if original_rest else ""

            corrected_full = correct_command.capitalize() + separator + original_rest

            command_info = self.KNOWN_COMMANDS[correct_command.lower()]

            logger.info(
                "Bekannter Tippfehler erkannt",
                extra={
                    "original": potential_command,
                    "suggestion": correct_command,
                    "confidence": confidence,
                },
            )

            return {
                "original": user_input,
                "suggestion": correct_command,
                "confidence": confidence,
                "full_suggestion": corrected_full,
                "description": command_info["description"],
                "example": command_info["example"],
            }

        # 2. Prüfe mit Fuzzy-Matching (langsamer, niedrigere Confidence)
        fuzzy_match = self._fuzzy_match_command(potential_command)
        if fuzzy_match:
            correct_command, similarity = fuzzy_match

            # Rekonstruiere korrigierten vollen Befehl (verwende Länge des korrekten Befehls)
            command_length = len(correct_command)
            original_rest = user_input[command_length:].lstrip()

            # Befehl mit ":" (z.B. "lerne:") braucht Leerzeichen nach dem Doppelpunkt
            if correct_command.endswith(":"):
                separator = " " if original_rest else ""
            # Befehl mit Leerzeichen (z.B. "was ist") ist bereits mit Leerzeichen
            elif correct_command.endswith(" "):
                separator = ""
            # Kein Leerzeichen am Ende -> füge eins hinzu
            else:
                separator = " " if original_rest else ""

            corrected_full = correct_command.capitalize() + separator + original_rest

            command_info = self.KNOWN_COMMANDS[correct_command.lower()]

            logger.info(
                "Fuzzy-Match gefunden",
                extra={
                    "original": potential_command,
                    "suggestion": correct_command,
                    "similarity": similarity,
                },
            )

            return {
                "original": user_input,
                "suggestion": correct_command,
                "confidence": similarity
                * 0.8,  # Reduzierte Confidence für Fuzzy-Matches
                "full_suggestion": corrected_full,
                "description": command_info["description"],
                "example": command_info["example"],
            }

        # Kein Tippfehler erkannt
        return None

    def _check_known_typos(self, text: str) -> Optional[Tuple[str, float, int]]:
        """
        Prüft gegen bekannte Tippfehler-Liste.

        Returns:
            (korrekter_befehl, confidence, typo_length) oder None
        """
        for correct_cmd, cmd_info in self.KNOWN_COMMANDS.items():
            for typo in cmd_info["typos"]:
                if text.startswith(typo):
                    # Hohe Confidence bei exaktem Typo-Match
                    return (correct_cmd, 0.95, len(typo))

        return None

    def _fuzzy_match_command(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Findet ähnlichste Befehle via Levenshtein-Ähnlichkeit.

        Returns:
            (korrekter_befehl, similarity) oder None
        """
        best_match = None
        best_similarity = 0.0

        for correct_cmd in self.KNOWN_COMMANDS.keys():
            # Extrahiere nur den Befehlsteil (ohne Doppelpunkt/Leerzeichen)
            cmd_prefix = (
                correct_cmd.split()[0]
                if " " in correct_cmd
                else correct_cmd.rstrip(":")
            )

            # Vergleiche mit Anfang der Benutzereingabe
            text_prefix = text.split()[0] if " " in text else text.rstrip(":")

            similarity = SequenceMatcher(None, text_prefix, cmd_prefix).ratio()

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = correct_cmd

        if best_match:
            return (best_match, best_similarity)

        return None

    def get_all_commands(self) -> List[Dict[str, str]]:
        """
        Gibt alle verfügbaren Befehle zurück (für Hilfe-Anzeige).

        Returns:
            Liste von Dicts mit description und example
        """
        return [
            {
                "command": cmd,
                "description": info["description"],
                "example": info["example"],
            }
            for cmd, info in self.KNOWN_COMMANDS.items()
        ]


# Singleton-Instanz für globalen Zugriff
_command_suggester_instance: Optional[CommandSuggester] = None


def get_command_suggester() -> CommandSuggester:
    """
    Gibt Singleton-Instanz des CommandSuggesters zurück.
    """
    global _command_suggester_instance
    if _command_suggester_instance is None:
        _command_suggester_instance = CommandSuggester()
    return _command_suggester_instance


if __name__ == "__main__":
    # Test-Code
    suggester = CommandSuggester()

    test_inputs = [
        "lehrne: ein Hund ist ein Tier",
        "lerne mustre: 'X mag Y' bedeutet LIKES",
        "wa ist ein Hund?",
        "defniere: hund/farbe = braun",
        "wer it Einstein?",
        "Lerne: normaler Befehl",  # Kein Tippfehler
        "irgendwas anderes",  # Kein Befehl
    ]

    print("=== Testing CommandSuggester ===\n")

    for test_input in test_inputs:
        print(f"Input: {test_input}")
        suggestion = suggester.suggest_command(test_input)

        if suggestion:
            print(f"  -> Vorschlag: {suggestion['suggestion']}")
            print(f"  -> Confidence: {suggestion['confidence']:.2f}")
            print(f"  -> Korrigiert: {suggestion['full_suggestion']}")
            print(f"  -> Beschreibung: {suggestion['description']}")
        else:
            print("  -> Kein Vorschlag")

        print()
