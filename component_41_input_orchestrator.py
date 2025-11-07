# component_41_input_orchestrator.py
"""
Input Orchestrator - Intelligente Segmentierung und Verarbeitung komplexer Eingaben

Verantwortlichkeiten:
- Segmentierung von Eingaben in Erklärungen und Fragen
- Semantische Klassifikation von Segmenten
- Erstellung von Multi-Step-Plänen für Logik-Rätsel
- Dynamische Anpassung an verschiedene Eingabetypen

Entwickelt für:
- Logik-Rätsel (Erklärung → Frage)
- Mehrere Erklärungen gefolgt von Fragen
- Natürlichsprachliche Verarbeitung ohne starre Muster
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from component_5_linguistik_strukturen import GoalType, MainGoal, SubGoal
from component_15_logging_config import get_logger

logger = get_logger(__name__)


class SegmentType(Enum):
    """Typ eines Eingabe-Segments."""

    EXPLANATION = "explanation"  # Erklärung, Kontext, deklarative Aussage
    QUESTION = "question"  # Frage, Anfrage
    COMMAND = "command"  # Befehl (explizit)
    UNKNOWN = "unknown"  # Unklarer Typ


@dataclass
class InputSegment:
    """
    Repräsentiert ein Segment der Benutzereingabe.

    Attributes:
        text: Der Text des Segments
        segment_type: Typ des Segments (EXPLANATION, QUESTION, etc.)
        confidence: Konfidenz der Klassifikation (0.0-1.0)
        metadata: Zusätzliche Metadaten (z.B. erkannte Entitäten)
    """

    text: str
    segment_type: SegmentType
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_explanation(self) -> bool:
        """Prüft ob Segment eine Erklärung ist."""
        return self.segment_type == SegmentType.EXPLANATION

    def is_question(self) -> bool:
        """Prüft ob Segment eine Frage ist."""
        return self.segment_type == SegmentType.QUESTION

    def is_command(self) -> bool:
        """Prüft ob Segment ein Befehl ist."""
        return self.segment_type == SegmentType.COMMAND


class InputOrchestrator:
    """
    Orchestriert die Verarbeitung komplexer Eingaben.

    Segmentiert Eingaben in Erklärungen und Fragen, klassifiziert diese
    und erstellt optimierte Multi-Step-Pläne für die Verarbeitung.
    """

    def __init__(self, preprocessor=None):
        """
        Initialisiert den Orchestrator.

        Args:
            preprocessor: Optional - LinguisticPreprocessor für erweiterte Analyse
        """
        self.preprocessor = preprocessor

        # Fragewörter für deutsche Sprache
        self.question_words = [
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

        # Explizite Befehls-Präfixe
        self.command_prefixes = [
            "lerne:",
            "definiere:",
            "ingestiere text:",
            "lerne muster:",
            "lese datei:",
            "ingestiere dokument:",
            "verarbeite pdf:",
            "lade datei:",
        ]

        logger.info("InputOrchestrator initialisiert")

    def should_orchestrate(self, text: str) -> bool:
        """
        Prüft ob eine Eingabe orchestriert werden sollte.

        Kriterien:
        - Mehr als ein Satz
        - Enthält sowohl Erklärungen als auch Fragen
        - Keine expliziten Befehle

        Args:
            text: Die zu prüfende Eingabe

        Returns:
            True wenn Orchestrierung sinnvoll ist
        """
        # Prüfe auf explizite Befehle (diese werden normal verarbeitet)
        text_lower = text.lower().strip()
        for prefix in self.command_prefixes:
            if text_lower.startswith(prefix):
                logger.debug(
                    f"Expliziter Befehl erkannt: {prefix} - überspringe Orchestrierung"
                )
                return False

        # Segmentiere zunächst
        segments = self._segment_text(text)

        # Orchestrierung nur bei mehreren Segmenten
        if len(segments) < 2:
            logger.debug(f"Nur {len(segments)} Segment(e) - überspringe Orchestrierung")
            return False

        # Klassifiziere Segmente
        classified_segments = [self.classify_segment(seg) for seg in segments]

        # Prüfe ob sowohl Erklärungen als auch Fragen vorhanden sind
        has_explanation = any(s.is_explanation() for s in classified_segments)
        has_question = any(s.is_question() for s in classified_segments)

        if has_explanation and has_question:
            logger.info(
                f"Orchestrierung aktiviert: {len(classified_segments)} Segmente, {sum(s.is_explanation() for s in classified_segments)} Erklärungen, {sum(s.is_question() for s in classified_segments)} Fragen"
            )
            return True

        logger.debug(
            "Keine Mischung aus Erklärungen und Fragen - überspringe Orchestrierung"
        )
        return False

    def orchestrate_input(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Orchestriert eine komplexe Eingabe.

        Returns:
            Dictionary mit:
            - segments: Liste von InputSegment
            - plan: MainGoal für orchestrierte Verarbeitung
            - metadata: Zusätzliche Informationen

            None wenn keine Orchestrierung notwendig
        """
        if not self.should_orchestrate(text):
            return None

        # Segmentiere und klassifiziere
        raw_segments = self._segment_text(text)
        segments = [self.classify_segment(seg) for seg in raw_segments]

        # Erstelle orchestrierten Plan
        plan = self._create_orchestrated_plan(segments)

        logger.info(
            f"Orchestrierung abgeschlossen: {len(segments)} Segmente verarbeitet"
        )

        return {
            "segments": segments,
            "plan": plan,
            "metadata": {
                "explanation_count": sum(s.is_explanation() for s in segments),
                "question_count": sum(s.is_question() for s in segments),
                "total_segments": len(segments),
            },
        }

    def _segment_text(self, text: str) -> List[str]:
        """
        Segmentiert Text in Sätze und Absätze.

        Strategie:
        1. Splitte an Absätzen (doppelte Zeilenumbrüche)
        2. Splitte an Satzgrenzen (., !, ?)
        3. Behalte Fragezeichen bei Fragen

        Args:
            text: Der zu segmentierende Text

        Returns:
            Liste von Text-Segmenten
        """
        # Normalisiere Whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Strategie 1: Splitte an doppelten Zeilenumbrüchen (Absätze)
        paragraphs = re.split(r"\n\s*\n", text)

        segments = []
        for paragraph in paragraphs:
            # Strategie 2: Splitte an Satzgrenzen
            # WICHTIG: Behalte Interpunktion (?, !, .) für Klassifikation
            sentence_pattern = r"([^.!?]+[.!?]+)"
            sentences = re.findall(sentence_pattern, paragraph)

            # Fallback: Wenn keine Satzgrenzen gefunden, nimm gesamten Absatz
            if not sentences:
                sentences = [paragraph]

            # Bereinige und füge hinzu
            for sentence in sentences:
                cleaned = sentence.strip()
                if cleaned:
                    segments.append(cleaned)

        logger.debug(f"Text segmentiert: {len(segments)} Segmente")
        return segments

    def classify_segment(self, text: str) -> InputSegment:
        """
        Klassifiziert ein Text-Segment.

        Nutzt mehrere Heuristiken:
        1. Fragezeichen → QUESTION
        2. Fragewörter am Anfang → QUESTION
        3. Explizite Befehle → COMMAND
        4. Deklarative Muster → EXPLANATION
        5. Fallback → EXPLANATION (konservativ)

        Args:
            text: Das zu klassifizierende Segment

        Returns:
            InputSegment mit Klassifikation
        """
        text_lower = text.lower().strip()

        # Heuristik 1: Fragezeichen → QUESTION
        if text_lower.endswith("?"):
            logger.debug(
                f"Segment klassifiziert als QUESTION (Fragezeichen): '{text[:50]}...'"
            )
            return InputSegment(
                text=text,
                segment_type=SegmentType.QUESTION,
                confidence=0.95,  # Hohe Konfidenz bei Fragezeichen
                metadata={"heuristic": "question_mark"},
            )

        # Heuristik 2: Fragewörter am Anfang → QUESTION
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in self.question_words:
            logger.debug(
                f"Segment klassifiziert als QUESTION (Fragewort '{first_word}'): '{text[:50]}...'"
            )
            return InputSegment(
                text=text,
                segment_type=SegmentType.QUESTION,
                confidence=0.90,
                metadata={"heuristic": "question_word", "question_word": first_word},
            )

        # Heuristik 3: Explizite Befehle → COMMAND
        for prefix in self.command_prefixes:
            if text_lower.startswith(prefix):
                logger.debug(
                    f"Segment klassifiziert als COMMAND ('{prefix}'): '{text[:50]}...'"
                )
                return InputSegment(
                    text=text,
                    segment_type=SegmentType.COMMAND,
                    confidence=1.0,  # Maximale Konfidenz bei expliziten Befehlen
                    metadata={"heuristic": "command_prefix", "command": prefix},
                )

        # Heuristik 4: Deklarative Muster → EXPLANATION
        # Prüfe auf typische deklarative Konstruktionen
        declarative_patterns = [
            r"\b(ist ein|ist eine|sind)\b",  # IS_A
            r"\b(kann|können)\b",  # CAPABLE_OF
            r"\b(hat|haben)\b",  # HAS_PROPERTY/PART_OF
            r"\b(liegt in|ist in)\b",  # LOCATED_IN
            r"\b(bedeutet|heißt|meint)\b",  # Definition
        ]

        has_declarative_pattern = any(
            re.search(pattern, text_lower) for pattern in declarative_patterns
        )

        if has_declarative_pattern:
            logger.debug(
                f"Segment klassifiziert als EXPLANATION (deklaratives Muster): '{text[:50]}...'"
            )
            return InputSegment(
                text=text,
                segment_type=SegmentType.EXPLANATION,
                confidence=0.85,
                metadata={"heuristic": "declarative_pattern"},
            )

        # Heuristik 5: Fallback → EXPLANATION (konservativ)
        # Wenn nichts anderes greift, gehe von Erklärung aus
        logger.debug(
            f"Segment klassifiziert als EXPLANATION (Fallback): '{text[:50]}...'"
        )
        return InputSegment(
            text=text,
            segment_type=SegmentType.EXPLANATION,
            confidence=0.70,  # Niedrigere Konfidenz bei Fallback
            metadata={"heuristic": "fallback"},
        )

    def _create_orchestrated_plan(self, segments: List[InputSegment]) -> MainGoal:
        """
        Erstellt einen orchestrierten Plan für segmentierte Eingaben.

        Strategie:
        1. Gruppiere zusammenhängende Erklärungen
        2. Verarbeite alle Erklärungen ZUERST (Lernen)
        3. Verarbeite dann Fragen (Reasoning mit gelerntem Wissen)

        Args:
            segments: Liste klassifizierter Segmente

        Returns:
            MainGoal mit orchestriertem Plan
        """
        # Trenne Erklärungen und Fragen
        explanations = [s for s in segments if s.is_explanation()]
        questions = [s for s in segments if s.is_question()]
        commands = [s for s in segments if s.is_command()]

        # Erstelle Haupt-Ziel
        plan = MainGoal(
            type=GoalType.PERFORM_TASK,
            description=f"Verarbeite {len(explanations)} Erklärung(en) und {len(questions)} Frage(n)",
        )

        # Sub-Goal 1: Verarbeite alle Erklärungen (Lernen)
        if explanations:
            # Kombiniere alle Erklärungen zu einem Text für Batch-Learning
            combined_explanations = ". ".join(e.text.rstrip(".") for e in explanations)

            plan.sub_goals.append(
                SubGoal(
                    description=f"Lerne Kontext: '{combined_explanations[:60]}...'",
                    metadata={
                        "orchestrated_type": "batch_learning",
                        "segment_texts": [e.text for e in explanations],
                        "segment_count": len(explanations),
                    },
                )
            )

        # Sub-Goal 2: Verarbeite alle Befehle
        if commands:
            for cmd in commands:
                plan.sub_goals.append(
                    SubGoal(
                        description=f"Führe Befehl aus: '{cmd.text[:60]}...'",
                        metadata={
                            "orchestrated_type": "command_execution",
                            "segment_text": cmd.text,
                        },
                    )
                )

        # Sub-Goal 3: Beantworte alle Fragen (mit gelerntem Kontext)
        if questions:
            for q in questions:
                plan.sub_goals.append(
                    SubGoal(
                        description=f"Beantworte Frage: '{q.text[:60]}...'",
                        metadata={
                            "orchestrated_type": "question_answering",
                            "segment_text": q.text,
                            "has_learned_context": len(explanations) > 0,
                        },
                    )
                )

        logger.info(
            f"Orchestrierter Plan erstellt: "
            f"{len(explanations)} Erklärungen, "
            f"{len(commands)} Befehle, "
            f"{len(questions)} Fragen"
        )

        return plan

    def get_segment_text(
        self, segment: InputSegment, strip_punctuation: bool = False
    ) -> str:
        """
        Holt den Text eines Segments.

        Args:
            segment: Das Segment
            strip_punctuation: Ob Interpunktion entfernt werden soll

        Returns:
            Text des Segments
        """
        text = segment.text
        if strip_punctuation:
            text = text.rstrip(".!?")
        return text.strip()
