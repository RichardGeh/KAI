# kai_sub_goal_strategies_dialogue.py
"""
Dialogue Management Strategies

Confirmation and Clarification strategies.

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
from typing import Any, Dict, Tuple

from component_5_linguistik_strukturen import (
    ContextAction,
    MeaningPointCategory,
    SubGoal,
)
from kai_sub_goal_strategy_base import SubGoalStrategy

logger = logging.getLogger(__name__)


class ConfirmationStrategy(SubGoalStrategy):
    """
    Strategy für Bestätigungs-Sub-Goals.

    Zuständig für:
    - Absichts-Bestätigung bei mittlerer Confidence
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        return "Bestätige die erkannte Absicht" in sub_goal_description

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        category = intent.category.value if intent.category else "unbekannt"
        confidence = intent.confidence

        # Spezielle Bestätigung für auto-erkannte Definitionen
        if intent.category == MeaningPointCategory.DEFINITION and intent.arguments.get(
            "auto_detected"
        ):
            subject = intent.arguments.get("subject", "etwas")
            relation_type = intent.arguments.get("relation_type", "IS_A")
            obj = intent.arguments.get("object", "etwas")

            relation_map = {
                "IS_A": f"'{subject}' ist ein '{obj}'",
                "HAS_PROPERTY": f"'{subject}' hat die Eigenschaft '{obj}'",
                "CAPABLE_OF": f"'{subject}' kann '{obj}'",
                "PART_OF": f"'{subject}' hat/gehört zu '{obj}'",
                "LOCATED_IN": f"'{subject}' liegt in '{obj}'",
            }

            fact_description = relation_map.get(
                relation_type, f"'{subject}' {relation_type} '{obj}'"
            )
            response = (
                f"Soll ich mir merken, dass {fact_description}? "
                f"(Konfidenz: {confidence:.0%})"
            )
        elif intent.category == MeaningPointCategory.QUESTION:
            topic = intent.arguments.get("topic", "etwas")
            action_desc = f"eine Frage über '{topic}' beantworten"
            response = (
                f"Ich glaube, du möchtest {action_desc}. "
                f"Ist das richtig? (Konfidenz: {confidence:.0%})"
            )
        elif intent.category == MeaningPointCategory.COMMAND:
            command = intent.arguments.get("command", "Befehl")
            action_desc = f"den Befehl '{command}' ausführen"
            response = (
                f"Ich glaube, du möchtest {action_desc}. "
                f"Ist das richtig? (Konfidenz: {confidence:.0%})"
            )
        else:
            action_desc = f"etwas vom Typ '{category}' verarbeiten"
            response = (
                f"Ich glaube, du möchtest {action_desc}. "
                f"Ist das richtig? (Konfidenz: {confidence:.0%})"
            )

        # Setze Kontext für nächste Eingabe
        self.worker.context.aktion = ContextAction.ERWARTE_BESTAETIGUNG
        self.worker.context.original_intent = intent
        self.worker.context.metadata["sub_goal_context"] = context

        self.worker._emit_context_update()

        logger.info(
            f"Confirmation requested: category={category}, confidence={confidence:.2f}"
        )

        return True, {"final_response": response}


# ============================================================================
# CLARIFICATION STRATEGY (Klärung)
# ============================================================================


class ClarificationStrategy(SubGoalStrategy):
    """
    Strategy für Klärungs-Sub-Goals.

    Zuständig für:
    - Rückfragen bei niedriger Confidence oder UNKNOWN-Kategorie
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        return (
            "Formuliere eine allgemeine Rückfrage zur Klärung" in sub_goal_description
        )

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        original_text = intent.text_span if intent.text_span else ""

        # Setze Kontext für Feedback-Loop
        self.worker.context.aktion = ContextAction.ERWARTE_FEEDBACK_ZU_CLARIFICATION
        self.worker.context.thema = "clarification_feedback"
        self.worker.context.original_intent = intent
        self.worker.context.metadata["original_query"] = original_text

        self.worker._emit_context_update()

        response = (
            "Ich bin nicht sicher, was du meinst. "
            "Kannst du es anders formulieren oder ein Beispiel geben? "
            "Du kannst auch mit 'Lerne Muster: \"...\" bedeutet KATEGORIE' ein Beispiel geben."
        )

        logger.info(f"Clarification requested for: '{original_text[:50]}...'")

        return True, {"final_response": response}


# ============================================================================
# FILE READER STRATEGY (Lese Datei)
# ============================================================================
