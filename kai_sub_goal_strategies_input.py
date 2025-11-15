# kai_sub_goal_strategies_input.py
"""
Input Processing Strategies

Pattern Learning and Ingestion strategies.

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
from typing import Any, Dict, Tuple

from component_5_linguistik_strukturen import MeaningPointCategory, SubGoal
from kai_sub_goal_strategy_base import SubGoalStrategy

logger = logging.getLogger(__name__)


class PatternLearningStrategy(SubGoalStrategy):
    """
    Strategy für Pattern-Learning Sub-Goals ("Lerne Muster:").

    Zuständig für:
    - Vektor-Erzeugung aus Beispielsätzen
    - Prototyp-Suche/Erstellung
    - Verknüpfung mit Extraktionsregeln
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        pattern_keywords = [
            "Verarbeite Beispielsatz zu Vektor",
            "Finde oder erstelle zugehörigen Muster-Prototypen",
            "Verknüpfe Prototyp mit Extraktionsregel",
        ]
        return any(kw in sub_goal_description for kw in pattern_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Verarbeite Beispielsatz zu Vektor" in description:
            return self._process_sentence_to_vector(intent)
        elif "Finde oder erstelle zugehörigen Muster-Prototypen" in description:
            return self._find_or_create_prototype(intent, context)
        elif "Verknüpfe Prototyp mit Extraktionsregel" in description:
            return self._link_prototype_to_rule(intent, context)

        return False, {"error": f"Unbekanntes Pattern-Learning-SubGoal: {description}"}

    def _process_sentence_to_vector(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """Erstellt Embedding-Vektor aus Beispielsatz."""
        sentence = intent.arguments.get("example_sentence")
        if not sentence:
            return False, {"error": "Beispielsatz nicht gefunden."}

        vector = self.worker.embedding_service.get_embedding(sentence)
        if not vector:
            return False, {"error": "Konnte keinen Embedding-Vektor erstellen."}

        return True, {"sentence_vector": vector}

    def _find_or_create_prototype(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Findet oder erstellt Prototyp für den Vektor."""
        vector = context.get("sentence_vector")
        if not vector:
            return False, {"error": "Vektor fehlt."}

        sentence = intent.arguments.get("example_sentence", "")
        if sentence.strip().endswith("?"):
            category = MeaningPointCategory.QUESTION.value.upper()
        else:
            category = MeaningPointCategory.DEFINITION.value.upper()

        logger.info(f"Kategorie für Beispielsatz abgeleitet: '{category}'")

        prototype_id = self.worker.prototyping_engine.process_vector(vector, category)
        if not prototype_id:
            return False, {"error": "Prototyp konnte nicht erstellt werden."}

        return True, {"prototype_id": prototype_id}

    def _link_prototype_to_rule(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verknüpft Prototyp mit Extraktionsregel."""
        prototype_id = context.get("prototype_id")
        relation_type = intent.arguments.get("relation_type")

        if not all([prototype_id, relation_type]):
            return False, {"error": "Daten fehlen."}

        success = self.worker.netzwerk.link_prototype_to_rule(
            prototype_id, relation_type
        )
        if not success:
            return False, {
                "error": f"Verknüpfung mit Regel '{relation_type}' fehlgeschlagen."
            }

        return True, {"linked_relation": relation_type}


# ============================================================================
# INGESTION STRATEGY (Ingestiere Text)
# ============================================================================


class IngestionStrategy(SubGoalStrategy):
    """
    Strategy für Text-Ingestion Sub-Goals.

    Zuständig für:
    - Text-Extraktion
    - Satz-Verarbeitung durch Ingestion-Pipeline
    - Berichts-Formulierung
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        ingestion_keywords = [
            "Extrahiere den zu ingestierenden Text",
            "Verarbeite Sätze durch die Ingestion-Pipeline",
            "Formuliere einen Ingestion-Bericht",
        ]
        return any(kw in sub_goal_description for kw in ingestion_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Extrahiere den zu ingestierenden Text" in description:
            return self._extract_text(intent)
        elif "Verarbeite Sätze durch die Ingestion-Pipeline" in description:
            return self._process_ingestion(context)
        elif "Formuliere einen Ingestion-Bericht" in description:
            return self._formulate_report(context)

        return False, {"error": f"Unbekanntes Ingestion-SubGoal: {description}"}

    def _extract_text(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """Extrahiert den zu ingestierenden Text."""
        text = intent.arguments.get("text_to_ingest")
        if not text:
            return False, {"error": "Kein Text zum Ingestieren gefunden."}
        return True, {"text_to_process": text}

    def _process_ingestion(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verarbeitet Text durch Ingestion-Pipeline."""
        from kai_ingestion_handler import KaiIngestionHandler

        text = context.get("text_to_process")
        if not text:
            return False, {"error": "Text aus vorigem Schritt fehlt."}

        # Erstelle Ingestion Handler
        ingestion_handler = KaiIngestionHandler(
            self.worker.netzwerk,
            self.worker.preprocessor,
            self.worker.prototyping_engine,
            self.worker.embedding_service,
        )

        stats = ingestion_handler.ingest_text(text)

        return True, {
            "facts_learned_count": stats["facts_created"],
            "learned_patterns": stats["learned_patterns"],
            "fallback_patterns": stats["fallback_patterns"],
        }

    def _formulate_report(self, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Formuliert Ingestion-Bericht."""
        count = context.get("facts_learned_count", 0)
        learned = context.get("learned_patterns", 0)
        fallback = context.get("fallback_patterns", 0)

        if count > 1:
            response = f"Ingestion abgeschlossen. Ich habe {count} neue Fakten gelernt"
        elif count == 1:
            response = "Ingestion abgeschlossen. Ich habe 1 neuen Fakt gelernt"
        else:
            response = (
                "Ingestion abgeschlossen. Ich konnte keine neuen Fakten extrahieren"
            )

        if learned > 0 or fallback > 0:
            details = []
            if learned > 0:
                details.append(f"{learned} aus gelernten Mustern")
            if fallback > 0:
                details.append(f"{fallback} aus neuen Mustern")
            response += f" ({', '.join(details)})"

        response += "."
        return True, {"final_response": response}


# ============================================================================
# DEFINITION STRATEGY (Definiere)
# ============================================================================
