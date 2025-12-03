# kai_response_production.py
"""
Production System Response Generation für KAI

Verantwortlichkeiten:
- Production System basierte Antwortgenerierung
- Integration mit component_54_production_system
- ProofTree-basierte transparente Generierung
- Signal-basierte UI-Updates

Extracted from kai_response_formatter.py für bessere Modularität.
"""
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProductionResponseGenerator:
    """
    Production System basierter Response Generator.

    Nutzt regelbasierte Generierung statt Template-basierter Pipeline.
    PHASE 5-6: Integration mit ProductionSystemEngine und ProofTree.
    """

    def __init__(self):
        """Initialisiert den Production System Generator."""

    def generate_with_production_system(
        self,
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
        query_type: str = "normal",
        confidence: Optional[float] = None,
        production_engine: Optional[Any] = None,
        signals: Optional[Any] = None,
    ) -> Any:  # Returns KaiResponse
        """
        Generiert Antwort mit Production System (statt Pipeline).

        PHASE 5: Integration mit ResponseFormatter

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen
            synonyms: Liste von Synonymen
            query_type: Typ der Query
            confidence: Optionale Konfidenz
            production_engine: Optional ProductionSystemEngine Instanz
            signals: Optional KaiSignals für UI-Updates (PHASE 5)

        Returns:
            KaiResponse mit Production-System generiertem Text
        """
        from component_54_production_system import (
            GenerationGoal,
            GenerationGoalType,
            ProductionSystemEngine,
            ResponseGenerationState,
            create_all_content_selection_rules,
        )
        from kai_response_pipeline import KaiResponse

        start_time = time.time()

        try:
            # 1. Erstelle Production Engine (falls nicht gegeben)
            if production_engine is None:
                production_engine = ProductionSystemEngine(signals=signals)
                # Füge Standard-Regeln hinzu
                production_engine.add_rules(create_all_content_selection_rules())
                # TODO: Füge Lexicalization-Regeln hinzu in späteren Phasen

            # 2. Konvertiere facts/bedeutungen in available_facts Format
            available_facts = []

            # Bedeutungen als Fakten
            for bed in bedeutungen:
                available_facts.append(
                    {
                        "relation_type": "DEFINITION",  # PHASE 6 FIX: relation -> relation_type
                        "subject": topic,
                        "object": bed,
                        "confidence": confidence or 0.9,
                        "source": "direct",
                    }
                )

            # Synonyme als Fakten
            for syn in synonyms:
                available_facts.append(
                    {
                        "relation_type": "SYNONYM",  # PHASE 6 FIX: relation -> relation_type
                        "subject": topic,
                        "object": syn,
                        "confidence": confidence or 0.85,
                        "source": "direct",
                    }
                )

            # Andere Fakten
            for relation_type, objects in facts.items():
                for obj in objects:
                    available_facts.append(
                        {
                            "relation_type": relation_type,  # PHASE 6 FIX: relation -> relation_type
                            "subject": topic,
                            "object": obj,
                            "confidence": confidence or 0.8,
                            "source": "graph",
                        }
                    )

            # 3. Erstelle initialen State
            primary_goal = GenerationGoal(
                goal_type=GenerationGoalType.ANSWER_QUESTION,
                target_entity=topic,
                constraints={"query_type": query_type},
            )

            state = ResponseGenerationState(
                primary_goal=primary_goal,
                available_facts=available_facts,
                constraints={"max_sentences": 5 if query_type == "normal" else 10},
                current_query=f"Generiere Antwort für: {topic}",  # PHASE 6: Query für ProofTree
            )

            # 4. Generiere Antwort
            final_state = production_engine.generate(state)

            # 5. Extrahiere Ergebnis
            generated_text = final_state.get_full_text()

            # PHASE 6: Extrahiere ProofTree
            proof_tree = final_state.proof_tree

            # 6. Erstelle Trace
            trace = [
                "Production System Generierung gestartet",
                f"Ziel: {primary_goal.goal_type.value}",
                f"Verfügbare Fakten: {len(available_facts)}",
                f"Cycles: {final_state.cycle_count}",
                f"Generierte Sätze: {len(final_state.text.completed_sentences)}",
            ]

            # 7. Berechne durchschnittliche Confidence aus verwendeten Fakten
            if available_facts:
                avg_confidence = sum(
                    f.get("confidence", 0.5) for f in available_facts
                ) / len(available_facts)
            else:
                avg_confidence = confidence or 0.5

            # 8. Response Time
            response_time = time.time() - start_time
            trace.append(f"Generierungszeit: {response_time:.3f}s")

            # PHASE 6: Füge ProofTree-Info zum Trace hinzu
            if proof_tree:
                num_steps = len(proof_tree.get_all_steps())
                trace.append(f"ProofTree: {num_steps} Regelanwendungen")

            # PHASE 6: Optional - Emit ProofTree Signal für UI
            if (
                signals is not None
                and hasattr(signals, "proof_tree_update")
                and proof_tree
            ):
                try:
                    signals.proof_tree_update.emit(proof_tree)
                    logger.debug("ProofTree signal emitted for UI update")
                except Exception as e:
                    logger.debug(f"Could not emit proof_tree_update signal: {e}")

            # 9. Erstelle KaiResponse
            response = KaiResponse(
                text=(
                    generated_text
                    if generated_text
                    else f"Keine Antwort generiert für '{topic}'."
                ),
                trace=trace,
                confidence=avg_confidence,
                strategy="production_system",
                proof_tree=proof_tree,  # PHASE 6: ProofTree inkludieren
            )

            logger.info(
                f"Production System generated response | cycles={final_state.cycle_count}, "
                f"sentences={len(final_state.text.completed_sentences)}, time={response_time:.3f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Error in production system generation: {e}", exc_info=True)

            # Fallback: Erstelle einfache Fehler-Response
            # (Die eigentliche Pipeline-Fallback-Logik ist in KaiResponseFormatter)
            return KaiResponse(
                text=f"Production System Fehler: {e}",
                trace=[f"Production System failed: {e}"],
                confidence=confidence or 0.5,
                strategy="production_error",
            )
