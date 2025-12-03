# kai_inference_abductive.py
"""
Abductive Reasoning Handler für KAI Inference System

Verantwortlichkeiten:
- Hypothesen-Generierung via Abductive Engine
- Hypothesen-Persistierung in Neo4j
- ProofTree-Generierung aus Hypothesen
"""
import logging
from typing import Any, Dict, Optional

from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine import Engine
from kai_exceptions import (
    AbductiveReasoningError,
    InferenceError,
    get_user_friendly_message,
)

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import ProofTree

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    logger.warning("Unified Proof Explanation System nicht verfügbar")


class AbductiveReasoningHandler:
    """
    Handler für Abductive Reasoning und Hypothesen-Generierung.

    Verwaltet:
    - Hypothesen-Generierung mit verschiedenen Strategien
    - Hypothesen-Persistierung in Neo4j
    - ProofTree-Generierung
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        engine: Engine,
        working_memory,  # WorkingMemory
        signals,  # KaiSignals
        fact_loader_callback,  # Callable für load_facts_from_graph
    ):
        """
        Initialisiert den Abductive Reasoning Handler.

        Args:
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            engine: Logic Engine für Backward-Chaining
            working_memory: WorkingMemory für Reasoning-Trace
            signals: KaiSignals für UI-Kommunikation
            fact_loader_callback: Callback-Funktion zum Laden von Fakten
        """
        self.netzwerk = netzwerk
        self.engine = engine
        self.working_memory = working_memory
        self.signals = signals
        self.load_facts_from_graph = fact_loader_callback

        # Lazy-Loading für Abductive Engine
        self._abductive_engine = None
        logger.info("AbductiveReasoningHandler initialisiert")

    @property
    def abductive_engine(self):
        """Lazy-Loading für Abductive Engine."""
        if self._abductive_engine is None:
            try:
                from component_14_abductive_engine import AbductiveEngine

                self._abductive_engine = AbductiveEngine(self.netzwerk, self.engine)
                logger.debug("Abductive Engine erfolgreich initialisiert")
            except Exception as e:
                logger.warning(f"Abductive Engine konnte nicht geladen werden: {e}")
        return self._abductive_engine

    def try_abductive_reasoning(
        self, topic: str, relation_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Versucht Abductive Reasoning zur Hypothesengenerierung.

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit Ergebnissen (markiert als "is_hypothesis") oder None
        """
        logger.info(
            f"[Abductive Reasoning] Versuche Hypothesen-Generierung für '{topic}'"
        )

        if not self.abductive_engine:
            logger.warning("[Abductive Reasoning] Engine nicht verfügbar")
            return None

        try:
            # Lade Kontextfakten für Hypothesengenerierung
            all_facts = self.load_facts_from_graph(topic)

            # Generiere Hypothesen
            observation = f"Beobachtung: Es wurde nach '{topic}' gefragt"
            hypotheses = self.abductive_engine.generate_hypotheses(
                observation=observation,
                context_facts=all_facts,
                strategies=["template", "analogy", "causal_chain"],
                max_hypotheses=5,
            )

            if hypotheses:
                # Nehme die beste Hypothese
                best_hypothesis = hypotheses[0]

                logger.info(
                    f"[Abductive Reasoning] [OK] Hypothese generiert: {best_hypothesis.explanation} "
                    f"(Konfidenz: {best_hypothesis.confidence:.2f})"
                )

                # Speichere Hypothese in Neo4j
                self.netzwerk.store_hypothesis(
                    hypothesis_id=best_hypothesis.id,
                    explanation=best_hypothesis.explanation,
                    observations=best_hypothesis.observations,
                    strategy=best_hypothesis.strategy,
                    confidence=best_hypothesis.confidence,
                    scores=best_hypothesis.scores,
                    abduced_facts=[
                        {"pred": f.pred, "args": f.args, "confidence": f.confidence}
                        for f in best_hypothesis.abduced_facts
                    ],
                    sources=best_hypothesis.sources,
                    reasoning_trace=best_hypothesis.reasoning_trace,
                )

                # Verknüpfe mit Konzept
                self.netzwerk.link_hypothesis_to_concepts(best_hypothesis.id, [topic])

                # Verknüpfe mit Beobachtungen
                self.netzwerk.link_hypothesis_to_observations(
                    best_hypothesis.id, best_hypothesis.observations
                )

                # Extrahiere abgeleitete Fakten
                inferred_facts = {}
                for fact in best_hypothesis.abduced_facts:
                    rel = fact.pred
                    obj = fact.args.get("object", "")
                    if rel not in inferred_facts:
                        inferred_facts[rel] = []
                    if obj and obj not in inferred_facts[rel]:
                        inferred_facts[rel].append(obj)

                # Formatiere Erklärung
                proof_trace = (
                    f"Abductive Reasoning ({best_hypothesis.strategy}):\n"
                    f"{best_hypothesis.explanation}\n"
                    f"Konfidenz: {best_hypothesis.confidence:.2f}\n"
                    f"Bewertung: Coverage={best_hypothesis.scores.get('coverage', 0):.2f}, "
                    f"Simplicity={best_hypothesis.scores.get('simplicity', 0):.2f}, "
                    f"Coherence={best_hypothesis.scores.get('coherence', 0):.2f}, "
                    f"Specificity={best_hypothesis.scores.get('specificity', 0):.2f}"
                )

                # Tracke in Working Memory
                self.working_memory.add_reasoning_state(
                    step_type="abductive_reasoning_success",
                    description=f"Abductive Reasoning erfolgreich für '{topic}'",
                    data={
                        "topic": topic,
                        "method": "abductive",
                        "strategy": best_hypothesis.strategy,
                        "confidence": best_hypothesis.confidence,
                        "hypotheses_generated": len(hypotheses),
                        "inferred_facts": inferred_facts,
                    },
                    confidence=best_hypothesis.confidence,
                )

                # PHASE 2 (Proof Tree): Generiere ProofTree für Abductive Reasoning
                if PROOF_SYSTEM_AVAILABLE:
                    try:
                        proof_tree = ProofTree(query=observation)
                        # Konvertiere alle Hypothesen zu ProofSteps
                        hypothesis_steps = (
                            self.abductive_engine.create_multi_hypothesis_proof_chain(
                                hypotheses[:3],  # Zeige nur Top 3 Hypothesen
                                query=observation,
                            )
                        )
                        for step in hypothesis_steps:
                            proof_tree.add_root_step(step)

                        # Emittiere ProofTree an UI
                        self.signals.proof_tree_update.emit(proof_tree)
                        logger.debug(
                            f"[Proof Tree] Abductive Reasoning ProofTree emittiert ({len(hypothesis_steps)} Hypothesen)"
                        )
                    except InferenceError as e:
                        logger.warning(
                            f"[Proof Tree] InferenceError beim Generieren des ProofTree: {e}",
                            exc_info=True,
                        )
                        user_msg = get_user_friendly_message(e)
                        logger.info(f"[Proof Tree] User-Message: {user_msg}")
                    except Exception as e:
                        logger.warning(
                            f"[Proof Tree] Unerwarteter Fehler beim Generieren des ProofTree: {e}",
                            exc_info=True,
                        )

                return {
                    "inferred_facts": inferred_facts,
                    "proof_trace": proof_trace,
                    "confidence": best_hypothesis.confidence,
                    "is_hypothesis": True,  # Markierung für Antwortgenerierung
                }
            else:
                logger.info(
                    f"[Abductive Reasoning] [X] Keine Hypothesen generiert für '{topic}'"
                )

        except AbductiveReasoningError as e:
            # Spezifischer Fehler beim Abductive Reasoning
            logger.warning(
                f"[Abductive Reasoning] AbductiveReasoningError: {e}", exc_info=True
            )
            user_msg = get_user_friendly_message(e)
            logger.info(f"[Abductive Reasoning] User-Message: {user_msg}")
        except Exception as e:
            # Wrap unerwarteter Fehler in AbductiveReasoningError
            logger.warning(
                f"[Abductive Reasoning] Unerwarteter Fehler: {e}", exc_info=True
            )

        return None
