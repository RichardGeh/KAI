# kai_inference_backward_chaining.py
"""
Backward Chaining Handler für KAI Inference System

Verantwortlichkeiten:
- Regelbasiertes Backward-Chaining mit Logic Engine
- Laden von Fakten aus Neo4j Graph
- Extraktion von Fakten aus Proof Steps
- Confidence-Decay für zeitbasierte Validierung
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine import Engine, Fact, Goal
from component_confidence_manager import get_confidence_manager
from kai_exceptions import InferenceError, get_user_friendly_message

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_9_logik_engine_proof import create_proof_tree_from_logic_engine

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    logger.warning("Unified Proof Explanation System nicht verfügbar")


class BackwardChainingHandler:
    """
    Handler für regelbasiertes Backward-Chaining mit Logic Engine.

    Verwaltet:
    - Backward-Chaining Inference
    - Fact-Loading aus Neo4j mit Confidence-Decay
    - Fact-Extraktion aus Proof Steps
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        engine: Engine,
        working_memory,  # WorkingMemory
        signals,  # KaiSignals
    ):
        """
        Initialisiert den Backward-Chaining Handler.

        Args:
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            engine: Logic Engine für Backward-Chaining
            working_memory: WorkingMemory für Reasoning-Trace
            signals: KaiSignals für UI-Kommunikation
        """
        self.netzwerk = netzwerk
        self.engine = engine
        self.working_memory = working_memory
        self.signals = signals

        # PHASE: Confidence-Based Learning
        self.confidence_manager = get_confidence_manager()
        logger.info("BackwardChainingHandler initialisiert mit ConfidenceManager")

    def try_backward_chaining(
        self, topic: str, relation_type: str
    ) -> Dict[str, Any] | None:
        """
        Versucht regelbasiertes Backward-Chaining.

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit Ergebnissen oder None
        """
        logger.info("[Backward-Chaining] Versuche regelbasiertes Backward-Chaining...")

        # Erstelle Goal für Backward-Chaining
        # Beispiel: "Was ist ein Hund?" -> Goal: IS_A(hund, ?x)
        goal = Goal(
            pred=relation_type,
            args={"subject": topic.lower(), "object": None},  # Object unbekannt
        )

        # Lade bekannte Fakten in die Engine
        # Hole alle Fakten aus dem Graph und wandle sie in Engine-Facts um
        all_facts = self.load_facts_from_graph(topic)

        for fact in all_facts:
            self.engine.add_fact(fact)

        # EPISODIC MEMORY FOR REASONING: Verwende tracked version
        # Dies erstellt eine InferenceEpisode und persistiert den Beweisbaum
        query_text = f"Was ist ein {topic}?"
        proof = self.engine.run_with_tracking(
            goal=goal, inference_type="backward_chaining", query=query_text, max_depth=5
        )

        if proof:
            logger.info(f"[Backward-Chaining] [OK] Beweis gefunden für '{topic}'")

            # Extrahiere abgeleitete Fakten aus dem Beweis
            inferred_facts = self.extract_facts_from_proof(proof)
            proof_trace = self.engine.format_proof_trace(proof)

            # Tracke in Working Memory
            self.working_memory.add_reasoning_state(
                step_type="backward_chaining_success",
                description=f"Multi-Hop-Schlussfolgerung erfolgreich für '{topic}'",
                data={
                    "topic": topic,
                    "method": proof.method,
                    "confidence": proof.confidence,
                    "inferred_facts": inferred_facts,
                },
                confidence=proof.confidence,
            )

            # PHASE 2 (Proof Tree): Generiere ProofTree für Backward-Chaining
            if PROOF_SYSTEM_AVAILABLE:
                try:
                    proof_tree = create_proof_tree_from_logic_engine(
                        proof, query=query_text
                    )
                    # Emittiere ProofTree an UI
                    self.signals.proof_tree_update.emit(proof_tree)
                    logger.debug(
                        f"[Proof Tree] Backward-Chaining ProofTree emittiert ({len(proof_tree.get_all_steps())} Schritte)"
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
                "confidence": proof.confidence,
            }

        logger.info(
            f"[Backward-Chaining] [X] Keine Schlussfolgerung möglich für '{topic}'"
        )
        return None

    def load_facts_from_graph(self, topic: str) -> List[Fact]:
        """
        Lädt relevante Fakten aus dem Neo4j-Graphen und wandelt sie in Engine-Facts um.

        UPDATED: Wendet Confidence-Decay für zeitbasierte Reduktion veralteter Fakten an

        Args:
            topic: Das zentrale Thema, für das Fakten geladen werden sollen

        Returns:
            Liste von Fact-Objekten mit angewendetem Confidence-Decay
        """
        facts = []

        # Lade direkte Fakten über das Thema MIT Confidence und Timestamps
        # FALLBACK: query_graph_for_facts_with_confidence existiert noch nicht in KonzeptNetzwerk
        # TODO: Implementiere diese Methode in component_1_netzwerk_core.py
        # Für jetzt: verwende query_graph_for_facts mit Dummy-Confidence
        try:
            fact_data_with_confidence = (
                self.netzwerk.query_graph_for_facts_with_confidence(topic)
            )
        except AttributeError:
            # Fallback: Konvertiere normale Facts zu Format mit Confidence
            logger.debug(
                "[Confidence-Decay] query_graph_for_facts_with_confidence nicht verfügbar, verwende Fallback"
            )
            fact_data = self.netzwerk.query_graph_for_facts(topic)
            fact_data_with_confidence = {}
            for relation_type, objects in fact_data.items():
                fact_data_with_confidence[relation_type] = [
                    {"target": obj, "confidence": 1.0, "timestamp": None}
                    for obj in objects
                ]

        for relation_type, targets_with_conf in fact_data_with_confidence.items():
            for target_info in targets_with_conf:
                target = target_info.get("target", "")
                fact_confidence = target_info.get("confidence", 1.0)
                # Timestamp falls vorhanden (TODO: Neo4j muss timestamps speichern)
                timestamp_str = target_info.get("timestamp")

                # PHASE: Confidence-Based Learning - Wende Confidence-Decay an
                if timestamp_str:
                    try:
                        # Parse timestamp (ISO format erwartet)
                        fact_timestamp = datetime.fromisoformat(timestamp_str)

                        # Wende Decay an
                        decay_metrics = self.confidence_manager.apply_decay(
                            fact_confidence, fact_timestamp
                        )

                        final_confidence = decay_metrics.value

                        if decay_metrics.decay_applied:
                            logger.debug(
                                f"[Confidence-Decay] {relation_type}({topic}, {target}): "
                                f"{fact_confidence:.3f} -> {final_confidence:.3f}"
                            )
                    except (ValueError, AttributeError) as e:
                        # Fallback wenn Timestamp nicht parsebar
                        logger.warning(
                            f"Konnte Timestamp nicht parsen für {topic}-{target}: {e}"
                        )
                        final_confidence = fact_confidence
                else:
                    # Kein Timestamp vorhanden - verwende ursprüngliche Confidence
                    final_confidence = fact_confidence

                fact = Fact(
                    pred=relation_type,
                    args={"subject": topic.lower(), "object": target.lower()},
                    confidence=final_confidence,
                    source="graph",
                )
                facts.append(fact)

        # Lade auch verwandte Fakten (1-Hop Nachbarn)
        # Dies hilft bei Multi-Hop-Reasoning
        for relation_type, targets_with_conf in fact_data_with_confidence.items():
            for target_info in targets_with_conf:
                target = target_info.get("target", "")
                # Lade Fakten über das Ziel-Objekt MIT Confidence
                try:
                    obj_facts_with_conf = (
                        self.netzwerk.query_graph_for_facts_with_confidence(target)
                    )
                except AttributeError:
                    # Fallback wie oben
                    obj_fact_data = self.netzwerk.query_graph_for_facts(target)
                    obj_facts_with_conf = {}
                    for obj_relation, obj_objects in obj_fact_data.items():
                        obj_facts_with_conf[obj_relation] = [
                            {"target": obj, "confidence": 1.0, "timestamp": None}
                            for obj in obj_objects
                        ]

                for obj_relation, obj_targets_with_conf in obj_facts_with_conf.items():
                    for obj_target_info in obj_targets_with_conf:
                        obj_target = obj_target_info.get("target", "")
                        obj_confidence = obj_target_info.get("confidence", 1.0)
                        obj_timestamp_str = obj_target_info.get("timestamp")

                        # PHASE: Confidence-Based Learning - Wende Decay auch hier an
                        if obj_timestamp_str:
                            try:
                                obj_timestamp = datetime.fromisoformat(
                                    obj_timestamp_str
                                )
                                decay_metrics = self.confidence_manager.apply_decay(
                                    obj_confidence, obj_timestamp
                                )
                                final_obj_confidence = decay_metrics.value
                            except (ValueError, AttributeError):
                                final_obj_confidence = obj_confidence
                        else:
                            final_obj_confidence = obj_confidence

                        fact = Fact(
                            pred=obj_relation,
                            args={
                                "subject": target.lower(),
                                "object": obj_target.lower(),
                            },
                            confidence=final_obj_confidence,
                            source="graph",
                        )
                        facts.append(fact)

        logger.debug(f"[Backward-Chaining] Geladen: {len(facts)} Fakten für '{topic}'")
        return facts

    def extract_facts_from_proof(self, proof) -> Dict[str, List[str]]:
        """
        Extrahiert strukturierte Fakten aus einem ProofStep.

        Args:
            proof: ProofStep-Objekt mit Beweisbaum

        Returns:
            Dictionary mit Relation-Types und zugehörigen Objekten
        """
        facts = {}

        # Sammle Fakten aus dem Beweis
        if proof.supporting_facts:
            for fact in proof.supporting_facts:
                relation = fact.pred
                obj = fact.args.get("object", "")

                if relation not in facts:
                    facts[relation] = []

                if obj and obj not in facts[relation]:
                    facts[relation].append(obj)

        # Rekursiv durch Subgoals
        for subproof in proof.subgoals:
            subfacts = self.extract_facts_from_proof(subproof)
            for relation, objects in subfacts.items():
                if relation not in facts:
                    facts[relation] = []
                facts[relation].extend([o for o in objects if o not in facts[relation]])

        return facts
