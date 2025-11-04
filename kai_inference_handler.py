# kai_inference_handler.py
"""
Inference Handler Module für KAI

Verantwortlichkeiten:
- Backward-Chaining Inference mit Logic Engine
- Graph-Traversal für Multi-Hop-Reasoning
- Abductive Reasoning für Hypothesengenerierung
- Proof Tree Generierung für Erklärbarkeit

PHASE: Confidence-Based Learning Integration
- Verwendet ConfidenceManager für einheitliche Confidence-Berechnung
- Confidence-Decay für veraltete Fakten
- Threshold-basierte Reasoning-Entscheidungen
"""
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine import Engine, Goal, Fact
from component_confidence_manager import get_confidence_manager

# Import exception utilities for user-friendly error messages
from kai_exceptions import (
    GraphTraversalError,
    InferenceError,
    AbductiveReasoningError,
    get_user_friendly_message,
)

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import (
        ProofTree,
        create_proof_tree_from_logic_engine,
    )

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    logger.warning("Unified Proof Explanation System nicht verfügbar")


class KaiInferenceHandler:
    """
    Handler für komplexe Schlussfolgerungen (Backward-Chaining, Graph-Traversal, Abductive Reasoning).

    Diese Klasse verwaltet:
    - Backward-Chaining mit Logic Engine
    - Graph-Traversal für transitive Relationen
    - Abductive Reasoning für Hypothesen
    - Proof Tree Generierung und Signale
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        engine: Engine,
        graph_traversal,  # GraphTraversal
        working_memory,  # WorkingMemory
        signals,  # KaiSignals
        enable_hybrid_reasoning: bool = True,
    ):
        """
        Initialisiert den Inference Handler.

        Args:
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            engine: Logic Engine für Backward-Chaining
            graph_traversal: GraphTraversal-Engine für Multi-Hop-Reasoning
            working_memory: WorkingMemory für Reasoning-Trace
            signals: KaiSignals für UI-Kommunikation (proof_tree_update)
            enable_hybrid_reasoning: Aktiviert Hybrid Reasoning Orchestrator (default: True)
        """
        self.netzwerk = netzwerk
        self.engine = engine
        self.graph_traversal = graph_traversal
        self.working_memory = working_memory
        self.signals = signals

        # Lazy-Loading für Abductive Engine (nur bei Bedarf)
        self._abductive_engine = None

        # PHASE: Confidence-Based Learning - Integriere ConfidenceManager
        self.confidence_manager = get_confidence_manager()
        logger.info("InferenceHandler initialisiert mit ConfidenceManager")

        # COMBINATORIAL REASONING: Integriere CombinatorialReasoner
        self._combinatorial_reasoner = None
        try:
            from component_40_combinatorial_reasoning import CombinatorialReasoner

            self._combinatorial_reasoner = CombinatorialReasoner()
            logger.info("CombinatorialReasoner successfully integrated")
        except ImportError:
            logger.warning("CombinatorialReasoner nicht verfügbar")

        # HYBRID REASONING: Integriere ReasoningOrchestrator
        self.enable_hybrid_reasoning = enable_hybrid_reasoning
        self._reasoning_orchestrator = None

        if self.enable_hybrid_reasoning:
            try:
                from kai_reasoning_orchestrator import ReasoningOrchestrator

                self._reasoning_orchestrator = ReasoningOrchestrator(
                    netzwerk=self.netzwerk,
                    logic_engine=self.engine,
                    graph_traversal=self.graph_traversal,
                    combinatorial_reasoner=self._combinatorial_reasoner,
                    working_memory=self.working_memory,
                    signals=self.signals,
                    probabilistic_engine=None,  # Lazy-loaded
                    abductive_engine=None,  # Lazy-loaded via property
                )
                logger.info("[OK] Hybrid Reasoning Orchestrator aktiviert")
            except Exception as e:
                logger.warning(f"Konnte Hybrid Reasoning nicht aktivieren: {e}")
                self.enable_hybrid_reasoning = False

    @property
    def abductive_engine(self):
        """Lazy-Loading für Abductive Engine."""
        if self._abductive_engine is None:
            try:
                from component_14_abductive_engine import AbductiveEngine

                self._abductive_engine = AbductiveEngine(self.netzwerk, self.engine)
                logger.debug("Abductive Engine erfolgreich initialisiert")
                # Update Orchestrator if exists
                if self._reasoning_orchestrator:
                    self._reasoning_orchestrator.abductive_engine = (
                        self._abductive_engine
                    )
            except Exception as e:
                logger.warning(f"Abductive Engine konnte nicht geladen werden: {e}")
        return self._abductive_engine

    @property
    def probabilistic_engine(self):
        """Lazy-Loading für Probabilistic Engine."""
        if (
            not hasattr(self, "_probabilistic_engine")
            or self._probabilistic_engine is None
        ):
            try:
                from component_16_probabilistic_engine import ProbabilisticEngine

                self._probabilistic_engine = ProbabilisticEngine()
                logger.debug("Probabilistic Engine erfolgreich initialisiert")
                # Update Orchestrator if exists
                if self._reasoning_orchestrator:
                    self._reasoning_orchestrator.probabilistic_engine = (
                        self._probabilistic_engine
                    )
            except Exception as e:
                logger.warning(f"Probabilistic Engine konnte nicht geladen werden: {e}")
                self._probabilistic_engine = None
        return self._probabilistic_engine

    def try_backward_chaining_inference(
        self, topic: str, relation_type: str = "IS_A"
    ) -> Optional[Dict[str, Any]]:
        """
        PHASE 3 & 7: Versucht eine Frage durch Backward-Chaining und Multi-Hop-Reasoning zu beantworten.

        Diese Methode wird aufgerufen, wenn die direkte Graph-Abfrage keine Ergebnisse liefert.

        HYBRID REASONING (NEW):
        - Wenn aktiviert, nutzt ReasoningOrchestrator für kombiniertes Reasoning
        - Kombiniert Logic + Graph + Probabilistic + Abductive
        - Weighted Confidence Fusion

        LEGACY FALLBACK:
        Sie versucht komplexere Schlussfolgerungen durch:
        1. Graph-Traversal für transitive Relationen (PHASE 7)
        2. Regelbasiertes Backward-Chaining
        3. Abductive Reasoning für Hypothesen (Fallback)

        Args:
            topic: Das Thema der Frage (z.B. "hund")
            relation_type: Der Typ der gesuchten Relation (default: "IS_A")

        Returns:
            Dictionary mit "inferred_facts", "proof_trace", "confidence" und optional "is_hypothesis"
            oder None wenn keine Schlussfolgerung möglich ist
        """
        logger.info(
            f"[Multi-Hop Reasoning] Versuche komplexe Schlussfolgerung für '{topic}'"
        )

        # HYBRID REASONING PATH (NEW)
        if self.enable_hybrid_reasoning and self._reasoning_orchestrator:
            logger.info("[Multi-Hop Reasoning] -> Nutze Hybrid Reasoning Orchestrator")
            result = self.try_hybrid_reasoning(topic, relation_type)
            if result:
                return result
            else:
                logger.info(
                    "[Multi-Hop Reasoning] Hybrid Reasoning lieferte keine Ergebnisse, falle zurück auf Legacy"
                )

        # LEGACY FALLBACK PATH (ORIGINAL)

        # PHASE 7: Versuche zuerst Graph-Traversal für transitive Relationen
        # Dies ist effizienter und direkter als regelbasiertes Backward-Chaining
        result = self._try_graph_traversal(topic, relation_type)
        if result:
            return result

        # PHASE 3: Fallback auf regelbasiertes Backward-Chaining
        result = self._try_backward_chaining(topic, relation_type)
        if result:
            return result

        # ABDUCTIVE REASONING: Wenn alles fehlschlägt, versuche Hypothesen zu generieren
        result = self._try_abductive_reasoning(topic, relation_type)
        if result:
            return result

        return None

    def try_hybrid_reasoning(
        self, topic: str, relation_type: str = "IS_A"
    ) -> Optional[Dict[str, Any]]:
        """
        HYBRID REASONING: Nutzt ReasoningOrchestrator für kombiniertes Reasoning.

        Kombiniert mehrere Reasoning-Strategien und aggregiert Ergebnisse:
        - Direct Facts (Fast Path)
        - Graph Traversal (Multi-Hop)
        - Logic Engine (Rule-based)
        - Probabilistic Enhancement
        - Abductive Fallback (Hypotheses)

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit aggregierten Ergebnissen oder None
        """
        if not self._reasoning_orchestrator:
            logger.warning("[Hybrid Reasoning] Orchestrator nicht verfügbar")
            return None

        try:
            # Lazy-load engines falls noch nicht geschehen
            if self.probabilistic_engine:
                pass  # Property triggers lazy loading
            if self.abductive_engine:
                pass  # Property triggers lazy loading

            # Query mit Hybrid Reasoning
            aggregated_result = (
                self._reasoning_orchestrator.query_with_hybrid_reasoning(
                    topic=topic,
                    relation_type=relation_type,
                    strategies=None,  # None = alle verfügbaren Strategien
                )
            )

            if aggregated_result:
                logger.info(
                    f"[Hybrid Reasoning] [OK] Erfolg mit {len(aggregated_result.strategies_used)} Strategien "
                    f"(Konfidenz: {aggregated_result.combined_confidence:.2f})"
                )

                # Konvertiere AggregatedResult zu Legacy-Format
                return {
                    "inferred_facts": aggregated_result.inferred_facts,
                    "proof_trace": aggregated_result.explanation,
                    "confidence": aggregated_result.combined_confidence,
                    "is_hypothesis": aggregated_result.is_hypothesis,
                    "hybrid": True,
                    "strategies_used": [
                        str(s.value) for s in aggregated_result.strategies_used
                    ],
                    "num_strategies": len(aggregated_result.strategies_used),
                }

            return None

        except Exception as e:
            logger.error(f"[Hybrid Reasoning] Fehler: {e}", exc_info=True)
            return None

    def _try_graph_traversal(
        self, topic: str, relation_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Versucht Graph-Traversal für transitive Relationen.

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit Ergebnissen oder None
        """
        logger.info(
            f"[Graph-Traversal] Versuche transitive Relationen für '{topic}' ({relation_type})"
        )

        try:
            # Finde alle transitiven Relationen des gesuchten Typs
            paths = self.graph_traversal.find_transitive_relations(
                topic, relation_type, max_depth=5
            )

            if paths:
                # Extrahiere alle Zielknoten aus den gefundenen Pfaden
                inferred_facts = {relation_type: []}
                for path in paths:
                    # Der letzte Knoten im Pfad ist das Ziel
                    target = path.nodes[-1]
                    if target not in inferred_facts[relation_type]:
                        inferred_facts[relation_type].append(target)

                # Generiere Erklärungstrace aus den Pfaden
                proof_trace_parts = []
                for i, path in enumerate(paths[:3], 1):  # Zeige nur erste 3 Pfade
                    proof_trace_parts.append(f"Pfad {i}: {path.explanation}")

                proof_trace = "\n".join(proof_trace_parts)

                # PHASE: Confidence-Based Learning - Berechne Confidence für Pfade
                # Verwende ConfidenceManager für Graph-Traversal-Confidence
                # Nutze die Confidence des besten (kürzesten) Pfads
                best_path = paths[0] if paths else None
                best_path.confidence if best_path else 1.0

                # Zusätzliche Validierung durch ConfidenceManager
                confidence_metrics = (
                    self.confidence_manager.calculate_graph_traversal_confidence(
                        [path.confidence for path in paths[:5]]  # Betrachte Top 5 Pfade
                    )
                )

                logger.info(
                    f"[Graph-Traversal] [OK] {len(inferred_facts[relation_type])} Fakten gefunden via Traversal "
                    f"(Confidence: {confidence_metrics.value:.2f})"
                )

                # Tracke in Working Memory
                self.working_memory.add_reasoning_state(
                    step_type="graph_traversal_success",
                    description=f"Graph-Traversal erfolgreich für '{topic}'",
                    data={
                        "topic": topic,
                        "method": "graph_traversal",
                        "relation_type": relation_type,
                        "num_paths": len(paths),
                        "inferred_facts": inferred_facts,
                        "confidence_explanation": confidence_metrics.explanation,
                    },
                    confidence=confidence_metrics.value,
                )

                # PHASE 2 (Proof Tree): Generiere ProofTree für Graph-Traversal
                if PROOF_SYSTEM_AVAILABLE:
                    try:
                        proof_tree = ProofTree(query=f"Was ist ein {topic}?")
                        # Konvertiere Pfade zu ProofSteps
                        for path in paths[:5]:  # Zeige nur erste 5 Pfade
                            proof_step = (
                                self.graph_traversal.create_proof_step_from_path(
                                    path, query=f"{topic} {relation_type}"
                                )
                            )
                            if proof_step:
                                proof_tree.add_root_step(proof_step)

                        # Emittiere ProofTree an UI
                        self.signals.proof_tree_update.emit(proof_tree)
                        logger.debug(
                            f"[Proof Tree] Graph-Traversal ProofTree emittiert ({len(proof_tree.root_steps)} Pfade)"
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
                    "confidence": confidence_metrics.value,
                }
            else:
                logger.info(
                    f"[Graph-Traversal] [X] Keine transitiven Relationen gefunden"
                )

        except GraphTraversalError as e:
            # Spezifischer Fehler bei Graph-Traversierung
            logger.warning(f"[Graph-Traversal] GraphTraversalError: {e}", exc_info=True)
            # Benutzerfreundliche Nachricht loggen
            user_msg = get_user_friendly_message(e)
            logger.info(f"[Graph-Traversal] User-Message: {user_msg}")
            # Graceful Degradation: Fallback auf Backward-Chaining
        except Exception as e:
            # Unerwarteter Fehler - wrap in GraphTraversalError
            logger.warning(
                f"[Graph-Traversal] Unerwarteter Fehler: {e}, falle zurück auf Backward-Chaining",
                exc_info=True,
            )

        return None

    def _try_backward_chaining(
        self, topic: str, relation_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Versucht regelbasiertes Backward-Chaining.

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit Ergebnissen oder None
        """
        logger.info(f"[Backward-Chaining] Versuche regelbasiertes Backward-Chaining...")

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

    def _try_abductive_reasoning(
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
