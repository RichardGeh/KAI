# kai_inference_handler.py
"""
Inference Handler Module für KAI (REFACTORED)

Verantwortlichkeiten:
- Koordination verschiedener Inference-Strategien
- Hybrid Reasoning Orchestration
- Delegation an spezialisierte Handler

ARCHITECTURAL REFACTORING (2025-12-01):
- Aufgeteilt in 4 spezialisierte Handler-Module
- Main Handler delegiert an: BackwardChainingHandler, GraphTraversalHandler,
  AbductiveReasoningHandler, ResonanceInferenceHandler
- 100% Backward-Kompatibilität durch Delegation
"""
import logging
from typing import Any, Dict, Optional

from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine import Engine
from component_confidence_manager import get_confidence_manager
from kai_inference_abductive import AbductiveReasoningHandler

# Import specialized handlers
from kai_inference_backward_chaining import BackwardChainingHandler
from kai_inference_graph_traversal import GraphTraversalHandler
from kai_inference_resonance import ResonanceInferenceHandler

logger = logging.getLogger(__name__)


class KaiInferenceHandler:
    """
    Koordinator für komplexe Schlussfolgerungen (Backward-Chaining, Graph-Traversal, Abductive, Resonance).

    Diese Klasse delegiert an spezialisierte Handler:
    - BackwardChainingHandler: Regelbasiertes Backward-Chaining
    - GraphTraversalHandler: Multi-Hop Graph-Traversal
    - AbductiveReasoningHandler: Hypothesen-Generierung
    - ResonanceInferenceHandler: Activation Spreading QA
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

        # SPATIAL REASONING: Integriere SpatialReasoner
        self._spatial_reasoner = None
        try:
            from component_42_spatial_reasoning import SpatialReasoner

            self._spatial_reasoner = SpatialReasoner(netzwerk=self.netzwerk)
            logger.info("SpatialReasoner successfully integrated")
        except ImportError:
            logger.warning("SpatialReasoner nicht verfügbar")

        # RESONANCE ENGINE: Integriere ResonanceEngine
        self._resonance_engine = None
        try:
            from component_44_resonance_engine import ResonanceEngine

            self._resonance_engine = ResonanceEngine(
                netzwerk=self.netzwerk, confidence_mgr=self.confidence_manager
            )
            logger.info("ResonanceEngine successfully integrated")
        except ImportError:
            logger.warning("ResonanceEngine nicht verfügbar")

        # Initialize specialized handlers
        self._backward_chaining_handler = BackwardChainingHandler(
            netzwerk=netzwerk,
            engine=engine,
            working_memory=working_memory,
            signals=signals,
        )

        self._graph_traversal_handler = GraphTraversalHandler(
            netzwerk=netzwerk,
            graph_traversal=graph_traversal,
            working_memory=working_memory,
            signals=signals,
        )

        self._abductive_handler = AbductiveReasoningHandler(
            netzwerk=netzwerk,
            engine=engine,
            working_memory=working_memory,
            signals=signals,
            fact_loader_callback=self._backward_chaining_handler.load_facts_from_graph,
        )

        self._resonance_handler = None
        if self._resonance_engine:
            # Try to get linguistik_engine for better concept extraction
            linguistik_engine = None
            try:
                from component_6_linguistik_engine import LinguistikEngine

                linguistik_engine = LinguistikEngine()
            except Exception:
                pass  # Use fallback extraction in handler

            self._resonance_handler = ResonanceInferenceHandler(
                netzwerk=netzwerk,
                resonance_engine=self._resonance_engine,
                working_memory=working_memory,
                signals=signals,
                linguistik_engine=linguistik_engine,
            )

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
                    spatial_reasoner=self._spatial_reasoner,
                    resonance_engine=self._resonance_engine,
                    working_memory=self.working_memory,
                    signals=self.signals,
                    probabilistic_engine=None,  # Lazy-loaded
                    abductive_engine=None,  # Lazy-loaded via property
                )
                logger.info("[OK] Hybrid Reasoning Orchestrator aktiviert")
            except Exception as e:
                logger.warning(f"Konnte Hybrid Reasoning nicht aktivieren: {e}")
                self.enable_hybrid_reasoning = False

        logger.info("KaiInferenceHandler initialisiert mit spezialisierten Handlern")

    @property
    def abductive_engine(self):
        """Lazy-Loading für Abductive Engine (Delegation an Handler)."""
        return self._abductive_handler.abductive_engine

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

        # RESONANCE-BASED QA PATH (NEW): Try resonance inference for question queries
        # This is especially useful for queries that benefit from spreading activation
        # and semantic connection detection (e.g., "Kann ein Pinguin fliegen?")
        if self._resonance_handler:
            # Construct a simple query string from topic and relation
            # This will be parsed by _handle_resonance_inference to extract concepts
            query_text = self._resonance_handler.construct_query_from_topic(
                topic, relation_type
            )
            logger.info(
                f"[Multi-Hop Reasoning] -> Versuche Resonance-Based QA für '{query_text}'"
            )

            result = self._resonance_handler.handle_resonance_inference(
                query=query_text,
                context={"topic": topic, "relation_type": relation_type},
            )

            if result and result.get("answer"):
                # Convert resonance result to standard format
                # Extract facts from the answer if possible
                inferred_facts = result.get("overlap", {})
                if not inferred_facts:
                    # Create simple fact structure from contradictions or answer
                    inferred_facts = {relation_type: []}

                return {
                    "inferred_facts": inferred_facts,
                    "proof_trace": result["answer"],
                    "confidence": result.get("confidence", 0.5),
                    "resonance_based": True,
                    "proof_tree": result.get("proof_tree"),
                }

        # LEGACY FALLBACK PATH (ORIGINAL)

        # PHASE 7: Versuche zuerst Graph-Traversal für transitive Relationen
        # Dies ist effizienter und direkter als regelbasiertes Backward-Chaining
        result = self._graph_traversal_handler.try_graph_traversal(topic, relation_type)
        if result:
            return result

        # PHASE 3: Fallback auf regelbasiertes Backward-Chaining
        result = self._backward_chaining_handler.try_backward_chaining(
            topic, relation_type
        )
        if result:
            return result

        # ABDUCTIVE REASONING: Wenn alles fehlschlägt, versuche Hypothesen zu generieren
        result = self._abductive_handler.try_abductive_reasoning(topic, relation_type)
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

    # ==================== DELEGATED METHODS (BACKWARD COMPATIBILITY) ====================

    def load_facts_from_graph(self, topic: str):
        """Delegiert an BackwardChainingHandler."""
        return self._backward_chaining_handler.load_facts_from_graph(topic)

    def extract_facts_from_proof(self, proof):
        """Delegiert an BackwardChainingHandler."""
        return self._backward_chaining_handler.extract_facts_from_proof(proof)

    # Private method aliases for backward compatibility (if needed by external code)
    def _try_graph_traversal(self, topic: str, relation_type: str):
        """Delegiert an GraphTraversalHandler."""
        return self._graph_traversal_handler.try_graph_traversal(topic, relation_type)

    def _try_backward_chaining(self, topic: str, relation_type: str):
        """Delegiert an BackwardChainingHandler."""
        return self._backward_chaining_handler.try_backward_chaining(
            topic, relation_type
        )

    def _try_abductive_reasoning(self, topic: str, relation_type: str):
        """Delegiert an AbductiveReasoningHandler."""
        return self._abductive_handler.try_abductive_reasoning(topic, relation_type)

    def _handle_resonance_inference(self, query: str, context: Optional[Dict] = None):
        """Delegiert an ResonanceInferenceHandler."""
        if self._resonance_handler:
            return self._resonance_handler.handle_resonance_inference(query, context)
        return None

    def _construct_query_from_topic(self, topic: str, relation_type: str) -> str:
        """Delegiert an ResonanceInferenceHandler."""
        if self._resonance_handler:
            return self._resonance_handler.construct_query_from_topic(
                topic, relation_type
            )
        # Fallback
        return f"Was ist ein {topic.lower()}?"
