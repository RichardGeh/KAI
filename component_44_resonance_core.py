"""
component_44_resonance_core.py

Core Resonance Engine - Wellenförmige Aktivierung mit Resonanz-Verstärkung

Implementiert spreading activation über den Knowledge Graph mit:
- Wave-based propagation (multi-hop)
- Resonance amplification (multiple paths -> boost)
- Dynamic confidence integration
- Context-aware filtering
- Pruning für Performance
- Explainable activation paths

Teil von Phase 4 Architecture Refactoring (2025-11-29)
Split from component_44_resonance_engine.py (1060 lines -> modular)

Author: KAI Development Team
Created: 2025-11-29
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from component_17_proof_explanation import ProofStep, ProofTree, StepType
from component_44_resonance_data_structures import (
    ActivationMap,
    ActivationType,
    ReasoningPath,
    ResonancePoint,
)
from infrastructure.cache_manager import cache_manager
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult
from kai_exceptions import Neo4jConnectionError, Neo4jQueryError

logger = logging.getLogger(__name__)


class ResonanceEngine(BaseReasoningEngine):
    """
    Cognitive Resonance Engine

    Implementiert spreading activation mit Resonanz-Verstärkung:
    1. Start mit initialem Konzept (activation=1.0)
    2. Wellenförmige Ausbreitung über Relationen
    3. Decay mit Distanz (decay_factor)
    4. Resonanz: Multiple Pfade verstärken Aktivierung
    5. Pruning: Top-N Konzepte pro Wave für Performance

    Integration:
    - Dynamic Confidence (component_confidence_manager)
    - Neo4j Knowledge Graph (component_1_netzwerk)
    - Context-Aware Filtering
    """

    def __init__(self, netzwerk, confidence_mgr=None):
        """
        Initialize Resonance Engine

        Args:
            netzwerk: KonzeptNetzwerk instance
            confidence_mgr: Optional ConfidenceManager für dynamische Confidences
        """
        self.netzwerk = netzwerk
        self.confidence_mgr = confidence_mgr

        # Hyperparameter (später: adaptive tuning via Meta-Learning)
        self.activation_threshold = 0.3  # Minimum für Weiterleitung
        self.decay_factor = 0.7  # Dämpfung pro Hop
        self.resonance_boost = 0.5  # Verstärkungsfaktor bei Resonanz
        self.max_waves = 5  # Maximum Propagation Tiefe
        self.max_concepts_per_wave = 100  # Pruning Limit

        # Tracking
        self._current_resonance_points: List[ResonancePoint] = []

        # Performance Optimization: Caching via CacheManager
        # Activation Maps Cache (TTL: 10 Minuten)
        cache_manager.register_cache("resonance_activations", maxsize=100, ttl=600)
        # Semantic Neighbors Cache (TTL: 10 Minuten, same as activations)
        cache_manager.register_cache("resonance_neighbors", maxsize=500, ttl=600)

        logger.info(
            "ResonanceEngine initialized with "
            f"threshold={self.activation_threshold}, "
            f"decay={self.decay_factor}, "
            f"resonance_boost={self.resonance_boost}, "
            f"caching enabled (activation_ttl=600s, neighbors_maxsize=500)"
        )

    def _generate_cache_key(
        self,
        start_word: str,
        query_context: Dict,
        allowed_relations: Optional[List[str]],
    ) -> str:
        """
        Generiert Cache-Key für Activation Maps

        Args:
            start_word: Start-Konzept
            query_context: Context dict
            allowed_relations: Liste erlaubter Relationstypen

        Returns:
            Hash-basierter Cache-Key
        """
        # Sortiere für konsistente Keys
        context_str = json.dumps(query_context, sort_keys=True)
        relations_str = json.dumps(sorted(allowed_relations or []))

        # Hash generieren (nur für Cache-Key, nicht für Security)
        key_string = f"{start_word}|{context_str}|{relations_str}"
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()

    def activate_concept(
        self,
        start_word: str,
        query_context: Optional[Dict] = None,
        allowed_relations: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> ActivationMap:
        """
        Wellenförmige Aktivierung mit Resonanz-Verstärkung

        Algorithm:
        1. Initialisiere start_word mit activation=1.0
        2. Für jede Wave (bis max_waves):
           a. Finde Nachbarn aller aktivierten Konzepte
           b. Berechne neue Aktivierung (alte * decay * confidence)
           c. RESONANZ: Wenn Konzept schon aktiviert, addiere boost
           d. Pruning: Behalte nur top-N Konzepte
        3. Baue Reasoning Paths für alle aktivierten Konzepte

        Args:
            start_word: Start-Konzept (lemmatisiert)
            query_context: Optional context dict für Filterung
            allowed_relations: Optional Liste erlaubter Relationstypen
            use_cache: Nutze Activation Maps Cache (default: True)

        Returns:
            ActivationMap mit allen Aktivierungen und Pfaden
        """
        if query_context is None:
            query_context = {}

        # Cache Lookup
        if use_cache:
            cache_key = self._generate_cache_key(
                start_word, query_context, allowed_relations
            )
            cached_result = cache_manager.get("resonance_activations", cache_key)
            if cached_result is not None:
                logger.debug(f"Cache HIT for activation '{start_word}'")
                return cached_result
            else:
                logger.debug(f"Cache MISS for activation '{start_word}'")

        # Reset tracking
        self._current_resonance_points = []

        # Initialisierung
        activation_map = {start_word: 1.0}
        activation_types = {start_word: ActivationType.DIRECT}
        wave_history = []
        all_paths = []
        visited_edges = set()  # Track visited edges to prevent duplicates

        logger.info(f"Starting spreading activation from '{start_word}'")

        for wave_depth in range(self.max_waves):
            new_activations = {}
            new_types = {}
            wave_paths = []

            # Für jedes aktivierte Konzept: finde Nachbarn
            active_concepts = [
                (concept, activation)
                for concept, activation in activation_map.items()
                if activation >= self.activation_threshold
            ]

            if not active_concepts:
                logger.debug(
                    f"Wave {wave_depth}: No active concepts above threshold, stopping"
                )
                break

            logger.debug(
                f"Wave {wave_depth}: Processing {len(active_concepts)} active concepts"
            )

            for concept, activation in active_concepts:
                neighbors = self._get_semantic_neighbors(
                    concept,
                    query_context,
                    current_activation=activation,
                    allowed_relations=allowed_relations,
                )

                for neighbor, rel_type, base_confidence in neighbors:
                    # Skip if this edge was already processed
                    edge_key = (concept, rel_type, neighbor)
                    if edge_key in visited_edges:
                        continue
                    visited_edges.add(edge_key)
                    # Dynamische Confidence (falls verfügbar)
                    if self.confidence_mgr:
                        try:
                            dynamic_conf = self.confidence_mgr.get_current_confidence(
                                relation=(concept, rel_type, neighbor)
                            )
                        except Exception as e:
                            logger.debug(f"Confidence lookup failed: {e}, using base")
                            dynamic_conf = base_confidence
                    else:
                        dynamic_conf = base_confidence

                    # Aktivierung nimmt mit Distanz ab
                    new_activation = activation * self.decay_factor * dynamic_conf

                    # RESONANZ: Verstärkung bei multiple paths
                    is_resonance = False
                    if neighbor in activation_map or neighbor in new_activations:
                        old_activation = activation_map.get(
                            neighbor, 0.0
                        ) + new_activations.get(neighbor, 0.0)
                        resonance = self.resonance_boost * old_activation
                        new_activation += resonance
                        is_resonance = True

                        # Track als Resonance Point
                        if resonance > 0.1:
                            self._mark_resonance(neighbor, resonance, wave_depth)

                    # Akkumuliere Aktivierung
                    if neighbor in new_activations:
                        new_activations[neighbor] += new_activation
                    else:
                        new_activations[neighbor] = new_activation

                    # Track Activation Type
                    if neighbor not in new_types:
                        new_types[neighbor] = (
                            ActivationType.RESONANCE
                            if is_resonance
                            else ActivationType.PROPAGATED
                        )

                    # Track Reasoning Path
                    path = ReasoningPath(
                        source=concept,
                        target=neighbor,
                        relations=[rel_type],
                        confidence_product=dynamic_conf,
                        wave_depth=wave_depth,
                        activation_contribution=new_activation,
                    )
                    wave_paths.append(path)

            # Pruning: Behalte nur top-N aktivierte Konzepte
            if len(new_activations) > self.max_concepts_per_wave:
                logger.debug(
                    f"Pruning: {len(new_activations)} -> {self.max_concepts_per_wave}"
                )
                sorted_items = sorted(
                    new_activations.items(), key=lambda x: x[1], reverse=True
                )[: self.max_concepts_per_wave]
                new_activations = dict(sorted_items)

                # Pruning auch für Types
                new_types = {k: v for k, v in new_types.items() if k in new_activations}

            # Update Activation Map
            activation_map.update(new_activations)
            activation_types.update(new_types)
            wave_history.append(new_activations.copy())
            all_paths.extend(wave_paths)

            logger.debug(
                f"Wave {wave_depth}: Activated {len(new_activations)} new concepts, "
                f"{len(wave_paths)} paths"
            )

            # Early stopping wenn keine neuen Aktivierungen
            if not new_activations:
                logger.debug(f"Wave {wave_depth}: No new activations, stopping")
                break

        # Erstelle ActivationMap
        result = ActivationMap(
            activations=activation_map,
            wave_history=wave_history,
            reasoning_paths=all_paths,
            resonance_points=self._current_resonance_points.copy(),
            max_activation=max(activation_map.values()) if activation_map else 0.0,
            concepts_activated=len(activation_map),
            waves_executed=len(wave_history),
            activation_types=activation_types,
        )

        logger.info(
            f"Spreading activation completed: {result.concepts_activated} concepts, "
            f"{result.waves_executed} waves, {len(result.resonance_points)} resonance points"
        )

        # Cache Write
        if use_cache:
            cache_key = self._generate_cache_key(
                start_word, query_context, allowed_relations
            )
            cache_manager.set("resonance_activations", cache_key, result)
            logger.debug(
                f"Cached activation map for '{start_word}' (key={cache_key[:8]}...)"
            )

        return result

    def _get_semantic_neighbors(
        self,
        concept: str,
        query_context: Dict,
        current_activation: float,
        allowed_relations: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Neo4j Query für semantische Nachbarn

        Berücksichtigt:
        - Context-Filterung (nur relevante Relationen)
        - Relation-Type Filtering (z.B. nur IS_A bei Taxonomie-Fragen)
        - Bidirektionalität
        - Aktivierungs-basiertes Pruning
        - Session-based Caching für Performance

        Args:
            concept: Aktuelles Konzept
            query_context: Context dict
            current_activation: Aktuelle Aktivierung des Konzepts
            allowed_relations: Optional Liste erlaubter Relationstypen

        Returns:
            Liste von (neighbor, relation_type, confidence) Tupeln
        """
        # Relation Filter
        if allowed_relations is None:
            allowed_relations = query_context.get("relation_types", [])

        # Cache Key generieren
        relations_str = json.dumps(sorted(allowed_relations or []))
        # Include min_confidence to avoid cache collisions with different thresholds
        min_conf_for_key = self.activation_threshold / max(current_activation, 0.01)
        cache_key = (
            f"{concept}|{current_activation:.3f}|{min_conf_for_key:.3f}|{relations_str}"
        )

        # Cache Lookup
        cached_neighbors = cache_manager.get("resonance_neighbors", cache_key)
        if cached_neighbors is not None:
            logger.debug(f"Neighbors cache HIT for '{concept}'")
            return cached_neighbors

        # Use facade method for semantic neighbor query
        # MIGRATION NOTE: Replaced direct session.run() with netzwerk.query_semantic_neighbors()
        # Original query lines 361-379 migrated to facade method (2025-12-03 Tier 2 Migration)
        try:
            if not hasattr(self.netzwerk, "driver") or not self.netzwerk.driver:
                logger.warning(
                    "KonzeptNetzwerk hat keinen driver - Resonance deaktiviert"
                )
                return []

            try:
                # Calculate effective minimum confidence based on activation
                # Original query: COALESCE(r.confidence, 0.7) * current_activation > activation_threshold
                # Rearranged: r.confidence > activation_threshold / current_activation
                min_confidence = self.activation_threshold / max(
                    current_activation, 0.01
                )

                # Use facade method - returns List[Dict[str, Any]]
                result_dicts = self.netzwerk.query_semantic_neighbors(
                    lemma=concept,
                    allowed_relations=allowed_relations if allowed_relations else [],
                    min_confidence=min_confidence,
                    limit=50,
                )

                # Convert facade result format to original tuple format for backward compatibility
                neighbors = [
                    (r["neighbor"], r["relation_type"], r["confidence"])
                    for r in result_dicts
                ]

            except Exception as e:
                # Neo4j-spezifische Fehler behandeln
                if "ServiceUnavailable" in str(type(e).__name__):
                    logger.error(
                        f"Neo4j-Verbindung fehlgeschlagen für '{concept}': {e}"
                    )
                    raise Neo4jConnectionError(
                        "Neo4j-Verbindung nicht verfügbar",
                        context={"concept": concept},
                        original_exception=e,
                    )
                else:
                    logger.error(f"Neo4j-Query fehlgeschlagen für '{concept}': {e}")
                    raise Neo4jQueryError(
                        "Fehler beim Abrufen von Nachbarn",
                        query=f"Facade: KonzeptNetzwerk.query_semantic_neighbors(lemma='{concept}', ...)",
                        parameters={
                            "lemma": concept,
                            "allowed_relations": allowed_relations,
                            "min_confidence": min_confidence,
                        },
                        original_exception=e,
                    )

            # Cache Write (CacheManager handles size limits automatically)
            cache_manager.set("resonance_neighbors", cache_key, neighbors)
            logger.debug(f"Cached {len(neighbors)} neighbors for '{concept}'")

            return neighbors

        except (Neo4jConnectionError, Neo4jQueryError):
            # Graceful degradation: Bei Neo4j-Fehler leere Liste zurückgeben
            logger.warning(
                f"Neo4j-Fehler bei '{concept}', fahre mit leerer Nachbarn-Liste fort"
            )
            return []
        except Exception as e:
            # Unerwarteter Fehler
            logger.error(
                f"Unerwarteter Fehler beim Abrufen von Nachbarn für '{concept}': {e}",
                exc_info=True,
            )
            return []

    def _mark_resonance(self, concept: str, resonance: float, wave_depth: int):
        """
        Markiert ein Konzept als Resonance Point

        Args:
            concept: Das Konzept
            resonance: Stärke der Resonanz
            wave_depth: Aktuelle Wave
        """
        # Prüfe ob schon vorhanden
        for rp in self._current_resonance_points:
            if rp.concept == concept:
                # Update existierenden Point
                rp.resonance_boost += resonance
                rp.num_paths += 1
                logger.debug(
                    f"Updated resonance for '{concept}': "
                    f"boost={rp.resonance_boost:.3f}, paths={rp.num_paths}"
                )
                return

        # Neuer Resonance Point
        rp = ResonancePoint(
            concept=concept,
            resonance_boost=resonance,
            wave_depth=wave_depth,
            num_paths=2,  # Mindestens 2 Pfade für Resonanz
        )
        self._current_resonance_points.append(rp)
        logger.debug(f"New resonance point: {rp}")

    def explain_activation(
        self, concept: str, activation_map: ActivationMap, max_paths: int = 3
    ) -> str:
        """
        Generiert natürlichsprachliche Erklärung der Aktivierung

        Args:
            concept: Das zu erklärende Konzept
            activation_map: Die ActivationMap
            max_paths: Maximale Anzahl anzuzeigender Pfade

        Returns:
            Deutsche Erklärung der Aktivierung
        """
        if concept not in activation_map.activations:
            return f"'{concept}' wurde nicht aktiviert."

        activation = activation_map.activations[concept]
        activation_type = activation_map.activation_types.get(
            concept, ActivationType.PROPAGATED
        )

        # Finde Pfade zu diesem Konzept
        paths = activation_map.get_paths_to(concept)
        paths_sorted = sorted(paths, key=lambda p: p.confidence_product, reverse=True)

        # Prüfe auf Resonanz
        is_resonance = activation_map.is_resonance_point(concept)

        # Baue Erklärung
        lines = []

        # Header
        lines.append(f"=== Aktivierung: '{concept}' ===")
        lines.append(f"Aktivierungslevel: {activation:.3f}")

        # Typ
        type_str = {
            ActivationType.DIRECT: "Direkt (Start-Konzept)",
            ActivationType.PROPAGATED: "Propagiert",
            ActivationType.RESONANCE: "Resonanz-verstärkt",
        }.get(activation_type, "Unbekannt")
        lines.append(f"Typ: {type_str}")

        # Resonanz-Info
        if is_resonance:
            resonance_point = next(
                rp for rp in activation_map.resonance_points if rp.concept == concept
            )
            lines.append(
                f"[R] RESONANZ: {resonance_point.num_paths} konvergierende Pfade, "
                f"Boost={resonance_point.resonance_boost:.3f}"
            )

        # Pfade
        if paths_sorted:
            lines.append(f"\nAktivierungspfade ({len(paths_sorted)} gesamt):")
            for i, path in enumerate(paths_sorted[:max_paths], 1):
                rel_str = ", ".join(path.relations)
                lines.append(f"  {i}. {path.source} --[{rel_str}]--> {concept}")
                lines.append(
                    f"     Wave {path.wave_depth}, "
                    f"Confidence: {path.confidence_product:.3f}, "
                    f"Beitrag: {path.activation_contribution:.3f}"
                )

            if len(paths_sorted) > max_paths:
                lines.append(f"  ... und {len(paths_sorted) - max_paths} weitere Pfade")

        return "\n".join(lines)

    def get_activation_summary(self, activation_map: ActivationMap) -> str:
        """
        Generiert Zusammenfassung der gesamten Aktivierung

        Args:
            activation_map: Die ActivationMap

        Returns:
            Deutsche Zusammenfassung
        """
        lines = []

        lines.append("=== Spreading Activation Zusammenfassung ===")
        lines.append(f"Aktivierte Konzepte: {activation_map.concepts_activated}")
        lines.append(f"Durchgeführte Waves: {activation_map.waves_executed}")
        lines.append(f"Resonanz-Punkte: {len(activation_map.resonance_points)}")
        lines.append(f"Max. Aktivierung: {activation_map.max_activation:.3f}")
        lines.append(f"Pfade gesamt: {len(activation_map.reasoning_paths)}")

        # Top aktivierte Konzepte
        top_concepts = activation_map.get_top_concepts(10)
        if top_concepts:
            lines.append("\nTop 10 aktivierte Konzepte:")
            for i, (concept, act) in enumerate(top_concepts, 1):
                marker = "[R]" if activation_map.is_resonance_point(concept) else "   "
                lines.append(f"  {marker}{i}. {concept}: {act:.3f}")

        # Resonanz-Punkte Details
        if activation_map.resonance_points:
            lines.append("\nResonanz-Punkte (Multiple Pfade):")
            for rp in sorted(
                activation_map.resonance_points,
                key=lambda x: x.resonance_boost,
                reverse=True,
            )[:5]:
                lines.append(
                    f"  [R] {rp.concept}: {rp.num_paths} Pfade, "
                    f"Boost={rp.resonance_boost:.3f}, Wave {rp.wave_depth}"
                )

        return "\n".join(lines)

    def set_hyperparameters(
        self,
        activation_threshold: Optional[float] = None,
        decay_factor: Optional[float] = None,
        resonance_boost: Optional[float] = None,
        max_waves: Optional[int] = None,
        max_concepts_per_wave: Optional[int] = None,
    ):
        """
        Setzt Hyperparameter (für adaptive Tuning via Meta-Learning)

        Args:
            activation_threshold: Minimum für Weiterleitung
            decay_factor: Dämpfung pro Hop
            resonance_boost: Verstärkungsfaktor bei Resonanz
            max_waves: Maximum Propagation Tiefe
            max_concepts_per_wave: Pruning Limit
        """
        if activation_threshold is not None:
            self.activation_threshold = activation_threshold
        if decay_factor is not None:
            self.decay_factor = decay_factor
        if resonance_boost is not None:
            self.resonance_boost = resonance_boost
        if max_waves is not None:
            self.max_waves = max_waves
        if max_concepts_per_wave is not None:
            self.max_concepts_per_wave = max_concepts_per_wave

        logger.info(
            f"Updated hyperparameters: threshold={self.activation_threshold}, "
            f"decay={self.decay_factor}, resonance={self.resonance_boost}, "
            f"max_waves={self.max_waves}, max_concepts={self.max_concepts_per_wave}"
        )

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Leert Caches

        Args:
            cache_type: 'activation', 'neighbors', oder None für beide
        """
        if cache_type == "activation" or cache_type is None:
            cache_manager.invalidate("resonance_activations")
            logger.info("Activation cache cleared")

        if cache_type == "neighbors" or cache_type is None:
            cache_manager.invalidate("resonance_neighbors")
            logger.info("Neighbors cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Gibt Cache-Statistiken zurück

        Returns:
            Dict mit Cache-Größen und Hit-Raten
        """
        activation_stats = cache_manager.get_stats("resonance_activations")
        neighbors_stats = cache_manager.get_stats("resonance_neighbors")

        return {
            "activation_cache": {
                "size": activation_stats["size"],
                "maxsize": activation_stats["maxsize"],
                "ttl": activation_stats["ttl"],
                "hits": activation_stats["hits"],
                "misses": activation_stats["misses"],
                "hit_rate": activation_stats["hit_rate"],
            },
            "neighbors_cache": {
                "size": neighbors_stats["size"],
                "maxsize": neighbors_stats["maxsize"],
                "ttl": neighbors_stats["ttl"],
                "hits": neighbors_stats["hits"],
                "misses": neighbors_stats["misses"],
                "hit_rate": neighbors_stats["hit_rate"],
            },
        }

    # ========================================================================
    # BaseReasoningEngine Interface Implementation
    # ========================================================================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Execute resonance-based reasoning on the query.

        Args:
            query: Natural language query (used as start concept if not in context)
            context: Context with:
                - "start_concepts": List of start concepts (default: [query])
                - "allowed_relations": Optional list of relation types to traverse
                - "query_context": Optional additional query context
                - "threshold": Optional activation threshold (default: self.activation_threshold)

        Returns:
            ReasoningResult with resonance points and activation map
        """
        try:
            # Extract parameters from context
            start_concepts = context.get("start_concepts")
            if not start_concepts:
                # Use query as start concept (normalized)
                start_concepts = [query.lower().strip()]

            allowed_relations = context.get("allowed_relations")
            query_context = context.get("query_context", {})
            threshold_override = context.get("threshold")

            # Store original threshold if override
            original_threshold = None
            if threshold_override is not None:
                original_threshold = self.activation_threshold
                self.activation_threshold = threshold_override

            # Spread activation
            activation_map = self.activate_concept(
                start_word=start_concepts[0] if len(start_concepts) == 1 else None,
                query_context=query_context,
                allowed_relations=allowed_relations,
            )

            # Restore original threshold if overridden
            if original_threshold is not None:
                self.activation_threshold = original_threshold

            # Extract resonance points (concepts with high activation)
            resonance_points = [
                point
                for point in self._current_resonance_points
                if point.activation >= self.activation_threshold
            ]

            # Sort by activation
            resonance_points.sort(key=lambda p: p.activation, reverse=True)

            # Build answer
            if resonance_points:
                top_concepts = ", ".join([rp.concept for rp in resonance_points[:5]])
                answer = (
                    f"Found {len(resonance_points)} resonance points: {top_concepts}"
                )
                confidence = max([rp.activation for rp in resonance_points])
            else:
                answer = "No resonance points found above threshold"
                confidence = 0.0

            # Build proof tree
            proof = ProofTree(conclusion=answer)
            proof.add_root_step(
                ProofStep(
                    step_type=StepType.PREMISE,
                    content=f"Starting resonance from: {', '.join(start_concepts)}",
                    confidence=1.0,
                )
            )
            proof.add_root_step(
                ProofStep(
                    step_type=StepType.RULE_APPLICATION,
                    content=f"Spreading activation over {len(activation_map.activations)} concepts",
                    confidence=0.9,
                )
            )
            for rp in resonance_points[:3]:  # Top 3 in proof
                proof.add_root_step(
                    ProofStep(
                        step_type=StepType.INFERENCE,
                        content=f"Resonance point: {rp.concept} (activation={rp.activation:.3f}, paths={rp.num_paths})",
                        confidence=rp.activation,
                    )
                )

            return ReasoningResult(
                success=len(resonance_points) > 0,
                answer=answer,
                confidence=confidence,
                proof_tree=proof,
                strategy_used="resonance_spreading_activation",
                metadata={
                    "num_resonance_points": len(resonance_points),
                    "num_concepts_activated": len(activation_map.activations),
                    "max_activation": (
                        max([rp.activation for rp in resonance_points])
                        if resonance_points
                        else 0.0
                    ),
                    "start_concepts": start_concepts,
                },
            )

        except Exception as e:
            logger.error(
                "Error in resonance reasoning",
                extra={"query": query, "error": str(e)},
                exc_info=True,
            )
            return ReasoningResult(
                success=False,
                answer=f"Resonance reasoning error: {str(e)}",
                confidence=0.0,
                strategy_used="resonance_reasoning",
            )

    def get_capabilities(self) -> List[str]:
        """Return list of reasoning capabilities."""
        return [
            "resonance",
            "spreading_activation",
            "concept_association",
            "pattern_emergence",
            "semantic_search",
            "context_aware_reasoning",
        ]

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for resonance reasoning.

        Returns:
            Cost estimate in [0.0, 1.0] range
            Base cost: 0.6 (medium-expensive due to graph traversal)
        """
        # Resonance reasoning can be expensive due to:
        # - Multi-wave spreading activation
        # - Graph traversal over multiple hops
        # - Resonance point detection
        # - Caching helps but initial queries are expensive
        base_cost = 0.6

        # Query length and complexity
        query_complexity = min(len(query) / 400.0, 0.15)

        # Number of waves affects cost (more waves = higher cost)
        wave_factor = min(self.max_waves / 10.0, 0.1)

        return min(base_cost + query_complexity + wave_factor, 1.0)
