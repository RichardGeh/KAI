"""
component_44_resonance_engine.py

Cognitive Resonance Core - Wellenförmige Aktivierung mit Resonanz-Verstärkung

Implementiert spreading activation über den Knowledge Graph mit:
- Wave-based propagation (multi-hop)
- Resonance amplification (multiple paths → boost)
- Dynamic confidence integration
- Context-aware filtering
- Pruning für Performance
- Explainable activation paths

Teil von Phase 2: Cognitive Resonance Core

Author: KAI Development Team
Created: 2025-11-07
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActivationType(Enum):
    """Typ der Aktivierung für besseres Tracking"""

    DIRECT = "direct"  # Start-Konzept
    PROPAGATED = "propagated"  # Via Spreading
    RESONANCE = "resonance"  # Via Resonanz-Verstärkung


@dataclass
class ReasoningPath:
    """
    Ein Pfad von Quelle zu Ziel im Aktivierungsnetzwerk

    Attributes:
        source: Start-Konzept des Pfades
        target: Ziel-Konzept des Pfades
        relations: Liste der Beziehungstypen im Pfad
        confidence_product: Produkt aller Confidences im Pfad
        wave_depth: Bei welcher Wave wurde dieser Pfad entdeckt?
        activation_contribution: Wie viel trägt dieser Pfad zur finalen Aktivierung bei?
    """

    source: str
    target: str
    relations: List[str]
    confidence_product: float
    wave_depth: int
    activation_contribution: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Path({self.source} → {self.target}, "
            f"relations={self.relations}, conf={self.confidence_product:.3f}, "
            f"wave={self.wave_depth})"
        )


@dataclass
class ResonancePoint:
    """
    Ein Konzept mit Resonanz-Verstärkung

    Attributes:
        concept: Das Konzept
        resonance_boost: Stärke der Resonanz-Verstärkung
        wave_depth: Wann trat Resonanz auf?
        num_paths: Anzahl konvergierender Pfade
    """

    concept: str
    resonance_boost: float
    wave_depth: int
    num_paths: int = 1

    def __repr__(self) -> str:
        return f"Resonance({self.concept}, boost={self.resonance_boost:.3f}, paths={self.num_paths})"


@dataclass
class ActivationMap:
    """
    Snapshot der Aktivierungszustände nach Spreading Activation

    Attributes:
        activations: Konzept → Aktivierungslevel
        wave_history: Aktivierungen pro Wave (für Animation/Debugging)
        reasoning_paths: Alle entdeckten Pfade
        resonance_points: Konzepte mit Resonanz-Verstärkung
        max_activation: Höchste erreichte Aktivierung
        concepts_activated: Anzahl aktivierter Konzepte
        waves_executed: Anzahl durchgeführter Waves
        activation_types: Konzept → Typ der Aktivierung
    """

    activations: Dict[str, float] = field(default_factory=dict)
    wave_history: List[Dict[str, float]] = field(default_factory=list)
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)
    resonance_points: List[ResonancePoint] = field(default_factory=list)
    max_activation: float = 0.0
    concepts_activated: int = 0
    waves_executed: int = 0
    activation_types: Dict[str, ActivationType] = field(default_factory=dict)

    def get_top_concepts(self, n: int = 10) -> List[Tuple[str, float]]:
        """Gibt die Top-N aktivierten Konzepte zurück"""
        return sorted(self.activations.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_paths_to(self, concept: str) -> List[ReasoningPath]:
        """Gibt alle Pfade zurück, die zu diesem Konzept führen"""
        return [p for p in self.reasoning_paths if p.target == concept]

    def is_resonance_point(self, concept: str) -> bool:
        """Prüft, ob ein Konzept ein Resonanz-Punkt ist"""
        return any(rp.concept == concept for rp in self.resonance_points)


class ResonanceEngine:
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

        logger.info(
            "ResonanceEngine initialized with "
            f"threshold={self.activation_threshold}, "
            f"decay={self.decay_factor}, "
            f"resonance_boost={self.resonance_boost}"
        )

    def activate_concept(
        self,
        start_word: str,
        query_context: Optional[Dict] = None,
        allowed_relations: Optional[List[str]] = None,
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

        Returns:
            ActivationMap mit allen Aktivierungen und Pfaden
        """
        if query_context is None:
            query_context = {}

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
                    f"Pruning: {len(new_activations)} → {self.max_concepts_per_wave}"
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

        # Cypher Query
        cypher = """
        MATCH (start:Wort {lemma: $lemma})
        MATCH (start)-[r]-(neighbor:Wort)
        WHERE
          // Relation-Type Filter
          (size($allowed_relations) = 0 OR type(r) IN $allowed_relations)
          AND
          // Aktivierungs-basiertes Pruning
          COALESCE(r.confidence, 0.7) * $current_activation > $activation_threshold
          AND
          // Nicht zum Start zurück
          neighbor.lemma <> $lemma

        RETURN DISTINCT
          neighbor.lemma as neighbor,
          type(r) as relation_type,
          COALESCE(r.confidence, 0.7) as base_confidence
        ORDER BY base_confidence DESC
        LIMIT 50
        """

        try:
            result = self.netzwerk.execute_query(
                cypher,
                {
                    "lemma": concept,
                    "allowed_relations": allowed_relations,
                    "current_activation": current_activation,
                    "activation_threshold": self.activation_threshold,
                },
            )

            neighbors = [
                (r["neighbor"], r["relation_type"], r["base_confidence"])
                for r in result
            ]

            logger.debug(f"Found {len(neighbors)} neighbors for '{concept}'")
            return neighbors

        except Exception as e:
            logger.error(f"Error fetching neighbors for '{concept}': {e}")
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
        lines.append(f"═══ Aktivierung: '{concept}' ═══")
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
                f"⭐ RESONANZ: {resonance_point.num_paths} konvergierende Pfade, "
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

        lines.append("═══ Spreading Activation Zusammenfassung ═══")
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
                marker = "⭐" if activation_map.is_resonance_point(concept) else "  "
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
                    f"  ⭐ {rp.concept}: {rp.num_paths} Pfade, "
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
