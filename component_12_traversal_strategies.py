"""
component_12_traversal_strategies.py

Traversal strategies for graph navigation.
Implements different pathfinding strategies: BFS, DFS, bidirectional search.

This module contains:
- BFS (Breadth-First Search) - finds shortest paths
- DFS (Depth-First Search) - explores all paths deeply
- Bidirectional search - searches from both ends simultaneously

Extracted from component_12_graph_traversal.py (Task 10, Phase 3 Architecture Refactoring)

Author: KAI Development Team
Date: 2025-11-28
"""

import threading
from collections import deque
from typing import List, Optional

from component_12_graph_traversal_core import (
    GraphPath,
    GraphTraversalCore,
    TraversalStrategy,
)
from component_15_logging_config import get_logger

logger = get_logger(__name__)


class TraversalStrategies:
    """
    Implements different graph traversal strategies.

    Provides:
    - BFS (Breadth-First Search): Finds shortest path
    - DFS (Depth-First Search): Finds any path quickly
    - Bidirectional: Searches from both ends (optimized)
    """

    def __init__(self, core: GraphTraversalCore):
        """
        Initialize traversal strategies.

        Args:
            core: GraphTraversalCore instance for shared utilities
        """
        self.core = core
        self.netzwerk = core.netzwerk
        self._lock = threading.RLock()

    def bfs_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """
        Breadth-First Search fur kurzesten Pfad.

        Explores all paths level-by-level, guaranteeing shortest path.

        Args:
            start: Startkonzept
            target: Zielkonzept
            allowed_relations: Erlaubte Relationstypen (None = alle)

        Returns:
            GraphPath (kurzester Pfad) oder None
        """
        logger.debug(
            f"BFS path search: {start} -> {target}, allowed_relations={allowed_relations}"
        )

        # Queue: (current_node, path_nodes, path_relations, path_confidences)
        queue = deque([(start, [start], [], [])])
        visited = {start}
        nodes_expanded = 0

        while queue:
            current, path_nodes, path_relations, path_confidences = queue.popleft()
            nodes_expanded += 1

            # Ziel erreicht?
            if current == target:
                logger.info(
                    f"BFS found path in {nodes_expanded} expansions: {path_nodes}"
                )
                # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                overall_confidence = min(path_confidences) if path_confidences else 1.0
                explanation = self.core.generate_path_explanation(
                    path_nodes, path_relations
                )
                return GraphPath(
                    nodes=path_nodes,
                    relations=path_relations,
                    confidence=overall_confidence,
                    explanation=explanation,
                )

            # Maximale Tiefe uberschritten?
            if len(path_relations) >= self.core.max_depth:
                continue

            # Erweitere Pfad
            facts_with_conf = self.netzwerk.query_graph_for_facts_with_confidence(
                current
            )

            for relation_type, targets_with_conf in facts_with_conf.items():
                # Filter nach erlaubten Relationen
                if allowed_relations and relation_type not in allowed_relations:
                    continue

                for target_info in targets_with_conf:
                    neighbor = target_info["target"]
                    base_confidence = target_info["confidence"]
                    timestamp = target_info.get("timestamp")

                    # Berechne dynamische Confidence
                    edge_confidence = self.core.calculate_dynamic_confidence(
                        subject=current,
                        relation=relation_type,
                        object_=neighbor,
                        base_confidence=base_confidence,
                        timestamp=timestamp,
                    )

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(
                            (
                                neighbor,
                                path_nodes + [neighbor],
                                path_relations + [relation_type],
                                path_confidences + [edge_confidence],
                            )
                        )

        logger.warning(f"BFS failed to find path after {nodes_expanded} expansions")
        return None  # Kein Pfad gefunden

    def dfs_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """
        Depth-First Search fur ersten gefundenen Pfad.

        Explores paths deeply before backtracking. May not find shortest path.

        Args:
            start: Startkonzept
            target: Zielkonzept
            allowed_relations: Erlaubte Relationstypen

        Returns:
            GraphPath (erster gefundener Pfad) oder None
        """
        logger.debug(
            f"DFS path search: {start} -> {target}, allowed_relations={allowed_relations}"
        )

        visited = set()
        nodes_expanded = [0]  # Use list to allow modification in nested function

        def dfs(
            current: str,
            path_nodes: List[str],
            path_relations: List[str],
            path_confidences: List[float],
        ) -> Optional[GraphPath]:
            nodes_expanded[0] += 1

            if current == target:
                logger.info(
                    f"DFS found path in {nodes_expanded[0]} expansions: {path_nodes}"
                )
                # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                overall_confidence = min(path_confidences) if path_confidences else 1.0
                explanation = self.core.generate_path_explanation(
                    path_nodes, path_relations
                )
                return GraphPath(
                    nodes=path_nodes,
                    relations=path_relations,
                    confidence=overall_confidence,
                    explanation=explanation,
                )

            if len(path_relations) >= self.core.max_depth:
                return None

            if current in visited:
                return None

            visited.add(current)

            facts_with_conf = self.netzwerk.query_graph_for_facts_with_confidence(
                current
            )

            for relation_type, targets_with_conf in facts_with_conf.items():
                if allowed_relations and relation_type not in allowed_relations:
                    continue

                for target_info in targets_with_conf:
                    neighbor = target_info["target"]
                    base_confidence = target_info["confidence"]
                    timestamp = target_info.get("timestamp")

                    # Berechne dynamische Confidence
                    edge_confidence = self.core.calculate_dynamic_confidence(
                        subject=current,
                        relation=relation_type,
                        object_=neighbor,
                        base_confidence=base_confidence,
                        timestamp=timestamp,
                    )

                    result = dfs(
                        neighbor,
                        path_nodes + [neighbor],
                        path_relations + [relation_type],
                        path_confidences + [edge_confidence],
                    )
                    if result:
                        return result

            visited.remove(current)
            return None

        result = dfs(start, [start], [], [])
        if result is None:
            logger.warning(
                f"DFS failed to find path after {nodes_expanded[0]} expansions"
            )
        return result

    def bidirectional_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """
        Bidirektionale Suche (von beiden Seiten gleichzeitig).

        Effizienter fur grosse Graphen, aber komplexer zu implementieren.
        TODO: Vollstandige Implementierung fur zukunftige Optimierung.

        Args:
            start: Startkonzept
            target: Zielkonzept
            allowed_relations: Erlaubte Relationstypen

        Returns:
            GraphPath oder None
        """
        # Fallback auf BFS
        logger.debug("Bidirectional search not fully implemented, using BFS")
        return self.bfs_path(start, target, allowed_relations)

    def find_path_between_concepts(
        self,
        start_concept: str,
        target_concept: str,
        allowed_relations: Optional[List[str]] = None,
        strategy: TraversalStrategy = TraversalStrategy.BREADTH_FIRST,
    ) -> Optional[GraphPath]:
        """
        Findet einen Pfad zwischen zwei Konzepten mit gewahlter Strategie.

        Beispiel: find_path_between_concepts("hund", "tier")
        Findet: hund --IS_A--> saugetier --IS_A--> tier

        Args:
            start_concept: Startkonzept
            target_concept: Zielkonzept
            allowed_relations: Erlaubte Relationstypen (None = alle)
            strategy: Traversierungsstrategie

        Returns:
            GraphPath oder None, wenn kein Pfad existiert
        """
        with self._lock:
            if strategy == TraversalStrategy.BREADTH_FIRST:
                return self.bfs_path(start_concept, target_concept, allowed_relations)
            elif strategy == TraversalStrategy.DEPTH_FIRST:
                return self.dfs_path(start_concept, target_concept, allowed_relations)
            elif strategy == TraversalStrategy.BIDIRECTIONAL:
                return self.bidirectional_path(
                    start_concept, target_concept, allowed_relations
                )
            else:
                raise ValueError(f"Unbekannte Strategie: {strategy}")

    def find_all_paths_between_concepts(
        self,
        start_concept: str,
        target_concept: str,
        allowed_relations: Optional[List[str]] = None,
        max_paths: int = 10,
    ) -> List[GraphPath]:
        """
        Findet ALLE Pfade zwischen zwei Konzepten (bis zu max_paths).

        Nützlich für alternative Erklärungen oder Hypothesenvergleich.

        Args:
            start_concept: Startkonzept
            target_concept: Zielkonzept
            allowed_relations: Erlaubte Relationstypen
            max_paths: Maximale Anzahl zurückzugebender Pfade

        Returns:
            Liste von GraphPath-Objekten (sortiert nach Länge)
        """
        all_paths = []
        visited_paths = set()  # Zum Deduplizieren

        def dfs(
            current: str,
            path_nodes: List[str],
            path_relations: List[str],
            path_confidences: List[float],
        ):
            if len(all_paths) >= max_paths:
                return  # Genug Pfade gefunden

            if current == target_concept:
                # Pfad zum Ziel gefunden
                path_tuple = tuple(path_nodes)  # Für Deduplizierung
                if path_tuple not in visited_paths:
                    visited_paths.add(path_tuple)
                    # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                    overall_confidence = (
                        min(path_confidences) if path_confidences else 1.0
                    )
                    explanation = self.core.generate_path_explanation(
                        path_nodes, path_relations
                    )
                    all_paths.append(
                        GraphPath(
                            nodes=path_nodes,
                            relations=path_relations,
                            confidence=overall_confidence,
                            explanation=explanation,
                        )
                    )
                return

            if len(path_relations) >= self.core.max_depth:
                return

            if current in path_nodes[:-1]:  # Zykluserkennung (nicht aktueller Knoten)
                return

            facts_with_conf = self.netzwerk.query_graph_for_facts_with_confidence(
                current
            )

            for relation_type, targets_with_conf in facts_with_conf.items():
                if allowed_relations and relation_type not in allowed_relations:
                    continue

                for target_info in targets_with_conf:
                    neighbor = target_info["target"]
                    base_confidence = target_info["confidence"]
                    timestamp = target_info.get("timestamp")

                    # Berechne dynamische Confidence
                    edge_confidence = self.core.calculate_dynamic_confidence(
                        subject=current,
                        relation=relation_type,
                        object_=neighbor,
                        base_confidence=base_confidence,
                        timestamp=timestamp,
                    )

                    dfs(
                        neighbor,
                        path_nodes + [neighbor],
                        path_relations + [relation_type],
                        path_confidences + [edge_confidence],
                    )

        with self._lock:
            dfs(start_concept, [start_concept], [], [])

            # Sortiere nach Pfadlänge
            all_paths.sort(key=lambda p: len(p.relations))

            return all_paths
