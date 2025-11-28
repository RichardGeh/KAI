"""
component_12_graph_traversal_core.py

Core graph traversal engine for multi-hop reasoning.
Provides fundamental graph navigation primitives and relation following.

This module contains:
- GraphPath data structure (represents paths through the knowledge graph)
- TraversalStrategy enum (defines available traversal strategies)
- Core helper methods (explanation generation, relation translation, dynamic confidence)

Extracted from component_12_graph_traversal.py (Task 10, Phase 3 Architecture Refactoring)

Author: KAI Development Team
Date: 2025-11-28
"""

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

# Import Unified Proof Explanation System
try:
    pass

    UNIFIED_PROOFS_AVAILABLE = True
except ImportError:
    UNIFIED_PROOFS_AVAILABLE = False

# Import Dynamic Confidence System
try:
    from component_47_dynamic_confidence import (
        get_dynamic_confidence_manager,
    )

    DYNAMIC_CONFIDENCE_AVAILABLE = True
except ImportError:
    DYNAMIC_CONFIDENCE_AVAILABLE = False

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class TraversalStrategy(Enum):
    """Strategien fur Graph-Traversierung"""

    BREADTH_FIRST = "breadth_first"  # Breite-zuerst (kurzeste Pfade)
    DEPTH_FIRST = "depth_first"  # Tiefe-zuerst (alle Pfade)
    BIDIRECTIONAL = "bidirectional"  # Von beiden Seiten (schnellster Weg)


@dataclass
class GraphPath:
    """Reprasentiert einen Pfad durch den Knowledge-Graph"""

    nodes: List[str]  # Konzepte im Pfad (z.B. ["hund", "saugetier", "tier"])
    relations: List[str]  # Relationen zwischen Knoten (z.B. ["IS_A", "IS_A"])
    confidence: float  # Gesamtkonfidenz des Pfads (min aller Kanten)
    explanation: str  # Menschenlesbare Erklarung

    def __repr__(self):
        path_str = " -> ".join(
            f"{self.nodes[i]} --{self.relations[i]}--> {self.nodes[i+1]}"
            for i in range(len(self.relations))
        )
        return f"Path({path_str}, confidence={self.confidence:.2f})"


class GraphTraversalCore:
    """
    Core graph traversal engine with fundamental navigation primitives.

    Provides:
    - Relation-to-German translation for explanations
    - Dynamic confidence calculation
    - Path explanation generation
    - Thread-safe operations
    """

    def __init__(self, netzwerk, use_dynamic_confidence: bool = True):
        """
        Initialize graph traversal core.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz fur Datenzugriff
            use_dynamic_confidence: Ob Dynamic Confidence System genutzt werden soll
        """
        self.netzwerk = netzwerk
        self.max_depth = 5  # Maximale Traversierungstiefe (verhindert Endlosschleifen)
        self.use_dynamic_confidence = (
            use_dynamic_confidence and DYNAMIC_CONFIDENCE_AVAILABLE
        )
        self._lock = threading.RLock()  # Thread safety

        # Initialisiere Dynamic Confidence Manager
        self.dynamic_conf_manager = None
        if self.use_dynamic_confidence:
            try:
                self.dynamic_conf_manager = get_dynamic_confidence_manager(netzwerk)
                logger.info("GraphTraversalCore: Dynamic Confidence System aktiviert")
            except Exception as e:
                logger.warning(
                    f"GraphTraversalCore: Fehler beim Initialisieren von Dynamic Confidence: {e}"
                )
                self.use_dynamic_confidence = False

    def calculate_dynamic_confidence(
        self,
        subject: str,
        relation: str,
        object_: str,
        base_confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """
        Berechnet dynamische Confidence fur eine Edge.

        Wenn Dynamic Confidence System aktiviert ist, wird Confidence on-the-fly
        basierend auf Temporal Decay und Usage Reinforcement berechnet.
        Sonst wird die Base Confidence zuruckgegeben.

        Args:
            subject: Subject der Relation
            relation: Relationstyp
            object_: Object der Relation
            base_confidence: Basis-Confidence aus DB
            timestamp: Zeitstempel der Fact-Erstellung (fur Decay)

        Returns:
            Dynamisch berechnete Confidence (oder base_confidence falls disabled)
        """
        if not self.use_dynamic_confidence or not self.dynamic_conf_manager:
            return base_confidence

        try:
            # Konvertiere timestamp von Neo4j Format falls notig
            if timestamp is not None and not isinstance(timestamp, datetime):
                # Neo4j gibt manchmal Millisekunden seit Epoch zuruck
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000.0)
                elif isinstance(timestamp, str):
                    # Parse ISO format
                    timestamp = datetime.fromisoformat(timestamp)

            # Berechne dynamische Confidence
            metrics = self.dynamic_conf_manager.calculate_dynamic_confidence(
                base_confidence=base_confidence,
                timestamp=timestamp,
                subject=subject,
                relation=relation,
                object_=object_,
            )

            logger.debug(
                f"Dynamic Confidence: {subject} -{relation}-> {object_}: "
                f"{base_confidence:.3f} -> {metrics.value:.3f} ({metrics.explanation})"
            )

            return metrics.value

        except Exception as e:
            logger.warning(
                f"Fehler bei Dynamic Confidence Berechnung: {e}, nutze Base Confidence"
            )
            return base_confidence

    def generate_transitive_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """Generiert Erklarung fur transitive Relation"""
        if not path_relations:
            return ""

        relation_type = path_relations[0]  # Alle gleich bei transitiver Relation
        relation_german = self._relation_to_german(relation_type)

        # Beispiel: "hund ist ein tier (uber saugetier)"
        if len(path_nodes) == 2:
            return f"'{path_nodes[0]}' {relation_german} '{path_nodes[1]}'"
        else:
            intermediate = " -> ".join(path_nodes[1:-1])
            return (
                f"'{path_nodes[0]}' {relation_german} '{path_nodes[-1]}' "
                f"(uber {intermediate})"
            )

    def generate_inverse_transitive_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """Generiert Erklarung fur inverse transitive Relation"""
        if not path_relations:
            return ""

        relation_type = path_relations[0]  # Alle gleich bei transitiver Relation
        relation_german = self._relation_to_german(relation_type)

        # Beispiel: "tier hat Nachfahre hund (uber saugetier)"
        # path_nodes: ["tier", "saugetier", "hund"]
        if len(path_nodes) == 2:
            return f"'{path_nodes[-1]}' {relation_german} '{path_nodes[0]}' (inverse)"
        else:
            intermediate = " <- ".join(path_nodes[1:-1])
            # Kehre die Richtung um fur bessere Lesbarkeit
            return (
                f"'{path_nodes[-1]}' {relation_german} '{path_nodes[0]}' "
                f"(uber {intermediate})"
            )

    def generate_path_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """Generiert Erklarung fur beliebigen Pfad"""
        if not path_relations:
            return ""

        steps = []
        for i in range(len(path_relations)):
            from_node = path_nodes[i]
            to_node = path_nodes[i + 1]
            relation = self._relation_to_german(path_relations[i])
            steps.append(f"{from_node} {relation} {to_node}")

        return " -> ".join(steps)

    def _relation_to_german(self, relation_type: str) -> str:
        """Ubersetzt Relationstyp in deutsche Phrase"""
        mapping = {
            "IS_A": "ist ein(e)",
            "HAS_PROPERTY": "hat die Eigenschaft",
            "CAPABLE_OF": "kann",
            "PART_OF": "ist Teil von",
            "LOCATED_IN": "befindet sich in",
            "HAS_TASTE": "schmeckt",
            "CAUSES": "verursacht",
            "USED_FOR": "wird verwendet fur",
            "KNOWS_THAT": "weiss dass",
        }
        return mapping.get(relation_type, relation_type.lower().replace("_", " "))
