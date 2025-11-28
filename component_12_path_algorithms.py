"""
component_12_path_algorithms.py

Path-finding algorithms and transitive inference logic.
Handles forward/inverse transitive relations, hierarchies, and proof generation.

This module contains:
- Transitive relation finding (forward: A->B->C)
- Inverse transitive relations (backward: C<-B<-A)
- Concept hierarchy building (ancestors + descendants)
- Inference chain explanation
- Unified proof step generation
- Epistemic path finding for meta-reasoning

Extracted from component_12_graph_traversal.py (Task 10, Phase 3 Architecture Refactoring)

Author: KAI Development Team
Date: 2025-11-28
"""

import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from component_12_graph_traversal_core import (
    UNIFIED_PROOFS_AVAILABLE,
    GraphPath,
    GraphTraversalCore,
)

if UNIFIED_PROOFS_AVAILABLE:
    from component_17_proof_explanation import (
        ProofStep as UnifiedProofStep,
        StepType,
        generate_explanation_text,
    )

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class PathAlgorithms:
    """
    Path-finding algorithms and transitive inference.

    Provides:
    - Transitive relation finding (forward + inverse)
    - Concept hierarchy construction
    - Inference chain explanation
    - Proof tree generation for reasoning chains
    """

    def __init__(self, core: GraphTraversalCore):
        """
        Initialize path algorithms.

        Args:
            core: GraphTraversalCore instance for shared utilities
        """
        self.core = core
        self.netzwerk = core.netzwerk
        self._lock = threading.RLock()

    def find_transitive_relations(
        self, start_concept: str, relation_type: str, max_depth: Optional[int] = None
    ) -> List[GraphPath]:
        """
        Findet alle transitiven Relationen eines bestimmten Typs.

        Beispiel: find_transitive_relations("hund", "IS_A")
        Findet: hund -> saugetier -> tier -> lebewesen

        Args:
            start_concept: Startkonzept (z.B. "hund")
            relation_type: Relationstyp (z.B. "IS_A")
            max_depth: Maximale Traversierungstiefe (Standard: self.core.max_depth)

        Returns:
            Liste von GraphPath-Objekten (sortiert nach Lange)
        """
        if max_depth is None:
            max_depth = self.core.max_depth

        paths = []
        visited = set()

        def dfs(
            current: str,
            path_nodes: List[str],
            path_relations: List[str],
            path_confidences: List[float],
            depth: int,
        ):
            """Depth-First Search fur transitive Relationen"""
            # depth reprasentiert die Anzahl der Hops (Relationen) im aktuellen Pfad
            if depth >= max_depth:
                return

            if current in visited:
                return  # Zyklen vermeiden

            visited.add(current)

            # Hole alle ausgehenden Relationen dieses Typs MIT Confidence
            facts_with_conf = self.netzwerk.query_graph_for_facts_with_confidence(
                current
            )
            targets_with_conf = facts_with_conf.get(relation_type, [])

            for target_info in targets_with_conf:
                target = target_info["target"]
                base_confidence = target_info["confidence"]
                timestamp = target_info.get("timestamp")

                # Berechne dynamische Confidence
                edge_confidence = self.core.calculate_dynamic_confidence(
                    subject=current,
                    relation=relation_type,
                    object_=target,
                    base_confidence=base_confidence,
                    timestamp=timestamp,
                )

                new_path_nodes = path_nodes + [target]
                new_path_relations = path_relations + [relation_type]
                new_path_confidences = path_confidences + [edge_confidence]

                # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                overall_confidence = (
                    min(new_path_confidences) if new_path_confidences else 1.0
                )

                # Erstelle Erklarung
                explanation = self.core.generate_transitive_explanation(
                    new_path_nodes, new_path_relations
                )

                # Erstelle GraphPath
                path = GraphPath(
                    nodes=new_path_nodes,
                    relations=new_path_relations,
                    confidence=overall_confidence,
                    explanation=explanation,
                )
                paths.append(path)

                # Rekursive Traversierung - incrementiere depth, da wir einen Hop hinzugefugt haben
                dfs(
                    target,
                    new_path_nodes,
                    new_path_relations,
                    new_path_confidences,
                    depth + 1,
                )

            visited.remove(current)  # Backtracking fur andere Pfade

        with self._lock:
            # Starte DFS vom Startkonzept
            dfs(start_concept, [start_concept], [], [], 0)

            # Sortiere nach Pfadlange (kurzeste zuerst)
            paths.sort(key=lambda p: len(p.relations))

            return paths

    def find_inverse_transitive_relations(
        self, start_concept: str, relation_type: str, max_depth: Optional[int] = None
    ) -> List[GraphPath]:
        """
        Findet alle inversen transitiven Relationen eines bestimmten Typs.

        Im Gegensatz zu find_transitive_relations(), das vorwarts traversiert
        (hund -> saugetier -> tier), traversiert diese Methode ruckwarts
        (tier <- saugetier <- hund).

        Beispiel: find_inverse_transitive_relations("tier", "IS_A")
        Findet: tier <- saugetier <- hund
               tier <- saugetier <- katze
               tier <- saugetier <- elefant

        Nutzlich fur:
        - Finden von Nachfahren/Unterklassen (descendants)
        - "Warum"-Fragen (z.B. "Warum ist ein Hund ein Tier?")
        - Ruckwarts-Reasoning

        Args:
            start_concept: Startkonzept (z.B. "tier")
            relation_type: Relationstyp (z.B. "IS_A")
            max_depth: Maximale Traversierungstiefe (Standard: self.core.max_depth)

        Returns:
            Liste von GraphPath-Objekten (sortiert nach Lange)
        """
        if max_depth is None:
            max_depth = self.core.max_depth

        paths = []
        visited = set()

        def dfs(
            current: str,
            path_nodes: List[str],
            path_relations: List[str],
            path_confidences: List[float],
            depth: int,
        ):
            """Depth-First Search fur inverse transitive Relationen"""
            if depth >= max_depth:
                return

            if current in visited:
                return  # Zyklen vermeiden

            visited.add(current)

            # Hole alle eingehenden Relationen dieses Typs MIT Confidence
            inverse_facts_with_conf = (
                self.netzwerk.query_inverse_relations_with_confidence(
                    current, relation_type
                )
            )
            sources_with_conf = inverse_facts_with_conf.get(relation_type, [])

            for source_info in sources_with_conf:
                source = source_info["source"]
                base_confidence = source_info["confidence"]
                timestamp = source_info.get("timestamp")

                # Berechne dynamische Confidence
                edge_confidence = self.core.calculate_dynamic_confidence(
                    subject=source,
                    relation=relation_type,
                    object_=current,
                    base_confidence=base_confidence,
                    timestamp=timestamp,
                )

                new_path_nodes = path_nodes + [source]
                new_path_relations = path_relations + [relation_type]
                new_path_confidences = path_confidences + [edge_confidence]

                # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                overall_confidence = (
                    min(new_path_confidences) if new_path_confidences else 1.0
                )

                # Erstelle Erklarung
                explanation = self.core.generate_inverse_transitive_explanation(
                    new_path_nodes, new_path_relations
                )

                # Erstelle GraphPath
                path = GraphPath(
                    nodes=new_path_nodes,
                    relations=new_path_relations,
                    confidence=overall_confidence,
                    explanation=explanation,
                )
                paths.append(path)

                # Rekursive Traversierung
                dfs(
                    source,
                    new_path_nodes,
                    new_path_relations,
                    new_path_confidences,
                    depth + 1,
                )

            visited.remove(current)  # Backtracking fur andere Pfade

        with self._lock:
            # Starte DFS vom Startkonzept
            dfs(start_concept, [start_concept], [], [], 0)

            # Sortiere nach Pfadlange (kurzeste zuerst)
            paths.sort(key=lambda p: len(p.relations))

            return paths

    def find_epistemic_paths(
        self, observer_id: str, max_depth: int = 3
    ) -> List[GraphPath]:
        """
        Spezialisierte Traversierung fur epistemische Pfade.

        Findet: Observer -[:KNOWS_THAT]-> MetaBelief -[:ABOUT_AGENT]-> Subject

        Dies ist eine spezialisierte Variante von find_transitive_relations
        fur epistemische Logik mit Meta-Knowledge.

        Args:
            observer_id: ID des Beobachters (Agent)
            max_depth: Maximale Traversierungstiefe (default: 3)

        Returns:
            Liste von GraphPath-Objekten mit epistemischen Pfaden

        Example:
            >>> # Alice knows that Bob knows the secret
            >>> paths = path_algos.find_epistemic_paths("alice", max_depth=2)
            >>> # Returns paths like: alice -[KNOWS_THAT]-> MetaBelief -[ABOUT_AGENT]-> bob
        """
        # Nutze existierende find_transitive_relations Logik
        # aber mit KNOWS_THAT statt IS_A
        return self.find_transitive_relations(
            start_concept=observer_id, relation_type="KNOWS_THAT", max_depth=max_depth
        )

    def get_concept_hierarchy(
        self, concept: str, relation_type: str = "IS_A"
    ) -> Dict[str, List[str]]:
        """
        Baut eine vollstandige Hierarchie fur ein Konzept auf.

        Beispiel: get_concept_hierarchy("saugetier", "IS_A")
        Ergebnis: {
            "ancestors": ["tier", "lebewesen"],
            "descendants": ["hund", "katze", "elefant"]
        }

        Args:
            concept: Das Konzept
            relation_type: Relationstyp (Standard: IS_A)

        Returns:
            Dict mit "ancestors" und "descendants" Listen
        """
        with self._lock:
            # Vorfahren (aufwarts) - folge ausgehenden Relationen
            ancestors_paths = self.find_transitive_relations(concept, relation_type)
            ancestors = []
            for path in ancestors_paths:
                ancestors.extend(path.nodes[1:])  # Ohne Startknoten

            # Nachfahren (abwarts) - folge eingehenden Relationen (inverse)
            descendants_paths = self.find_inverse_transitive_relations(
                concept, relation_type
            )
            descendants = []
            for path in descendants_paths:
                descendants.extend(path.nodes[1:])  # Ohne Startknoten

            return {
                "ancestors": list(
                    dict.fromkeys(ancestors)
                ),  # Deduplizieren, Reihenfolge erhalten
                "descendants": list(
                    dict.fromkeys(descendants)
                ),  # Deduplizieren, Reihenfolge erhalten
            }

    def explain_inference_chain(
        self, conclusion: str, premise: str, relation_type: str
    ) -> Optional[str]:
        """
        Generiert eine Erklarung fur eine mehrstufige Schlussfolgerung.

        Beispiel: explain_inference_chain("hund", "tier", "IS_A")
        Ergebnis: "Ein Hund ist ein Tier, weil: Hund -> Saugetier (IS_A) -> Tier (IS_A)"

        Args:
            conclusion: Das abgeleitete Konzept (z.B. "hund")
            premise: Das Zielkonzept (z.B. "tier")
            relation_type: Der Relationstyp (z.B. "IS_A")

        Returns:
            Menschenlesbare Erklarung oder None
        """
        # Import strategies to find path
        from component_12_traversal_strategies import TraversalStrategies

        strategies = TraversalStrategies(self.core)
        path = strategies.find_path_between_concepts(
            conclusion, premise, allowed_relations=[relation_type]
        )

        if not path:
            return None

        # Generiere detaillierte Erklarung
        relation_german = self.core._relation_to_german(relation_type)

        steps = []
        for i in range(len(path.relations)):
            from_node = path.nodes[i]
            to_node = path.nodes[i + 1]
            steps.append(f"{from_node} {relation_german} {to_node}")

        chain = " -> ".join(steps)

        explanation = (
            f"'{path.nodes[0]}' {relation_german} '{path.nodes[-1]}', " f"weil: {chain}"
        )

        return explanation

    # ==================== UNIFIED PROOF EXPLANATION INTEGRATION ====================

    def create_proof_step_from_path(
        self, path: GraphPath, query: str = ""
    ) -> Optional[UnifiedProofStep]:
        """
        Konvertiert einen GraphPath in einen UnifiedProofStep.

        Args:
            path: GraphPath-Objekt
            query: Die ursprungliche Anfrage (optional)

        Returns:
            UnifiedProofStep oder None
        """
        if not UNIFIED_PROOFS_AVAILABLE or not path:
            return None

        # Generiere Step-ID
        step_id = f"graph_traversal_{uuid.uuid4().hex[:8]}"

        # Erstelle Inputs (alle Zwischenknoten)
        inputs = [
            f"{path.nodes[i]} --{path.relations[i]}--> {path.nodes[i+1]}"
            for i in range(len(path.relations))
        ]

        # Output ist die finale Schlussfolgerung
        output = f"{path.nodes[0]} {path.relations[0] if path.relations else ''} {path.nodes[-1]}"

        # Generiere Erklarung
        explanation = generate_explanation_text(
            step_type=StepType.GRAPH_TRAVERSAL,
            inputs=inputs,
            output=output,
            metadata={
                "hops": len(path.relations),
                "path": path.nodes,
                "relations": path.relations,
            },
        )

        return UnifiedProofStep(
            step_id=step_id,
            step_type=StepType.GRAPH_TRAVERSAL,
            inputs=inputs,
            rule_name=None,
            output=output,
            confidence=path.confidence,
            explanation_text=explanation,
            parent_steps=[],
            bindings={f"node_{i}": node for i, node in enumerate(path.nodes)},
            metadata={
                "path_nodes": path.nodes,
                "path_relations": path.relations,
                "hops": len(path.relations),
                "original_explanation": path.explanation,
            },
            source_component="component_12_graph_traversal",
            timestamp=datetime.now(),
        )

    def create_multi_hop_proof_chain(
        self, paths: List[GraphPath], query: str = ""
    ) -> List[UnifiedProofStep]:
        """
        Erstellt eine Proof-Kette aus mehreren Pfaden.

        Nutzlich wenn mehrere alternative Erklarungen existieren.

        Args:
            paths: Liste von GraphPath-Objekten
            query: Die ursprungliche Anfrage

        Returns:
            Liste von UnifiedProofStep-Objekten
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            return []

        proof_steps = []
        for i, path in enumerate(paths):
            step = self.create_proof_step_from_path(path, query)
            if step:
                # Markiere alternative Pfade
                step.metadata["alternative_rank"] = i + 1
                step.metadata["total_alternatives"] = len(paths)
                proof_steps.append(step)

        return proof_steps

    def explain_with_proof_step(
        self, start_concept: str, target_concept: str, relation_type: str
    ) -> Optional[UnifiedProofStep]:
        """
        Erklart eine Verbindung zwischen Konzepten und gibt einen UnifiedProofStep zuruck.

        Dies ist die Hauptschnittstelle fur Integration mit dem Reasoning System.

        Args:
            start_concept: Startkonzept
            target_concept: Zielkonzept
            relation_type: Relationstyp

        Returns:
            UnifiedProofStep mit vollstandiger Erklarung oder None
        """
        from component_12_traversal_strategies import TraversalStrategies

        strategies = TraversalStrategies(self.core)
        path = strategies.find_path_between_concepts(
            start_concept, target_concept, allowed_relations=[relation_type]
        )

        if not path:
            return None

        if not UNIFIED_PROOFS_AVAILABLE:
            # Fallback: Returniere None wenn Unified System nicht verfugbar
            return None

        # Erstelle UnifiedProofStep
        step = self.create_proof_step_from_path(path)

        # Erweitere mit detaillierter Erklarung
        if step:
            step.explanation_text = (
                self.explain_inference_chain(
                    start_concept, target_concept, relation_type
                )
                or step.explanation_text
            )

            # Fuge zusatzliche Metadaten hinzu
            step.metadata["query_type"] = "multi_hop_reasoning"
            step.metadata["relation_type"] = relation_type

        return step

    def create_decomposed_proof_steps(self, path: GraphPath) -> List[UnifiedProofStep]:
        """
        Zerlegt einen Pfad in einzelne ProofSteps (einen pro Hop).

        Nutzlich fur detaillierte Schritt-fur-Schritt-Erklarungen.

        Args:
            path: GraphPath-Objekt

        Returns:
            Liste von UnifiedProofStep-Objekten (einer pro Relation)
        """
        if not UNIFIED_PROOFS_AVAILABLE or not path.relations:
            return []

        steps = []
        for i in range(len(path.relations)):
            step_id = f"graph_hop_{i+1}_{uuid.uuid4().hex[:6]}"

            from_node = path.nodes[i]
            to_node = path.nodes[i + 1]
            relation = path.relations[i]

            # Erstelle Input/Output
            input_str = from_node
            output_str = f"{from_node} --{relation}--> {to_node}"

            # Generiere Erklarung fur diesen einzelnen Hop
            explanation = f"Direkter Fakt: {from_node} {self.core._relation_to_german(relation)} {to_node}"

            step = UnifiedProofStep(
                step_id=step_id,
                step_type=StepType.FACT_MATCH,  # Jeder einzelne Hop ist ein Fakt
                inputs=[input_str],
                rule_name=None,
                output=output_str,
                confidence=1.0,  # Einzelne Fakten haben volle Konfidenz
                explanation_text=explanation,
                parent_steps=(
                    [steps[-1].step_id] if steps else []
                ),  # Verkette mit vorherigem
                bindings={"from": from_node, "to": to_node},
                metadata={
                    "hop_number": i + 1,
                    "total_hops": len(path.relations),
                    "relation": relation,
                },
                source_component="component_12_graph_traversal",
            )

            steps.append(step)

        return steps
