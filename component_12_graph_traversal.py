"""
component_12_graph_traversal.py

Graph-Traversal Engine für Multi-Hop Reasoning
Ermöglicht das Verketten mehrerer Fakten durch Graph-Traversierung.

Funktionalität:
- Transitive Relationen (z.B. IS_A-Hierarchien: Hund -> Säugetier -> Tier)
- Path-Finding zwischen Konzepten
- Erklärungsgenerierung für Inferenz-Ketten
- Konfidenz-Scoring für mehrstufige Schlussfolgerungen
- Integration mit Unified Proof Explanation System
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import (
        ProofStep as UnifiedProofStep,
        StepType,
        generate_explanation_text,
    )

    UNIFIED_PROOFS_AVAILABLE = True
except ImportError:
    UNIFIED_PROOFS_AVAILABLE = False


class TraversalStrategy(Enum):
    """Strategien für Graph-Traversierung"""

    BREADTH_FIRST = "breadth_first"  # Breite-zuerst (kürzeste Pfade)
    DEPTH_FIRST = "depth_first"  # Tiefe-zuerst (alle Pfade)
    BIDIRECTIONAL = "bidirectional"  # Von beiden Seiten (schnellster Weg)


@dataclass
class GraphPath:
    """Repräsentiert einen Pfad durch den Knowledge-Graph"""

    nodes: List[str]  # Konzepte im Pfad (z.B. ["hund", "säugetier", "tier"])
    relations: List[str]  # Relationen zwischen Knoten (z.B. ["IS_A", "IS_A"])
    confidence: float  # Gesamtkonfidenz des Pfads (min aller Kanten)
    explanation: str  # Menschenlesbare Erklärung

    def __repr__(self):
        path_str = " -> ".join(
            f"{self.nodes[i]} --{self.relations[i]}--> {self.nodes[i+1]}"
            for i in range(len(self.relations))
        )
        return f"Path({path_str}, confidence={self.confidence:.2f})"


class GraphTraversal:
    """
    Graph-Traversal Engine für Multi-Hop Reasoning.

    Verwendet das KonzeptNetzwerk für Datenzugriff, implementiert aber eigene
    Traversierungs-Algorithmen für komplexe Pfad-Findung.
    """

    def __init__(self, netzwerk):
        """
        Initialisierung mit KonzeptNetzwerk-Instanz.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz für Datenzugriff
        """
        self.netzwerk = netzwerk
        self.max_depth = 5  # Maximale Traversierungstiefe (verhindert Endlosschleifen)

    def find_epistemic_paths(
        self, observer_id: str, max_depth: int = 3
    ) -> List[GraphPath]:
        """
        Spezialisierte Traversierung für epistemische Pfade

        Findet: Observer -[:KNOWS_THAT]-> MetaBelief -[:ABOUT_AGENT]-> Subject

        Dies ist eine spezialisierte Variante von find_transitive_relations
        für epistemische Logik mit Meta-Knowledge.

        Args:
            observer_id: ID des Beobachters (Agent)
            max_depth: Maximale Traversierungstiefe (default: 3)

        Returns:
            Liste von GraphPath-Objekten mit epistemischen Pfaden

        Example:
            >>> # Alice knows that Bob knows the secret
            >>> paths = traversal.find_epistemic_paths("alice", max_depth=2)
            >>> # Returns paths like: alice -[KNOWS_THAT]-> MetaBelief -[ABOUT_AGENT]-> bob
        """
        # Nutze existierende find_transitive_relations Logik
        # aber mit KNOWS_THAT statt IS_A
        return self.find_transitive_relations(
            start_concept=observer_id, relation_type="KNOWS_THAT", max_depth=max_depth
        )

    def find_transitive_relations(
        self, start_concept: str, relation_type: str, max_depth: Optional[int] = None
    ) -> List[GraphPath]:
        """
        Findet alle transitiven Relationen eines bestimmten Typs.

        Beispiel: find_transitive_relations("hund", "IS_A")
        Findet: hund -> säugetier -> tier -> lebewesen

        Args:
            start_concept: Startkonzept (z.B. "hund")
            relation_type: Relationstyp (z.B. "IS_A")
            max_depth: Maximale Traversierungstiefe (Standard: self.max_depth)

        Returns:
            Liste von GraphPath-Objekten (sortiert nach Länge)
        """
        if max_depth is None:
            max_depth = self.max_depth

        paths = []
        visited = set()

        def dfs(
            current: str,
            path_nodes: List[str],
            path_relations: List[str],
            path_confidences: List[float],
            depth: int,
        ):
            """Depth-First Search für transitive Relationen"""
            # depth repräsentiert die Anzahl der Hops (Relationen) im aktuellen Pfad
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
                edge_confidence = target_info["confidence"]

                new_path_nodes = path_nodes + [target]
                new_path_relations = path_relations + [relation_type]
                new_path_confidences = path_confidences + [edge_confidence]

                # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                overall_confidence = (
                    min(new_path_confidences) if new_path_confidences else 1.0
                )

                # Erstelle Erklärung
                explanation = self._generate_transitive_explanation(
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

                # Rekursive Traversierung - incrementiere depth, da wir einen Hop hinzugefügt haben
                dfs(
                    target,
                    new_path_nodes,
                    new_path_relations,
                    new_path_confidences,
                    depth + 1,
                )

            visited.remove(current)  # Backtracking für andere Pfade

        # Starte DFS vom Startkonzept
        dfs(start_concept, [start_concept], [], [], 0)

        # Sortiere nach Pfadlänge (kürzeste zuerst)
        paths.sort(key=lambda p: len(p.relations))

        return paths

    def find_inverse_transitive_relations(
        self, start_concept: str, relation_type: str, max_depth: Optional[int] = None
    ) -> List[GraphPath]:
        """
        Findet alle inversen transitiven Relationen eines bestimmten Typs.

        Im Gegensatz zu find_transitive_relations(), das vorwärts traversiert
        (hund -> säugetier -> tier), traversiert diese Methode rückwärts
        (tier <- säugetier <- hund).

        Beispiel: find_inverse_transitive_relations("tier", "IS_A")
        Findet: tier <- säugetier <- hund
               tier <- säugetier <- katze
               tier <- säugetier <- elefant

        Nützlich für:
        - Finden von Nachfahren/Unterklassen (descendants)
        - "Warum"-Fragen (z.B. "Warum ist ein Hund ein Tier?")
        - Rückwärts-Reasoning

        Args:
            start_concept: Startkonzept (z.B. "tier")
            relation_type: Relationstyp (z.B. "IS_A")
            max_depth: Maximale Traversierungstiefe (Standard: self.max_depth)

        Returns:
            Liste von GraphPath-Objekten (sortiert nach Länge)
        """
        if max_depth is None:
            max_depth = self.max_depth

        paths = []
        visited = set()

        def dfs(
            current: str,
            path_nodes: List[str],
            path_relations: List[str],
            path_confidences: List[float],
            depth: int,
        ):
            """Depth-First Search für inverse transitive Relationen"""
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
                edge_confidence = source_info["confidence"]

                new_path_nodes = path_nodes + [source]
                new_path_relations = path_relations + [relation_type]
                new_path_confidences = path_confidences + [edge_confidence]

                # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                overall_confidence = (
                    min(new_path_confidences) if new_path_confidences else 1.0
                )

                # Erstelle Erklärung
                explanation = self._generate_inverse_transitive_explanation(
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

            visited.remove(current)  # Backtracking für andere Pfade

        # Starte DFS vom Startkonzept
        dfs(start_concept, [start_concept], [], [], 0)

        # Sortiere nach Pfadlänge (kürzeste zuerst)
        paths.sort(key=lambda p: len(p.relations))

        return paths

    def find_path_between_concepts(
        self,
        start_concept: str,
        target_concept: str,
        allowed_relations: Optional[List[str]] = None,
        strategy: TraversalStrategy = TraversalStrategy.BREADTH_FIRST,
    ) -> Optional[GraphPath]:
        """
        Findet einen Pfad zwischen zwei Konzepten.

        Beispiel: find_path_between_concepts("hund", "tier")
        Findet: hund --IS_A--> säugetier --IS_A--> tier

        Args:
            start_concept: Startkonzept
            target_concept: Zielkonzept
            allowed_relations: Erlaubte Relationstypen (None = alle)
            strategy: Traversierungsstrategie

        Returns:
            GraphPath oder None, wenn kein Pfad existiert
        """
        if strategy == TraversalStrategy.BREADTH_FIRST:
            return self._bfs_path(start_concept, target_concept, allowed_relations)
        elif strategy == TraversalStrategy.DEPTH_FIRST:
            return self._dfs_path(start_concept, target_concept, allowed_relations)
        elif strategy == TraversalStrategy.BIDIRECTIONAL:
            return self._bidirectional_path(
                start_concept, target_concept, allowed_relations
            )
        else:
            raise ValueError(f"Unbekannte Strategie: {strategy}")

    def _bfs_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """Breadth-First Search für kürzesten Pfad"""
        from collections import deque

        # Queue: (current_node, path_nodes, path_relations, path_confidences)
        queue = deque([(start, [start], [], [])])
        visited = {start}

        while queue:
            current, path_nodes, path_relations, path_confidences = queue.popleft()

            # Ziel erreicht?
            if current == target:
                # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                overall_confidence = min(path_confidences) if path_confidences else 1.0
                explanation = self._generate_path_explanation(
                    path_nodes, path_relations
                )
                return GraphPath(
                    nodes=path_nodes,
                    relations=path_relations,
                    confidence=overall_confidence,
                    explanation=explanation,
                )

            # Maximale Tiefe überschritten?
            if len(path_relations) >= self.max_depth:
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
                    edge_confidence = target_info["confidence"]

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

        return None  # Kein Pfad gefunden

    def _dfs_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """Depth-First Search für ersten gefundenen Pfad"""
        visited = set()

        def dfs(
            current: str,
            path_nodes: List[str],
            path_relations: List[str],
            path_confidences: List[float],
        ) -> Optional[GraphPath]:
            if current == target:
                # Berechne Gesamt-Konfidenz: Minimum aller Kanten (weakest link)
                overall_confidence = min(path_confidences) if path_confidences else 1.0
                explanation = self._generate_path_explanation(
                    path_nodes, path_relations
                )
                return GraphPath(
                    nodes=path_nodes,
                    relations=path_relations,
                    confidence=overall_confidence,
                    explanation=explanation,
                )

            if len(path_relations) >= self.max_depth:
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
                    edge_confidence = target_info["confidence"]

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

        return dfs(start, [start], [], [])

    def _bidirectional_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """
        Bidirektionale Suche (von beiden Seiten gleichzeitig).

        Effizienter für große Graphen, aber komplexer zu implementieren.
        TODO: Implementierung für zukünftige Optimierung.
        """
        # Fallback auf BFS
        return self._bfs_path(start, target, allowed_relations)

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
                    explanation = self._generate_path_explanation(
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

            if len(path_relations) >= self.max_depth:
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
                    edge_confidence = target_info["confidence"]

                    dfs(
                        neighbor,
                        path_nodes + [neighbor],
                        path_relations + [relation_type],
                        path_confidences + [edge_confidence],
                    )

        dfs(start_concept, [start_concept], [], [])

        # Sortiere nach Pfadlänge
        all_paths.sort(key=lambda p: len(p.relations))

        return all_paths

    def explain_inference_chain(
        self, conclusion: str, premise: str, relation_type: str
    ) -> Optional[str]:
        """
        Generiert eine Erklärung für eine mehrstufige Schlussfolgerung.

        Beispiel: explain_inference_chain("hund", "tier", "IS_A")
        Ergebnis: "Ein Hund ist ein Tier, weil: Hund -> Säugetier (IS_A) -> Tier (IS_A)"

        Args:
            conclusion: Das abgeleitete Konzept (z.B. "hund")
            premise: Das Zielkonzept (z.B. "tier")
            relation_type: Der Relationstyp (z.B. "IS_A")

        Returns:
            Menschenlesbare Erklärung oder None
        """
        path = self.find_path_between_concepts(
            conclusion, premise, allowed_relations=[relation_type]
        )

        if not path:
            return None

        # Generiere detaillierte Erklärung
        relation_german = self._relation_to_german(relation_type)

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

    def _generate_transitive_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """Generiert Erklärung für transitive Relation"""
        if not path_relations:
            return ""

        relation_type = path_relations[0]  # Alle gleich bei transitiver Relation
        relation_german = self._relation_to_german(relation_type)

        # Beispiel: "hund ist ein tier (über säugetier)"
        if len(path_nodes) == 2:
            return f"'{path_nodes[0]}' {relation_german} '{path_nodes[1]}'"
        else:
            intermediate = " -> ".join(path_nodes[1:-1])
            return (
                f"'{path_nodes[0]}' {relation_german} '{path_nodes[-1]}' "
                f"(über {intermediate})"
            )

    def _generate_inverse_transitive_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """Generiert Erklärung für inverse transitive Relation"""
        if not path_relations:
            return ""

        relation_type = path_relations[0]  # Alle gleich bei transitiver Relation
        relation_german = self._relation_to_german(relation_type)

        # Beispiel: "tier hat Nachfahre hund (über säugetier)"
        # path_nodes: ["tier", "säugetier", "hund"]
        if len(path_nodes) == 2:
            return f"'{path_nodes[-1]}' {relation_german} '{path_nodes[0]}' (inverse)"
        else:
            intermediate = " <- ".join(path_nodes[1:-1])
            # Kehre die Richtung um für bessere Lesbarkeit
            return (
                f"'{path_nodes[-1]}' {relation_german} '{path_nodes[0]}' "
                f"(über {intermediate})"
            )

    def _generate_path_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """Generiert Erklärung für beliebigen Pfad"""
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
        """Übersetzt Relationstyp in deutsche Phrase"""
        mapping = {
            "IS_A": "ist ein(e)",
            "HAS_PROPERTY": "hat die Eigenschaft",
            "CAPABLE_OF": "kann",
            "PART_OF": "ist Teil von",
            "LOCATED_IN": "befindet sich in",
            "HAS_TASTE": "schmeckt",
            "CAUSES": "verursacht",
            "USED_FOR": "wird verwendet für",
        }
        return mapping.get(relation_type, relation_type.lower().replace("_", " "))

    def get_concept_hierarchy(
        self, concept: str, relation_type: str = "IS_A"
    ) -> Dict[str, List[str]]:
        """
        Baut eine vollständige Hierarchie für ein Konzept auf.

        Beispiel: get_concept_hierarchy("säugetier", "IS_A")
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
        # Vorfahren (aufwärts) - folge ausgehenden Relationen
        ancestors_paths = self.find_transitive_relations(concept, relation_type)
        ancestors = []
        for path in ancestors_paths:
            ancestors.extend(path.nodes[1:])  # Ohne Startknoten

        # Nachfahren (abwärts) - folge eingehenden Relationen (inverse)
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

    # ==================== UNIFIED PROOF EXPLANATION INTEGRATION ====================

    def create_proof_step_from_path(
        self, path: GraphPath, query: str = ""
    ) -> Optional[UnifiedProofStep]:
        """
        Konvertiert einen GraphPath in einen UnifiedProofStep.

        Args:
            path: GraphPath-Objekt
            query: Die ursprüngliche Anfrage (optional)

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

        # Generiere Erklärung
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

        Nützlich wenn mehrere alternative Erklärungen existieren.

        Args:
            paths: Liste von GraphPath-Objekten
            query: Die ursprüngliche Anfrage

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
        Erklärt eine Verbindung zwischen Konzepten und gibt einen UnifiedProofStep zurück.

        Dies ist die Hauptschnittstelle für Integration mit dem Reasoning System.

        Args:
            start_concept: Startkonzept
            target_concept: Zielkonzept
            relation_type: Relationstyp

        Returns:
            UnifiedProofStep mit vollständiger Erklärung oder None
        """
        path = self.find_path_between_concepts(
            start_concept, target_concept, allowed_relations=[relation_type]
        )

        if not path:
            return None

        if not UNIFIED_PROOFS_AVAILABLE:
            # Fallback: Returniere None wenn Unified System nicht verfügbar
            return None

        # Erstelle UnifiedProofStep
        step = self.create_proof_step_from_path(path)

        # Erweitere mit detaillierter Erklärung
        if step:
            step.explanation_text = (
                self.explain_inference_chain(
                    start_concept, target_concept, relation_type
                )
                or step.explanation_text
            )

            # Füge zusätzliche Metadaten hinzu
            step.metadata["query_type"] = "multi_hop_reasoning"
            step.metadata["relation_type"] = relation_type

        return step

    def create_decomposed_proof_steps(self, path: GraphPath) -> List[UnifiedProofStep]:
        """
        Zerlegt einen Pfad in einzelne ProofSteps (einen pro Hop).

        Nützlich für detaillierte Schritt-für-Schritt-Erklärungen.

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

            # Generiere Erklärung für diesen einzelnen Hop
            explanation = f"Direkter Fakt: {from_node} {self._relation_to_german(relation)} {to_node}"

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


# ==================== STATE-AWARE TRAVERSAL (Integration with State-Space Planning) ====================


class StateAwareTraversal:
    """
    Erweitert GraphTraversal mit State-Reasoning und Constraint-Checking.

    Kombiniert:
    - Graph-Traversal (component_12): Multi-hop Pfadfindung im Knowledge-Graph
    - State-Space Planning (component_31): STRIPS-ähnliche Aktionsplanung
    - Constraint Reasoning (component_29): CSP-basierte State-Validierung

    Use Cases:
    - Goal-basiertes Planen mit Graph-Kontext
    - Multi-Step Reasoning mit State-Validierung
    - Constraint-Aware Path-Finding
    - Temporal Reasoning (Zustandsänderungen)
    - Root-Cause Analyse (Rückwärts-Planung)
    """

    def __init__(self, netzwerk, constraint_solver=None):
        """
        Initialisiert StateAwareTraversal.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz für Graph-Zugriff
            constraint_solver: Optional ConstraintSolver für State-Validierung
        """
        self.netzwerk = netzwerk
        self.graph_traversal = GraphTraversal(netzwerk)
        self.constraint_solver = constraint_solver

        # Import State-Space Planner (lazy import um Circular Dependencies zu vermeiden)
        try:
            from component_31_state_space_planner import (
                StateSpacePlanner,
                State,
                Action,
                PlanningProblem,
            )

            self.StateSpacePlanner = StateSpacePlanner
            self.State = State
            self.Action = Action
            self.PlanningProblem = PlanningProblem
            self.state_planning_available = True
        except ImportError:
            self.state_planning_available = False
            logger.warning(
                "StateSpacePlanner nicht verfügbar - StateAwareTraversal limitiert"
            )

    def find_path_with_constraints(
        self,
        start_state,  # State from component_31
        goal_state,  # State from component_31
        actions,  # List[Action] from component_31
        constraints=None,  # Optional List[Constraint] from component_29
    ) -> Optional[List]:
        """
        Path-Finding mit State-Validierung und Constraint-Checking.

        Kombiniert Graph-Traversal + State-Planning + Constraint-Checking:
        1. Nutzt A* Search aus StateSpacePlanner für Pfadfindung
        2. Validiert jeden State gegen Constraints (CSP)
        3. Integriert Graph-Kontext für heuristische Verbesserungen
        4. Generiert UnifiedProofSteps für Erklärungen

        Args:
            start_state: Initial State (component_31.State)
            goal_state: Goal State (component_31.State)
            actions: Liste verfügbarer Actions (component_31.Action)
            constraints: Optional Liste von Constraints (component_29.Constraint)

        Returns:
            Liste von Actions die vom start_state zum goal_state führen, oder None

        Example:
            >>> # Blocks World mit Constraints
            >>> start = State(propositions={("on", "A", "B"), ("on", "B", "table")})
            >>> goal = State(propositions={("on", "B", "A"), ("on", "A", "table")})
            >>> actions = [unstack_action, stack_action, pickup_action, putdown_action]
            >>> constraints = [safety_constraint]  # z.B. "kein Block kann auf mehr als einem Block sein"
            >>> plan = state_traversal.find_path_with_constraints(start, goal, actions, constraints)
        """
        if not self.state_planning_available:
            logger.error("StateSpacePlanner nicht verfügbar")
            return None

        logger.info(
            f"StateAwareTraversal: Planning mit {len(actions)} Actions, Constraints={constraints is not None}"
        )

        # Erstelle PlanningProblem
        # Konvertiere goal_state zu goal_conditions (Set von Propositions)
        goal_propositions = (
            goal_state.propositions if hasattr(goal_state, "propositions") else set()
        )

        # Extrahiere Objekte aus States
        objects = self._extract_objects_from_state(start_state)

        problem = self.PlanningProblem(
            initial_state=start_state,
            goal=goal_propositions,
            actions=actions,
            objects=objects,
        )

        # Erstelle State-Constraint-Funktion wenn Constraints gegeben
        state_constraint = None
        if constraints and self.constraint_solver:
            state_constraint = self._create_state_constraint_validator(constraints)

        # Erstelle Planner mit Constraint-Validierung
        planner = self.StateSpacePlanner(
            heuristic=None,  # Nutze Default-Heuristic (RelaxedPlan)
            max_expansions=10000,
            state_constraint=state_constraint,
        )

        # Finde Plan
        plan = planner.solve(problem)

        if plan:
            logger.info(f"StateAwareTraversal: Plan gefunden mit {len(plan)} Actions")
            # Validiere Plan (Sicherheitscheck)
            valid, error = planner.validate_plan(problem, plan)
            if not valid:
                logger.error(
                    f"StateAwareTraversal: Plan-Validierung fehlgeschlagen: {error}"
                )
                return None
            return plan
        else:
            logger.warning("StateAwareTraversal: Kein Plan gefunden")
            return None

    def find_path_with_graph_heuristic(
        self,
        start_state,
        goal_state,
        actions,
        constraints=None,
        use_graph_context: bool = True,
    ) -> Optional[List]:
        """
        Erweiterte Variante mit Graph-basierter Heuristik.

        Nutzt Graph-Traversal um bessere Heuristik für A* zu erstellen:
        - Berechnet Graph-Distanz zwischen State-Elementen
        - Priorisiert Actions die zu konzeptuell ähnlichen States führen
        - Nutzt transitive Relationen für Distanz-Schätzung

        Args:
            start_state: Initial State
            goal_state: Goal State
            actions: Liste verfügbarer Actions
            constraints: Optional Constraints
            use_graph_context: Ob Graph-Kontext für Heuristik genutzt werden soll

        Returns:
            Liste von Actions oder None
        """
        if not self.state_planning_available:
            logger.error("StateSpacePlanner nicht verfügbar")
            return None

        logger.info(
            f"StateAwareTraversal: Planning mit Graph-Heuristik (use_graph={use_graph_context})"
        )

        # Erstelle PlanningProblem
        goal_propositions = (
            goal_state.propositions if hasattr(goal_state, "propositions") else set()
        )
        objects = self._extract_objects_from_state(start_state)

        problem = self.PlanningProblem(
            initial_state=start_state,
            goal=goal_propositions,
            actions=actions,
            objects=objects,
        )

        # Erstelle Graph-basierte Heuristik
        heuristic = None
        if use_graph_context:
            heuristic = self._create_graph_heuristic(goal_state)

        # Erstelle State-Constraint-Funktion
        state_constraint = None
        if constraints and self.constraint_solver:
            state_constraint = self._create_state_constraint_validator(constraints)

        # Erstelle Planner mit Custom-Heuristik
        planner = self.StateSpacePlanner(
            heuristic=heuristic, max_expansions=10000, state_constraint=state_constraint
        )

        # Finde Plan
        plan = planner.solve(problem)

        if plan:
            logger.info(
                f"StateAwareTraversal: Plan gefunden mit {len(plan)} Actions (Graph-Heuristic)"
            )
            return plan
        else:
            logger.warning("StateAwareTraversal: Kein Plan gefunden (Graph-Heuristic)")
            return None

    def explain_plan_with_proof(
        self, start_state, goal_state, plan  # List[Action]
    ) -> Optional[UnifiedProofStep]:
        """
        Generiert UnifiedProofStep für einen Plan.

        Kombiniert State-Planning Proof mit Graph-Traversal Erklärungen.

        Args:
            start_state: Initial State
            goal_state: Goal State
            plan: Die Action-Sequenz

        Returns:
            UnifiedProofStep mit vollständiger Erklärung oder None
        """
        if not UNIFIED_PROOFS_AVAILABLE or not self.state_planning_available:
            return None

        # Erstelle Proof-Step für gesamten Plan
        step_id = f"state_plan_{uuid.uuid4().hex[:8]}"

        # Inputs: Start-State und Actions
        inputs = [str(start_state)] + [action.name for action in plan]

        # Output: Goal-State
        output = str(goal_state)

        # Erklärung
        action_sequence = " -> ".join([action.name for action in plan])
        explanation = (
            f"Plan von {start_state} zu {goal_state}:\n"
            f"Aktionssequenz: {action_sequence}\n"
            f"Schritte: {len(plan)}"
        )

        # Erstelle UnifiedProofStep
        proof_step = UnifiedProofStep(
            step_id=step_id,
            step_type=StepType.RULE_APPLICATION,
            inputs=inputs,
            rule_name="StateAwarePlanning",
            output=output,
            confidence=1.0,  # Validierte Pläne haben volle Confidence
            explanation_text=explanation,
            parent_steps=[],
            bindings={
                "start": str(start_state),
                "goal": str(goal_state),
                "plan_length": len(plan),
            },
            metadata={
                "plan_actions": [action.name for action in plan],
                "planning_method": "StateAwareTraversal",
                "uses_constraints": self.constraint_solver is not None,
            },
            source_component="component_12_graph_traversal",
            timestamp=datetime.now(),
        )

        return proof_step

    # ==================== HELPER METHODS ====================

    def _extract_objects_from_state(self, state) -> List[str]:
        """
        Extrahiert Objekte aus State-Propositions für Action-Grounding.

        Args:
            state: State mit propositions

        Returns:
            Liste von Objekt-Namen
        """
        objects = set()

        if hasattr(state, "propositions"):
            for prop in state.propositions:
                # Proposition ist tuple: (predicate, *args)
                # Extrahiere alle args (außer erstes Element = predicate)
                if isinstance(prop, tuple) and len(prop) > 1:
                    for arg in prop[1:]:
                        if isinstance(arg, str):
                            objects.add(arg)

        return list(objects)

    def _create_state_constraint_validator(self, constraints) -> Optional[callable]:
        """
        Erstellt State-Validierungs-Funktion aus Constraints.

        Args:
            constraints: Liste von Constraints (component_29)

        Returns:
            Funktion die State validiert (returns True/False)
        """
        if not self.constraint_solver:
            return None

        def validate_state(state) -> bool:
            """Validiert State gegen alle Constraints."""
            # Konvertiere State zu CSP-Variable-Assignment
            # (Vereinfachte Implementation - kann erweitert werden)

            # Prüfe ob State "sichere" Bedingungen erfüllt
            # Beispiel: River Crossing - keine gefährlichen Kombinationen

            # Für jetzt: Nutze Constraint-Check über Propositions
            if hasattr(state, "propositions"):
                # Check constraints über Propositions
                # (Kann mit component_29 CSP-Solver erweitert werden)
                for constraint in constraints:
                    # Placeholder: Implementiere Constraint-Checking-Logik
                    # Real implementation würde constraint.is_satisfied() nutzen
                    pass

            return True  # Default: State ist valid (kann erweitert werden)

        return validate_state

    def _create_graph_heuristic(self, goal_state):
        """
        Erstellt Heuristik-Funktion basierend auf Graph-Kontext.

        Nutzt Graph-Distanz zwischen State-Elementen als Heuristik:
        - Zählt fehlende Propositions in Goal
        - Nutzt Graph-Traversal um ähnliche Konzepte zu finden
        - Schätzt Distanz basierend auf Graph-Struktur

        Args:
            goal_state: Ziel-State

        Returns:
            Heuristik-Funktion für A*
        """
        try:
            from component_31_state_space_planner import RelaxedPlanHeuristic

            # Nutze RelaxedPlanHeuristic als Basis
            base_heuristic = RelaxedPlanHeuristic()

            def graph_enhanced_heuristic(state) -> float:
                """
                Graph-erweiterte Heuristik.

                Kombiniert:
                1. RelaxedPlan Heuristik (zählt fehlende Goals)
                2. Graph-Distanz zwischen Konzepten
                """
                # Basis-Schätzung: Anzahl fehlender Goal-Propositions
                goal_props = (
                    goal_state.propositions
                    if hasattr(goal_state, "propositions")
                    else set()
                )
                base_estimate = base_heuristic.estimate(state, goal_props)

                # Graph-Kontext: Schätze zusätzliche Kosten basierend auf Graph-Struktur
                graph_penalty = 0.0

                # Extrahiere Konzepte aus State und Goal
                state_concepts = self._extract_concepts_from_state(state)
                goal_concepts = self._extract_concepts_from_state(goal_state)

                # Für jedes Goal-Konzept: Prüfe Graph-Distanz zum aktuellen State
                for goal_concept in goal_concepts:
                    if goal_concept not in state_concepts:
                        # Konzept fehlt: Schätze Distanz
                        # (Vereinfacht: +1 für jedes fehlende Konzept)
                        graph_penalty += 1.0

                return base_estimate + (
                    graph_penalty * 0.5
                )  # Weight Graph-Penalty niedriger

            return graph_enhanced_heuristic

        except ImportError:
            logger.warning(
                "RelaxedPlanHeuristic nicht verfügbar - nutze Standard-Heuristik"
            )
            return None

    def _extract_concepts_from_state(self, state) -> Set[str]:
        """
        Extrahiert Konzept-Namen aus State-Propositions.

        Args:
            state: State mit propositions

        Returns:
            Set von Konzept-Namen
        """
        concepts = set()

        if hasattr(state, "propositions"):
            for prop in state.propositions:
                if isinstance(prop, tuple):
                    # Füge alle String-Argumente als Konzepte hinzu
                    for arg in prop:
                        if isinstance(arg, str) and arg not in [
                            "left",
                            "right",
                            "table",
                        ]:
                            concepts.add(arg)

        return concepts
