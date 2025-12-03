"""
component_12_graph_traversal.py - Facade

Graph-Traversal Engine fur Multi-Hop Reasoning (Unified Facade)
Ermoglicht das Verketten mehrerer Fakten durch Graph-Traversierung.

This is a facade providing backward compatibility after refactoring into:
- component_12_graph_traversal_core.py: Core engine and utilities
- component_12_traversal_strategies.py: BFS, DFS, bidirectional strategies
- component_12_path_algorithms.py: Transitive inference and proof generation

Funktionalitat:
- Transitive Relationen (z.B. IS_A-Hierarchien: Hund -> Saugetier -> Tier)
- Path-Finding zwischen Konzepten
- Erklarungsgenerierung fur Inferenz-Ketten
- Konfidenz-Scoring fur mehrstufige Schlussfolgerungen
- Integration mit Unified Proof Explanation System

Refactored: 2025-11-28 (Task 10, Phase 3 Architecture Refactoring)
Original: component_12_graph_traversal.py (1,484 lines)
"""

import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Re-export all public types for backward compatibility
from component_12_graph_traversal_core import (
    UNIFIED_PROOFS_AVAILABLE,
    GraphPath,
    GraphTraversalCore,
    TraversalStrategy,
)
from component_12_path_algorithms import PathAlgorithms
from component_12_traversal_strategies import TraversalStrategies
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult

if UNIFIED_PROOFS_AVAILABLE:
    from component_17_proof_explanation import ProofStep as UnifiedProofStep

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class GraphTraversal(BaseReasoningEngine):
    """
    Graph-Traversal Engine fur Multi-Hop Reasoning (Facade).

    Verwendet das KonzeptNetzwerk fur Datenzugriff, implementiert aber eigene
    Traversierungs-Algorithmen fur komplexe Pfad-Findung.

    This facade delegates to:
    - GraphTraversalCore: Core utilities and helpers
    - TraversalStrategies: BFS, DFS, bidirectional search
    - PathAlgorithms: Transitive inference and proof generation

    Implements BaseReasoningEngine for integration with reasoning orchestrator.
    """

    def __init__(self, netzwerk, use_dynamic_confidence: bool = True):
        """
        Initialisierung mit KonzeptNetzwerk-Instanz.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz fur Datenzugriff
            use_dynamic_confidence: Ob Dynamic Confidence System genutzt werden soll (Default: True)
        """
        self.netzwerk = netzwerk
        self._lock = threading.RLock()

        # Initialize core components
        self.core = GraphTraversalCore(netzwerk, use_dynamic_confidence)
        self.strategies = TraversalStrategies(self.core)
        self.path_algorithms = PathAlgorithms(self.core)

        # Expose max_depth for backward compatibility
        self.max_depth = self.core.max_depth

        # Expose dynamic confidence manager for backward compatibility
        self.use_dynamic_confidence = self.core.use_dynamic_confidence
        self.dynamic_conf_manager = self.core.dynamic_conf_manager

    # ==================== TRANSITIVE RELATIONS ====================

    def find_transitive_relations(
        self, start_concept: str, relation_type: str, max_depth: Optional[int] = None
    ) -> List[GraphPath]:
        """
        Findet alle transitiven Relationen eines bestimmten Typs.

        Delegates to PathAlgorithms.find_transitive_relations()
        """
        return self.path_algorithms.find_transitive_relations(
            start_concept, relation_type, max_depth
        )

    def find_inverse_transitive_relations(
        self, start_concept: str, relation_type: str, max_depth: Optional[int] = None
    ) -> List[GraphPath]:
        """
        Findet alle inversen transitiven Relationen eines bestimmten Typs.

        Delegates to PathAlgorithms.find_inverse_transitive_relations()
        """
        return self.path_algorithms.find_inverse_transitive_relations(
            start_concept, relation_type, max_depth
        )

    def find_epistemic_paths(
        self, observer_id: str, max_depth: int = 3
    ) -> List[GraphPath]:
        """
        Spezialisierte Traversierung fur epistemische Pfade.

        Delegates to PathAlgorithms.find_epistemic_paths()
        """
        return self.path_algorithms.find_epistemic_paths(observer_id, max_depth)

    # ==================== PATH FINDING ====================

    def find_path_between_concepts(
        self,
        start_concept: str,
        target_concept: str,
        allowed_relations: Optional[List[str]] = None,
        strategy: TraversalStrategy = TraversalStrategy.BREADTH_FIRST,
    ) -> Optional[GraphPath]:
        """
        Findet einen Pfad zwischen zwei Konzepten.

        Delegates to TraversalStrategies.find_path_between_concepts()
        """
        return self.strategies.find_path_between_concepts(
            start_concept, target_concept, allowed_relations, strategy
        )

    def find_all_paths_between_concepts(
        self,
        start_concept: str,
        target_concept: str,
        allowed_relations: Optional[List[str]] = None,
        max_paths: int = 10,
    ) -> List[GraphPath]:
        """
        Findet ALLE Pfade zwischen zwei Konzepten (bis zu max_paths).

        Delegates to TraversalStrategies.find_all_paths_between_concepts()
        """
        return self.strategies.find_all_paths_between_concepts(
            start_concept, target_concept, allowed_relations, max_paths
        )

    # ==================== HIERARCHY & EXPLANATION ====================

    def get_concept_hierarchy(
        self, concept: str, relation_type: str = "IS_A"
    ) -> Dict[str, List[str]]:
        """
        Baut eine vollstandige Hierarchie fur ein Konzept auf.

        Delegates to PathAlgorithms.get_concept_hierarchy()
        """
        return self.path_algorithms.get_concept_hierarchy(concept, relation_type)

    def explain_inference_chain(
        self, conclusion: str, premise: str, relation_type: str
    ) -> Optional[str]:
        """
        Generiert eine Erklarung fur eine mehrstufige Schlussfolgerung.

        Delegates to PathAlgorithms.explain_inference_chain()
        """
        return self.path_algorithms.explain_inference_chain(
            conclusion, premise, relation_type
        )

    # ==================== UNIFIED PROOF EXPLANATION INTEGRATION ====================

    def create_proof_step_from_path(
        self, path: GraphPath, query: str = ""
    ) -> Optional[UnifiedProofStep]:
        """
        Konvertiert einen GraphPath in einen UnifiedProofStep.

        Delegates to PathAlgorithms.create_proof_step_from_path()
        """
        return self.path_algorithms.create_proof_step_from_path(path, query)

    def create_multi_hop_proof_chain(
        self, paths: List[GraphPath], query: str = ""
    ) -> List[UnifiedProofStep]:
        """
        Erstellt eine Proof-Kette aus mehreren Pfaden.

        Delegates to PathAlgorithms.create_multi_hop_proof_chain()
        """
        return self.path_algorithms.create_multi_hop_proof_chain(paths, query)

    def explain_with_proof_step(
        self, start_concept: str, target_concept: str, relation_type: str
    ) -> Optional[UnifiedProofStep]:
        """
        Erklart eine Verbindung zwischen Konzepten und gibt einen UnifiedProofStep zuruck.

        Delegates to PathAlgorithms.explain_with_proof_step()
        """
        return self.path_algorithms.explain_with_proof_step(
            start_concept, target_concept, relation_type
        )

    def create_decomposed_proof_steps(self, path: GraphPath) -> List[UnifiedProofStep]:
        """
        Zerlegt einen Pfad in einzelne ProofSteps (einen pro Hop).

        Delegates to PathAlgorithms.create_decomposed_proof_steps()
        """
        return self.path_algorithms.create_decomposed_proof_steps(path)

    # ==================== PRIVATE HELPER METHODS (for backward compatibility) ====================

    def _calculate_dynamic_confidence(
        self,
        subject: str,
        relation: str,
        object_: str,
        base_confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """
        Berechnet dynamische Confidence fur eine Edge.

        Delegates to GraphTraversalCore.calculate_dynamic_confidence()
        """
        return self.core.calculate_dynamic_confidence(
            subject, relation, object_, base_confidence, timestamp
        )

    def _generate_transitive_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """
        Generiert Erklarung fur transitive Relation.

        Delegates to GraphTraversalCore.generate_transitive_explanation()
        """
        return self.core.generate_transitive_explanation(path_nodes, path_relations)

    def _generate_inverse_transitive_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """
        Generiert Erklarung fur inverse transitive Relation.

        Delegates to GraphTraversalCore.generate_inverse_transitive_explanation()
        """
        return self.core.generate_inverse_transitive_explanation(
            path_nodes, path_relations
        )

    def _generate_path_explanation(
        self, path_nodes: List[str], path_relations: List[str]
    ) -> str:
        """
        Generiert Erklarung fur beliebigen Pfad.

        Delegates to GraphTraversalCore.generate_path_explanation()
        """
        return self.core.generate_path_explanation(path_nodes, path_relations)

    def _relation_to_german(self, relation_type: str) -> str:
        """
        Ubersetzt Relationstyp in deutsche Phrase.

        Delegates to GraphTraversalCore._relation_to_german()
        """
        return self.core._relation_to_german(relation_type)

    # ==================== BASE REASONING ENGINE INTERFACE ====================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Execute multi-hop graph reasoning on the query.

        Args:
            query: Natural language query to reason about
            context: Context with start_concept, target_concept, relation_type, etc.

        Returns:
            ReasoningResult with answer, confidence, and proof tree
        """
        # Extract parameters from context
        start_concept = context.get("start_concept", context.get("subject"))
        target_concept = context.get("target_concept", context.get("object"))
        relation_type = context.get("relation_type", "IS_A")
        context.get("max_depth", self.max_depth)

        if not start_concept or not target_concept:
            return ReasoningResult(
                success=False,
                answer="Start- oder Zielkonzept fehlt fur Graph-Traversierung",
                confidence=0.0,
                strategy_used="graph_traversal",
            )

        # Find path between concepts
        path = self.find_path_between_concepts(
            start_concept=start_concept,
            target_concept=target_concept,
            allowed_relations=[relation_type] if relation_type else None,
            strategy=TraversalStrategy.BREADTH_FIRST,
        )

        if path:
            # Generate explanation and proof tree
            explanation = self.explain_inference_chain(
                target_concept, start_concept, relation_type
            )
            proof_tree = None

            if UNIFIED_PROOFS_AVAILABLE:
                proof_step = self.create_proof_step_from_path(path, query)
                if proof_step:
                    from component_17_proof_explanation import ProofTree

                    proof_tree = ProofTree(query=query)
                    proof_tree.add_root_step(proof_step)

            return ReasoningResult(
                success=True,
                answer=explanation or f"Pfad gefunden: {' -> '.join(path.nodes)}",
                confidence=path.confidence,
                proof_tree=proof_tree,
                strategy_used="graph_traversal_bfs",
                metadata={
                    "path_length": len(path.nodes),
                    "path_nodes": path.nodes,
                    "path_relations": path.relations,
                },
            )
        else:
            return ReasoningResult(
                success=False,
                answer=f"Kein Pfad gefunden zwischen {start_concept} und {target_concept}",
                confidence=0.0,
                strategy_used="graph_traversal",
                metadata={
                    "start": start_concept,
                    "target": target_concept,
                    "relation": relation_type,
                },
            )

    def get_capabilities(self) -> List[str]:
        """Return list of reasoning capabilities."""
        return [
            "graph_traversal",
            "multi_hop_inference",
            "transitive_relations",
            "path_finding",
            "hierarchical_reasoning",
        ]

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for graph traversal.

        Cost depends on:
        - Graph size (estimated from netzwerk)
        - Max depth setting
        - Query complexity

        Returns:
            Cost estimate in [0.0, 1.0] range
        """
        # Graph traversal is generally medium cost
        base_cost = 0.4

        # Max depth affects cost (deeper = more expensive)
        depth_cost = min(self.max_depth / 10.0, 0.2)

        # Query complexity
        query_complexity = min(len(query) / 200.0, 0.1)

        total_cost = base_cost + depth_cost + query_complexity

        return min(total_cost, 1.0)

    def _bfs_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """
        Breadth-First Search fur kurzesten Pfad.

        Delegates to TraversalStrategies.bfs_path()
        """
        return self.strategies.bfs_path(start, target, allowed_relations)

    def _dfs_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """
        Depth-First Search fur ersten gefundenen Pfad.

        Delegates to TraversalStrategies.dfs_path()
        """
        return self.strategies.dfs_path(start, target, allowed_relations)

    def _bidirectional_path(
        self, start: str, target: str, allowed_relations: Optional[List[str]]
    ) -> Optional[GraphPath]:
        """
        Bidirektionale Suche (von beiden Seiten gleichzeitig).

        Delegates to TraversalStrategies.bidirectional_path()
        """
        return self.strategies.bidirectional_path(start, target, allowed_relations)


# ==================== STATE-AWARE TRAVERSAL ====================


class StateAwareTraversal:
    """
    Erweitert GraphTraversal mit State-Reasoning und Constraint-Checking.

    Kombiniert:
    - Graph-Traversal (component_12): Multi-hop Pfadfindung im Knowledge-Graph
    - State-Space Planning (component_31): STRIPS-ahnliche Aktionsplanung
    - Constraint Reasoning (component_29): CSP-basierte State-Validierung

    Use Cases:
    - Goal-basiertes Planen mit Graph-Kontext
    - Multi-Step Reasoning mit State-Validierung
    - Constraint-Aware Path-Finding
    - Temporal Reasoning (Zustandsanderungen)
    - Root-Cause Analyse (Ruckwarts-Planung)
    """

    def __init__(self, netzwerk, constraint_solver=None):
        """
        Initialisiert StateAwareTraversal.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz fur Graph-Zugriff
            constraint_solver: Optional ConstraintSolver fur State-Validierung
        """
        self.netzwerk = netzwerk
        self.graph_traversal = GraphTraversal(netzwerk)
        self.constraint_solver = constraint_solver

        # Import State-Space Planner (lazy import um Circular Dependencies zu vermeiden)
        try:
            from component_31_state_space_planner import (
                Action,
                PlanningProblem,
                State,
                StateSpacePlanner,
            )

            self.StateSpacePlanner = StateSpacePlanner
            self.State = State
            self.Action = Action
            self.PlanningProblem = PlanningProblem
            self.state_planning_available = True
        except ImportError:
            self.state_planning_available = False
            logger.warning(
                "StateSpacePlanner nicht verfugbar - StateAwareTraversal limitiert"
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

        See original implementation in component_12_graph_traversal.py (lines 1116-1200)
        """
        if not self.state_planning_available:
            logger.error("StateSpacePlanner nicht verfugbar")
            return None

        logger.info(
            f"StateAwareTraversal: Planning mit {len(actions)} Actions, Constraints={constraints is not None}"
        )

        # Erstelle PlanningProblem
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

        See original implementation in component_12_graph_traversal.py (lines 1202-1274)
        """
        if not self.state_planning_available:
            logger.error("StateSpacePlanner nicht verfugbar")
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
        Generiert UnifiedProofStep fur einen Plan.

        See original implementation in component_12_graph_traversal.py (lines 1276-1336)
        """
        if not UNIFIED_PROOFS_AVAILABLE or not self.state_planning_available:
            return None

        # Erstelle Proof-Step fur gesamten Plan
        step_id = f"state_plan_{uuid.uuid4().hex[:8]}"

        # Inputs: Start-State und Actions
        inputs = [str(start_state)] + [action.name for action in plan]

        # Output: Goal-State
        output = str(goal_state)

        # Erklarung
        action_sequence = " -> ".join([action.name for action in plan])
        explanation = (
            f"Plan von {start_state} zu {goal_state}:\n"
            f"Aktionssequenz: {action_sequence}\n"
            f"Schritte: {len(plan)}"
        )

        from component_17_proof_explanation import StepType

        # Erstelle UnifiedProofStep
        proof_step = UnifiedProofStep(
            step_id=step_id,
            step_type=StepType.RULE_APPLICATION,
            inputs=inputs,
            rule_name="StateAwarePlanning",
            output=output,
            confidence=1.0,  # Validierte Plane haben volle Confidence
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
        """Extrahiert Objekte aus State-Propositions fur Action-Grounding."""
        objects = set()

        if hasattr(state, "propositions"):
            for prop in state.propositions:
                if isinstance(prop, tuple) and len(prop) > 1:
                    for arg in prop[1:]:
                        if isinstance(arg, str):
                            objects.add(arg)

        return list(objects)

    def _create_state_constraint_validator(self, constraints) -> Optional[callable]:
        """Erstellt State-Validierungs-Funktion aus Constraints."""
        if not self.constraint_solver:
            return None

        def validate_state(state) -> bool:
            """Validiert State gegen alle Constraints."""
            # Placeholder implementation - can be extended with component_29
            return True  # Default: State ist valid

        return validate_state

    def _create_graph_heuristic(self, goal_state):
        """Erstellt Heuristik-Funktion basierend auf Graph-Kontext."""
        try:
            from component_31_state_space_planner import RelaxedPlanHeuristic

            base_heuristic = RelaxedPlanHeuristic()

            def graph_enhanced_heuristic(state) -> float:
                """Graph-erweiterte Heuristik."""
                goal_props = (
                    goal_state.propositions
                    if hasattr(goal_state, "propositions")
                    else set()
                )
                base_estimate = base_heuristic.estimate(state, goal_props)

                # Graph-Kontext: Schatze zusatzliche Kosten
                graph_penalty = 0.0

                state_concepts = self._extract_concepts_from_state(state)
                goal_concepts = self._extract_concepts_from_state(goal_state)

                for goal_concept in goal_concepts:
                    if goal_concept not in state_concepts:
                        graph_penalty += 1.0

                return base_estimate + (graph_penalty * 0.5)

            return graph_enhanced_heuristic

        except ImportError:
            logger.warning(
                "RelaxedPlanHeuristic nicht verfugbar - nutze Standard-Heuristik"
            )
            return None

    def _extract_concepts_from_state(self, state):
        """Extrahiert Konzept-Namen aus State-Propositions."""
        concepts = set()

        if hasattr(state, "propositions"):
            for prop in state.propositions:
                if isinstance(prop, tuple):
                    for arg in prop:
                        if isinstance(arg, str) and arg not in [
                            "left",
                            "right",
                            "table",
                        ]:
                            concepts.add(arg)

        return concepts


# Export all public APIs
__all__ = [
    "GraphTraversal",
    "GraphPath",
    "TraversalStrategy",
    "StateAwareTraversal",
]
