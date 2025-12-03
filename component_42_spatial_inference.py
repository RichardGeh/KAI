"""
Component 42: Spatial Inference Engine

Core spatial reasoning and relation inference.

This module handles:
- Transitive spatial inference (A north of B, B north of C -> A north of C)
- Symmetric relation inference (A adjacent to B -> B adjacent to A)
- Spatial query processing with caching
- Confidence decay for inferred relations

Author: KAI Development Team
Date: 2025-11-27
"""

import threading
from typing import Dict, List, Optional, Tuple

from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofStep, StepType
from component_42_spatial_types import (
    SpatialReasoningResult,
    SpatialRelation,
    SpatialRelationType,
)
from infrastructure.cache_manager import cache_manager

logger = get_logger(__name__)

# Constants for spatial reasoning
TRANSITIVE_CONFIDENCE_DECAY = 0.9  # Confidence reduction for transitive inference


class SpatialInferenceEngine:
    """
    Core spatial inference engine for KAI.

    Provides methods for:
    - Spatial relation inference (transitive, symmetric)
    - Position-based reasoning
    - Query processing with caching
    - Proof tree generation for inferred relations

    Thread Safety:
        This class is thread-safe. All shared mutable state is protected
        by locks to support concurrent access from multiple threads.
    """

    def __init__(self, netzwerk=None):
        """
        Initialize the spatial inference engine.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph access
        """
        self.netzwerk = netzwerk
        self._lock = threading.RLock()  # For thread safety

        # Cache for spatial queries (5 minute TTL) via CacheManager
        cache_manager.register_cache("spatial_queries", maxsize=100, ttl=300)

        # Supported spatial relations
        self.spatial_relation_types = {rt.value for rt in SpatialRelationType}

        # Performance metrics
        self._performance_metrics = {
            "queries_total": 0,
            "queries_cached": 0,
            "transitive_inferences": 0,
            "symmetric_inferences": 0,
        }

        logger.info(
            "SpatialInferenceEngine initialized",
            extra={
                "relation_types": len(self.spatial_relation_types),
                "cache_size": 100,
                "cache_ttl": 300,
            },
        )

    def infer_spatial_relations(
        self, subject: str, relation_type: Optional[SpatialRelationType] = None
    ) -> SpatialReasoningResult:
        """
        Infer spatial relations for a given subject.

        Args:
            subject: The entity to reason about
            relation_type: Optional filter for specific relation type

        Returns:
            SpatialReasoningResult with inferred relations
        """
        with self._lock:
            cache_key = f"{subject}:{relation_type.value if relation_type else 'ALL'}"

            # Update metrics
            self._performance_metrics["queries_total"] += 1

            # Check cache
            cached_result = cache_manager.get("spatial_queries", cache_key)
            if cached_result is not None:
                self._performance_metrics["queries_cached"] += 1
                logger.debug(
                    "Cache hit for spatial query",
                    extra={
                        "cache_key": cache_key,
                        "cache_hit_rate": self._performance_metrics["queries_cached"]
                        / self._performance_metrics["queries_total"],
                    },
                )
                return cached_result

            logger.info(
                "Inferring spatial relations",
                extra={
                    "subject": subject,
                    "relation_type": relation_type.value if relation_type else "ALL",
                    "cache_key": cache_key,
                },
            )

            result = SpatialReasoningResult(query=subject)

            try:
                # Query direct relations from graph
                direct_relations = self._query_direct_relations(subject, relation_type)
                result.relations.extend(direct_relations)
                result.reasoning_steps.append(
                    f"Found {len(direct_relations)} direct spatial relations"
                )

                # Apply transitive inference
                transitive_relations = self._infer_transitive_relations(
                    subject, direct_relations, relation_type
                )
                result.relations.extend(transitive_relations)
                if transitive_relations:
                    result.reasoning_steps.append(
                        f"Inferred {len(transitive_relations)} transitive relations"
                    )

                # Apply symmetric inference
                symmetric_relations = self._infer_symmetric_relations(
                    subject, result.relations
                )
                result.relations.extend(symmetric_relations)
                if symmetric_relations:
                    result.reasoning_steps.append(
                        f"Inferred {len(symmetric_relations)} symmetric relations"
                    )

                # Calculate overall confidence
                if result.relations:
                    result.confidence = sum(
                        r.confidence for r in result.relations
                    ) / len(result.relations)

                logger.info(
                    "Spatial reasoning complete: %d relations, confidence=%.2f",
                    len(result.relations),
                    result.confidence,
                )

            except Exception as e:
                logger.error(
                    "Error during spatial reasoning: %s", str(e), exc_info=True
                )
                result.error = str(e)
                result.confidence = 0.0
                # DO NOT cache errors - allow retry
                return result

            # Only cache successful results
            cache_manager.set("spatial_queries", cache_key, result)

            return result

    def _query_direct_relations(
        self, subject: str, relation_type: Optional[SpatialRelationType]
    ) -> List[SpatialRelation]:
        """
        Query direct spatial relations from the knowledge graph.

        Args:
            subject: Entity to query
            relation_type: Optional relation type filter

        Returns:
            List of direct spatial relations
        """
        relations = []

        # Determine which relation types to query
        if relation_type:
            relation_types_to_query = [relation_type.value]
        else:
            relation_types_to_query = list(self.spatial_relation_types)

        # Query graph for each relation type
        for rel_type_str in relation_types_to_query:
            try:
                # Query facts from graph
                facts = self.netzwerk.query_graph_for_facts(subject)

                # Extract spatial relations
                if rel_type_str in facts:
                    for obj in facts[rel_type_str]:
                        rel_type_enum = SpatialRelationType(rel_type_str)
                        relations.append(
                            SpatialRelation(
                                subject=subject,
                                object=obj,
                                relation_type=rel_type_enum,
                                confidence=1.0,  # Direct facts have full confidence
                            )
                        )
            except Exception as e:
                logger.warning("Error querying relation %s: %s", rel_type_str, str(e))

        return relations

    def _infer_transitive_relations(
        self,
        subject: str,
        known_relations: List[SpatialRelation],
        relation_type_filter: Optional[SpatialRelationType],
    ) -> List[SpatialRelation]:
        """
        Infer transitive spatial relations.

        For transitive relations R: If A R B and B R C, then A R C.
        Examples: NORTH_OF, INSIDE, CONTAINS, etc.

        Args:
            subject: Starting entity
            known_relations: Already known relations
            relation_type_filter: Optional filter

        Returns:
            List of inferred transitive relations
        """
        inferred = []

        # Only process transitive relation types
        transitive_types = {
            rt
            for rt in SpatialRelationType
            if rt.is_transitive
            and (not relation_type_filter or rt == relation_type_filter)
        }

        for rel_type in transitive_types:
            # Find all A R B relations
            for rel in known_relations:
                if rel.relation_type == rel_type:
                    # Query for B R C relations
                    second_hop = self._query_direct_relations(rel.object, rel_type)

                    for second_rel in second_hop:
                        # A R B, B R C => A R C
                        inferred_rel = SpatialRelation(
                            subject=subject,
                            object=second_rel.object,
                            relation_type=rel_type,
                            confidence=min(rel.confidence, second_rel.confidence)
                            * TRANSITIVE_CONFIDENCE_DECAY,
                            metadata={
                                "inferred_via": "transitivity",
                                "intermediate": rel.object,
                            },
                        )
                        inferred.append(inferred_rel)

        return inferred

    def infer_transitive_with_proof(
        self,
        subject: str,
        known_relations: List[SpatialRelation],
        relation_type_filter: Optional[SpatialRelationType] = None,
    ) -> Tuple[List[SpatialRelation], List[ProofStep]]:
        """
        Infer transitive spatial relations WITH ProofStep generation.

        For transitive relations R: If A R B and B R C, then A R C.

        Args:
            subject: Starting entity
            known_relations: Already known relations
            relation_type_filter: Optional filter

        Returns:
            Tuple (inferred_relations, proof_steps)
        """
        inferred = []
        proof_steps = []

        # Only process transitive relation types
        transitive_types = {
            rt
            for rt in SpatialRelationType
            if rt.is_transitive
            and (not relation_type_filter or rt == relation_type_filter)
        }

        for rel_type in transitive_types:
            # Find all A R B relations
            for rel in known_relations:
                if rel.relation_type == rel_type:
                    # Query for B R C relations
                    second_hop = self._query_direct_relations(rel.object, rel_type)

                    for second_rel in second_hop:
                        # A R B, B R C => A R C
                        intermediate = rel.object
                        final_object = second_rel.object

                        inferred_rel = SpatialRelation(
                            subject=subject,
                            object=final_object,
                            relation_type=rel_type,
                            confidence=min(rel.confidence, second_rel.confidence) * 0.9,
                            metadata={
                                "inferred_via": "transitivity",
                                "intermediate": intermediate,
                            },
                        )
                        inferred.append(inferred_rel)

                        # Erstelle ProofStep für diese transitive Inferenz
                        proof_step = ProofStep(
                            step_id=f"spatial_transitive_{subject}_{rel_type.value}_{final_object}",
                            step_type=StepType.SPATIAL_TRANSITIVE_INFERENCE,
                            inputs=[
                                f"{subject} {rel_type.value} {intermediate}",
                                f"{intermediate} {rel_type.value} {final_object}",
                            ],
                            rule_name=f"Transitivity({rel_type.value})",
                            output=f"{subject} {rel_type.value} {final_object}",
                            confidence=inferred_rel.confidence,
                            explanation_text=(
                                f"Durch transitive Schlussfolgerung: "
                                f"Da '{subject}' {rel_type.value} '{intermediate}' liegt, "
                                f"und '{intermediate}' {rel_type.value} '{final_object}' liegt, "
                                f"muss '{subject}' auch {rel_type.value} '{final_object}' liegen."
                            ),
                            metadata={
                                "inference_type": "transitivity",
                                "intermediate_entity": intermediate,
                                "relation_type": rel_type.value,
                            },
                            source_component="spatial_inference",
                        )
                        proof_steps.append(proof_step)

        # Update metrics
        with self._lock:
            self._performance_metrics["transitive_inferences"] += len(inferred)

        logger.info(
            "Transitive inference complete",
            extra={
                "relations_inferred": len(inferred),
                "proof_steps_generated": len(proof_steps),
                "total_transitive_inferences": self._performance_metrics.get(
                    "transitive_inferences", 0
                ),
            },
        )

        return inferred, proof_steps

    def _infer_symmetric_relations(
        self, subject: str, known_relations: List[SpatialRelation]
    ) -> List[SpatialRelation]:
        """
        Infer symmetric spatial relations.

        For symmetric relations R: If A R B, then B R A.
        Examples: ADJACENT_TO, NEIGHBOR_ORTHOGONAL, etc.

        Args:
            subject: Entity to reason about
            known_relations: Known relations

        Returns:
            List of inferred symmetric relations (from subject's perspective)
        """
        inferred = []

        # Query relations where subject is the object (B R A)
        for rel_type in SpatialRelationType:
            if not rel_type.is_symmetric:
                continue

            try:
                # Find all X R subject relations
                # This requires a reverse query (find nodes that have relation to subject)
                # We'll use the graph traversal for this
                reverse_facts = self.netzwerk.find_incoming_relations(
                    subject, rel_type.value
                )

                for source_entity in reverse_facts:
                    # X R subject => subject R X (symmetric)
                    inferred_rel = SpatialRelation(
                        subject=subject,
                        object=source_entity,
                        relation_type=rel_type,
                        confidence=1.0,  # Symmetric relations preserve confidence
                        metadata={"inferred_via": "symmetry"},
                    )
                    inferred.append(inferred_rel)

            except AttributeError:
                # If find_incoming_relations doesn't exist, skip
                # We'll implement this in the netzwerk integration
                logger.debug("Reverse query not available for symmetry inference")
                continue

        return inferred

    def check_spatial_consistency(
        self, relations: List[SpatialRelation]
    ) -> Tuple[bool, List[str]]:
        """
        Check if a set of spatial relations is consistent.

        Detects contradictions like:
        - A NORTH_OF B and B NORTH_OF A (impossible)
        - A INSIDE B and B INSIDE A (impossible)
        - Circular transitive relations

        Args:
            relations: Set of spatial relations to check

        Returns:
            Tuple of (is_consistent, list_of_violations)
        """
        violations = []

        # Build relation graph
        relation_graph: Dict[Tuple[str, str], set] = {}

        for rel in relations:
            key = (rel.subject, rel.object)
            if key not in relation_graph:
                relation_graph[key] = set()
            relation_graph[key].add(rel.relation_type)

        # Check for contradictions
        for (subj, obj), rel_types in relation_graph.items():
            # Check inverse key
            inverse_key = (obj, subj)
            if inverse_key in relation_graph:
                inverse_rels = relation_graph[inverse_key]

                # For each relation type, check for contradictions
                for rel_type in rel_types:
                    # Non-symmetric relations with their inverse is a contradiction
                    if not rel_type.is_symmetric and rel_type.inverse in inverse_rels:
                        violations.append(
                            f"Contradiction: {subj} {rel_type.value} {obj} AND "
                            f"{obj} {rel_type.inverse.value} {subj}"
                        )

                    # Same non-symmetric relation in both directions is also a contradiction
                    if not rel_type.is_symmetric and rel_type in inverse_rels:
                        violations.append(
                            f"Contradiction: {subj} {rel_type.value} {obj} AND "
                            f"{obj} {rel_type.value} {subj}"
                        )

        is_consistent = len(violations) == 0

        if not is_consistent:
            logger.warning(
                "Spatial consistency check failed: %d violations", len(violations)
            )

        return is_consistent, violations

    def clear_cache(self):
        """Clear the query cache."""
        with self._lock:
            cache_manager.invalidate("spatial_queries")
            logger.info("Spatial inference cache cleared")

    def get_performance_metrics(self) -> Dict[str, any]:
        """
        Returns performance metrics for the spatial inference engine.

        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            metrics = self._performance_metrics.copy()

            # Add cache statistics from CacheManager
            cache_stats = cache_manager.get_stats("spatial_queries")
            metrics["cache_size"] = cache_stats["size"]
            metrics["cache_max_size"] = cache_stats["maxsize"]
            metrics["cache_hits_cm"] = cache_stats["hits"]
            metrics["cache_misses_cm"] = cache_stats["misses"]
            metrics["cache_hit_rate_cm"] = cache_stats["hit_rate"]

            if metrics["queries_total"] > 0:
                metrics["cache_hit_rate"] = (
                    metrics["queries_cached"] / metrics["queries_total"]
                )
            else:
                metrics["cache_hit_rate"] = 0.0

            return metrics

    def log_performance_summary(self) -> None:
        """Logs a summary of performance metrics."""
        metrics = self.get_performance_metrics()

        logger.info(
            "SpatialInferenceEngine Performance Summary",
            extra={
                "queries_total": metrics["queries_total"],
                "cache_hit_rate": f"{metrics['cache_hit_rate']:.2%}",
                "transitive_inferences": metrics["transitive_inferences"],
                "symmetric_inferences": metrics["symmetric_inferences"],
                "cache_usage": f"{metrics['cache_size']}/{metrics['cache_max_size']}",
            },
        )
