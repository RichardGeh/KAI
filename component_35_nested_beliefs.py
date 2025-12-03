"""
component_35_nested_beliefs.py

[FACADE] Nested Belief Structures and Meta-Knowledge for Epistemic Reasoning

This is a BACKWARD COMPATIBILITY FACADE that delegates to split modules:
- component_35_nested_beliefs_core.py: Core nested belief operations (K^n, signatures, paths)
- component_35_nested_beliefs_inference.py: Common knowledge inference and propagation (C operators)

SPLIT INTO 2 MODULES (2025-11-28):
Original file was 1,070 lines, split into:
- nested_beliefs_core.py: 551 lines (core operations)
- nested_beliefs_inference.py: 497 lines (common knowledge)
- This facade: ~100 lines

All existing imports continue to work:
    from component_35_nested_beliefs import NestedBeliefsHandler

Autor: KAI Development Team
Erstellt: 2025-11-28 (Facade for split modules)
"""

from typing import Any, Dict, List

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger
from component_35_nested_beliefs_core import NestedBeliefsCore
from component_35_nested_beliefs_inference import CommonKnowledgeInference

logger = get_logger(__name__)


# ============================================================================
# Nested Beliefs Handler (Facade)
# ============================================================================


class NestedBeliefsHandler:
    """
    [FACADE] Nested Belief Structures and Meta-Knowledge Management

    This facade coordinates two specialized modules:
    1. NestedBeliefsCore: Core nested belief operations
    2. CommonKnowledgeInference: Common knowledge operations

    Handles higher-order epistemic reasoning:
    - K^n operator: Nested knowledge chains
    - C operator: Common knowledge (simple and full)
    - Meta-knowledge propagation (legacy and batch)
    - Graph traversal for belief paths
    - Theory of mind reasoning

    Thread Safety:
        All methods are thread-safe (delegated to underlying modules).

    Integration:
        - Uses KonzeptNetzwerk for Neo4j persistence
        - Uses CacheManager for performance
        - Coordinates with EpistemicEngineCore for K operator
        - Coordinates with BeliefTracker for knowledge updates
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        k_operator_func,
        add_knowledge_func,
        add_nested_knowledge_func,
    ):
        """
        Initialize nested beliefs handler.

        Args:
            netzwerk: KonzeptNetzwerk instance for Neo4j access
            k_operator_func: Function reference to K operator (from core)
            add_knowledge_func: Function reference to add_knowledge (from tracker)
            add_nested_knowledge_func: Function reference for recursive nested knowledge

        Raises:
            ValueError: If netzwerk is None
        """
        if netzwerk is None:
            raise ValueError("netzwerk cannot be None")

        # Initialize core nested beliefs module
        self._core = NestedBeliefsCore(
            netzwerk=netzwerk,
            k_operator_func=k_operator_func,
            add_knowledge_func=add_knowledge_func,
        )

        # Initialize common knowledge inference module
        self._inference = CommonKnowledgeInference(
            netzwerk=netzwerk,
            k_operator_func=k_operator_func,
            k_n_operator_func=self._core.K_n,  # Pass K_n from core
            add_knowledge_func=add_knowledge_func,
            add_nested_knowledge_func=self._core.add_nested_knowledge,  # Pass from core
        )

        # Store references for facade access
        self.netzwerk = netzwerk
        self._k_operator = k_operator_func
        self._add_knowledge = add_knowledge_func
        self._add_nested_knowledge_ref = add_nested_knowledge_func

        logger.info("NestedBeliefsHandler facade initialized (delegates to 2 modules)")

    # ========================================================================
    # Core Nested Beliefs Operations (delegate to _core)
    # ========================================================================

    def _create_nested_signature(
        self, agent_chain: List[str], proposition_id: str
    ) -> str:
        """Delegate to NestedBeliefsCore"""
        return self._core._create_nested_signature(agent_chain, proposition_id)

    def K_n(
        self,
        observer_id: str,
        nested_knowledge: List[str],
        proposition_id: str,
        _depth: int = 0,
        _max_depth: int = 10,
    ) -> bool:
        """Delegate to NestedBeliefsCore"""
        return self._core.K_n(
            observer_id, nested_knowledge, proposition_id, _depth, _max_depth
        )

    def add_nested_knowledge(
        self, observer_id: str, nested_chain: List[str], proposition_id: str
    ) -> bool:
        """Delegate to NestedBeliefsCore"""
        return self._core.add_nested_knowledge(
            observer_id, nested_chain, proposition_id
        )

    def query_meta_knowledge_paths(
        self, observer_id: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Delegate to NestedBeliefsCore"""
        return self._core.query_meta_knowledge_paths(observer_id, max_depth)

    # ========================================================================
    # Common Knowledge Inference Operations (delegate to _inference)
    # ========================================================================

    def _has_meta_knowledge(
        self, observer_id: str, group: List[str], prop_id: str, level: int
    ) -> bool:
        """Delegate to CommonKnowledgeInference"""
        return self._inference._has_meta_knowledge(observer_id, group, prop_id, level)

    def C_simple(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 3
    ) -> bool:
        """Delegate to CommonKnowledgeInference"""
        return self._inference.C_simple(agent_ids, proposition_id, max_depth)

    def establish_common_knowledge(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 3
    ) -> bool:
        """Delegate to CommonKnowledgeInference"""
        return self._inference.establish_common_knowledge(
            agent_ids, proposition_id, max_depth
        )

    def C(
        self, agent_ids: List[str], proposition_id: str, max_iterations: int = 10
    ) -> bool:
        """Delegate to CommonKnowledgeInference"""
        return self._inference.C(agent_ids, proposition_id, max_iterations)

    def propagate_common_knowledge(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 2
    ) -> int:
        """Delegate to CommonKnowledgeInference (LEGACY)"""
        return self._inference.propagate_common_knowledge(
            agent_ids, proposition_id, max_depth
        )

    def propagate_common_knowledge_batch(
        self,
        agent_ids: List[str],
        proposition_id: str,
        max_depth: int = 2,
        batch_size: int = 100,
    ) -> int:
        """Delegate to CommonKnowledgeInference (BATCH)"""
        return self._inference.propagate_common_knowledge_batch(
            agent_ids, proposition_id, max_depth, batch_size
        )
