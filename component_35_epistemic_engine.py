"""
component_35_epistemic_engine.py

Epistemic Logic Engine für KAI - Facade for Modular Architecture

This file provides backward compatibility for the refactored epistemic engine.
The original monolithic implementation has been split into three focused modules:

1. component_35_epistemic_engine_core.py (~500 lines)
   - Core epistemic reasoning (K, M, E operators)
   - Base data structures and state management
   - Agent creation and management

2. component_35_belief_tracker.py (~500 lines)
   - Belief state tracking and updates
   - Add/remove/query knowledge
   - Group knowledge operations
   - Consistency checking

3. component_35_nested_beliefs.py (~900 lines)
   - Nested belief structures (K^n operator)
   - Common knowledge (C operator)
   - Meta-knowledge propagation
   - Graph traversal for belief paths

This facade delegates all operations to the specialized modules while maintaining
the original EpistemicEngine interface for backward compatibility.

Autor: KAI Development Team
Erstellt: 2025-11-01 | Refactored: 2025-11-28 (Task 12 - Phase 4)
"""

from typing import Any, Dict, List, Optional

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger
from component_35_belief_tracker import BeliefTracker

# Import specialized modules
from component_35_epistemic_engine_core import (
    Agent,
    EpistemicEngineCore,
    EpistemicState,
    MetaProposition,
    ModalOperator,
    Proposition,
)
from component_35_nested_beliefs import NestedBeliefsHandler

logger = get_logger(__name__)


# ============================================================================
# Facade: EpistemicEngine
# ============================================================================


class EpistemicEngine:
    """
    Epistemic Logic Engine - Facade for Modular Architecture

    This class provides backward compatibility with the original monolithic
    EpistemicEngine implementation while delegating to specialized modules:
    - EpistemicEngineCore: K, M, E operators and state management
    - BeliefTracker: Belief updates and consistency
    - NestedBeliefsHandler: K^n, C operators and meta-knowledge

    All existing code using EpistemicEngine will continue to work without
    modification.

    Thread Safety:
        All operations are thread-safe via delegation to thread-safe modules.

    Example:
        >>> netzwerk = KonzeptNetzwerk()
        >>> engine = EpistemicEngine(netzwerk)
        >>> engine.create_agent("alice", "Alice")
        >>> engine.add_knowledge("alice", "sky_is_blue")
        >>> engine.K("alice", "sky_is_blue")
        True
    """

    def __init__(self, netzwerk: KonzeptNetzwerk):
        """
        Initialize epistemic engine facade.

        Args:
            netzwerk: KonzeptNetzwerk instance for Neo4j access

        Raises:
            ValueError: If netzwerk is None
        """
        if netzwerk is None:
            raise ValueError("netzwerk cannot be None")

        self.netzwerk = netzwerk

        # Initialize core engine
        self._core = EpistemicEngineCore(netzwerk)

        # Initialize belief tracker (shares state with core)
        self._belief_tracker = BeliefTracker(netzwerk, self._core.current_state)

        # Initialize nested beliefs handler (needs references to core/tracker methods)
        self._nested_beliefs = NestedBeliefsHandler(
            netzwerk,
            k_operator_func=self._core.K,
            add_knowledge_func=self._belief_tracker.add_knowledge,
            add_nested_knowledge_func=None,  # Set below to avoid circular ref
        )

        # Set recursive reference for nested knowledge
        self._nested_beliefs._add_nested_knowledge_ref = (
            self._nested_beliefs.add_nested_knowledge
        )

        logger.info(
            "EpistemicEngine facade initialized (modular architecture)",
            extra={
                "modules": [
                    "EpistemicEngineCore",
                    "BeliefTracker",
                    "NestedBeliefsHandler",
                ]
            },
        )

    # ========================================================================
    # State Management (from Core)
    # ========================================================================

    @property
    def current_state(self) -> Optional[EpistemicState]:
        """Get current epistemic state"""
        return self._core.current_state

    @current_state.setter
    def current_state(self, state: Optional[EpistemicState]) -> None:
        """Set current epistemic state"""
        self._core.current_state = state
        self._belief_tracker.current_state = state  # Keep tracker in sync

    def clear_cache(self) -> None:
        """Clear all TTL caches (useful for testing or after bulk updates)"""
        self._core.clear_cache()

    def load_state_from_graph(self) -> Optional[EpistemicState]:
        """Lädt aktuellen epistemischen Zustand aus Neo4j"""
        return self._core.load_state_from_graph()

    def persist_state_to_graph(self, state: EpistemicState) -> bool:
        """Persistiert epistemischen Zustand nach Neo4j"""
        return self._core.persist_state_to_graph(state)

    def create_agent(
        self, agent_id: str, name: str, reasoning_capacity: int = 5
    ) -> Agent:
        """Erstelle neuen Agenten (in-memory + graph)"""
        return self._core.create_agent(agent_id, name, reasoning_capacity)

    # ========================================================================
    # Core Modal Operators (from Core)
    # ========================================================================

    def K(self, agent_id: str, proposition_id: str) -> bool:
        """
        Modal Operator K: "Agent knows proposition"

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            True wenn Agent die Proposition kennt
        """
        return self._core.K(agent_id, proposition_id)

    def M(self, agent_id: str, proposition_id: str) -> bool:
        """
        Modal Operator M: "Agent considers proposition possible"

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            True wenn Agent die Proposition für möglich hält
        """
        return self._core.M(agent_id, proposition_id)

    def E(self, agent_ids: List[str], proposition_id: str) -> bool:
        """
        Modal Operator E: "Everyone in group knows proposition"

        Args:
            agent_ids: Liste von Agent-IDs
            proposition_id: ID der Proposition

        Returns:
            True wenn ALLE Agenten die Proposition kennen
        """
        return self._core.E(agent_ids, proposition_id)

    # ========================================================================
    # Belief Management (from BeliefTracker)
    # ========================================================================

    def add_knowledge(
        self, agent_id: str, proposition_id: str, certainty: float = 1.0
    ) -> bool:
        """
        Füge Wissen zu Agent hinzu

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition
            certainty: Gewissheit (0.0 - 1.0)

        Returns:
            True bei Erfolg
        """
        return self._belief_tracker.add_knowledge(agent_id, proposition_id, certainty)

    def add_negated_knowledge(self, agent_id: str, proposition_id: str) -> bool:
        """
        Agent weiß, dass Proposition FALSCH ist

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            True bei Erfolg
        """
        return self._belief_tracker.add_negated_knowledge(agent_id, proposition_id)

    def add_group_knowledge(self, agent_ids: List[str], proposition_id: str) -> bool:
        """
        Füge Wissen zu allen Agenten in Gruppe hinzu

        Args:
            agent_ids: Liste von Agent-IDs
            proposition_id: ID der Proposition

        Returns:
            True bei Erfolg
        """
        return self._belief_tracker.add_group_knowledge(agent_ids, proposition_id)

    # ========================================================================
    # Nested Beliefs & Meta-Knowledge (from NestedBeliefsHandler)
    # ========================================================================

    def K_n(
        self,
        observer_id: str,
        nested_knowledge: List[str],
        proposition_id: str,
        _depth: int = 0,
        _max_depth: int = 10,
    ) -> bool:
        """
        K^n Operator: Nested knowledge

        Args:
            observer_id: Der äußerste Beobachter
            nested_knowledge: Chain von Agenten
            proposition_id: Die Basis-Proposition
            _depth: Internal recursion depth
            _max_depth: Maximum recursion depth

        Returns:
            True wenn verschachteltes Wissen existiert
        """
        return self._nested_beliefs.K_n(
            observer_id, nested_knowledge, proposition_id, _depth, _max_depth
        )

    def add_nested_knowledge(
        self, observer_id: str, nested_chain: List[str], proposition_id: str
    ) -> bool:
        """
        Füge verschachteltes Wissen hinzu

        Args:
            observer_id: Der äußerste Beobachter
            nested_chain: Chain von Agenten
            proposition_id: Basis-Proposition

        Returns:
            True bei Erfolg
        """
        return self._nested_beliefs.add_nested_knowledge(
            observer_id, nested_chain, proposition_id
        )

    def C_simple(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 3
    ) -> bool:
        """
        Modal Operator C (Simple): "Common knowledge in group"

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_depth: Anzahl E-Iterationen

        Returns:
            True wenn Common Knowledge approximiert erfüllt ist
        """
        return self._nested_beliefs.C_simple(agent_ids, proposition_id, max_depth)

    def establish_common_knowledge(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 3
    ) -> bool:
        """
        Etabliere Common Knowledge in Gruppe

        Args:
            agent_ids: Liste von Agent-IDs
            proposition_id: Proposition
            max_depth: Maximale Meta-Level Tiefe

        Returns:
            True bei Erfolg
        """
        return self._nested_beliefs.establish_common_knowledge(
            agent_ids, proposition_id, max_depth
        )

    def C(
        self, agent_ids: List[str], proposition_id: str, max_iterations: int = 10
    ) -> bool:
        """
        Modal Operator C (Full): Common Knowledge via Fixed-Point Iteration

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_iterations: Max Anzahl Iterationen

        Returns:
            True wenn Common Knowledge erreicht
        """
        return self._nested_beliefs.C(agent_ids, proposition_id, max_iterations)

    def propagate_common_knowledge(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 2
    ) -> int:
        """
        Propagiere Common Knowledge durch Gruppe (LEGACY - use batch version)

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_depth: Maximale Meta-Level Tiefe

        Returns:
            Anzahl der erstellten Meta-Knowledge Nodes
        """
        return self._nested_beliefs.propagate_common_knowledge(
            agent_ids, proposition_id, max_depth
        )

    def propagate_common_knowledge_batch(
        self,
        agent_ids: List[str],
        proposition_id: str,
        max_depth: int = 2,
        batch_size: int = 100,
    ) -> int:
        """
        Propagiere Common Knowledge mit BATCH OPERATIONS (O(1) DB queries)

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_depth: Maximale Meta-Level Tiefe
            batch_size: Maximum entries per batch

        Returns:
            Anzahl der erstellten Meta-Knowledge Nodes
        """
        return self._nested_beliefs.propagate_common_knowledge_batch(
            agent_ids, proposition_id, max_depth, batch_size
        )

    def query_meta_knowledge_paths(
        self, observer_id: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Finde alle Meta-Knowledge-Pfade ausgehend von observer

        Args:
            observer_id: ID des Beobachters
            max_depth: Maximale Tiefe der Pfade

        Returns:
            Liste von Dicts mit path, proposition, meta_level
        """
        return self._nested_beliefs.query_meta_knowledge_paths(observer_id, max_depth)

    # ========================================================================
    # Internal Helper (for compatibility - delegates to nested beliefs)
    # ========================================================================

    def _create_nested_signature(
        self, agent_chain: List[str], proposition_id: str
    ) -> str:
        """
        Erstelle String-Signatur für verschachteltes Wissen

        Args:
            agent_chain: Liste von Agenten
            proposition_id: Basis-Proposition

        Returns:
            Verschachtelte Signatur
        """
        return self._nested_beliefs._create_nested_signature(
            agent_chain, proposition_id
        )

    def _has_meta_knowledge(
        self, observer_id: str, group: List[str], prop_id: str, level: int
    ) -> bool:
        """Helper: Check ob Agent Meta-Knowledge auf gegebenem Level hat"""
        return self._nested_beliefs._has_meta_knowledge(
            observer_id, group, prop_id, level
        )

    def _ensure_state(self) -> None:
        """Ensure current_state is initialized"""
        self._core._ensure_state()


# ============================================================================
# Re-export Data Structures for Backward Compatibility
# ============================================================================

__all__ = [
    "EpistemicEngine",
    "ModalOperator",
    "Proposition",
    "Agent",
    "EpistemicState",
    "MetaProposition",
]


# ============================================================================
# Mini-Test (preserved from original for compatibility)
# ============================================================================

if __name__ == "__main__":
    # Mini-Test
    print("Starting K and M Operator Mini-Test...")

    from component_1_netzwerk import KonzeptNetzwerk

    netzwerk = KonzeptNetzwerk()
    engine = EpistemicEngine(netzwerk)

    # Setup
    print("Creating agent 'alice'...")
    engine.create_agent("alice", "Alice")

    print("Adding knowledge: alice knows sky_is_blue")
    engine.add_knowledge("alice", "sky_is_blue")

    # Test K operator
    print("\nTesting K operator...")

    result1 = engine.K("alice", "sky_is_blue")
    print(f"K(alice, sky_is_blue) = {result1}")
    assert result1 is True, "Expected True for known proposition"

    result2 = engine.K("alice", "grass_is_red")
    print(f"K(alice, grass_is_red) = {result2}")
    assert result2 is False, "Expected False for unknown proposition"

    print("[OK] K operator test passed")

    # Test M operator
    print("\nTesting M operator...")

    # Alice weiß, dass Mond KEIN Käse ist
    print("Adding negated knowledge: alice knows moon is NOT cheese")
    engine.add_negated_knowledge("alice", "moon_is_cheese")

    result3 = engine.M("alice", "moon_is_cheese")
    print(f"M(alice, moon_is_cheese) = {result3}")
    assert result3 is False, "Expected False (Alice knows moon is NOT cheese)"

    # Alice hat kein Wissen über Mars -> hält es für möglich
    result4 = engine.M("alice", "mars_is_red")
    print(f"M(alice, mars_is_red) = {result4}")
    assert result4 is True, "Expected True (Alice has no contradicting knowledge)"

    # Alice weiß, dass Himmel blau ist -> NOT(sky_is_blue) ist unmöglich
    result5 = engine.M("alice", "sky_is_blue")
    print(f"M(alice, sky_is_blue) = {result5}")
    assert result5 is True, "Expected True (Alice knows sky IS blue, so it's possible)"

    print("[OK] M operator test passed")

    # Test E operator
    print("\nTesting E operator...")

    # Create additional agents
    print("Creating agents 'bob' and 'carol'...")
    engine.create_agent("bob", "Bob")
    engine.create_agent("carol", "Carol")

    # Test group knowledge
    group = ["alice", "bob", "carol"]
    print(f"Adding group knowledge to {group}: meeting_at_3pm")
    engine.add_group_knowledge(group, "meeting_at_3pm")

    result6 = engine.E(group, "meeting_at_3pm")
    print(f"E({group}, meeting_at_3pm) = {result6}")
    assert result6 is True, "Expected True (all agents know meeting_at_3pm)"

    result7 = engine.E(group, "meeting_at_4pm")
    print(f"E({group}, meeting_at_4pm) = {result7}")
    assert result7 is False, "Expected False (not all agents know meeting_at_4pm)"

    # Test empty group
    result8 = engine.E([], "meeting_at_3pm")
    print(f"E([], meeting_at_3pm) = {result8}")
    assert result8 is False, "Expected False for empty group"

    print("[OK] E operator test passed")

    # Test C operator (simple)
    print("\nTesting C operator (simple)...")

    # Establish common knowledge for group
    print(f"Establishing common knowledge for {group}: game_rules (max_depth=2)")
    engine.establish_common_knowledge(group, "game_rules", max_depth=2)

    result9 = engine.C_simple(group, "game_rules", max_depth=2)
    print(f"C_simple({group}, game_rules, max_depth=2) = {result9}")
    assert result9 is True, "Expected True for established common knowledge"

    # Test without common knowledge (just everyone knows, but no meta-knowledge)
    print("\nAdding individual knowledge without meta-knowledge: random_fact")
    engine.add_group_knowledge(group, "random_fact")

    result10 = engine.C_simple(group, "random_fact", max_depth=2)
    print(f"C_simple({group}, random_fact, max_depth=2) = {result10}")
    # This should be False because we didn't establish meta-knowledge
    # (C requires meta-knowledge, not just E)
    print(f"Note: Result is {result10} (depends on meta-knowledge presence)")

    print("[OK] C operator (simple) test passed")

    print("\n[OK] All tests passed!")
