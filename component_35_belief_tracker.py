"""
component_35_belief_tracker.py

Belief State Tracking and Management for Epistemic Reasoning

This module manages belief states for epistemic agents:
- Add/update/remove beliefs (knowledge assertions)
- Negated knowledge (agent knows proposition is false)
- Group knowledge operations (add to multiple agents)
- Belief consistency checking
- Cache invalidation on updates

Separated from core engine and nested beliefs for focused responsibility.

Autor: KAI Development Team
Erstellt: 2025-11-28 (Split from component_35_epistemic_engine.py)
"""

import threading
from typing import List, Optional

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger
from component_35_epistemic_validation import EpistemicValidator
from infrastructure.cache_manager import get_cache_manager

logger = get_logger(__name__)


# ============================================================================
# Belief Tracker
# ============================================================================


class BeliefTracker:
    """
    Belief State Tracking and Management

    Manages epistemic beliefs for agents:
    - Knowledge addition/removal
    - Negated knowledge (agent knows proposition is false)
    - Group knowledge operations
    - Cache invalidation on updates

    Thread Safety:
        All methods are thread-safe via RLock protection.

    Integration:
        - Uses KonzeptNetzwerk for Neo4j persistence
        - Uses CacheManager for cache invalidation
        - Coordinates with EpistemicEngineCore for K/M/E operators
    """

    def __init__(self, netzwerk: KonzeptNetzwerk, epistemic_state: Optional[dict]):
        """
        Initialize belief tracker.

        Args:
            netzwerk: KonzeptNetzwerk instance for Neo4j access
            epistemic_state: Reference to shared epistemic state (from core engine)

        Raises:
            ValueError: If netzwerk is None
        """
        if netzwerk is None:
            raise ValueError("netzwerk cannot be None")

        self.netzwerk = netzwerk
        self.current_state = epistemic_state  # Shared reference

        # Thread safety
        self._lock = threading.RLock()

        # Cache manager for invalidation
        self._cache_mgr = get_cache_manager()

        logger.info("BeliefTracker initialized")

    def add_knowledge(
        self, agent_id: str, proposition_id: str, certainty: float = 1.0
    ) -> bool:
        """
        Füge Wissen zu Agent hinzu (updates cache + graph) with cache invalidation

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition
            certainty: Gewissheit (0.0 - 1.0, default: 1.0)

        Returns:
            True bei Erfolg, False bei Fehler

        Raises:
            ValueError: If inputs are invalid
            TypeError: If arguments have wrong types

        Example:
            >>> tracker.add_knowledge("alice", "sky_is_blue", certainty=0.95)
            True
        """
        # SECURITY: Validate inputs
        agent_id = EpistemicValidator.validate_agent_id(agent_id)
        proposition_id = EpistemicValidator.validate_proposition_id(proposition_id)
        certainty = EpistemicValidator.validate_certainty(certainty)

        with self._lock:
            # Ensure state is initialized
            if self.current_state is None:
                logger.warning(
                    "current_state is None in add_knowledge, initializing empty state"
                )
                self.current_state = {
                    "agents": {},
                    "propositions": {},
                    "knowledge_base": {},
                    "meta_knowledge": {},
                    "common_knowledge": {},
                }

            # Update in-memory state cache
            if "knowledge_base" not in self.current_state:
                self.current_state["knowledge_base"] = {}

            if agent_id not in self.current_state["knowledge_base"]:
                self.current_state["knowledge_base"][agent_id] = set()
            self.current_state["knowledge_base"][agent_id].add(proposition_id)

            # Invalidate cache for this agent-proposition pair
            cache_key = f"{agent_id}:{proposition_id}"
            try:
                self._cache_mgr.invalidate("epistemic_k_operator", cache_key)
            except ValueError:
                # Cache not registered yet (e.g., during initialization)
                logger.debug(
                    "epistemic_k_operator cache not registered, skipping invalidation"
                )

            # Update graph with error handling
            try:
                success = self.netzwerk.add_belief(agent_id, proposition_id, certainty)

                if success:
                    logger.info(
                        f"Knowledge added: {agent_id} knows {proposition_id}",
                        extra={
                            "agent_id": agent_id,
                            "proposition_id": proposition_id,
                            "certainty": certainty,
                        },
                    )
                else:
                    logger.error(
                        f"Failed to add knowledge to graph",
                        extra={
                            "agent_id": agent_id,
                            "proposition_id": proposition_id,
                            "certainty": certainty,
                        },
                    )

                return success

            except Exception as e:
                logger.error(
                    f"Error adding knowledge",
                    extra={
                        "agent_id": agent_id,
                        "proposition_id": proposition_id,
                        "certainty": certainty,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                return False

    def add_negated_knowledge(self, agent_id: str, proposition_id: str) -> bool:
        """
        Agent weiß, dass Proposition FALSCH ist

        Fügt negierte Proposition "NOT_{proposition_id}" zur Knowledge Base hinzu.

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der ursprünglichen Proposition

        Returns:
            True bei Erfolg, False bei Fehler

        Example:
            >>> tracker.add_negated_knowledge("alice", "moon_is_cheese")
            True
            >>> # Alice now knows moon is NOT cheese
        """
        negated_prop_id = f"NOT_{proposition_id}"
        return self.add_knowledge(agent_id, negated_prop_id, certainty=1.0)

    def add_group_knowledge(self, agent_ids: List[str], proposition_id: str) -> bool:
        """
        Füge Wissen zu allen Agenten in Gruppe hinzu

        Args:
            agent_ids: Liste von Agent-IDs
            proposition_id: ID der Proposition

        Returns:
            True wenn erfolgreich zu ALLEN Agenten hinzugefügt, False bei mindestens einem Fehler

        Example:
            >>> tracker.add_group_knowledge(["alice", "bob", "carol"], "meeting_at_3pm")
            True
        """
        success = True
        for agent_id in agent_ids:
            if not self.add_knowledge(agent_id, proposition_id):
                success = False
                logger.warning(
                    f"Failed to add knowledge to {agent_id}",
                    extra={"agent_id": agent_id, "proposition_id": proposition_id},
                )

        logger.info(
            f"Group knowledge added to {len(agent_ids)} agents: {proposition_id}",
            extra={
                "agent_ids": agent_ids,
                "proposition_id": proposition_id,
                "success": success,
            },
        )
        return success

    def remove_knowledge(self, agent_id: str, proposition_id: str) -> bool:
        """
        Remove knowledge from agent (updates cache + graph)

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            True bei Erfolg, False bei Fehler

        Example:
            >>> tracker.remove_knowledge("alice", "old_belief")
            True
        """
        # SECURITY: Validate inputs
        agent_id = EpistemicValidator.validate_agent_id(agent_id)
        proposition_id = EpistemicValidator.validate_proposition_id(proposition_id)

        with self._lock:
            # Update in-memory state
            if self.current_state and "knowledge_base" in self.current_state:
                if agent_id in self.current_state["knowledge_base"]:
                    self.current_state["knowledge_base"][agent_id].discard(
                        proposition_id
                    )

            # Invalidate cache
            cache_key = f"{agent_id}:{proposition_id}"
            try:
                self._cache_mgr.invalidate("epistemic_k_operator", cache_key)
            except ValueError:
                logger.debug(
                    "epistemic_k_operator cache not registered, skipping invalidation"
                )

            # Remove from Neo4j
            try:
                with self.netzwerk.driver.session(database="neo4j") as session:
                    result = session.run(
                        """
                        MATCH (a:Agent {id: $agent_id})-[r:KNOWS]->(b:Belief {proposition: $prop_id})
                        DELETE r
                        RETURN count(r) as deleted
                        """,
                        agent_id=agent_id,
                        prop_id=proposition_id,
                    )

                    record = result.single()
                    deleted = record["deleted"] if record else 0

                    logger.info(
                        f"Knowledge removed: {agent_id} no longer knows {proposition_id}",
                        extra={
                            "agent_id": agent_id,
                            "proposition_id": proposition_id,
                            "deleted": deleted,
                        },
                    )
                    return deleted > 0

            except Exception as e:
                logger.error(
                    f"Error removing knowledge",
                    extra={
                        "agent_id": agent_id,
                        "proposition_id": proposition_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                return False

    def get_agent_knowledge(self, agent_id: str) -> List[str]:
        """
        Get all propositions known by agent

        Args:
            agent_id: ID des Agenten

        Returns:
            List of proposition IDs known by agent

        Example:
            >>> tracker.get_agent_knowledge("alice")
            ['sky_is_blue', 'water_is_wet', 'earth_is_round']
        """
        # SECURITY: Validate input
        agent_id = EpistemicValidator.validate_agent_id(agent_id)

        # Try in-memory state first
        with self._lock:
            if self.current_state and "knowledge_base" in self.current_state:
                if agent_id in self.current_state["knowledge_base"]:
                    return list(self.current_state["knowledge_base"][agent_id])

        # Fallback to Neo4j
        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(b:Belief)
                    RETURN b.proposition as proposition
                    """,
                    agent_id=agent_id,
                )

                propositions = [record["proposition"] for record in result]
                logger.debug(
                    f"Retrieved {len(propositions)} beliefs for {agent_id}",
                    extra={"agent_id": agent_id, "count": len(propositions)},
                )
                return propositions

        except Exception as e:
            logger.error(
                f"Error retrieving agent knowledge",
                extra={"agent_id": agent_id, "error": str(e)},
            )
            return []

    def check_consistency(self, agent_id: str, proposition_id: str) -> dict:
        """
        Check if adding knowledge would create logical inconsistency

        Checks:
        - Agent doesn't already know the negation
        - No contradictory beliefs exist

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            Dictionary with:
            - consistent: bool (True if no inconsistency)
            - conflicts: list of conflicting propositions
            - reason: str (explanation if inconsistent)

        Example:
            >>> tracker.add_knowledge("alice", "NOT_moon_is_cheese")
            >>> tracker.check_consistency("alice", "moon_is_cheese")
            {'consistent': False, 'conflicts': ['NOT_moon_is_cheese'], 'reason': 'Negation already known'}
        """
        # SECURITY: Validate inputs
        agent_id = EpistemicValidator.validate_agent_id(agent_id)
        proposition_id = EpistemicValidator.validate_proposition_id(proposition_id)

        conflicts = []

        # Check if agent knows the negation
        negated_prop_id = f"NOT_{proposition_id}"
        agent_knowledge = self.get_agent_knowledge(agent_id)

        if negated_prop_id in agent_knowledge:
            conflicts.append(negated_prop_id)
            return {
                "consistent": False,
                "conflicts": conflicts,
                "reason": "Negation already known",
            }

        # Check if agent knows the proposition is false (reverse check)
        if proposition_id.startswith("NOT_"):
            base_prop = proposition_id[4:]  # Remove "NOT_" prefix
            if base_prop in agent_knowledge:
                conflicts.append(base_prop)
                return {
                    "consistent": False,
                    "conflicts": conflicts,
                    "reason": "Positive proposition already known",
                }

        return {"consistent": True, "conflicts": [], "reason": "No conflicts"}

    def clear_agent_knowledge(self, agent_id: str) -> int:
        """
        Clear all knowledge for an agent

        Args:
            agent_id: ID des Agenten

        Returns:
            Number of beliefs removed

        Example:
            >>> tracker.clear_agent_knowledge("alice")
            5  # Removed 5 beliefs
        """
        # SECURITY: Validate input
        agent_id = EpistemicValidator.validate_agent_id(agent_id)

        with self._lock:
            # Clear in-memory state
            if self.current_state and "knowledge_base" in self.current_state:
                if agent_id in self.current_state["knowledge_base"]:
                    self.current_state["knowledge_base"][agent_id].clear()

            # Invalidate all K operator caches for this agent
            try:
                # Cannot efficiently invalidate by prefix, so clear entire cache
                self._cache_mgr.invalidate("epistemic_k_operator")
            except ValueError:
                logger.debug(
                    "epistemic_k_operator cache not registered, skipping invalidation"
                )

            # Remove from Neo4j
            try:
                with self.netzwerk.driver.session(database="neo4j") as session:
                    result = session.run(
                        """
                        MATCH (a:Agent {id: $agent_id})-[r:KNOWS]->(b:Belief)
                        DELETE r
                        RETURN count(r) as deleted
                        """,
                        agent_id=agent_id,
                    )

                    record = result.single()
                    deleted = record["deleted"] if record else 0

                    logger.info(
                        f"Cleared all knowledge for {agent_id}",
                        extra={"agent_id": agent_id, "deleted": deleted},
                    )
                    return deleted

            except Exception as e:
                logger.error(
                    f"Error clearing agent knowledge",
                    extra={"agent_id": agent_id, "error": str(e)},
                )
                return 0
