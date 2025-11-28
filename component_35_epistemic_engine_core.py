"""
component_35_epistemic_engine_core.py

Core Epistemic Reasoning Engine for KAI - Foundational Logic and Orchestration

This module provides the foundational epistemic logic infrastructure:
- Modal operators: K (knows), M (believes possible), E (everyone knows)
- Core data structures and state management
- Base epistemic reasoning queries
- Cache management for performance

Separated from belief tracking and nested beliefs for clarity.

Autor: KAI Development Team
Erstellt: 2025-11-28 (Split from component_35_epistemic_engine.py)
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from neo4j.exceptions import Neo4jError, ServiceUnavailable, TransientError

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger
from component_35_epistemic_validation import EpistemicValidator
from infrastructure.cache_manager import get_cache_manager

logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class ModalOperator(Enum):
    """Modale Operatoren für epistemische Logik"""

    KNOWS = "K"  # Agent knows (certainty)
    BELIEVES = "M"  # Agent believes possible (uncertainty)
    EVERYONE_KNOWS = "E"  # Everyone in group knows
    COMMON = "C"  # Common knowledge in group


@dataclass
class Proposition:
    """Atomare Proposition (z.B. 'has_blue_eyes', 'date_is_july_16')"""

    id: str
    content: str
    truth_value: Optional[bool] = None


@dataclass
class Agent:
    """Repräsentiert einen epistemischen Agenten"""

    id: str
    name: str
    reasoning_capacity: int = 5  # max meta-level
    knowledge: Set[str] = field(default_factory=set)  # Set von Proposition IDs


@dataclass
class EpistemicState:
    """Vollständiger epistemischer Zustand aller Agenten"""

    agents: Dict[str, Agent]  # agent_id -> Agent
    propositions: Dict[str, Proposition]  # prop_id -> Proposition
    knowledge_base: Dict[str, Set[str]]  # agent_id -> Set[prop_id]
    meta_knowledge: Dict[
        str, Dict[int, Set[str]]
    ]  # agent_id -> level -> Set[meta_prop_id]
    common_knowledge: Dict[str, Set[str]]  # group_id -> Set[prop_id]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetaProposition:
    """Meta-level proposition: "A knows that B knows P" """

    id: str
    observer_id: str  # A
    subject_id: str  # B
    proposition_id: str  # P
    meta_level: (
        int  # 1 = "A knows that B knows", 2 = "A knows that B knows that C knows"
    )
    certainty: float = 1.0


# ============================================================================
# Core Epistemic Engine
# ============================================================================


class EpistemicEngineCore:
    """
    Core Epistemic Reasoning Engine

    Provides foundational epistemic logic operations:
    - K operator: Agent knows proposition
    - M operator: Agent believes proposition possible
    - E operator: Everyone in group knows proposition
    - Agent creation and management
    - Epistemic state management

    Uses:
    - Neo4j for persistent storage via KonzeptNetzwerk
    - CacheManager for high-performance caching
    - Thread-safe operations with RLock

    Thread Safety:
        All methods are thread-safe via RLock protection on shared state.
    """

    def __init__(self, netzwerk: KonzeptNetzwerk):
        """
        Initialize core epistemic engine.

        Args:
            netzwerk: KonzeptNetzwerk instance for Neo4j access

        Raises:
            ValueError: If netzwerk is None
        """
        if netzwerk is None:
            raise ValueError("netzwerk cannot be None")

        self.netzwerk = netzwerk
        self.current_state: Optional[EpistemicState] = None
        self._proposition_cache: Dict[str, Proposition] = {}
        self._agent_cache: Dict[str, Agent] = {}

        # Thread safety for shared state
        self._state_lock = threading.RLock()

        # Use centralized cache manager instead of scattered TTLCaches
        self._cache_mgr = get_cache_manager()

        # Register caches for epistemic operations
        try:
            self._cache_mgr.register_cache(
                "epistemic_k_operator", maxsize=1000, ttl=600
            )
            self._cache_mgr.register_cache(
                "epistemic_k_n_operator", maxsize=1000, ttl=600
            )
            self._cache_mgr.register_cache("epistemic_meta_paths", maxsize=100, ttl=600)
        except ValueError:
            # Caches already registered (e.g., in tests with shared instance)
            logger.debug("Epistemic caches already registered, reusing existing")

        logger.info(
            "EpistemicEngineCore initialized with CacheManager",
            extra={
                "k_cache": "epistemic_k_operator (1000, 600s)",
                "k_n_cache": "epistemic_k_n_operator (1000, 600s)",
                "meta_paths_cache": "epistemic_meta_paths (100, 600s)",
            },
        )

    def clear_cache(self) -> None:
        """Clear all epistemic caches (useful for testing or after bulk updates)"""
        self._cache_mgr.invalidate("epistemic_k_operator")
        self._cache_mgr.invalidate("epistemic_k_n_operator")
        self._cache_mgr.invalidate("epistemic_meta_paths")
        logger.info("All epistemic caches cleared")

    def load_state_from_graph(self) -> Optional[EpistemicState]:
        """Lädt aktuellen epistemischen Zustand aus Neo4j"""
        # TODO: Implementiere in Phase 2
        return None

    def persist_state_to_graph(self, state: EpistemicState) -> bool:
        """Persistiert epistemischen Zustand nach Neo4j"""
        # TODO: Implementiere in Phase 2
        return False

    def create_agent(
        self, agent_id: str, name: str, reasoning_capacity: int = 5
    ) -> Agent:
        """
        Erstelle neuen Agenten (in-memory + graph).

        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            reasoning_capacity: Maximum meta-level depth (default: 5)

        Returns:
            Created Agent instance

        Raises:
            ValueError: If agent_id or name invalid
        """
        # Validate inputs
        agent_id = EpistemicValidator.validate_agent_id(agent_id)

        with self._state_lock:
            agent = Agent(id=agent_id, name=name, reasoning_capacity=reasoning_capacity)
            self._agent_cache[agent_id] = agent

            # Persistiere zu Neo4j
            self.netzwerk.create_agent(agent_id, name, reasoning_capacity)
            logger.debug(f"Agent created: {agent_id} (capacity={reasoning_capacity})")

            return agent

    def _ensure_state(self) -> None:
        """Ensure current_state is initialized (thread-safe)"""
        with self._state_lock:
            if self.current_state is None:
                self.current_state = EpistemicState(
                    agents={},
                    propositions={},
                    knowledge_base={},
                    meta_knowledge={},
                    common_knowledge={},
                )
                logger.debug("EpistemicState initialized")

    def K(self, agent_id: str, proposition_id: str) -> bool:
        """
        Modal Operator K: "Agent knows proposition"

        Checks knowledge in three layers:
        1. CacheManager (fast path, TTL 10 min)
        2. In-memory state (current_state)
        3. Neo4j graph (authoritative)

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            True wenn Agent die Proposition kennt, False sonst

        Raises:
            ValueError: If agent_id or proposition_id invalid
            TypeError: If arguments have wrong types

        Example:
            >>> engine.add_knowledge("alice", "sky_is_blue")
            >>> engine.K("alice", "sky_is_blue")
            True
        """
        # SECURITY: Validate inputs
        agent_id = EpistemicValidator.validate_agent_id(agent_id)
        proposition_id = EpistemicValidator.validate_proposition_id(proposition_id)

        # Fast path 1: CacheManager check
        cache_key = f"{agent_id}:{proposition_id}"
        cached = self._cache_mgr.get("epistemic_k_operator", cache_key)
        if cached is not None:
            logger.debug(
                f"K({agent_id}, {proposition_id}) = {cached} (cache hit)",
                extra={
                    "agent_id": agent_id,
                    "proposition_id": proposition_id,
                    "source": "cache",
                },
            )
            return cached

        # Fast path 2: In-memory state check
        with self._state_lock:
            if self.current_state:
                if agent_id in self.current_state.knowledge_base:
                    if proposition_id in self.current_state.knowledge_base[agent_id]:
                        self._cache_mgr.set(
                            "epistemic_k_operator", cache_key, True
                        )  # Update cache
                        logger.debug(
                            f"K({agent_id}, {proposition_id}) = True (state)",
                            extra={
                                "agent_id": agent_id,
                                "proposition_id": proposition_id,
                                "source": "state",
                            },
                        )
                        return True

        # Authoritative: Query Neo4j with error handling
        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(b:Belief {proposition: $prop_id})
                    RETURN count(b) > 0 AS knows
                    """,
                    agent_id=agent_id,
                    prop_id=proposition_id,
                )

                record = result.single()
                knows = record["knows"] if record else False

                # Update cache
                self._cache_mgr.set("epistemic_k_operator", cache_key, knows)

                logger.debug(
                    f"K({agent_id}, {proposition_id}) = {knows} (neo4j)",
                    extra={
                        "agent_id": agent_id,
                        "proposition_id": proposition_id,
                        "knows": knows,
                        "source": "neo4j",
                    },
                )
                return knows

        except ServiceUnavailable as e:
            logger.error(
                f"Neo4j unavailable in K operator",
                extra={
                    "agent_id": agent_id,
                    "proposition_id": proposition_id,
                    "error": str(e),
                },
            )
            return False

        except TransientError as e:
            logger.warning(
                f"Transient Neo4j error in K operator (retryable)",
                extra={
                    "agent_id": agent_id,
                    "proposition_id": proposition_id,
                    "error": str(e),
                },
            )
            return False

        except Neo4jError as e:
            logger.error(
                f"Neo4j error in K operator",
                extra={
                    "agent_id": agent_id,
                    "proposition_id": proposition_id,
                    "error": str(e),
                    "error_code": getattr(e, "code", None),
                },
            )
            return False

        except Exception as e:
            logger.critical(
                f"Unexpected error in K operator",
                extra={
                    "agent_id": agent_id,
                    "proposition_id": proposition_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    def M(self, agent_id: str, proposition_id: str) -> bool:
        """
        Modal Operator M: "Agent considers proposition possible"

        In Kripke Semantics: M(p) = NOT K(NOT p)
        "Agent glaubt P ist möglich" = "Agent weiß nicht, dass NOT P wahr ist"

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            True wenn Agent die Proposition für möglich hält

        Example:
            >>> engine.add_negated_knowledge("alice", "moon_is_cheese")
            >>> engine.M("alice", "moon_is_cheese")
            False  # Alice knows moon is NOT cheese
        """
        # Erstelle negierte Proposition ID
        negated_prop_id = f"NOT_{proposition_id}"

        # M(p) = NOT K(NOT p)
        knows_negation = self.K(agent_id, negated_prop_id)
        believes_possible = not knows_negation

        logger.debug(
            f"M({agent_id}, {proposition_id}) = {believes_possible}",
            extra={
                "agent_id": agent_id,
                "proposition_id": proposition_id,
                "negated_prop_id": negated_prop_id,
                "knows_negation": knows_negation,
                "believes_possible": believes_possible,
            },
        )

        return believes_possible

    def E(self, agent_ids: List[str], proposition_id: str) -> bool:
        """
        Modal Operator E: "Everyone in group knows proposition"

        E_G(p) = ∀a ∈ G: K_a(p)

        Args:
            agent_ids: Liste von Agent-IDs in der Gruppe
            proposition_id: ID der Proposition

        Returns:
            True wenn ALLE Agenten die Proposition kennen

        Example:
            >>> engine.add_group_knowledge(["alice", "bob"], "meeting_at_3pm")
            >>> engine.E(["alice", "bob"], "meeting_at_3pm")
            True
        """
        if not agent_ids:
            logger.warning("E() called with empty group")
            return False

        for agent_id in agent_ids:
            if not self.K(agent_id, proposition_id):
                logger.debug(
                    f"E({agent_ids}, {proposition_id}) = False (agent {agent_id} doesn't know)",
                    extra={
                        "agent_ids": agent_ids,
                        "proposition_id": proposition_id,
                        "failing_agent": agent_id,
                    },
                )
                return False

        logger.debug(
            f"E({agent_ids}, {proposition_id}) = True",
            extra={"agent_ids": agent_ids, "proposition_id": proposition_id},
        )
        return True
