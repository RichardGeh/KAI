"""
component_35_nested_beliefs_core.py

Core Nested Belief Operations

This module handles core nested belief tracking:
- K^n operator: Nested knowledge chains
- Nested knowledge signatures (string representation)
- Meta-knowledge path queries
- Basic nested knowledge operations

SPLIT FROM: component_35_nested_beliefs.py (2025-11-28)
Part of Phase 4 architectural refactoring to maintain <800 line limit

Autor: KAI Development Team
Erstellt: 2025-11-28 (Split from component_35_nested_beliefs.py)
"""

import threading
from typing import Any, Dict, List

from neo4j.exceptions import Neo4jError, ServiceUnavailable, TransientError

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger
from component_35_epistemic_validation import EpistemicValidator
from infrastructure.cache_manager import get_cache_manager

logger = get_logger(__name__)


# ============================================================================
# Core Nested Beliefs Operations
# ============================================================================


class NestedBeliefsCore:
    """
    Core Nested Belief Operations

    Handles fundamental nested belief tracking:
    - K^n operator for nested knowledge chains
    - Nested knowledge signature creation
    - Meta-knowledge path queries
    - Basic nested knowledge operations

    Thread Safety:
        All methods are thread-safe via RLock protection.

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
    ):
        """
        Initialize core nested beliefs handler.

        Args:
            netzwerk: KonzeptNetzwerk instance for Neo4j access
            k_operator_func: Function reference to K operator (from core)
            add_knowledge_func: Function reference to add_knowledge (from tracker)

        Raises:
            ValueError: If netzwerk is None
        """
        if netzwerk is None:
            raise ValueError("netzwerk cannot be None")

        self.netzwerk = netzwerk
        self._k_operator = k_operator_func
        self._add_knowledge = add_knowledge_func

        # Thread safety
        self._lock = threading.RLock()

        # Cache manager
        self._cache_mgr = get_cache_manager()

        logger.info("NestedBeliefsCore initialized")

    def _create_nested_signature(
        self, agent_chain: List[str], proposition_id: str
    ) -> str:
        """
        Erstelle String-Signatur für verschachteltes Wissen

        Baut verschachtelte K-Operatoren von innen nach außen auf.

        Args:
            agent_chain: Liste von Agenten [A, B, C, ...] (äußerste zuerst)
            proposition_id: Basis-Proposition

        Returns:
            Verschachtelte Signatur, z.B. "K(bob, K(carol, p))"

        Example:
            _create_nested_signature(["bob", "carol"], "p") -> "K(bob, K(carol, p))"
        """
        if not agent_chain:
            return proposition_id

        # Baue von innen nach außen: K(A, K(B, K(C, p)))
        signature = proposition_id
        for agent_id in reversed(agent_chain):
            signature = f"K({agent_id}, {signature})"

        logger.debug(
            f"Created nested signature: {signature}",
            extra={"agent_chain": agent_chain, "proposition_id": proposition_id},
        )
        return signature

    def K_n(
        self,
        observer_id: str,
        nested_knowledge: List[str],
        proposition_id: str,
        _depth: int = 0,
        _max_depth: int = 10,
    ) -> bool:
        """
        K^n Operator: Nested knowledge with caching + recursion protection + error handling

        Prüft rekursiv verschachteltes Wissen wie:
        - K_n("alice", ["bob"], "p") = "Alice knows that Bob knows p"
        - K_n("alice", ["bob", "carol"], "p") = "Alice knows that Bob knows that Carol knows p"

        Args:
            observer_id: Der äußerste Beobachter (A)
            nested_knowledge: Chain von Agenten [B, C, ...] (äußerste zuerst)
            proposition_id: Die Basis-Proposition (p)
            _depth: Internal recursion depth (do not set manually)
            _max_depth: Maximum allowed recursion depth (default: 10)

        Returns:
            True wenn verschachteltes Wissen existiert, False bei Fehler oder unbekannt

        Raises:
            RecursionError: If max_depth exceeded
            ValueError: If inputs are invalid
            TypeError: If arguments have wrong types

        Example:
            # Bob knows secret_password
            add_knowledge("bob", "secret_password")

            # Alice knows that Bob knows secret_password
            add_nested_knowledge("alice", ["bob"], "secret_password")

            # Check if Alice knows that Bob knows the secret
            K_n("alice", ["bob"], "secret_password")  # Returns True
        """
        # SECURITY: Validate inputs
        observer_id = EpistemicValidator.validate_agent_id(observer_id)
        proposition_id = EpistemicValidator.validate_proposition_id(proposition_id)

        # SECURITY: Check recursion depth
        if _depth > _max_depth:
            raise RecursionError(
                f"K_n recursion depth exceeded: {_depth} > {_max_depth}. "
                f"Possible circular reference in nested_knowledge: {nested_knowledge}"
            )

        # Validate nested knowledge list
        if len(nested_knowledge) > _max_depth:
            raise ValueError(
                f"nested_knowledge chain too long: {len(nested_knowledge)} > {_max_depth}"
            )

        # Detect circular references
        if len(nested_knowledge) != len(set(nested_knowledge)):
            logger.warning(
                "Circular reference detected in nested_knowledge",
                extra={
                    "nested_knowledge": nested_knowledge,
                    "duplicates": [
                        x for x in nested_knowledge if nested_knowledge.count(x) > 1
                    ],
                },
            )

        if not nested_knowledge:
            # Base case: K(observer, prop)
            return self._k_operator(observer_id, proposition_id)

        # Cache check for K_n
        cache_key = f"{observer_id}:{':'.join(nested_knowledge)}:{proposition_id}"
        cached = self._cache_mgr.get("epistemic_k_n_operator", cache_key)
        if cached is not None:
            logger.debug(
                f"K_n({observer_id}, {nested_knowledge}, {proposition_id}) = {cached} (cache hit)",
                extra={
                    "observer_id": observer_id,
                    "nested_knowledge": nested_knowledge,
                    "proposition_id": proposition_id,
                    "source": "cache",
                },
            )
            return cached

        # Recursive case: K(observer, K(nested[0], ...))
        # Query Neo4j für MetaBelief
        next_subject = nested_knowledge[0]
        remaining_chain = nested_knowledge[1:]
        meta_level = len(nested_knowledge)

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                # Erstelle Signatur für verschachteltes Wissen
                if remaining_chain:
                    # Noch weitere Verschachtelung
                    nested_sig = self._create_nested_signature(
                        remaining_chain, proposition_id
                    )
                else:
                    # Innerste Ebene: subject knows proposition
                    nested_sig = f"K({next_subject}, {proposition_id})"

                # Query: (observer)-[:KNOWS_THAT]->(mb:MetaBelief {meta_level: X})-[:ABOUT_AGENT]->(subject)
                #        WHERE mb.proposition = nested_knowledge_signature
                result = session.run(
                    """
                    MATCH (observer:Agent {id: $observer_id})-[:KNOWS_THAT]->(mb:MetaBelief)
                    WHERE mb.meta_level = $meta_level
                      AND mb.proposition = $nested_sig
                    MATCH (mb)-[:ABOUT_AGENT]->(subject:Agent {id: $subject_id})
                    RETURN count(mb) > 0 AS has_knowledge
                    """,
                    observer_id=observer_id,
                    subject_id=next_subject,
                    meta_level=meta_level,
                    nested_sig=nested_sig,
                )

                record = result.single()
                has_knowledge = record["has_knowledge"] if record else False

                # Update cache
                self._cache_mgr.set("epistemic_k_n_operator", cache_key, has_knowledge)

                logger.debug(
                    f"K_n({observer_id}, {nested_knowledge}, {proposition_id}) = {has_knowledge}",
                    extra={
                        "observer_id": observer_id,
                        "nested_knowledge": nested_knowledge,
                        "proposition_id": proposition_id,
                        "nested_sig": nested_sig,
                        "has_knowledge": has_knowledge,
                        "source": "neo4j",
                    },
                )
                return has_knowledge

        except ServiceUnavailable as e:
            logger.error(
                f"Neo4j unavailable in K_n operator",
                extra={
                    "observer_id": observer_id,
                    "nested_knowledge": nested_knowledge,
                    "proposition_id": proposition_id,
                    "error": str(e),
                },
            )
            return False

        except TransientError as e:
            logger.warning(
                f"Transient Neo4j error in K_n operator",
                extra={
                    "observer_id": observer_id,
                    "nested_knowledge": nested_knowledge,
                    "proposition_id": proposition_id,
                    "error": str(e),
                },
            )
            return False

        except Neo4jError as e:
            logger.error(
                f"Neo4j error in K_n operator",
                extra={
                    "observer_id": observer_id,
                    "nested_knowledge": nested_knowledge,
                    "proposition_id": proposition_id,
                    "error": str(e),
                    "error_code": getattr(e, "code", None),
                },
            )
            return False

        except Exception as e:
            logger.critical(
                f"Unexpected error in K_n operator",
                extra={
                    "observer_id": observer_id,
                    "nested_knowledge": nested_knowledge,
                    "proposition_id": proposition_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    def add_nested_knowledge(
        self, observer_id: str, nested_chain: List[str], proposition_id: str
    ) -> bool:
        """
        Füge verschachteltes Wissen hinzu

        Erstellt MetaBelief-Nodes für verschachteltes Wissen.

        Args:
            observer_id: Der äußerste Beobachter
            nested_chain: Chain von Agenten [B, C, ...] (äußerste zuerst)
            proposition_id: Basis-Proposition

        Returns:
            True bei Erfolg, False bei Fehler

        Raises:
            ValueError: If inputs are invalid or chain too long
            TypeError: If arguments have wrong types

        Example:
            # Alice knows that Bob knows the secret
            add_nested_knowledge("alice", ["bob"], "secret_password")

            # Carol knows that Bob knows that Alice knows the secret
            add_nested_knowledge("carol", ["bob", "alice"], "secret_password")
        """
        # SECURITY: Validate inputs
        observer_id = EpistemicValidator.validate_agent_id(observer_id)
        proposition_id = EpistemicValidator.validate_proposition_id(proposition_id)

        # Validate nested chain
        if len(nested_chain) > 10:
            raise ValueError(f"nested_chain too long: {len(nested_chain)} > 10")

        for agent in nested_chain:
            EpistemicValidator.validate_agent_id(agent)

        if not nested_chain:
            # Base case: Einfaches Wissen ohne Verschachtelung
            return self._add_knowledge(observer_id, proposition_id)

        subject_id = nested_chain[0]
        remaining_chain = nested_chain[1:]
        meta_level = len(nested_chain)

        # Erstelle nested signature
        if remaining_chain:
            nested_sig = self._create_nested_signature(remaining_chain, proposition_id)
        else:
            nested_sig = f"K({subject_id}, {proposition_id})"

        # Erstelle MetaBelief in Neo4j
        success = self.netzwerk.add_meta_belief(
            observer_id, subject_id, nested_sig, meta_level
        )

        if success:
            logger.info(
                f"Nested knowledge added: {observer_id} -> {nested_chain} -> {proposition_id}",
                extra={
                    "observer_id": observer_id,
                    "nested_chain": nested_chain,
                    "proposition_id": proposition_id,
                    "nested_sig": nested_sig,
                    "meta_level": meta_level,
                },
            )

        return success

    def query_meta_knowledge_paths(
        self, observer_id: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Finde alle Meta-Knowledge-Pfade ausgehend von observer with caching + error handling

        Nutzt Graph-Traversal für effiziente Suche

        Args:
            observer_id: ID des Beobachters
            max_depth: Maximale Tiefe der Pfade (default: 3)

        Returns:
            Liste von Dicts mit:
            - path: [observer -> subject1 -> subject2 -> ...]
            - proposition: Die finale Proposition
            - meta_level: Tiefe des Pfads
            Returns empty list on error

        Raises:
            ValueError: If max_depth is invalid

        Example:
            >>> paths = query_meta_knowledge_paths("alice", max_depth=2)
            >>> paths[0]
            {'path': ['alice', 'bob'], 'proposition': 'K(bob, secret)', 'meta_level': 1}
        """
        # SECURITY: Validate max_depth to prevent injection attacks
        max_depth = EpistemicValidator.validate_max_depth(
            max_depth, min_val=1, max_val=10
        )

        # Cache check
        cache_key = f"{observer_id}:depth_{max_depth}"
        cached = self._cache_mgr.get("epistemic_meta_paths", cache_key)
        if cached is not None:
            logger.debug(
                f"query_meta_knowledge_paths({observer_id}, max_depth={max_depth}) - cache hit ({len(cached)} paths)",
                extra={
                    "observer_id": observer_id,
                    "max_depth": max_depth,
                    "source": "cache",
                    "paths_count": len(cached),
                },
            )
            return cached

        paths: List[Dict[str, Any]] = []

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                # SECURITY FIX: Build query with validated parameter (not string interpolation)
                # max_depth is validated above, so f-string is safe here
                query = f"""
                    MATCH path = (observer:Agent {{id: $observer_id}})
                                 -[:KNOWS_THAT*1..{max_depth}]->(mb:MetaBelief)
                    UNWIND nodes(path) AS node
                    WITH path, mb,
                         [n IN nodes(path) WHERE n:Agent | n.id] AS agent_chain,
                         length(path) AS depth
                    RETURN agent_chain, mb.proposition AS proposition, depth
                    ORDER BY depth ASC
                    """
                result = session.run(query, observer_id=observer_id)

                for record in result:
                    paths.append(
                        {
                            "path": record["agent_chain"],
                            "proposition": record["proposition"],
                            "meta_level": record["depth"],
                        }
                    )

            # Update cache
            self._cache_mgr.set("epistemic_meta_paths", cache_key, paths)

            logger.debug(
                f"Found {len(paths)} meta-knowledge paths from {observer_id} (neo4j)",
                extra={
                    "observer_id": observer_id,
                    "max_depth": max_depth,
                    "paths_found": len(paths),
                    "source": "neo4j",
                },
            )
            return paths

        except ServiceUnavailable as e:
            logger.error(
                f"Neo4j unavailable in query_meta_knowledge_paths",
                extra={
                    "observer_id": observer_id,
                    "max_depth": max_depth,
                    "error": str(e),
                },
            )
            return []

        except TransientError as e:
            logger.warning(
                f"Transient Neo4j error in query_meta_knowledge_paths",
                extra={
                    "observer_id": observer_id,
                    "max_depth": max_depth,
                    "error": str(e),
                },
            )
            return []

        except Neo4jError as e:
            logger.error(
                f"Neo4j error in query_meta_knowledge_paths",
                extra={
                    "observer_id": observer_id,
                    "max_depth": max_depth,
                    "error": str(e),
                    "error_code": getattr(e, "code", None),
                },
            )
            return []

        except Exception as e:
            logger.critical(
                f"Unexpected error in query_meta_knowledge_paths",
                extra={
                    "observer_id": observer_id,
                    "max_depth": max_depth,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise
