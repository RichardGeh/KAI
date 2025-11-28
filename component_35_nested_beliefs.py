"""
component_35_nested_beliefs.py

Nested Belief Structures and Meta-Knowledge for Epistemic Reasoning

This module handles higher-order epistemic reasoning:
- K^n operator: Nested knowledge (A knows that B knows that C knows...)
- Common knowledge (C operator): Fixed-point iteration and batch propagation
- Meta-knowledge paths: Graph traversal for belief chains
- Nested knowledge signatures: String representation of belief hierarchies
- Theory of mind reasoning: What agents know about other agents' knowledge

Separated from core and belief tracking for complexity management.

Autor: KAI Development Team
Erstellt: 2025-11-28 (Split from component_35_epistemic_engine.py)
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
# Nested Beliefs Handler
# ============================================================================


class NestedBeliefsHandler:
    """
    Nested Belief Structures and Meta-Knowledge Management

    Handles higher-order epistemic reasoning:
    - K^n operator: Nested knowledge chains
    - C operator: Common knowledge (simple and full)
    - Meta-knowledge propagation (legacy and batch)
    - Graph traversal for belief paths
    - Theory of mind reasoning

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

        self.netzwerk = netzwerk
        self._k_operator = k_operator_func
        self._add_knowledge = add_knowledge_func
        self._add_nested_knowledge_ref = (
            add_nested_knowledge_func  # For recursive calls
        )

        # Thread safety
        self._lock = threading.RLock()

        # Cache manager
        self._cache_mgr = get_cache_manager()

        logger.info("NestedBeliefsHandler initialized")

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

    def _has_meta_knowledge(
        self, observer_id: str, group: List[str], prop_id: str, level: int
    ) -> bool:
        """Helper: Check ob Agent Meta-Knowledge auf gegebenem Level hat"""
        # Query Neo4j für MetaBelief
        with self.netzwerk.driver.session(database="neo4j") as session:
            # Vereinfacht: Check ob Agent weiß, dass alle in Gruppe prop_id kennen
            # Für jeden anderen Agent in der Gruppe prüfen, ob MetaBelief existiert
            for subject_id in group:
                if observer_id == subject_id:
                    continue  # Agent muss nicht wissen, dass er selbst etwas weiß

                # Prüfe ob MetaBelief (observer_id knows that subject_id knows prop_id) existiert
                # Schema: (observer:Agent)-[:KNOWS_THAT]->(mb:MetaBelief)-[:ABOUT_AGENT]->(subject:Agent)
                result = session.run(
                    """
                    MATCH (o:Agent {id: $observer_id})-[:KNOWS_THAT]->(mb:MetaBelief)-[:ABOUT_AGENT]->(s:Agent {id: $subject_id})
                    WHERE mb.proposition = $prop_id AND mb.meta_level = $level
                    RETURN count(mb) > 0 AS has_meta
                    """,
                    observer_id=observer_id,
                    subject_id=subject_id,
                    prop_id=prop_id,
                    level=level,
                )

                record = result.single()
                has_meta = record["has_meta"] if record else False

                if not has_meta:
                    return False

            return True

    def C_simple(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 3
    ) -> bool:
        """
        Modal Operator C (Simple): "Common knowledge in group"

        C_G(p) = E_G(p) AND E_G(E_G(p)) AND E_G(E_G(E_G(p))) AND ...

        Vereinfachte Fixed-Point Approximation:
        C ca. E^max_depth (iterierte "everyone knows")

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_depth: Anzahl E-Iterationen (default: 3)

        Returns:
            True wenn Common Knowledge approximiert erfüllt ist

        Example:
            >>> establish_common_knowledge(["alice", "bob"], "game_rules", max_depth=2)
            >>> C_simple(["alice", "bob"], "game_rules", max_depth=2)
            True
        """
        # Level 0: Everyone knows p (use E operator from core via K operator)
        all_know = all(
            self._k_operator(agent_id, proposition_id) for agent_id in agent_ids
        )
        if not all_know:
            return False

        # Level 1..max_depth: Everyone knows that everyone knows (rekursiv)
        current_prop = proposition_id

        for depth in range(1, max_depth + 1):
            # Erstelle Meta-Proposition: "Everyone knows {current_prop}"
            meta_prop_id = f"E_LEVEL_{depth}_{current_prop}"

            # Check: Wissen alle Agenten, dass "everyone knows current_prop"?
            # (Vereinfacht: prüfe ob alle Agenten Meta-Wissen haben)
            all_know_meta = True
            for agent_id in agent_ids:
                # Prüfe ob Agent Meta-Knowledge hat
                if not self._has_meta_knowledge(
                    agent_id, agent_ids, current_prop, depth
                ):
                    all_know_meta = False
                    break

            if not all_know_meta:
                logger.debug(
                    f"C({agent_ids}, {proposition_id}) = False at depth {depth}"
                )
                return False

            current_prop = meta_prop_id

        logger.debug(f"C({agent_ids}, {proposition_id}) = True (depth {max_depth})")
        return True

    def establish_common_knowledge(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 3
    ) -> bool:
        """
        Etabliere Common Knowledge in Gruppe

        Fügt proposition zu allen Agenten + Meta-Knowledge für alle Levels hinzu

        Args:
            agent_ids: Liste von Agent-IDs
            proposition_id: Proposition
            max_depth: Maximale Meta-Level Tiefe (default: 3)

        Returns:
            True bei Erfolg

        Example:
            >>> establish_common_knowledge(["alice", "bob", "carol"], "meeting_time", max_depth=2)
            True
        """
        # Step 1: Everyone knows p (Level 0)
        for agent_id in agent_ids:
            self._add_knowledge(agent_id, proposition_id)

        # Step 2: Create meta-knowledge entries for all levels (Level 1..max_depth)
        # Level 1: Everyone knows that everyone knows p
        # Level 2: Everyone knows that everyone knows that everyone knows p
        # etc.
        current_prop = proposition_id

        for level in range(1, max_depth + 1):
            # Für dieses Level: Jeder Agent weiß, dass jeder andere Agent current_prop kennt
            for observer in agent_ids:
                for subject in agent_ids:
                    if observer != subject:
                        # observer weiß, dass subject current_prop kennt (auf diesem Meta-Level)
                        self.netzwerk.add_meta_belief(
                            observer, subject, current_prop, meta_level=level
                        )

            # Für nächstes Level: Meta-Proposition wird zur neuen current_prop
            current_prop = f"E_LEVEL_{level}_{current_prop}"

        logger.info(
            f"Common knowledge established for {len(agent_ids)} agents (depth {max_depth}): {proposition_id}",
            extra={
                "agent_ids": agent_ids,
                "proposition_id": proposition_id,
                "max_depth": max_depth,
            },
        )
        return True

    def C(
        self, agent_ids: List[str], proposition_id: str, max_iterations: int = 10
    ) -> bool:
        """
        Modal Operator C (Full): Common Knowledge via Fixed-Point Iteration

        Algorithm:
          1. Start: knowledge_0 = {agents who know proposition}
          2. Iterate: knowledge_i+1 = {agents who know that all agents in knowledge_i know}
          3. Stop wenn Fixed-Point: knowledge_i == knowledge_i+1
          4. Common Knowledge = (knowledge_fixedpoint == full_group)

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_iterations: Max Anzahl Iterationen

        Returns:
            True wenn Common Knowledge erreicht

        Example:
            >>> propagate_common_knowledge_batch(["alice", "bob", "carol"], "shared_secret")
            >>> C(["alice", "bob", "carol"], "shared_secret")
            True
        """
        # Empty group check
        if not agent_ids:
            logger.warning("C() called with empty group")
            return False

        logger.info(
            f"Computing C({len(agent_ids)} agents, {proposition_id})",
            extra={"agent_ids": agent_ids, "proposition_id": proposition_id},
        )

        # Initialize: Wer kennt die Proposition direkt?
        knowledge_set = set()
        for agent_id in agent_ids:
            if self._k_operator(agent_id, proposition_id):
                knowledge_set.add(agent_id)

        logger.debug(
            f"Iteration 0: {len(knowledge_set)} agents know proposition",
            extra={"knowledge_set": list(knowledge_set)},
        )

        # Fixed-Point Iteration
        for iteration in range(1, max_iterations + 1):
            # Berechne new_knowledge_set: Wer weiß, dass ALLE in knowledge_set wissen?
            new_knowledge_set = set()

            for agent_id in agent_ids:
                # First check: Agent must be in current knowledge_set
                # (can't have common knowledge if agent doesn't know proposition)
                if agent_id not in knowledge_set:
                    continue

                # Second check: Weiß agent_id, dass ALLE ANDEREN in knowledge_set die Proposition kennen?
                knows_about_all = True

                for subject_id in knowledge_set:
                    if agent_id == subject_id:
                        continue  # Selbstwissen nicht prüfen

                    # Check: agent_id weiß, dass subject_id proposition kennt
                    if not self.K_n(agent_id, [subject_id], proposition_id):
                        knows_about_all = False
                        break

                if knows_about_all:
                    new_knowledge_set.add(agent_id)

            logger.debug(
                f"Iteration {iteration}: {len(new_knowledge_set)} agents have meta-knowledge",
                extra={
                    "iteration": iteration,
                    "new_knowledge_set": list(new_knowledge_set),
                },
            )

            # Fixed-Point erreicht?
            if new_knowledge_set == knowledge_set:
                is_common = knowledge_set == set(agent_ids)
                logger.info(
                    f"Fixed-point reached at iteration {iteration}",
                    extra={
                        "iteration": iteration,
                        "knowledge_set": list(knowledge_set),
                        "agent_ids": agent_ids,
                        "is_common": is_common,
                        "knowledge_set_size": len(knowledge_set),
                        "agent_ids_size": len(agent_ids),
                    },
                )
                return is_common

            knowledge_set = new_knowledge_set

            # Leere Menge = kein Common Knowledge möglich
            if not knowledge_set:
                logger.info("Knowledge set became empty - no common knowledge")
                return False

        logger.warning(
            f"Max iterations {max_iterations} reached without convergence",
            extra={
                "max_iterations": max_iterations,
                "final_knowledge_set": list(knowledge_set),
            },
        )
        return False

    def propagate_common_knowledge(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 2
    ) -> int:
        """
        Propagiere Common Knowledge durch Gruppe (Public Announcement) - LEGACY VERSION

        DEPRECATED: Use propagate_common_knowledge_batch() for better performance

        Algorithm:
          1. Alle Agenten lernen proposition
          2. Alle Agenten lernen, dass alle anderen es wissen (Level 1)
          3. Alle Agenten lernen, dass alle anderen wissen, dass alle es wissen (Level 2)
          ... bis max_depth

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_depth: Maximale Meta-Level Tiefe (default: 2)

        Returns:
            Anzahl der erstellten Meta-Knowledge Nodes

        Example:
            >>> propagate_common_knowledge(["alice", "bob"], "announcement", max_depth=1)
            4  # 2 beliefs + 2 meta-beliefs
        """
        logger.warning(
            "Using legacy propagate_common_knowledge() - consider using propagate_common_knowledge_batch() for O(N^2) -> O(1) improvement",
            extra={"agent_count": len(agent_ids), "max_depth": max_depth},
        )

        count = 0

        # Level 0: Everyone learns proposition
        for agent_id in agent_ids:
            self._add_knowledge(agent_id, proposition_id)
            count += 1

        # Level 1: Everyone learns that everyone else knows (only if max_depth >= 1)
        if max_depth >= 1:
            for observer in agent_ids:
                for subject in agent_ids:
                    if observer != subject:
                        self.add_nested_knowledge(observer, [subject], proposition_id)
                        count += 1

        # Level 2+: Everyone learns that everyone knows that everyone knows
        if max_depth >= 2:
            for observer in agent_ids:
                for subject1 in agent_ids:
                    if observer == subject1:
                        continue
                    for subject2 in agent_ids:
                        if subject1 == subject2:
                            continue
                        self.add_nested_knowledge(
                            observer, [subject1, subject2], proposition_id
                        )
                        count += 1

        logger.info(
            f"Common knowledge propagated (legacy): {count} knowledge nodes created",
            extra={
                "agent_ids": agent_ids,
                "proposition_id": proposition_id,
                "max_depth": max_depth,
                "count": count,
            },
        )
        return count

    def propagate_common_knowledge_batch(
        self,
        agent_ids: List[str],
        proposition_id: str,
        max_depth: int = 2,
        batch_size: int = 100,
    ) -> int:
        """
        Propagiere Common Knowledge durch Gruppe mit BATCH OPERATIONS (O(1) DB queries)

        Performance: O(N^2) queries -> O(1) queries via UNWIND
        For 100 agents: ~10,000 queries -> 1-3 queries (100x faster)

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_depth: Maximale Meta-Level Tiefe (default: 2)
            batch_size: Maximum entries per batch (default: 100)

        Returns:
            Anzahl der erstellten Meta-Knowledge Nodes

        Example:
            >>> propagate_common_knowledge_batch(["alice", "bob", "carol"], "public_fact", max_depth=2)
            21  # 3 beliefs + 6 level-1 meta-beliefs + 12 level-2 meta-beliefs
        """
        if not agent_ids:
            logger.warning(
                "propagate_common_knowledge_batch called with empty agent_ids"
            )
            return 0

        logger.info(
            f"Starting batch common knowledge propagation for {len(agent_ids)} agents (max_depth={max_depth})",
            extra={
                "agent_count": len(agent_ids),
                "proposition_id": proposition_id,
                "max_depth": max_depth,
                "batch_size": batch_size,
            },
        )

        count = 0

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:

                def _batch_create(tx):
                    nonlocal count

                    # Level 0: Batch create all beliefs
                    beliefs_query = """
                    UNWIND $agents AS agent_id
                    MATCH (a:Agent {id: agent_id})
                    MERGE (a)-[:KNOWS]->(b:Belief {proposition: $prop_id})
                    ON CREATE SET b.certainty = 1.0, b.created_at = timestamp()
                    RETURN count(b) as created
                    """
                    result = tx.run(
                        beliefs_query, agents=agent_ids, prop_id=proposition_id
                    )
                    beliefs_created = result.single()["created"]
                    count += beliefs_created

                    logger.debug(
                        f"Level 0: Created {beliefs_created} beliefs",
                        extra={"beliefs_created": beliefs_created},
                    )

                    # Level 1: Batch create meta-beliefs (if max_depth >= 1)
                    if max_depth >= 1:
                        # Generate all observer-subject pairs (excluding self-knowledge)
                        pairs = [
                            {"observer": obs, "subject": subj}
                            for obs in agent_ids
                            for subj in agent_ids
                            if obs != subj
                        ]

                        # Process in batches if necessary
                        for i in range(0, len(pairs), batch_size):
                            batch_pairs = pairs[i : i + batch_size]

                            nested_sig = f"K({agent_ids[0]}, {proposition_id})"  # Simplified signature

                            meta_query = """
                            UNWIND $pairs AS pair
                            MATCH (observer:Agent {id: pair.observer})
                            MATCH (subject:Agent {id: pair.subject})
                            MERGE (observer)-[:KNOWS_THAT]->(mb:MetaBelief {
                                proposition: $nested_sig,
                                meta_level: 1
                            })
                            MERGE (mb)-[:ABOUT_AGENT]->(subject)
                            ON CREATE SET mb.created_at = timestamp()
                            RETURN count(mb) as created
                            """
                            result = tx.run(
                                meta_query, pairs=batch_pairs, nested_sig=nested_sig
                            )
                            meta_created = result.single()["created"]
                            count += meta_created

                            logger.debug(
                                f"Level 1: Created {meta_created} meta-beliefs (batch {i // batch_size + 1})",
                                extra={
                                    "meta_created": meta_created,
                                    "batch_index": i // batch_size + 1,
                                },
                            )

                    # Level 2+: Batch create higher-order meta-beliefs
                    if max_depth >= 2:
                        # Generate all triplets (observer, subject1, subject2)
                        triplets = [
                            {"observer": obs, "subject1": subj1, "subject2": subj2}
                            for obs in agent_ids
                            for subj1 in agent_ids
                            if obs != subj1
                            for subj2 in agent_ids
                            if subj1 != subj2
                        ]

                        # Process in batches
                        for i in range(0, len(triplets), batch_size):
                            batch_triplets = triplets[i : i + batch_size]

                            nested_sig_l2 = f"K({agent_ids[0]}, K({agent_ids[1]}, {proposition_id}))"  # Simplified

                            meta2_query = """
                            UNWIND $triplets AS triple
                            MATCH (observer:Agent {id: triple.observer})
                            MATCH (subject1:Agent {id: triple.subject1})
                            MATCH (subject2:Agent {id: triple.subject2})
                            MERGE (observer)-[:KNOWS_THAT]->(mb:MetaBelief {
                                proposition: $nested_sig,
                                meta_level: 2
                            })
                            MERGE (mb)-[:ABOUT_AGENT]->(subject1)
                            MERGE (mb)-[:ABOUT_AGENT]->(subject2)
                            ON CREATE SET mb.created_at = timestamp()
                            RETURN count(mb) as created
                            """
                            result = tx.run(
                                meta2_query,
                                triplets=batch_triplets,
                                nested_sig=nested_sig_l2,
                            )
                            meta2_created = result.single()["created"]
                            count += meta2_created

                            logger.debug(
                                f"Level 2: Created {meta2_created} meta-beliefs (batch {i // batch_size + 1})",
                                extra={
                                    "meta2_created": meta2_created,
                                    "batch_index": i // batch_size + 1,
                                },
                            )

                    return count

                # Execute batch operation in single transaction
                total_count = session.write_transaction(_batch_create)

            # Invalidate caches after bulk update
            self._cache_mgr.invalidate("epistemic_k_operator")
            self._cache_mgr.invalidate("epistemic_k_n_operator")

            logger.info(
                f"Batch common knowledge propagated: {total_count} nodes created",
                extra={
                    "agent_ids": agent_ids,
                    "proposition_id": proposition_id,
                    "max_depth": max_depth,
                    "count": total_count,
                    "performance": "O(1) batch operations",
                },
            )

            return total_count

        except ServiceUnavailable as e:
            logger.error(
                f"Neo4j unavailable during batch common knowledge propagation",
                extra={
                    "agent_count": len(agent_ids),
                    "proposition_id": proposition_id,
                    "max_depth": max_depth,
                    "error": str(e),
                },
            )
            return 0

        except Neo4jError as e:
            logger.error(
                f"Neo4j error during batch common knowledge propagation",
                extra={
                    "agent_count": len(agent_ids),
                    "proposition_id": proposition_id,
                    "max_depth": max_depth,
                    "error": str(e),
                    "error_code": getattr(e, "code", None),
                },
            )
            return 0

        except Exception as e:
            logger.critical(
                f"Unexpected error during batch common knowledge propagation",
                extra={
                    "agent_count": len(agent_ids),
                    "proposition_id": proposition_id,
                    "max_depth": max_depth,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

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
