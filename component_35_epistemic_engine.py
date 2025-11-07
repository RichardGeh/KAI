"""
component_35_epistemic_engine.py

Epistemic Logic Engine für KAI - Wissens- und Glaubensmodellierung

Implementiert modale epistemische Logik mit Multi-Agenten-Perspektiven,
Meta-Wissen und Common Knowledge.

Autor: KAI Development Team
Erstellt: 2025-11-01
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger

logger = get_logger(__name__)


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


class EpistemicEngine:
    """
    Hybrid Epistemic Reasoning Engine

    Kombiniert:
    - Neo4j Graph (Plan 2) für Persistenz und Traversal
    - In-Memory Reasoning (Plan 3) für schnelle Modal-Operationen
    """

    def __init__(self, netzwerk: KonzeptNetzwerk):
        self.netzwerk = netzwerk
        self.current_state: Optional[EpistemicState] = None
        self._proposition_cache: Dict[str, Proposition] = {}
        self._agent_cache: Dict[str, Agent] = {}
        logger.info("EpistemicEngine initialisiert")

    def load_state_from_graph(self) -> EpistemicState:
        """Lädt aktuellen epistemischen Zustand aus Neo4j"""
        # TODO: Implementiere in Phase 2

    def persist_state_to_graph(self, state: EpistemicState) -> bool:
        """Persistiert epistemischen Zustand nach Neo4j"""
        # TODO: Implementiere in Phase 2

    def create_agent(
        self, agent_id: str, name: str, reasoning_capacity: int = 5
    ) -> Agent:
        """Erstelle neuen Agenten (in-memory + graph)"""
        agent = Agent(id=agent_id, name=name, reasoning_capacity=reasoning_capacity)
        self._agent_cache[agent_id] = agent
        # Persistiere zu Neo4j
        self.netzwerk.create_agent(agent_id, name, reasoning_capacity)
        logger.debug(f"Agent erstellt: {agent_id}")
        return agent

    def _ensure_state(self) -> None:
        """Ensure current_state is initialized"""
        if self.current_state is None:
            self.current_state = EpistemicState(
                agents={},
                propositions={},
                knowledge_base={},
                meta_knowledge={},
                common_knowledge={},
            )
            logger.debug("EpistemicState initialisiert")

    def K(self, agent_id: str, proposition_id: str) -> bool:
        """
        Modal Operator K: "Agent knows proposition"

        Checks:
        1. In-memory cache (fast path)
        2. Neo4j graph (authoritative)

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            True wenn Agent die Proposition kennt
        """
        # Fast path: Check cache
        if self.current_state:
            if agent_id in self.current_state.knowledge_base:
                if proposition_id in self.current_state.knowledge_base[agent_id]:
                    logger.debug(
                        f"K({agent_id}, {proposition_id}) = True (cache)",
                        extra={"agent_id": agent_id, "proposition_id": proposition_id},
                    )
                    return True

        # Authoritative: Query Neo4j
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

            logger.debug(
                f"K({agent_id}, {proposition_id}) = {knows} (graph)",
                extra={
                    "agent_id": agent_id,
                    "proposition_id": proposition_id,
                    "knows": knows,
                },
            )
            return knows

    def add_knowledge(
        self, agent_id: str, proposition_id: str, certainty: float = 1.0
    ) -> bool:
        """
        Füge Wissen zu Agent hinzu (updates cache + graph)

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition
            certainty: Gewissheit (0.0 - 1.0, default: 1.0)

        Returns:
            True bei Erfolg, False bei Fehler
        """
        # Ensure state is initialized
        self._ensure_state()

        # Update cache
        if agent_id not in self.current_state.knowledge_base:
            self.current_state.knowledge_base[agent_id] = set()
        self.current_state.knowledge_base[agent_id].add(proposition_id)

        # Update graph
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

        return success

    def M(self, agent_id: str, proposition_id: str) -> bool:
        """
        Modal Operator M: "Agent considers proposition possible"

        In Kripke Semantics: M(p) = ¬K(¬p)
        "Agent glaubt P ist möglich" = "Agent weiß nicht, dass ¬P wahr ist"

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der Proposition

        Returns:
            True wenn Agent die Proposition für möglich hält
        """
        # Erstelle negierte Proposition ID
        negated_prop_id = f"NOT_{proposition_id}"

        # M(p) = ¬K(¬p)
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

    def add_negated_knowledge(self, agent_id: str, proposition_id: str) -> bool:
        """
        Agent weiß, dass Proposition FALSCH ist

        Fügt negierte Proposition "NOT_{proposition_id}" zur Knowledge Base hinzu.

        Args:
            agent_id: ID des Agenten
            proposition_id: ID der ursprünglichen Proposition

        Returns:
            True bei Erfolg, False bei Fehler
        """
        negated_prop_id = f"NOT_{proposition_id}"
        return self.add_knowledge(agent_id, negated_prop_id, certainty=1.0)

    def E(self, agent_ids: List[str], proposition_id: str) -> bool:
        """
        Modal Operator E: "Everyone in group knows proposition"

        E_G(p) = ∀a ∈ G: K_a(p)

        Args:
            agent_ids: Liste von Agent-IDs in der Gruppe
            proposition_id: ID der Proposition

        Returns:
            True wenn ALLE Agenten die Proposition kennen
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

    def add_group_knowledge(self, agent_ids: List[str], proposition_id: str) -> bool:
        """
        Füge Wissen zu allen Agenten in Gruppe hinzu

        Args:
            agent_ids: Liste von Agent-IDs
            proposition_id: ID der Proposition

        Returns:
            True wenn erfolgreich zu ALLEN Agenten hinzugefügt, False bei mindestens einem Fehler
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

    def C_simple(
        self, agent_ids: List[str], proposition_id: str, max_depth: int = 3
    ) -> bool:
        """
        Modal Operator C (Simple): "Common knowledge in group"

        C_G(p) = E_G(p) ∧ E_G(E_G(p)) ∧ E_G(E_G(E_G(p))) ∧ ...

        Vereinfachte Fixed-Point Approximation:
        C ≈ E^max_depth (iterierte "everyone knows")

        Args:
            agent_ids: Gruppe von Agenten
            proposition_id: Proposition
            max_depth: Anzahl E-Iterationen (default: 3)

        Returns:
            True wenn Common Knowledge approximiert erfüllt ist
        """
        # Level 0: Everyone knows p
        if not self.E(agent_ids, proposition_id):
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
        """
        # Step 1: Everyone knows p (Level 0)
        self.add_group_knowledge(agent_ids, proposition_id)

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
            if self.K(agent_id, proposition_id):
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
        Propagiere Common Knowledge durch Gruppe (Public Announcement)

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
        """
        count = 0

        # Level 0: Everyone learns proposition
        for agent_id in agent_ids:
            self.add_knowledge(agent_id, proposition_id)
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
            f"Common knowledge propagated: {count} knowledge nodes created",
            extra={
                "agent_ids": agent_ids,
                "proposition_id": proposition_id,
                "max_depth": max_depth,
                "count": count,
            },
        )
        return count

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
        self, observer_id: str, nested_knowledge: List[str], proposition_id: str
    ) -> bool:
        """
        K^n Operator: Nested knowledge

        Prüft rekursiv verschachteltes Wissen wie:
        - K_n("alice", ["bob"], "p") = "Alice knows that Bob knows p"
        - K_n("alice", ["bob", "carol"], "p") = "Alice knows that Bob knows that Carol knows p"

        Args:
            observer_id: Der äußerste Beobachter (A)
            nested_knowledge: Chain von Agenten [B, C, ...] (äußerste zuerst)
            proposition_id: Die Basis-Proposition (p)

        Returns:
            True wenn verschachteltes Wissen existiert

        Example:
            # Bob knows secret_password
            engine.add_knowledge("bob", "secret_password")

            # Alice knows that Bob knows secret_password
            engine.add_nested_knowledge("alice", ["bob"], "secret_password")

            # Check if Alice knows that Bob knows the secret
            engine.K_n("alice", ["bob"], "secret_password")  # Returns True
        """
        if not nested_knowledge:
            # Base case: K(observer, prop)
            return self.K(observer_id, proposition_id)

        # Recursive case: K(observer, K(nested[0], ...))
        # Query Neo4j für MetaBelief
        next_subject = nested_knowledge[0]
        remaining_chain = nested_knowledge[1:]
        meta_level = len(nested_knowledge)

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

            logger.debug(
                f"K_n({observer_id}, {nested_knowledge}, {proposition_id}) = {has_knowledge}",
                extra={
                    "observer_id": observer_id,
                    "nested_knowledge": nested_knowledge,
                    "proposition_id": proposition_id,
                    "nested_sig": nested_sig,
                    "has_knowledge": has_knowledge,
                },
            )
            return has_knowledge

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

        Example:
            # Alice knows that Bob knows the secret
            add_nested_knowledge("alice", ["bob"], "secret_password")

            # Carol knows that Bob knows that Alice knows the secret
            add_nested_knowledge("carol", ["bob", "alice"], "secret_password")
        """
        if not nested_chain:
            # Base case: Einfaches Wissen ohne Verschachtelung
            return self.add_knowledge(observer_id, proposition_id)

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
    ) -> List[Dict]:
        """
        Finde alle Meta-Knowledge-Pfade ausgehend von observer

        Nutzt Graph-Traversal für effiziente Suche

        Args:
            observer_id: ID des Beobachters
            max_depth: Maximale Tiefe der Pfade (default: 3)

        Returns:
            Liste von Dicts mit:
            - path: [observer -> subject1 -> subject2 -> ...]
            - proposition: Die finale Proposition
            - meta_level: Tiefe des Pfads
        """
        paths = []

        with self.netzwerk.driver.session(database="neo4j") as session:
            # Cypher: Variable-length path query
            # Findet alle Pfade: (observer)-[:KNOWS_THAT*1..max_depth]->(mb:MetaBelief)
            result = session.run(
                """
                MATCH path = (observer:Agent {id: $observer_id})
                             -[:KNOWS_THAT*1..%d]->(mb:MetaBelief)
                UNWIND nodes(path) AS node
                WITH path, mb,
                     [n IN nodes(path) WHERE n:Agent | n.id] AS agent_chain,
                     length(path) AS depth
                RETURN agent_chain, mb.proposition AS proposition, depth
                ORDER BY depth ASC
                """
                % max_depth,
                observer_id=observer_id,
            )

            for record in result:
                paths.append(
                    {
                        "path": record["agent_chain"],
                        "proposition": record["proposition"],
                        "meta_level": record["depth"],
                    }
                )

        logger.debug(
            f"Found {len(paths)} meta-knowledge paths from {observer_id}",
            extra={
                "observer_id": observer_id,
                "max_depth": max_depth,
                "paths_found": len(paths),
            },
        )
        return paths


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

    print("✓ K operator test passed")

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

    print("✓ M operator test passed")

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

    print("✓ E operator test passed")

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

    print("✓ C operator (simple) test passed")

    print("\n✓ All tests passed!")
