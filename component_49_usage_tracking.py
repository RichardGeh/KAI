# component_49_usage_tracking.py
"""
Usage Tracking für KAI - Phase 1.3: Cognitive Resonance

Implementiert detailliertes Tracking von:
- Relation Usage: Wie oft wurden Relations in Reasoning verwendet?
- Concept Activation: Wie aktiv war ein Konzept in Queries?
- Query Patterns: Welche Queries aktivieren welche Konzepte?

Design:
- Integriert mit Dynamic Confidence (Phase 1.1)
- Nutzt Enhanced Schema (Phase 1.2)
- Automatisch aufgerufen von allen Reasoning Engines
- Performance-optimiert mit Batch-Updates
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ==================== DATA STRUCTURES ====================


@dataclass
class UsageStats:
    """
    Statistiken über Relation/Concept Usage.

    Attributes:
        usage_count: Anzahl der Nutzungen
        first_used: Zeitpunkt der ersten Nutzung
        last_used: Zeitpunkt der letzten Nutzung
        queries: Liste der Query-IDs, die diesen Fakt genutzt haben
        activation_levels: Durchschnittliche Aktivierungslevel (für Concepts)
        contexts: Kontexte, in denen genutzt wurde
        reinforcement_score: Aggregierter Reinforcement-Score
    """

    usage_count: int = 0
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None
    queries: List[str] = field(default_factory=list)
    activation_levels: List[float] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    reinforcement_score: float = 0.0

    @property
    def average_activation(self) -> float:
        """Durchschnittliche Aktivierung."""
        if not self.activation_levels:
            return 0.0
        return sum(self.activation_levels) / len(self.activation_levels)

    @property
    def days_since_last_use(self) -> Optional[int]:
        """Tage seit letzter Nutzung."""
        if not self.last_used:
            return None
        return (datetime.now() - self.last_used).days

    @property
    def usage_frequency(self) -> float:
        """Nutzungshäufigkeit (uses per day)."""
        if not self.first_used or not self.last_used:
            return 0.0

        days_active = max(1, (self.last_used - self.first_used).days)
        return self.usage_count / days_active


@dataclass
class QueryUsageRecord:
    """
    Tracking-Record für eine einzelne Query.

    Erfasst welche Relations/Concepts in einer Query verwendet wurden.
    """

    query_id: str
    timestamp: datetime
    query_type: str  # "reasoning", "search", "learning"
    activated_concepts: List[str] = field(default_factory=list)
    used_relations: List[tuple] = field(
        default_factory=list
    )  # (subject, relation, object)
    activation_levels: Dict[str, float] = field(default_factory=dict)
    reasoning_depth: int = 0


# ==================== USAGE TRACKING MANAGER ====================


class UsageTrackingManager:
    """
    Manager für Usage Tracking.

    Koordiniert alle Tracking-Aktivitäten und speichert Usage-Daten in Neo4j.
    """

    def __init__(self, netzwerk):
        """
        Initialisiert den UsageTrackingManager.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz
        """
        self.netzwerk = netzwerk

        # Bounded buffer with automatic eviction (CRITICAL FIX: Issue #3)
        # maxlen=500 ensures bounded growth, preventing memory leaks
        self._usage_buffer = deque(maxlen=500)  # Hard limit
        self._buffer_size = 50  # Soft limit for batching
        self._buffer_lock = threading.Lock()

        # Thread safety for Neo4j access
        self._neo4j_lock = threading.RLock()

        logger.info(
            "UsageTrackingManager initialisiert",
            extra={"max_buffer_size": 500, "batch_size": 50},
        )

    # ==================== RELATION USAGE TRACKING ====================

    def track_relation_usage(
        self,
        subject: str,
        relation: str,
        object_: str,
        query_id: str,
        context: Optional[str] = None,
    ) -> bool:
        """
        Tracked die Nutzung einer Relation in einer Query.

        Updates:
        - relation.usage_count += 1
        - relation.last_reinforced = now()
        - Verknüpft mit Query-Record
        - Integriert mit Dynamic Confidence System

        Args:
            subject: Subject der Relation
            relation: Relationstyp
            object_: Object der Relation
            query_id: ID der Query die diese Relation genutzt hat
            context: Optional - Kontext der Nutzung

        Returns:
            True bei Erfolg, False bei Fehler

        Example:
            >>> manager.track_relation_usage(
            ...     subject="hund",
            ...     relation="IS_A",
            ...     object_="säugetier",
            ...     query_id=query_id,
            ...     context="multi_hop_reasoning"
            ... )
        """
        if not self.netzwerk or not self.netzwerk.driver:
            logger.warning("Kein Netzwerk verfügbar für Relation Usage Tracking")
            return False

        try:
            import re

            safe_relation = re.sub(r"[^a-zA-Z0-9_]", "", relation.upper())

            with self._neo4j_lock:
                # Use driver directly for usage tracking (no facade method exists yet)
                with self.netzwerk.driver.session(database="neo4j") as session:
                    # Use WHERE clause with type() function
                    session.run(
                        """
                        MATCH (s:Konzept {name: $subject})-[r]->(o:Konzept {name: $object})
                        WHERE type(r) = $relation
                        SET r.usage_count = COALESCE(r.usage_count, 0) + 1,
                            r.last_reinforced = datetime({timezone: 'UTC'}),
                            r.context = CASE
                                WHEN $context IS NOT NULL AND r.context IS NOT NULL
                                    AND NOT $context IN r.context
                                THEN r.context + $context
                                WHEN $context IS NOT NULL AND r.context IS NULL
                                THEN [$context]
                                ELSE r.context
                            END

                        // Verknüpfe mit Query Record
                        WITH r
                        MERGE (q:QueryRecord {id: $query_id})
                        ON CREATE SET q.timestamp = datetime({timezone: 'UTC'})
                        MERGE (q)-[used:USED_RELATION]->(f:Fact {
                            subject: $subject,
                            relation: $relation,
                            object: $object
                        })
                        ON CREATE SET used.timestamp = datetime({timezone: 'UTC'})
                        """,
                        subject=subject.lower(),
                        object=object_.lower(),
                        relation=safe_relation,
                        query_id=query_id,
                        context=context,
                    )

            logger.debug(
                f"Relation usage tracked: {subject} -{relation}-> {object_} (query={query_id})"
            )
            return True

        except ValueError as e:
            logger.error(
                "Validation error beim Tracken von Relation Usage",
                exc_info=True,
                extra={
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                    "query_id": query_id,
                    "error_type": "ValueError",
                },
            )
            return False
        except Exception as e:
            logger.error(
                "Neo4j Fehler beim Tracken von Relation Usage",
                exc_info=True,
                extra={
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                    "query_id": query_id,
                    "error_type": type(e).__name__,
                    "driver_available": self.netzwerk.driver is not None,
                },
            )
            return False

    # ==================== CONCEPT ACTIVATION TRACKING ====================

    def track_concept_activation(
        self,
        concept: str,
        activation_level: float,
        query_id: str,
        context: Optional[str] = None,
    ) -> bool:
        """
        Tracked die Aktivierung eines Konzepts in einer Query.

        Activation Level:
        - 1.0: Direkt in Query erwähnt (z.B. "Was ist ein Hund?")
        - 0.8: Inferiert via 1-Hop Reasoning
        - 0.6: Inferiert via 2-Hop Reasoning
        - 0.4: Inferiert via 3+ Hop Reasoning
        - 0.2: Indirekt relevant

        Updates:
        - konzept.usage_frequency += 1
        - konzept.last_used = now()
        - Verknüpft mit Query-Record

        Args:
            concept: Das Konzept
            activation_level: Aktivierungslevel (0.0-1.0)
            query_id: ID der Query
            context: Optional - Kontext

        Returns:
            True bei Erfolg

        Example:
            >>> manager.track_concept_activation(
            ...     concept="hund",
            ...     activation_level=1.0,
            ...     query_id=query_id,
            ...     context="direct_query"
            ... )
        """
        if not self.netzwerk or not self.netzwerk.driver:
            logger.warning("Kein Netzwerk verfügbar für Concept Activation Tracking")
            return False

        if not 0.0 <= activation_level <= 1.0:
            logger.warning(
                f"Ungültiger Activation Level: {activation_level}, clipping zu [0, 1]"
            )
            activation_level = max(0.0, min(1.0, activation_level))

        try:
            with self._neo4j_lock:
                # Use driver directly for usage tracking (no facade method exists yet)
                with self.netzwerk.driver.session(database="neo4j") as session:
                    session.run(
                        """
                        MATCH (k:Konzept {name: $concept})
                        SET k.usage_frequency = COALESCE(k.usage_frequency, 0) + 1,
                            k.last_used = datetime({timezone: 'UTC'})

                        // Verknüpfe mit Query Record
                        WITH k
                        MERGE (q:QueryRecord {id: $query_id})
                        ON CREATE SET q.timestamp = datetime({timezone: 'UTC'})
                        MERGE (q)-[activated:ACTIVATED_CONCEPT]->(k)
                        ON CREATE SET activated.timestamp = datetime({timezone: 'UTC'}),
                                      activated.activation_level = $activation_level,
                                      activated.context = $context
                        ON MATCH SET activated.activation_level =
                            (activated.activation_level + $activation_level) / 2.0
                        """,
                        concept=concept.lower(),
                        query_id=query_id,
                        activation_level=activation_level,
                        context=context,
                    )

            logger.debug(
                f"Concept activation tracked: {concept} (level={activation_level:.2f}, query={query_id})"
            )
            return True

        except ValueError as e:
            logger.error(
                "Validation error beim Tracken von Concept Activation",
                exc_info=True,
                extra={
                    "concept": concept,
                    "activation_level": activation_level,
                    "query_id": query_id,
                    "error_type": "ValueError",
                },
            )
            return False
        except Exception as e:
            logger.error(
                "Neo4j Fehler beim Tracken von Concept Activation",
                exc_info=True,
                extra={
                    "concept": concept,
                    "activation_level": activation_level,
                    "query_id": query_id,
                    "error_type": type(e).__name__,
                    "driver_available": self.netzwerk.driver is not None,
                },
            )
            return False

    # ==================== BATCH TRACKING ====================

    def track_query_usage(
        self,
        query_id: str,
        query_type: str,
        activated_concepts: List[str],
        used_relations: List[tuple],
        activation_levels: Optional[Dict[str, float]] = None,
        reasoning_depth: int = 0,
    ) -> bool:
        """
        Tracked alle Aktivitäten einer Query in einem Batch.

        Performance-optimiert: Ein Query anstatt vieler einzelner.

        Args:
            query_id: Query-ID
            query_type: Typ der Query ("reasoning", "search", "learning")
            activated_concepts: Liste aktivierter Konzepte
            used_relations: Liste genutzter Relations (tuples)
            activation_levels: Optional - Aktivierungslevel pro Konzept
            reasoning_depth: Maximale Reasoning-Tiefe

        Returns:
            True bei Erfolg

        Example:
            >>> manager.track_query_usage(
            ...     query_id="query_123",
            ...     query_type="reasoning",
            ...     activated_concepts=["hund", "säugetier", "tier"],
            ...     used_relations=[("hund", "IS_A", "säugetier"), ("säugetier", "IS_A", "tier")],
            ...     activation_levels={"hund": 1.0, "säugetier": 0.8, "tier": 0.6},
            ...     reasoning_depth=2
            ... )
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return False

        if activation_levels is None:
            activation_levels = {concept: 1.0 for concept in activated_concepts}

        try:
            # Track alle Concepts
            for concept in activated_concepts:
                level = activation_levels.get(concept, 0.5)
                self.track_concept_activation(concept, level, query_id)

            # Track alle Relations
            for subject, relation, object_ in used_relations:
                self.track_relation_usage(subject, relation, object_, query_id)

            # Update Query Record mit Metadaten
            # Use driver directly for query metadata (no facade method exists yet)
            with self.netzwerk.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (q:QueryRecord {id: $query_id})
                    SET q.query_type = $query_type,
                        q.reasoning_depth = $reasoning_depth,
                        q.concept_count = $concept_count,
                        q.relation_count = $relation_count
                    """,
                    query_id=query_id,
                    query_type=query_type,
                    reasoning_depth=reasoning_depth,
                    concept_count=len(activated_concepts),
                    relation_count=len(used_relations),
                )

            logger.info(
                f"Query usage tracked: {query_id} "
                f"({len(activated_concepts)} concepts, {len(used_relations)} relations)"
            )
            return True

        except Exception as e:
            logger.error(f"Fehler beim Batch-Tracking: {e}", exc_info=True)
            return False

    # ==================== STATISTICS RETRIEVAL ====================

    def get_usage_statistics(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        object_: Optional[str] = None,
        concept: Optional[str] = None,
    ) -> UsageStats:
        """
        Holt Usage-Statistiken für eine Relation oder ein Konzept.

        Args:
            subject: Subject der Relation (für Relation-Stats)
            relation: Relationstyp (für Relation-Stats)
            object_: Object der Relation (für Relation-Stats)
            concept: Konzept-Name (für Concept-Stats)

        Returns:
            UsageStats-Objekt mit allen Statistiken

        Example:
            >>> # Relation Stats
            >>> stats = manager.get_usage_statistics(
            ...     subject="hund",
            ...     relation="IS_A",
            ...     object_="säugetier"
            ... )
            >>> print(f"Usage: {stats.usage_count}x, Avg Activation: {stats.average_activation:.2f}")

            >>> # Concept Stats
            >>> stats = manager.get_usage_statistics(concept="hund")
            >>> print(f"Frequency: {stats.usage_frequency:.2f} uses/day")
        """
        if concept:
            return self._get_concept_usage_statistics(concept)
        elif subject and relation and object_:
            return self._get_relation_usage_statistics(subject, relation, object_)
        else:
            logger.warning(
                "get_usage_statistics benötigt entweder concept oder (subject, relation, object)"
            )
            return UsageStats()

    def _get_relation_usage_statistics(
        self, subject: str, relation: str, object_: str
    ) -> UsageStats:
        """Holt Usage Stats für eine Relation."""
        if not self.netzwerk or not self.netzwerk.driver:
            return UsageStats()

        try:
            import re

            safe_relation = re.sub(r"[^a-zA-Z0-9_]", "", relation.upper())

            # Use driver directly for usage statistics (no facade method exists yet)
            with self.netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    f"""
                    MATCH (s:Konzept {{name: $subject}})-[r:{safe_relation}]->(o:Konzept {{name: $object}})
                    OPTIONAL MATCH (q:QueryRecord)-[:USED_RELATION]->(f:Fact {{
                        subject: $subject,
                        relation: $relation,
                        object: $object
                    }})
                    WITH r,
                         count(DISTINCT q) AS usage_count,
                         collect(DISTINCT q.id) AS query_ids,
                         r.context AS contexts
                    RETURN COALESCE(r.usage_count, usage_count, 0) AS usage_count,
                           r.last_reinforced AS last_used,
                           query_ids,
                           contexts
                    """,
                    subject=subject.lower(),
                    object=object_.lower(),
                    relation=relation,
                )

                record = result.single()

                if record:
                    last_used = record["last_used"]
                    if last_used and hasattr(last_used, "to_native"):
                        last_used = last_used.to_native()

                    return UsageStats(
                        usage_count=record["usage_count"] or 0,
                        last_used=last_used,
                        queries=record["query_ids"] or [],
                        contexts=record["contexts"] or [],
                    )

                return UsageStats()

        except Exception as e:
            logger.error(f"Fehler beim Abrufen von Relation Stats: {e}", exc_info=True)
            return UsageStats()

    def _get_concept_usage_statistics(self, concept: str) -> UsageStats:
        """Holt Usage Stats für ein Konzept."""
        if not self.netzwerk or not self.netzwerk.driver:
            return UsageStats()

        try:
            # Use driver directly for usage statistics (no facade method exists yet)
            with self.netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (k:Konzept {name: $concept})
                    OPTIONAL MATCH (q:QueryRecord)-[activated:ACTIVATED_CONCEPT]->(k)
                    WITH k,
                         count(DISTINCT q) AS query_count,
                         collect(DISTINCT q.id) AS query_ids,
                         collect(activated.activation_level) AS activation_levels,
                         k.contexts AS contexts
                    RETURN COALESCE(k.usage_frequency, query_count, 0) AS usage_count,
                           k.first_seen AS first_used,
                           k.last_used AS last_used,
                           query_ids,
                           activation_levels,
                           contexts
                    """,
                    concept=concept.lower(),
                )

                record = result.single()

                if record:
                    first_used = record["first_used"]
                    last_used = record["last_used"]

                    if first_used and hasattr(first_used, "to_native"):
                        first_used = first_used.to_native()
                    if last_used and hasattr(last_used, "to_native"):
                        last_used = last_used.to_native()

                    activation_levels = [
                        level
                        for level in record["activation_levels"]
                        if level is not None
                    ]

                    return UsageStats(
                        usage_count=record["usage_count"] or 0,
                        first_used=first_used,
                        last_used=last_used,
                        queries=record["query_ids"] or [],
                        activation_levels=activation_levels,
                        contexts=record["contexts"] or [],
                    )

                return UsageStats()

        except Exception as e:
            logger.error(f"Fehler beim Abrufen von Concept Stats: {e}", exc_info=True)
            return UsageStats()

    # ==================== ANALYTICS ====================

    def get_most_used_relations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Holt die am häufigsten genutzten Relations.

        Args:
            limit: Maximale Anzahl Relations

        Returns:
            Liste von Dicts mit {subject, relation, object, usage_count}
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return []

        try:
            # Use driver directly for analytics (no facade method exists yet)
            with self.netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (s:Konzept)-[r]->(o:Konzept)
                    WHERE r.usage_count IS NOT NULL
                    RETURN s.name AS subject,
                           type(r) AS relation,
                           o.name AS object,
                           r.usage_count AS usage_count
                    ORDER BY r.usage_count DESC
                    LIMIT $limit
                    """,
                    limit=limit,
                )

                return [record.data() for record in result]

        except Exception as e:
            logger.error(f"Fehler bei get_most_used_relations: {e}", exc_info=True)
            return []

    def get_most_activated_concepts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Holt die am häufigsten aktivierten Konzepte.

        Args:
            limit: Maximale Anzahl Konzepte

        Returns:
            Liste von Dicts mit {concept, usage_frequency, last_used}
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return []

        try:
            # Use driver directly for analytics (no facade method exists yet)
            with self.netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (k:Konzept)
                    WHERE k.usage_frequency IS NOT NULL
                    RETURN k.name AS concept,
                           k.usage_frequency AS usage_frequency,
                           k.last_used AS last_used
                    ORDER BY k.usage_frequency DESC
                    LIMIT $limit
                    """,
                    limit=limit,
                )

                return [record.data() for record in result]

        except Exception as e:
            logger.error(f"Fehler bei get_most_activated_concepts: {e}", exc_info=True)
            return []


# ==================== GLOBAL INSTANCE ====================

_global_usage_tracker: Optional[UsageTrackingManager] = None
_usage_tracker_lock = threading.Lock()


def get_usage_tracker(netzwerk=None) -> UsageTrackingManager:
    """
    Gibt die globale UsageTrackingManager-Instanz zurück.

    Thread-safe mit double-check locking pattern.

    Args:
        netzwerk: KonzeptNetzwerk-Instanz (nur beim ersten Aufruf erforderlich)

    Returns:
        Globale UsageTrackingManager-Instanz

    Raises:
        ValueError: Wenn beim ersten Aufruf kein netzwerk übergeben wurde
    """
    global _global_usage_tracker

    # Double-check locking pattern for thread safety
    if _global_usage_tracker is None:
        with _usage_tracker_lock:
            if _global_usage_tracker is None:
                if netzwerk is None:
                    raise ValueError(
                        "Beim ersten Aufruf muss netzwerk-Parameter übergeben werden"
                    )

                _global_usage_tracker = UsageTrackingManager(netzwerk)
                logger.info("Globale UsageTrackingManager-Instanz erstellt")

    return _global_usage_tracker


# ==================== BEISPIEL-USAGE ====================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== Usage Tracking System Demo ===\n")

    # Mock Netzwerk für Demo
    class MockNetzwerk:
        driver = None

    netzwerk = MockNetzwerk()
    manager = UsageTrackingManager(netzwerk)

    print("Beispiel 1: Track Relation Usage")
    print("  manager.track_relation_usage('hund', 'IS_A', 'säugetier', 'query_1')")
    print()

    print("Beispiel 2: Track Concept Activation")
    print("  manager.track_concept_activation('hund', 1.0, 'query_1')")
    print()

    print("Beispiel 3: Batch Query Tracking")
    print("  manager.track_query_usage(")
    print("      query_id='query_2',")
    print("      query_type='reasoning',")
    print("      activated_concepts=['hund', 'säugetier', 'tier'],")
    print("      used_relations=[('hund', 'IS_A', 'säugetier')]")
    print("  )")
    print()

    print("Beispiel 4: Get Usage Statistics")
    print("  stats = manager.get_usage_statistics(concept='hund')")
    print(
        "  print(f'Usage: {stats.usage_count}x, Frequency: {stats.usage_frequency:.2f}/day')"
    )
