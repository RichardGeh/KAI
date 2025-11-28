# component_1_query_engine.py
"""
Fact queries and semantic search for Neo4j knowledge graph.

This module handles all query operations including:
- Fact queries (outgoing and inverse relations)
- Confidence-aware queries
- Synonym-aware search
- Semantic similarity search
- Production rule queries and statistics

Extracted from monolithic component_1_netzwerk_core.py as part of architecture
refactoring (Task 5).
"""

import json
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import PerformanceLogger, get_logger
from infrastructure import get_cache_manager

logger = get_logger(__name__)


class QueryEngine:
    """
    Handles fact queries and semantic search in Neo4j.

    Responsibilities:
    - Query facts (outgoing/inverse relations)
    - Confidence-aware queries
    - Synonym-aware search
    - Semantic similarity (via embedding service)
    - Production rule queries
    - Fact caching with TTL

    Thread Safety:
        This class is thread-safe. All cache operations are protected by the
        CacheManager's internal locking.

    Attributes:
        driver: Neo4j driver instance
        word_manager: WordManager for word operations
        cache_mgr: CacheManager for caching queries
        _lock: Thread lock for critical operations
    """

    def __init__(self, driver: Driver, word_manager):
        """
        Initialize query engine.

        Args:
            driver: Neo4j driver instance
            word_manager: WordManager instance for word operations

        Raises:
            ValueError: If driver or word_manager is None
        """
        if not driver:
            raise ValueError("Driver cannot be None")
        if not word_manager:
            raise ValueError("WordManager cannot be None")

        self.driver = driver
        self.word_manager = word_manager
        self._lock = threading.RLock()

        # Register caches with CacheManager
        self.cache_mgr = get_cache_manager()

        # Cache for fact queries (5 minutes TTL)
        self.cache_mgr.register_cache(
            "netzwerk_facts", maxsize=500, ttl=300, overwrite=True
        )

        logger.debug("QueryEngine initialisiert mit CacheManager")

    def query_graph_for_facts(self, topic: str) -> Dict[str, List[str]]:
        """
        Query the graph for all known outgoing facts (relationships) for a topic.

        Performance optimization: Uses TTL cache (5 minutes) for frequently queried topics.

        Note: This method searches only for facts about the directly queried topic.
        For extended search across all synonyms, use query_facts_with_synonyms().

        Args:
            topic: The concept to query

        Returns:
            Dict with {relation_type: [target_concepts]}
        """
        if not self.driver:
            logger.error(
                "query_graph_for_facts: Kein DB-Driver verfügbar",
                extra={"topic": topic},
            )
            return {}

        # Cache key
        cache_key = f"facts:{topic.lower()}"

        # Check cache
        cached = self.cache_mgr.get("netzwerk_facts", cache_key)
        if cached is not None:
            logger.debug("Cache-Hit für query_graph_for_facts", extra={"topic": topic})
            return cached

        try:
            with PerformanceLogger(logger.logger, "query_graph_for_facts", topic=topic):
                with self.driver.session(database="neo4j") as session:
                    self.word_manager.ensure_wort_und_konzept(topic)

                    # Main query: All outgoing relations
                    result = session.run(
                        """
                        MATCH (s:Konzept {name: $topic})-[r]->(o:Konzept)
                        RETURN type(r) AS relation, o.name AS object
                        """,
                        topic=topic.lower(),
                    )

                    facts: Dict[str, List[str]] = defaultdict(list)
                    for record in result:
                        facts[record["relation"]].append(record["object"])

                    # Additional synonym query (bidirectional)
                    # Finds all words in the same synonym group
                    synonym_result = session.run(
                        """
                        MATCH (w:Wort {lemma: $topic})-[:TEIL_VON]->(sg:Synonymgruppe)<-[:TEIL_VON]-(syn:Wort)
                        WHERE syn.lemma <> $topic
                        RETURN DISTINCT syn.lemma AS synonym
                        """,
                        topic=topic.lower(),
                    )

                    synonyms: List[str] = [
                        record["synonym"] for record in synonym_result
                    ]
                    if synonyms:
                        # Overwrite or merge with existing TEIL_VON facts
                        facts["TEIL_VON"] = list(
                            set(synonyms + facts.get("TEIL_VON", []))
                        )

                    fact_count: int = sum(len(v) for v in facts.values())
                    logger.debug(
                        "Fakten abgerufen (Cache-Miss)",
                        extra={
                            "topic": topic,
                            "fact_count": fact_count,
                            "relation_types": list(facts.keys()),
                        },
                    )

                    # Store in cache
                    result_dict = dict(facts)
                    self.cache_mgr.set("netzwerk_facts", cache_key, result_dict)

                    return result_dict

        except Exception as e:
            # Specific exception for query errors
            logger.log_exception(e, "Fehler in query_graph_for_facts", topic=topic)
            # Graceful degradation: return empty dict
            return {}

    def query_inverse_relations(
        self, topic: str, relation_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Query the graph for all incoming relations for a concept.

        Unlike query_graph_for_facts() which finds outgoing relations
        (e.g., "hund" -> IS_A -> "säugetier"), this method finds incoming
        relations (e.g., "säugetier" <- IS_A <- "hund").

        Useful for:
        - Finding descendants in hierarchies
        - "Why" questions (e.g., "Why is X a Y?")
        - Backward reasoning

        Args:
            topic: The target concept
            relation_type: Optional - only relations of this type (e.g., "IS_A")

        Returns:
            Dict with {relation_type: [source_concepts]}
            Example: {"IS_A": ["hund", "katze", "elefant"]} for topic="säugetier"
        """
        if not self.driver:
            logger.error(
                "query_inverse_relations: Kein DB-Driver verfügbar",
                extra={"topic": topic},
            )
            return {}

        try:
            with PerformanceLogger(
                logger.logger,
                "query_inverse_relations",
                topic=topic,
                relation_type=relation_type,
            ):
                with self.driver.session(database="neo4j") as session:
                    self.word_manager.ensure_wort_und_konzept(topic)

                    # Query: All incoming relations
                    if relation_type:
                        # Only specific relation
                        result = session.run(
                            """
                            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
                            WHERE type(r) = $relation_type
                            RETURN type(r) AS relation, s.name AS subject
                            """,
                            topic=topic.lower(),
                            relation_type=relation_type,
                        )
                    else:
                        # All relations
                        result = session.run(
                            """
                            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
                            RETURN type(r) AS relation, s.name AS subject
                            """,
                            topic=topic.lower(),
                        )

                    inverse_facts: Dict[str, List[str]] = defaultdict(list)
                    for record in result:
                        inverse_facts[record["relation"]].append(record["subject"])

                    fact_count: int = sum(len(v) for v in inverse_facts.values())
                    logger.debug(
                        "Inverse Relationen abgerufen",
                        extra={
                            "topic": topic,
                            "fact_count": fact_count,
                            "relation_types": list(inverse_facts.keys()),
                        },
                    )

                    return dict(inverse_facts)

        except Exception as e:
            logger.log_exception(e, "Fehler in query_inverse_relations", topic=topic)
            return {}

    def query_graph_for_facts_with_confidence(
        self, topic: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query the graph for all outgoing facts AND their confidence values and timestamps.

        Unlike query_graph_for_facts() which returns only target concepts, this method
        also returns confidence values and timestamps from the relations.

        Args:
            topic: The concept to query

        Returns:
            Dict with {relation_type: [{"target": str, "confidence": float, "timestamp": str}]}
            Example: {
                "IS_A": [
                    {"target": "säugetier", "confidence": 0.85, "timestamp": "2025-01-15T10:30:00"},
                    {"target": "tier", "confidence": 0.9, "timestamp": "2025-01-20T14:00:00"}
                ],
                "HAS_PROPERTY": [
                    {"target": "vierbeinig", "confidence": 1.0, "timestamp": None}
                ]
            }
        """
        if not self.driver:
            logger.error(
                "query_graph_for_facts_with_confidence: Kein DB-Driver verfügbar",
                extra={"topic": topic},
            )
            return {}

        try:
            with PerformanceLogger(
                logger.logger, "query_graph_for_facts_with_confidence", topic=topic
            ):
                with self.driver.session(database="neo4j") as session:
                    self.word_manager.ensure_wort_und_konzept(topic)

                    # Query: All outgoing relations with confidence AND timestamp
                    result = session.run(
                        """
                        MATCH (s:Konzept {name: $topic})-[r]->(o:Konzept)
                        RETURN type(r) AS relation,
                               o.name AS object,
                               COALESCE(r.confidence, 1.0) AS confidence,
                               toString(r.timestamp) AS timestamp
                        """,
                        topic=topic.lower(),
                    )

                    facts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for record in result:
                        facts[record["relation"]].append(
                            {
                                "target": record["object"],
                                "confidence": float(record["confidence"]),
                                "timestamp": record[
                                    "timestamp"
                                ],  # Can be None, otherwise ISO string
                            }
                        )

                    fact_count: int = sum(len(v) for v in facts.values())
                    logger.debug(
                        "Fakten mit Confidence und Timestamps abgerufen",
                        extra={
                            "topic": topic,
                            "fact_count": fact_count,
                            "relation_types": list(facts.keys()),
                        },
                    )

                    return dict(facts)

        except Exception as e:
            logger.log_exception(
                e, "Fehler in query_graph_for_facts_with_confidence", topic=topic
            )
            return {}

    def query_inverse_relations_with_confidence(
        self, topic: str, relation_type: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query the graph for incoming relations AND their confidence values.

        Unlike query_inverse_relations() which returns only source concepts, this method
        also returns confidence values from the relations.

        Args:
            topic: The target concept
            relation_type: Optional - only relations of this type (e.g., "IS_A")

        Returns:
            Dict with {relation_type: [{"source": str, "confidence": float}]}
            Example: {
                "IS_A": [
                    {"source": "hund", "confidence": 0.85},
                    {"source": "katze", "confidence": 0.9}
                ]
            }
        """
        if not self.driver:
            logger.error(
                "query_inverse_relations_with_confidence: Kein DB-Driver verfügbar",
                extra={"topic": topic},
            )
            return {}

        try:
            with PerformanceLogger(
                logger.logger,
                "query_inverse_relations_with_confidence",
                topic=topic,
                relation_type=relation_type,
            ):
                with self.driver.session(database="neo4j") as session:
                    self.word_manager.ensure_wort_und_konzept(topic)

                    # Query: All incoming relations with confidence
                    if relation_type:
                        result = session.run(
                            """
                            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
                            WHERE type(r) = $relation_type
                            RETURN type(r) AS relation,
                                   s.name AS subject,
                                   COALESCE(r.confidence, 1.0) AS confidence
                            """,
                            topic=topic.lower(),
                            relation_type=relation_type,
                        )
                    else:
                        result = session.run(
                            """
                            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
                            RETURN type(r) AS relation,
                                   s.name AS subject,
                                   COALESCE(r.confidence, 1.0) AS confidence
                            """,
                            topic=topic.lower(),
                        )

                    inverse_facts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for record in result:
                        inverse_facts[record["relation"]].append(
                            {
                                "source": record["subject"],
                                "confidence": float(record["confidence"]),
                            }
                        )

                    fact_count: int = sum(len(v) for v in inverse_facts.values())
                    logger.debug(
                        "Inverse Relationen mit Confidence abgerufen",
                        extra={
                            "topic": topic,
                            "fact_count": fact_count,
                            "relation_types": list(inverse_facts.keys()),
                        },
                    )

                    return dict(inverse_facts)

        except Exception as e:
            logger.log_exception(
                e, "Fehler in query_inverse_relations_with_confidence", topic=topic
            )
            return {}

    def query_facts_with_synonyms(self, topic: str) -> Dict[str, Any]:
        """
        Robust synonym-aware fact search.

        This method finds ALL facts about a topic AND all its synonyms.
        This allows KAI to merge knowledge across different names for the same concept
        and answer intelligently.

        Example:
        - "Auto" is synonym to "PKW"
        - Fact: "Auto IS_A Fahrzeug"
        - Fact: "PKW HAS_PROPERTY schnell"
        - Query "Was ist ein Auto?" -> Returns both facts

        Args:
            topic: The queried topic

        Returns:
            Dictionary with:
            - "primary_topic": The queried term
            - "synonyms": List of all synonyms
            - "facts": All facts (merged across topic + synonyms)
            - "sources": Mapping of which fact came from which term
            - "bedeutungen": List of all meanings/definitions
        """
        if not self.driver:
            return {
                "primary_topic": topic,
                "synonyms": [],
                "facts": {},
                "sources": {},
                "bedeutungen": [],
            }

        with self.driver.session(database="neo4j") as session:
            self.word_manager.ensure_wort_und_konzept(topic)
            topic_lower: str = topic.lower()

            # STEP 1: Find all synonyms
            synonym_result = session.run(
                """
                MATCH (w:Wort {lemma: $topic})-[:TEIL_VON]->(sg:Synonymgruppe)<-[:TEIL_VON]-(syn:Wort)
                WHERE syn.lemma <> $topic
                RETURN DISTINCT syn.lemma AS synonym
                """,
                topic=topic_lower,
            )
            synonyms: List[str] = [record["synonym"] for record in synonym_result]

            # STEP 2: Collect facts about main topic AND all synonyms (concept level)
            all_topics: List[str] = [topic_lower] + synonyms
            combined_facts: Dict[str, List[str]] = defaultdict(list)
            fact_sources: Dict[str, List[str]] = (
                {}
            )  # Tracking which fact came from which term

            for search_topic in all_topics:
                result = session.run(
                    """
                    MATCH (s:Konzept {name: $topic})-[r]->(o:Konzept)
                    RETURN type(r) AS relation, o.name AS object
                    """,
                    topic=search_topic,
                )

                for record in result:
                    relation: str = record["relation"]
                    obj: str = record["object"]

                    # Prevent duplicates
                    if obj not in combined_facts[relation]:
                        combined_facts[relation].append(obj)

                        # Tracking for transparency
                        fact_key: str = f"{relation}:{obj}"
                        if fact_key not in fact_sources:
                            fact_sources[fact_key] = []
                        fact_sources[fact_key].append(search_topic)

            # STEP 3: Get meanings/definitions (word level)
            # Search HAT_BEDEUTUNG relations for all topics including synonyms
            bedeutungen_list: List[str] = []
            for search_topic in all_topics:
                bedeutung_result = session.run(
                    """
                    MATCH (w:Wort {lemma: $topic})-[:HAT_BEDEUTUNG]->(b:Bedeutung)
                    RETURN DISTINCT b.text AS bedeutung
                    """,
                    topic=search_topic,
                )

                for record in bedeutung_result:
                    bedeutung_text: str = record["bedeutung"]
                    if bedeutung_text and bedeutung_text not in bedeutungen_list:
                        bedeutungen_list.append(bedeutung_text)

            # STEP 4: Add synonyms as TEIL_VON facts (if any)
            if synonyms:
                combined_facts["TEIL_VON"] = list(
                    set(synonyms + combined_facts.get("TEIL_VON", []))
                )

            return {
                "primary_topic": topic_lower,
                "synonyms": synonyms,
                "facts": dict(combined_facts),
                "sources": fact_sources,
                "bedeutungen": bedeutungen_list,
            }

    def find_similar_words(
        self,
        query_word: str,
        embedding_service=None,
        similarity_threshold: float = 0.75,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find similar words for typo tolerance using semantic embeddings.

        Algorithm:
        1. Generate embedding for query_word
        2. Generate embeddings for all known words
        3. Calculate cosine similarity
        4. Return top-N candidates

        Args:
            query_word: The word to search (possibly with typo)
            embedding_service: EmbeddingService instance
            similarity_threshold: Minimum similarity (0.0 to 1.0)
            max_results: Maximum number of results

        Returns:
            List of dicts: [{"word": str, "similarity": float}, ...]
            Sorted by similarity (highest first)
        """
        if (
            not self.driver
            or not embedding_service
            or not embedding_service.is_available()
        ):
            logger.warning(
                "find_similar_words: Driver oder Embedding-Service nicht verfügbar"
            )
            return []

        try:
            # Get all known words
            known_words: List[str] = self.word_manager.get_all_known_words()

            if not known_words:
                logger.debug("find_similar_words: Keine bekannten Wörter gefunden")
                return []

            # Generate embedding for query_word (not used, only for validation)
            try:
                _ = embedding_service.get_embedding(query_word.lower())
            except Exception as e:
                logger.error(f"find_similar_words: Fehler beim Query-Embedding: {e}")
                return []

            # Calculate similarities
            similarities: List[Dict[str, Any]] = []

            for known_word in known_words:
                # Skip exact match
                if known_word.lower() == query_word.lower():
                    continue

                try:
                    similarity: float = embedding_service.get_similarity(
                        query_word.lower(), known_word
                    )

                    if similarity >= similarity_threshold:
                        similarities.append(
                            {"word": known_word, "similarity": similarity}
                        )
                except Exception as e:
                    logger.debug(f"find_similar_words: Fehler bei '{known_word}': {e}")
                    continue

            # Sort by similarity (highest first) and limit
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            result: List[Dict[str, Any]] = similarities[:max_results]

            logger.debug(
                f"find_similar_words: '{query_word}' -> {len(result)} Kandidaten "
                f"(von {len(known_words)} bekannten Wörtern)"
            )

            return result

        except Exception as e:
            logger.error(f"find_similar_words: Unerwarteter Fehler: {e}", exc_info=True)
            return []

    def create_production_rule(
        self,
        name: str,
        category: str,
        utility: float = 1.0,
        specificity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Persist a production rule in Neo4j.

        Args:
            name: Unique rule name
            category: Category (content_selection, lexicalization, discourse, syntax)
            utility: Static utility (preference)
            specificity: Rule specificity
            metadata: Additional metadata (tags, description, etc.)

        Returns:
            True if successful, False otherwise

        Note:
            The Condition/Action callables cannot be persisted.
            This method stores only metadata and statistics.
            Metadata is stored as JSON string.
        """
        if not self.driver:
            logger.error("create_production_rule: Kein DB-Driver verfügbar")
            return False

        try:
            # Serialize metadata as JSON string for Neo4j
            metadata_json = json.dumps(metadata or {})

            with self.driver.session(database="neo4j") as session:
                query = """
                MERGE (pr:ProductionRule {name: $name})
                SET pr.category = $category,
                    pr.utility = $utility,
                    pr.specificity = $specificity,
                    pr.metadata = $metadata,
                    pr.application_count = COALESCE(pr.application_count, 0),
                    pr.success_count = COALESCE(pr.success_count, 0),
                    pr.created_at = COALESCE(pr.created_at, datetime()),
                    pr.last_updated = datetime()
                RETURN pr.name AS name
                """

                result = session.run(
                    query,
                    name=name,
                    category=category,
                    utility=utility,
                    specificity=specificity,
                    metadata=metadata_json,
                )

                record = result.single()

                if record:
                    logger.info(
                        "ProductionRule erstellt/aktualisiert",
                        extra={
                            "name": name,
                            "category": category,
                            "utility": utility,
                            "specificity": specificity,
                        },
                    )
                    return True
                else:
                    logger.warning(
                        "create_production_rule: Regel konnte nicht erstellt werden",
                        extra={"name": name},
                    )
                    return False

        except Exception as e:
            logger.log_exception(
                e,
                "create_production_rule: Fehler",
                name=name,
                category=category,
            )
            return False

    def get_production_rules(
        self,
        category: Optional[str] = None,
        min_utility: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load production rules from Neo4j.

        Args:
            category: Optional - filter by category
            min_utility: Optional - minimum utility
            limit: Optional - maximum number of rules

        Returns:
            List of rule dicts with metadata and statistics
        """
        if not self.driver:
            logger.error("get_production_rules: Kein DB-Driver verfügbar")
            return []

        try:
            with self.driver.session(database="neo4j") as session:
                # Build query dynamically
                query_parts = ["MATCH (pr:ProductionRule)"]
                where_clauses = []
                params = {}

                if category:
                    where_clauses.append("pr.category = $category")
                    params["category"] = category

                if min_utility is not None:
                    where_clauses.append("pr.utility >= $min_utility")
                    params["min_utility"] = min_utility

                if where_clauses:
                    query_parts.append("WHERE " + " AND ".join(where_clauses))

                query_parts.append(
                    """
                    RETURN pr.name AS name,
                           pr.category AS category,
                           pr.utility AS utility,
                           pr.specificity AS specificity,
                           pr.metadata AS metadata,
                           pr.application_count AS application_count,
                           pr.success_count AS success_count,
                           pr.last_applied AS last_applied,
                           pr.created_at AS created_at,
                           pr.last_updated AS last_updated
                    ORDER BY pr.utility DESC, pr.specificity DESC
                    """
                )

                if limit:
                    query_parts.append(f"LIMIT {limit}")

                query = "\n".join(query_parts)

                result = session.run(query, **params)

                rules = []
                for record in result:
                    # Deserialize metadata from JSON string
                    metadata = {}
                    if record["metadata"]:
                        try:
                            metadata = json.loads(record["metadata"])
                        except json.JSONDecodeError:
                            metadata = {}

                    rule_dict = {
                        "name": record["name"],
                        "category": record["category"],
                        "utility": record["utility"],
                        "specificity": record["specificity"],
                        "metadata": metadata,
                        "application_count": record["application_count"] or 0,
                        "success_count": record["success_count"] or 0,
                        "last_applied": (
                            record["last_applied"].isoformat()
                            if record["last_applied"]
                            else None
                        ),
                        "created_at": (
                            record["created_at"].isoformat()
                            if record["created_at"]
                            else None
                        ),
                        "last_updated": (
                            record["last_updated"].isoformat()
                            if record["last_updated"]
                            else None
                        ),
                    }
                    rules.append(rule_dict)

                logger.debug(
                    f"Loaded {len(rules)} production rules",
                    extra={"category": category, "min_utility": min_utility},
                )

                return rules

        except Exception as e:
            logger.log_exception(
                e,
                "get_production_rules: Fehler",
                category=category,
                min_utility=min_utility,
            )
            return []

    def update_rule_stats(
        self,
        rule_name: str,
        applied: bool = True,
        success: Optional[bool] = None,
    ) -> bool:
        """
        Update the statistics of a production rule.

        Args:
            rule_name: Rule name
            applied: True if rule was applied
            success: Optional - True if rule was successful

        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("update_rule_stats: Kein DB-Driver verfügbar")
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                # Build update statements
                set_clauses = []
                if applied:
                    set_clauses.append(
                        "pr.application_count = pr.application_count + 1"
                    )
                    set_clauses.append("pr.last_applied = datetime()")

                if success is not None and success:
                    set_clauses.append("pr.success_count = pr.success_count + 1")

                if not set_clauses:
                    # Nothing to do
                    return True

                query = f"""
                MATCH (pr:ProductionRule {{name: $rule_name}})
                SET {', '.join(set_clauses)}
                RETURN pr.name AS name,
                       pr.application_count AS application_count,
                       pr.success_count AS success_count
                """

                result = session.run(query, rule_name=rule_name)

                record = result.single()

                if record:
                    logger.debug(
                        "Rule stats updated",
                        extra={
                            "name": record["name"],
                            "applications": record["application_count"],
                            "successes": record["success_count"],
                        },
                    )
                    return True
                else:
                    logger.warning(
                        "update_rule_stats: Regel nicht gefunden",
                        extra={"rule_name": rule_name},
                    )
                    return False

        except Exception as e:
            logger.log_exception(
                e,
                "update_rule_stats: Fehler",
                rule_name=rule_name,
            )
            return False

    def get_production_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all production rules.

        Returns:
            Dict with statistics (rule count, categories, top rules, etc.)
        """
        if not self.driver:
            logger.error("get_production_rule_statistics: Kein DB-Driver verfügbar")
            return {}

        try:
            with self.driver.session(database="neo4j") as session:
                # Overall statistics
                stats_query = """
                MATCH (pr:ProductionRule)
                RETURN count(pr) AS total_rules,
                       sum(pr.application_count) AS total_applications,
                       sum(pr.success_count) AS total_successes,
                       avg(pr.utility) AS avg_utility,
                       avg(pr.specificity) AS avg_specificity
                """

                result = session.run(stats_query)
                record = result.single()

                if not record or record["total_rules"] == 0:
                    return {
                        "total_rules": 0,
                        "total_applications": 0,
                        "total_successes": 0,
                        "categories": {},
                        "top_rules": [],
                    }

                # Category distribution
                category_query = """
                MATCH (pr:ProductionRule)
                RETURN pr.category AS category, count(pr) AS count
                ORDER BY count DESC
                """

                category_result = session.run(category_query)
                categories = {r["category"]: r["count"] for r in category_result}

                # Top 10 rules (by applications)
                top_rules_query = """
                MATCH (pr:ProductionRule)
                WHERE pr.application_count > 0
                RETURN pr.name AS name,
                       pr.category AS category,
                       pr.application_count AS applications,
                       pr.success_count AS successes
                ORDER BY pr.application_count DESC
                LIMIT 10
                """

                top_result = session.run(top_rules_query)
                top_rules = [
                    {
                        "name": r["name"],
                        "category": r["category"],
                        "applications": r["applications"],
                        "successes": r["successes"],
                    }
                    for r in top_result
                ]

                stats = {
                    "total_rules": record["total_rules"],
                    "total_applications": record["total_applications"] or 0,
                    "total_successes": record["total_successes"] or 0,
                    "avg_utility": record["avg_utility"] or 0.0,
                    "avg_specificity": record["avg_specificity"] or 0.0,
                    "categories": categories,
                    "top_rules": top_rules,
                }

                logger.debug("Production rule statistics retrieved", extra=stats)

                return stats

        except Exception as e:
            logger.log_exception(e, "get_production_rule_statistics: Fehler")
            return {}

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Invalidate caches.

        Args:
            cache_type: Optional - 'facts' or None for all caches
        """
        if cache_type == "facts" or cache_type is None:
            # Clear entire facts cache
            cache_names = self.cache_mgr.list_caches()
            if "netzwerk_facts" in cache_names:
                count = self.cache_mgr.invalidate("netzwerk_facts")
                logger.debug(f"Fakten-Cache geleert ({count} Einträge)")

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about the caches.

        Returns:
            Dict with cache statistics for different cache types
        """
        stats = {}

        # Get stats for facts cache
        cache_names = self.cache_mgr.list_caches()
        if "netzwerk_facts" in cache_names:
            stats["fact_cache"] = self.cache_mgr.get_stats("netzwerk_facts")

        return stats
