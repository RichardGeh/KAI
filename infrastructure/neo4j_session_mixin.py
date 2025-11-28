"""
infrastructure/neo4j_session_mixin.py

Shared Neo4j session management for KAI components.

This module provides a reusable mixin class for components that need to interact
with the Neo4j knowledge graph. It eliminates the duplication of session management
and error handling code that currently exists in 8+ components.

Key Features:
    - Thread-safe Neo4j session management
    - Consistent error handling with kai_exceptions
    - Optional query result caching
    - Performance logging for database operations
    - Parameterized query support (SQL injection prevention)

Usage:
    from infrastructure.neo4j_session_mixin import Neo4jSessionMixin
    from neo4j import Driver

    class MyComponent(Neo4jSessionMixin):
        def __init__(self, driver: Driver):
            super().__init__(driver)

        def my_query(self, word: str):
            query = "MATCH (w:Wort {lemma: $lemma}) RETURN w"
            result = self._safe_run(
                query,
                operation="fetch_word",
                lemma=word
            )
            return result

Replaces Duplicate Code:
    This mixin consolidates the _safe_session_run() pattern found in:
    - component_1_netzwerk_core.py
    - component_1_netzwerk_memory.py
    - component_1_netzwerk_patterns.py
    - component_42_spatial_reasoner.py
    - component_44_resonance_engine.py
    - component_46_meta_learning.py
    - component_1_netzwerk_production_rules.py
    - component_1_netzwerk_feedback.py
"""

import threading
from typing import Any, Dict, List, Optional

from cachetools import TTLCache
from neo4j import Driver, Result, Session

from component_15_logging_config import PerformanceLogger, get_logger
from kai_exceptions import (
    DatabaseException,
    Neo4jConnectionError,
    Neo4jQueryError,
    Neo4jWriteError,
    wrap_exception,
)

logger = get_logger(__name__)


class Neo4jSessionMixin:
    """
    Mixin class providing shared Neo4j session management and error handling.

    Components that need Neo4j access should inherit from this mixin and call
    super().__init__(driver) in their __init__ method.

    Thread Safety:
        This mixin is thread-safe. The _safe_run method uses a lock to ensure
        that concurrent access from multiple threads is properly synchronized.

    Caching:
        Optional query result caching can be enabled by setting enable_cache=True.
        Cache is stored per-instance with configurable TTL and maxsize.

    Attributes:
        driver: Neo4j driver instance
        _session_lock: Thread lock for session access
        _query_cache: Optional TTL cache for query results
        _enable_cache: Whether caching is enabled

    Example:
        class SpatialNeo4jRepository(Neo4jSessionMixin):
            def __init__(self, driver: Driver):
                super().__init__(driver, enable_cache=True, cache_ttl=300)

            def store_relation(self, entity1: str, relation: str, entity2: str):
                query = '''
                MERGE (e1:Entity {name: $entity1})
                MERGE (e2:Entity {name: $entity2})
                MERGE (e1)-[r:SPATIAL_RELATION {type: $relation}]->(e2)
                RETURN r
                '''
                return self._safe_run(
                    query,
                    operation="store_spatial_relation",
                    entity1=entity1,
                    entity2=entity2,
                    relation=relation
                )
    """

    def __init__(
        self,
        driver: Driver,
        enable_cache: bool = False,
        cache_ttl: int = 300,
        cache_maxsize: int = 100,
    ):
        """
        Initialize Neo4j session mixin.

        Args:
            driver: Neo4j driver instance to use for database access
            enable_cache: Whether to enable query result caching (default: False)
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 minutes)
            cache_maxsize: Maximum number of cached results (default: 100)

        Raises:
            Neo4jConnectionError: If driver is None or connection cannot be verified
        """
        if driver is None:
            raise Neo4jConnectionError(
                "Neo4j driver cannot be None",
                context={"component": self.__class__.__name__},
            )

        self.driver: Driver = driver
        self._session_lock = threading.RLock()  # Reentrant lock for nested calls
        self._enable_cache = enable_cache

        # Optional query cache (disabled by default to avoid memory issues)
        self._query_cache: Optional[TTLCache] = None
        if enable_cache:
            self._query_cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
            logger.debug(
                "Query cache enabled",
                extra={
                    "component": self.__class__.__name__,
                    "ttl": cache_ttl,
                    "maxsize": cache_maxsize,
                },
            )

    def _safe_run(
        self,
        query: str,
        operation: str,
        use_cache: bool = False,
        **params: Any,
    ) -> List[Dict[str, Any]]:
        """
        Safely execute a Neo4j query with error handling and optional caching.

        This method provides:
        - Thread-safe session management
        - Comprehensive error handling with kai_exceptions
        - Performance logging for slow queries
        - Optional result caching
        - Parameterized query support (prevents injection)

        Args:
            query: Cypher query to execute (use $param syntax for parameters)
            operation: Human-readable operation name (for logging/errors)
            use_cache: Whether to use cache for this query (requires enable_cache=True)
            **params: Query parameters (passed to query as $param)

        Returns:
            List of result dictionaries from the query
            Empty list if query returns no results

        Raises:
            Neo4jConnectionError: If database connection is lost
            Neo4jQueryError: If query execution fails (syntax error, constraint violation)
            Neo4jWriteError: If write operation fails

        Example:
            # Simple query
            results = self._safe_run(
                "MATCH (w:Wort {lemma: $lemma}) RETURN w",
                operation="fetch_word",
                lemma="hund"
            )

            # Cached query
            results = self._safe_run(
                "MATCH (w:Wort) RETURN count(w) as total",
                operation="count_words",
                use_cache=True
            )

            # Write operation
            self._safe_run(
                "CREATE (w:Wort {lemma: $lemma, pos: $pos})",
                operation="create_word",
                lemma="katze",
                pos="NOUN"
            )

        Performance:
            Queries taking >1 second are logged as warnings with performance stats.
            Consider adding indexes or optimizing such queries.
        """
        # Check cache if enabled
        cache_key = None
        if use_cache and self._enable_cache and self._query_cache is not None:
            # Create cache key from query + params (sorted for consistency)
            cache_key = (query, tuple(sorted(params.items())))
            if cache_key in self._query_cache:
                logger.debug(
                    "Cache hit for query",
                    extra={
                        "operation": operation,
                        "cache_size": len(self._query_cache),
                    },
                )
                return self._query_cache[cache_key]

        # Execute query with error handling and performance tracking
        with PerformanceLogger(
            logger, operation, extra={"query_preview": query[:100]}
        ):
            try:
                with self._session_lock:
                    with self.driver.session() as session:
                        result: Result = session.run(query, params)
                        data = [record.data() for record in result]

                        # Log summary
                        logger.debug(
                            f"Query executed: {operation}",
                            extra={
                                "operation": operation,
                                "result_count": len(data),
                                "params_count": len(params),
                            },
                        )

                        # Cache result if enabled
                        if (
                            cache_key is not None
                            and self._query_cache is not None
                        ):
                            self._query_cache[cache_key] = data

                        return data

            except Exception as e:
                # Classify error type
                error_msg = str(e).lower()

                # Connection errors
                if any(
                    keyword in error_msg
                    for keyword in ["connection", "network", "timeout", "refused"]
                ):
                    raise wrap_exception(
                        e,
                        Neo4jConnectionError,
                        f"Database connection failed during '{operation}'",
                        operation=operation,
                        query=query[:200],  # Limit query length in error
                    )

                # Write errors (constraint violations, etc.)
                if any(
                    keyword in error_msg
                    for keyword in ["constraint", "duplicate", "unique"]
                ):
                    raise wrap_exception(
                        e,
                        Neo4jWriteError,
                        f"Write operation failed during '{operation}'",
                        operation=operation,
                        query=query[:200],
                    )

                # Generic query errors (syntax, semantic, etc.)
                raise wrap_exception(
                    e,
                    Neo4jQueryError,
                    f"Query execution failed during '{operation}'",
                    operation=operation,
                    query=query[:200],
                    params=str(params)[:100],  # Limit params in error
                )

    def clear_query_cache(self) -> None:
        """
        Clear the query result cache.

        Should be called after write operations that may invalidate cached reads.

        Example:
            def create_word(self, lemma: str):
                self._safe_run(
                    "CREATE (w:Wort {lemma: $lemma})",
                    operation="create_word",
                    lemma=lemma
                )
                self.clear_query_cache()  # Invalidate cached word counts
        """
        if self._query_cache is not None:
            self._query_cache.clear()
            logger.debug(
                "Query cache cleared",
                extra={"component": self.__class__.__name__},
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the query cache.

        Returns:
            Dictionary with cache statistics:
            - enabled: Whether caching is enabled
            - size: Current number of cached queries
            - maxsize: Maximum cache size
            - ttl: Cache TTL in seconds

        Example:
            stats = self.get_cache_stats()
            print(f"Cache hit rate: {stats['size']}/{stats['maxsize']}")
        """
        if not self._enable_cache or self._query_cache is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": len(self._query_cache),
            "maxsize": self._query_cache.maxsize,
            "ttl": self._query_cache.ttl,
        }
