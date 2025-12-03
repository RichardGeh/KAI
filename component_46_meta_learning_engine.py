"""
component_46_meta_learning_engine.py

Meta-Learning Engine: Strategy Selection & Neo4j Persistence

Extracted from component_46_meta_learning.py as part of Phase 4 architectural refactoring.

Functions:
- Strategy selection with epsilon-greedy exploration
- Context-based strategy matching
- Neo4j persistence (atomic transactions)
- Configuration management

Author: KAI Development Team
Last Updated: 2025-11-28 (Modular Refactoring)
"""

import random
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from neo4j.exceptions import Neo4jError

from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from component_15_logging_config import get_logger
from component_46_ab_testing_manager import ABTestingManager
from component_46_performance_tracker import (
    PerformanceTracker,
    StrategyPerformance,
    StrategyUsageEpisode,
)
from infrastructure.cache_manager import cache_manager
from kai_exceptions import KAIException

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class MetaLearningConfig:
    """Konfiguration für Meta-Learning Engine"""

    # Exploration/Exploitation
    epsilon: float = 0.1  # 10% exploration
    epsilon_decay: float = 0.995  # Decay über Zeit
    min_epsilon: float = 0.05

    # Learning rates (extracted from magic numbers)
    confidence_learning_rate: float = (
        0.1  # For exponential moving average in update_from_usage
    )
    confidence_alpha: float = 0.1
    success_rate_alpha: float = 0.1

    # Pattern matching
    pattern_similarity_threshold: float = 0.85
    max_patterns_per_strategy: int = 50

    # Performance thresholds
    min_queries_for_confidence: int = 5  # Mindest-Queries bevor Strategy bevorzugt wird

    # Scoring weights (extracted from magic numbers in _match_query_patterns)
    pattern_similarity_weight: float = 0.6  # Weight for similarity score
    pattern_success_weight: float = 0.4  # Weight for success rate
    performance_success_weight: float = 0.6  # Weight for success rate in perf score
    performance_confidence_weight: float = 0.4  # Weight for confidence in perf score
    performance_speed_bonus_max: float = 0.1  # Max bonus for fast response times
    performance_speed_threshold: float = 5.0  # Response time threshold (seconds)

    # Overall scoring weights
    pattern_weight: float = 0.4
    performance_weight: float = 0.4
    context_weight: float = 0.2

    # Text processing
    query_truncate_length: int = 100  # Max length for query text in patterns
    max_failure_modes: int = 20  # Max number of failure modes to track per strategy

    # Neo4j persistence
    persist_every_n_queries: int = 10

    # Cache
    cache_ttl_seconds: int = 600  # 10 Minuten (für Strategy Stats)
    query_pattern_cache_ttl: int = 300  # 5 Minuten (für Query Patterns)

    # Memory management
    max_usage_history_size: int = 10000  # Limit für usage_history (prevent memory leak)
    max_query_patterns_size: int = 1000  # Limit für query_patterns gesamt


# ============================================================================
# Exception Handling
# ============================================================================


class MetaLearningException(KAIException):
    """Base exception für Meta-Learning Fehler"""


class StrategySelectionException(MetaLearningException):
    """Exception bei Strategy-Auswahl"""


class PersistenceException(MetaLearningException):
    """Exception bei Neo4j Persistence"""


# ============================================================================
# Meta-Learning Engine
# ============================================================================


class MetaLearningEngine:
    """
    Meta-Reasoning Engine für Strategy-Auswahl und Performance-Tracking

    Funktionen:
    1. Trackt Performance jeder Reasoning-Strategy
    2. Lernt Query-Patterns für jede Strategy
    3. Wählt optimale Strategy für neue Queries
    4. Epsilon-Greedy Exploration
    5. Persistiert Statistiken in Neo4j
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        embedding_service: EmbeddingService,
        config: Optional[MetaLearningConfig] = None,
    ):
        self.netzwerk = netzwerk
        self.embedding_service = embedding_service
        self.config = config or MetaLearningConfig()

        # Thread Safety: Locks for shared state
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._cache_lock = threading.Lock()  # Separate lock for cache operations

        # Performance Tracker (delegates pattern/performance tracking)
        self.performance_tracker = PerformanceTracker(
            embedding_service=embedding_service,
            pattern_similarity_threshold=self.config.pattern_similarity_threshold,
            max_patterns_per_strategy=self.config.max_patterns_per_strategy,
            max_failure_modes=self.config.max_failure_modes,
            query_truncate_length=self.config.query_truncate_length,
        )

        # A/B Testing Manager (delegates dual-system tracking)
        self.ab_testing_manager = ABTestingManager(
            performance_tracker=self.performance_tracker,
            record_strategy_usage_callback=self.record_strategy_usage,
        )

        # Usage History (protected by _lock, bounded with deque)
        self.usage_history: deque = deque(maxlen=self.config.max_usage_history_size)

        # Counters (protected by _lock)
        self.total_queries: int = 0
        self.queries_since_last_persist: int = 0

        # Performance Optimization: Caching via CacheManager
        # Strategy Stats Cache (TTL: 10 Minuten) für schnellen Zugriff
        cache_manager.register_cache(
            "meta_learning_stats", maxsize=50, ttl=self.config.cache_ttl_seconds
        )
        # Query Pattern Cache (TTL: 5 Minuten)
        cache_manager.register_cache(
            "meta_learning_patterns",
            maxsize=100,
            ttl=self.config.query_pattern_cache_ttl,
        )

        # Neo4j Indexes
        self._ensure_neo4j_indexes()

        # Load from Neo4j
        self._load_persisted_stats()

        logger.info(
            "MetaLearningEngine initialized with %d strategies",
            len(self.performance_tracker.strategy_stats),
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _sanitize_for_logging(self, text: str, max_len: int = 100) -> str:
        """
        Sanitize text for Windows cp1252 logging (remove forbidden Unicode chars).

        Per CLAUDE.md policy: Windows console supports only cp1252, Unicode chars
        like -> x / != <= >= cause UnicodeEncodeError.

        Args:
            text: Text to sanitize
            max_len: Maximum length (truncate longer text)

        Returns:
            Sanitized ASCII-safe text
        """
        # Replace forbidden Unicode characters with ASCII equivalents
        replacements = {
            "\u2192": "->",  # →
            "\u00d7": "x",  # ×
            "\u00f7": "/",  # ÷
            "\u2260": "!=",  # ≠
            "\u2264": "<=",  # ≤
            "\u2265": ">=",  # ≥
            "\u2713": "[OK]",  # ✓
            "\u2717": "[FAIL]",  # ✗
            "\u2227": "AND",  # ∧
            "\u2228": "OR",  # ∨
            "\u00ac": "NOT",  # ¬
        }

        sanitized = text
        for unicode_char, ascii_replacement in replacements.items():
            sanitized = sanitized.replace(unicode_char, ascii_replacement)

        # Truncate and ensure ASCII-safe (replace unknown chars with ?)
        return sanitized[:max_len].encode("ascii", errors="replace").decode("ascii")

    def _ensure_neo4j_indexes(self) -> None:
        """
        Create Neo4j indexes for performance.

        Creates index on StrategyPerformance.strategy_name for fast MERGE/MATCH operations.
        """
        try:
            with self.netzwerk.driver.session() as session:
                session.run(
                    """
                    CREATE INDEX strategy_perf_name IF NOT EXISTS
                    FOR (sp:StrategyPerformance)
                    ON (sp.strategy_name)
                """
                )
                logger.debug("Neo4j indexes verified for MetaLearning")
        except Neo4jError as e:
            logger.warning("Failed to create Neo4j indexes: %s", e)
        except Exception as e:
            logger.error("Unexpected error creating Neo4j indexes: %s", e)

    def _execute_query(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute generic Neo4j query"""
        try:
            with self.netzwerk.driver.session() as session:
                result = session.run(cypher, params or {})
                return [dict(record) for record in result]
        except Neo4jError as e:
            logger.error("Neo4j query failed: %s", e, exc_info=True)
            raise PersistenceException(f"Neo4j query failed: {e}") from e
        except Exception as e:
            logger.error("Unexpected error in Neo4j query: %s", e, exc_info=True)
            raise

    # ========================================================================
    # Core Functions: Strategy Usage Recording
    # ========================================================================

    def record_strategy_usage(
        self,
        strategy: str,
        query: str,
        result: Dict[str, Any],
        response_time: float,
        context: Optional[Dict[str, Any]] = None,
        user_feedback: Optional[str] = None,
    ) -> None:
        """
        Track jede Strategy-Verwendung

        Args:
            strategy: Name der verwendeten Strategy
            query: User-Query
            result: Ergebnis der Strategy (muss 'confidence' enthalten)
            response_time: Zeit in Sekunden
            context: Optional context dict
            user_feedback: 'correct', 'incorrect', 'neutral'

        Raises:
            ValueError: If strategy name is invalid
            KeyError: If result dict is missing required keys
        """
        # Validate strategy name (security: prevent Cypher injection)
        self.performance_tracker.validate_strategy_name(strategy)

        with self._lock:
            try:
                # Get or create Strategy stats (atomic check-then-act)
                stats = self.performance_tracker.get_or_create_strategy_stats(strategy)

                # Extrahiere Confidence
                if "confidence" not in result:
                    raise KeyError("Result dict must contain 'confidence' key")
                confidence = result["confidence"]

                # Determine Success basierend auf Feedback
                success = None
                if user_feedback == "correct":
                    success = True
                elif user_feedback == "incorrect":
                    success = False
                    # Extract failure mode
                    failure_mode = self.performance_tracker.extract_failure_pattern(
                        query, result
                    )
                    if failure_mode and failure_mode not in stats.failure_modes:
                        stats.failure_modes.append(failure_mode)
                        if len(stats.failure_modes) > self.config.max_failure_modes:
                            stats.failure_modes.pop(0)

                # Update Stats (pass config learning rate)
                stats.update_from_usage(
                    confidence,
                    response_time,
                    success,
                    learning_rate=self.config.confidence_learning_rate,
                )

                # Query Embedding für Pattern-Learning
                query_embedding = self.embedding_service.get_embedding(query)

                # Record Episode (deque automatically evicts oldest if maxlen exceeded)
                episode = StrategyUsageEpisode(
                    timestamp=datetime.now(),
                    strategy_name=strategy,
                    query=query,
                    query_embedding=query_embedding,
                    context=context or {},
                    result_confidence=confidence,
                    response_time=response_time,
                    user_feedback=user_feedback,
                    failure_reason=result.get("error") if not success else None,
                )
                self.usage_history.append(episode)

                # Update Query Patterns (delegate to PerformanceTracker)
                self.performance_tracker.update_query_patterns(
                    strategy, query, query_embedding, success
                )

                # Increment counters (atomic)
                self.total_queries += 1
                self.queries_since_last_persist += 1

                # Decay epsilon on successful feedback
                if user_feedback == "correct":
                    self.config.epsilon = max(
                        self.config.min_epsilon,
                        self.config.epsilon * self.config.epsilon_decay,
                    )

                # Persist to Neo4j periodisch
                if (
                    self.queries_since_last_persist
                    >= self.config.persist_every_n_queries
                ):
                    self._persist_all_stats()
                    self.queries_since_last_persist = 0

                # Sanitize for logging (Windows cp1252 safe)
                safe_query = self._sanitize_for_logging(query)
                logger.debug(
                    "Recorded usage for strategy '%s': query='%s', confidence=%.2f, "
                    "response_time=%.3fs, feedback=%s",
                    strategy,
                    safe_query,
                    confidence,
                    response_time,
                    user_feedback,
                )

            except (ValueError, KeyError) as e:
                logger.error(
                    "Invalid input in record_strategy_usage: %s", e, exc_info=True
                )
                raise
            except Exception as e:
                logger.critical(
                    "Unexpected error recording strategy usage: %s", e, exc_info=True
                )
                raise MetaLearningException(f"Failed to record usage: {e}") from e

    def record_strategy_usage_with_feedback(
        self,
        strategy_name: str,
        query: str,
        success: bool,
        confidence: float,
        response_time: float = 0.0,
        user_feedback: str = "neutral",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Vereinfachte Methode für FeedbackHandler

        Speziell für User-Feedback-Loop optimiert.

        Args:
            strategy_name: Name der Strategy
            query: User-Query
            success: True wenn korrekt, False wenn inkorrekt
            confidence: Confidence-Wert
            response_time: Optional Response-Zeit
            user_feedback: 'correct', 'incorrect', 'unsure', 'partially_correct'
            context: Optional Context
        """
        # Erstelle vereinfachtes result dict
        result = {"confidence": confidence, "success": success}

        # Map user_feedback zu internem Format
        feedback_map = {
            "correct": "correct",
            "incorrect": "incorrect",
            "unsure": "neutral",
            "partially_correct": "neutral",
        }
        internal_feedback = feedback_map.get(user_feedback, "neutral")

        # Rufe Hauptmethode auf
        self.record_strategy_usage(
            strategy=strategy_name,
            query=query,
            result=result,
            response_time=response_time,
            context=context,
            user_feedback=internal_feedback,
        )

        logger.info(
            f"Recorded user feedback | strategy={strategy_name}, "
            f"feedback={user_feedback}, success={success}"
        )

    # ========================================================================
    # Core Functions: Strategy Selection (Meta-Reasoning)
    # ========================================================================

    def select_best_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """
        Meta-Reasoning: Welche Strategy ist am besten für diese Query?

        Side-effect free: Does NOT mutate epsilon (decay moved to record_strategy_usage).

        Args:
            query: User-Query
            context: Optional context dict
            available_strategies: Liste verfügbarer Strategien (None = alle)

        Returns:
            (strategy_name, confidence_score)
        """
        with self._lock:
            try:
                context = context or {}

                # Epsilon-Greedy: Exploration vs. Exploitation
                if random.random() < self.config.epsilon:
                    # EXPLORATION: Random strategy
                    strategies = available_strategies or list(
                        self.performance_tracker.strategy_stats.keys()
                    )
                    if not strategies:
                        return ("direct_answer", 0.5)  # Fallback

                    selected = random.choice(strategies)
                    logger.debug("Epsilon-greedy EXPLORATION: selected '%s'", selected)
                    return (selected, 0.3)  # Low confidence für exploration

                # EXPLOITATION: Best strategy basierend auf Scoring
                query_embedding = self.embedding_service.get_embedding(query)

                strategy_scores: Dict[str, float] = {}
                strategies_to_evaluate = available_strategies or list(
                    self.performance_tracker.strategy_stats.keys()
                )

                if not strategies_to_evaluate:
                    return ("direct_answer", 0.5)

                for strategy_name in strategies_to_evaluate:
                    stats = self.performance_tracker.get_strategy_stats(strategy_name)
                    if stats is None:
                        # Neue Strategy: neutral score
                        strategy_scores[strategy_name] = 0.5
                        continue

                    # 1. Pattern-basierte Scoring (delegate to PerformanceTracker)
                    pattern_score = self.performance_tracker.match_query_patterns(
                        query_embedding,
                        strategy_name,
                        pattern_similarity_weight=self.config.pattern_similarity_weight,
                        pattern_success_weight=self.config.pattern_success_weight,
                    )

                    # 2. Performance-basierte Scoring (delegate to PerformanceTracker)
                    perf_score = self.performance_tracker.calculate_performance_score(
                        stats,
                        min_queries_for_confidence=self.config.min_queries_for_confidence,
                        success_weight=self.config.performance_success_weight,
                        confidence_weight=self.config.performance_confidence_weight,
                        speed_bonus_max=self.config.performance_speed_bonus_max,
                        speed_threshold=self.config.performance_speed_threshold,
                    )

                    # 3. Context-basierte Scoring
                    context_score = self._match_context_requirements(
                        context, strategy_name
                    )

                    # Aggregierte Score (weighted sum)
                    aggregated_score = (
                        pattern_score * self.config.pattern_weight
                        + perf_score * self.config.performance_weight
                        + context_score * self.config.context_weight
                    )

                    strategy_scores[strategy_name] = aggregated_score

                    logger.debug(
                        "Strategy '%s' scores: pattern=%.2f, perf=%.2f, "
                        "context=%.2f -> total=%.2f",
                        strategy_name,
                        pattern_score,
                        perf_score,
                        context_score,
                        aggregated_score,
                    )

                # Select best strategy
                best_strategy = max(strategy_scores, key=strategy_scores.get)
                best_score = strategy_scores[best_strategy]

                logger.info(
                    "Selected strategy '%s' with score %.2f (epsilon=%.3f)",
                    best_strategy,
                    best_score,
                    self.config.epsilon,
                )

                return (best_strategy, best_score)

            except (ValueError, KeyError) as e:
                logger.error(
                    "Invalid input in select_best_strategy: %s", e, exc_info=True
                )
                return ("direct_answer", 0.3)
            except Exception as e:
                logger.critical(
                    "Unexpected error selecting strategy: %s", e, exc_info=True
                )
                raise StrategySelectionException(
                    f"Failed to select strategy: {e}"
                ) from e

    # ========================================================================
    # Context Matching
    # ========================================================================

    def _match_context_requirements(
        self, context: Dict[str, Any], strategy_name: str
    ) -> float:
        """
        Context-basierte Scoring: Passt Strategy zu Context?

        Heuristiken:
        - temporal_reasoning -> benötigt temporale Keywords
        - graph_traversal -> benötigt Relationen
        - probabilistic -> benötigt Unsicherheit
        - constraint_reasoning -> benötigt Constraints

        Returns:
            Score 0.0-1.0
        """
        # Default: neutral
        score = 0.5

        # Temporal keywords
        if "temporal_required" in context and context["temporal_required"]:
            if strategy_name in ["temporal_reasoning", "causal_reasoning"]:
                score = 0.9
            else:
                score = 0.3

        # Graph-based
        if "requires_graph" in context and context["requires_graph"]:
            if strategy_name in ["graph_traversal", "abductive_reasoning"]:
                score = 0.9
            else:
                score = 0.4

        # Probabilistic
        if "uncertainty" in context and context["uncertainty"]:
            if strategy_name == "probabilistic_reasoning":
                score = 0.95
            else:
                score = 0.5

        # Constraint satisfaction
        if "constraints" in context and len(context.get("constraints", [])) > 0:
            if strategy_name == "constraint_reasoning":
                score = 0.95
            else:
                score = 0.4

        # Kombinatorische Probleme
        if "combinatorial" in context and context["combinatorial"]:
            if strategy_name == "combinatorial_reasoning":
                score = 0.95
            else:
                score = 0.3

        return score

    # ========================================================================
    # Neo4j Persistence
    # ========================================================================

    def _persist_all_stats(self) -> None:
        """
        Persistiere alle Strategy-Statistiken in Neo4j (atomic transaction).

        Uses single transaction for all strategies to prevent partial updates.
        """
        with self._lock:
            try:
                with self.netzwerk.driver.session() as session:
                    with session.begin_transaction() as tx:
                        for (
                            strategy_name,
                            stats,
                        ) in self.performance_tracker.strategy_stats.items():
                            query = """
                            MERGE (sp:StrategyPerformance {strategy_name: $strategy_name})
                            SET sp.queries_handled = $queries_handled,
                                sp.success_count = $success_count,
                                sp.failure_count = $failure_count,
                                sp.success_rate = $success_rate,
                                sp.avg_confidence = $avg_confidence,
                                sp.avg_response_time = $avg_response_time,
                                sp.failure_modes = $failure_modes,
                                sp.last_used = datetime($last_used),
                                sp.updated_at = datetime()
                            RETURN sp
                            """

                            tx.run(
                                query,
                                {
                                    "strategy_name": strategy_name,
                                    "queries_handled": stats.queries_handled,
                                    "success_count": stats.success_count,
                                    "failure_count": stats.failure_count,
                                    "success_rate": stats.success_rate,
                                    "avg_confidence": stats.avg_confidence,
                                    "avg_response_time": stats.avg_response_time,
                                    "failure_modes": stats.failure_modes,
                                    "last_used": (
                                        stats.last_used.isoformat()
                                        if stats.last_used
                                        else None
                                    ),
                                },
                            )

                        tx.commit()

                logger.info(
                    "Persisted stats for %d strategies to Neo4j (atomic transaction)",
                    len(self.performance_tracker.strategy_stats),
                )

            except Neo4jError as e:
                logger.error(
                    "Neo4j error persisting stats (rolled back): %s", e, exc_info=True
                )
                raise PersistenceException(f"Failed to persist stats: {e}") from e
            except Exception as e:
                logger.critical(
                    "Unexpected error persisting stats: %s", e, exc_info=True
                )
                raise

    def _load_persisted_stats(self) -> None:
        """Lade Strategy-Statistiken aus Neo4j"""
        with self._lock:
            try:
                query = """
                MATCH (sp:StrategyPerformance)
                RETURN sp.strategy_name AS strategy_name,
                       sp.queries_handled AS queries_handled,
                       sp.success_count AS success_count,
                       sp.failure_count AS failure_count,
                       sp.success_rate AS success_rate,
                       sp.avg_confidence AS avg_confidence,
                       sp.avg_response_time AS avg_response_time,
                       sp.failure_modes AS failure_modes,
                       sp.last_used AS last_used
                """

                results = self._execute_query(query)

                for record in results:
                    strategy_name = record["strategy_name"]

                    stats = StrategyPerformance(
                        strategy_name=strategy_name,
                        queries_handled=record["queries_handled"] or 0,
                        success_count=record["success_count"] or 0,
                        failure_count=record["failure_count"] or 0,
                        success_rate=record["success_rate"] or 0.5,
                        avg_confidence=record["avg_confidence"] or 0.0,
                        avg_response_time=record["avg_response_time"] or 0.0,
                        total_response_time=0.0,  # Wird neu berechnet
                        typical_query_patterns=[],  # TODO: Load from separate nodes
                        failure_modes=record["failure_modes"] or [],
                        last_used=None,  # TODO: Parse datetime
                    )

                    # Recalculate total_response_time
                    stats.total_response_time = (
                        stats.avg_response_time * stats.queries_handled
                    )

                    self.performance_tracker.strategy_stats[strategy_name] = stats

                logger.info(
                    "Loaded %d strategy statistics from Neo4j",
                    len(self.performance_tracker.strategy_stats),
                )

            except PersistenceException:
                # Re-raise PersistenceException from _execute_query
                raise
            except (KeyError, TypeError, ValueError) as e:
                logger.error(
                    "Invalid data loading persisted stats: %s", e, exc_info=True
                )
            except Exception as e:
                logger.warning("Could not load persisted stats (starting fresh): %s", e)

    # ========================================================================
    # Utility Functions (Delegating & Cache Management)
    # ========================================================================

    def get_strategy_stats(
        self, strategy_name: str, use_cache: bool = True
    ) -> Optional[StrategyPerformance]:
        """
        Hole Performance-Stats für Strategy

        Args:
            strategy_name: Name der Strategy
            use_cache: Nutze TTL-Cache (default: True)

        Returns:
            StrategyPerformance oder None
        """
        # Cache Lookup (thread-safe)
        if use_cache:
            with self._cache_lock:
                cached_stats = cache_manager.get("meta_learning_stats", strategy_name)
                if cached_stats is not None:
                    logger.debug("Stats cache HIT for strategy '%s'", strategy_name)
                    return cached_stats

        # Get from PerformanceTracker (thread-safe)
        stats = self.performance_tracker.get_strategy_stats(strategy_name)

        # Cache Write (thread-safe)
        if use_cache and stats is not None:
            with self._cache_lock:
                cache_manager.set("meta_learning_stats", strategy_name, stats)
                logger.debug("Cached stats for strategy '%s'", strategy_name)

        return stats

    def get_all_stats(self) -> Dict[str, StrategyPerformance]:
        """Hole alle Strategy-Stats (delegate to PerformanceTracker)"""
        return self.performance_tracker.get_all_stats()

    def get_top_strategies(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Hole Top N Strategies basierend auf Performance

        Returns:
            List of (strategy_name, score) tuples
        """
        with self._lock:
            strategy_scores = []

            for name, stats in self.performance_tracker.strategy_stats.items():
                score = self.performance_tracker.calculate_performance_score(stats)
                strategy_scores.append((name, score))

            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            return strategy_scores[:n]

    def reset_epsilon(self, new_epsilon: float = 0.1) -> None:
        """Reset Epsilon für neue Exploration-Phase"""
        with self._lock:
            self.config.epsilon = new_epsilon
            logger.info("Reset epsilon to %.3f", new_epsilon)

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Leert Caches

        Args:
            cache_type: 'stats', 'patterns', oder None für beide
        """
        if cache_type == "stats" or cache_type is None:
            with self._cache_lock:
                cache_manager.invalidate("meta_learning_stats")
                logger.info("Strategy stats cache cleared")

        if cache_type == "patterns" or cache_type is None:
            with self._cache_lock:
                cache_manager.invalidate("meta_learning_patterns")
                logger.info("Query pattern cache cleared")

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt Cache-Statistiken zurück

        Returns:
            Dict mit Cache-Größen und TTLs
        """
        with self._cache_lock:
            stats_cache_stats = cache_manager.get_stats("meta_learning_stats")
            patterns_cache_stats = cache_manager.get_stats("meta_learning_patterns")

            cache_stats = {
                "stats_cache": {
                    "size": stats_cache_stats["size"],
                    "maxsize": stats_cache_stats["maxsize"],
                    "ttl": stats_cache_stats["ttl"],
                    "hits": stats_cache_stats["hits"],
                    "misses": stats_cache_stats["misses"],
                    "hit_rate": stats_cache_stats["hit_rate"],
                },
                "pattern_cache": {
                    "size": patterns_cache_stats["size"],
                    "maxsize": patterns_cache_stats["maxsize"],
                    "ttl": patterns_cache_stats["ttl"],
                    "hits": patterns_cache_stats["hits"],
                    "misses": patterns_cache_stats["misses"],
                    "hit_rate": patterns_cache_stats["hit_rate"],
                },
            }

        with self._lock:
            cache_stats["in_memory_stats"] = {
                "strategies": len(self.performance_tracker.strategy_stats),
                "total_queries": self.total_queries,
                "usage_history_size": len(self.usage_history),
            }

        return cache_stats

    # ========================================================================
    # Delegated Methods (ABTestingManager)
    # ========================================================================

    def record_generation_system_usage(self, *args, **kwargs) -> None:
        """Delegate to ABTestingManager"""
        return self.ab_testing_manager.record_generation_system_usage(*args, **kwargs)

    def get_generation_system_comparison(self) -> Dict[str, Any]:
        """Delegate to ABTestingManager"""
        return self.ab_testing_manager.get_generation_system_comparison()
