"""
component_46_performance_tracker.py

Performance Tracking for Reasoning Strategies

Extracted from component_46_meta_learning.py as part of Phase 4 architectural refactoring.

Functions:
- StrategyPerformance tracking (queries, success rate, confidence, response time)
- QueryPattern learning and matching
- Pattern similarity scoring
- Failure mode extraction

Author: KAI Development Team
Last Updated: 2025-11-28 (Modular Refactoring)
"""

import math
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from component_11_embedding_service import EmbeddingService
from component_15_logging_config import get_logger

logger = get_logger(__name__)

# ============================================================================
# Input Validation
# ============================================================================

# Whitelist pattern for strategy names (security: prevent Cypher injection)
STRATEGY_NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class StrategyPerformance:
    """Performance-Statistiken für eine Reasoning-Strategy"""

    strategy_name: str
    queries_handled: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.5  # Initial neutral
    avg_confidence: float = 0.0
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
    typical_query_patterns: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def update_from_usage(
        self,
        confidence: float,
        response_time: float,
        success: Optional[bool] = None,
        learning_rate: float = 0.1,
    ) -> None:
        """Update statistiken basierend auf neuem usage"""
        self.queries_handled += 1

        # Update Confidence (exponential moving average)
        self.avg_confidence = (
            1 - learning_rate
        ) * self.avg_confidence + learning_rate * confidence

        # Update Response Time
        self.total_response_time += response_time
        self.avg_response_time = self.total_response_time / self.queries_handled

        # Update Success Rate (wenn Feedback vorhanden)
        if success is not None:
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1

            # Recalculate success rate mit Laplace smoothing
            self.success_rate = (self.success_count + 1) / (self.queries_handled + 2)

        self.last_used = datetime.now()


@dataclass
class QueryPattern:
    """Erkanntes Query-Pattern für Strategy-Matching"""

    pattern_text: str
    embedding: Optional[List[float]] = None
    associated_strategy: str = ""
    success_count: int = 0
    total_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count


@dataclass
class StrategyUsageEpisode:
    """Einzelne Strategy-Verwendung für Tracking"""

    timestamp: datetime
    strategy_name: str
    query: str
    query_embedding: List[float]
    context: Dict[str, Any]
    result_confidence: float
    response_time: float
    user_feedback: Optional[str] = None  # 'correct', 'incorrect', 'neutral'
    failure_reason: Optional[str] = None


# ============================================================================
# Performance Tracker
# ============================================================================


class PerformanceTracker:
    """
    Tracks performance statistics for reasoning strategies.

    Functions:
    - Maintains StrategyPerformance statistics (success rate, confidence, response time)
    - Learns query patterns for each strategy
    - Scores pattern matches using embeddings
    - Extracts failure patterns for debugging
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        pattern_similarity_threshold: float = 0.85,
        max_patterns_per_strategy: int = 50,
        max_failure_modes: int = 20,
        query_truncate_length: int = 100,
    ):
        """
        Initialize PerformanceTracker.

        Args:
            embedding_service: EmbeddingService for pattern matching
            pattern_similarity_threshold: Threshold for pattern matching (default: 0.85)
            max_patterns_per_strategy: Max patterns to track per strategy (default: 50)
            max_failure_modes: Max failure modes to track (default: 20)
            query_truncate_length: Max query text length in patterns (default: 100)
        """
        self.embedding_service = embedding_service

        # Configuration
        self.pattern_similarity_threshold = pattern_similarity_threshold
        self.max_patterns_per_strategy = max_patterns_per_strategy
        self.max_failure_modes = max_failure_modes
        self.query_truncate_length = query_truncate_length

        # Thread Safety
        self._lock = threading.RLock()

        # In-Memory State (protected by _lock)
        self.strategy_stats: Dict[str, StrategyPerformance] = {}
        self.query_patterns: Dict[str, List[QueryPattern]] = defaultdict(list)

        logger.info("PerformanceTracker initialized")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def validate_strategy_name(self, name: str) -> None:
        """
        Validate strategy name against whitelist pattern (security: prevent Cypher injection).

        Args:
            name: Strategy name to validate

        Raises:
            ValueError: If strategy name contains invalid characters
        """
        if not STRATEGY_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid strategy name '{name}': must match pattern "
                f"[a-z_][a-z0-9_]{{0,63}} (lowercase, underscore, max 64 chars)"
            )

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Berechne Cosine Similarity zwischen zwei Vektoren.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity [0.0, 1.0] or 0.0 if vectors are invalid
        """
        try:
            # Validate inputs
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                logger.warning(
                    "Invalid vectors for cosine similarity: len(vec1)=%d, len(vec2)=%d",
                    len(vec1) if vec1 else 0,
                    len(vec2) if vec2 else 0,
                )
                return 0.0

            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))

            # Handle zero vectors
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)

        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning("Error computing cosine similarity: %s", e)
            return 0.0

    # ========================================================================
    # Pattern Learning & Matching
    # ========================================================================

    def update_query_patterns(
        self,
        strategy: str,
        query: str,
        query_embedding: List[float],
        success: Optional[bool],
    ) -> None:
        """
        Update Query-Patterns für Strategy.

        Args:
            strategy: Strategy name
            query: User query text
            query_embedding: Query embedding vector
            success: True if successful, False if failed, None if unknown

        Note: Must be called within _lock context.
        """
        try:
            # Finde ähnliche existierende Patterns
            existing_pattern = None
            max_similarity = 0.0

            for pattern in self.query_patterns[strategy]:
                if pattern.embedding is None:
                    continue

                similarity = self.cosine_similarity(query_embedding, pattern.embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    existing_pattern = pattern

            # Wenn ähnliches Pattern existiert, update
            if existing_pattern and max_similarity >= self.pattern_similarity_threshold:
                existing_pattern.total_count += 1
                if success:
                    existing_pattern.success_count += 1
            else:
                # Neues Pattern erstellen
                new_pattern = QueryPattern(
                    pattern_text=query[: self.query_truncate_length],
                    embedding=query_embedding,
                    associated_strategy=strategy,
                    success_count=1 if success else 0,
                    total_count=1,
                )
                self.query_patterns[strategy].append(new_pattern)

                # Limit Patterns pro Strategy
                if len(self.query_patterns[strategy]) > self.max_patterns_per_strategy:
                    # Remove Pattern mit niedrigster Success Rate
                    self.query_patterns[strategy].sort(key=lambda p: p.success_rate)
                    self.query_patterns[strategy].pop(0)

        except (TypeError, ValueError) as e:
            logger.error("Invalid data in update_query_patterns: %s", e, exc_info=True)
        except Exception as e:
            logger.critical(
                "Unexpected error updating query patterns: %s", e, exc_info=True
            )

    def match_query_patterns(
        self,
        query_embedding: List[float],
        strategy_name: str,
        pattern_similarity_weight: float = 0.6,
        pattern_success_weight: float = 0.4,
    ) -> float:
        """
        Pattern-basierte Scoring: Ähnlichkeit zu erfolgreichen früheren Queries.

        Args:
            query_embedding: Query embedding vector
            strategy_name: Strategy to match against
            pattern_similarity_weight: Weight for similarity score (default: 0.6)
            pattern_success_weight: Weight for success rate (default: 0.4)

        Returns:
            Score 0.0-1.0
        """
        if strategy_name not in self.query_patterns:
            return 0.5  # Neutral

        patterns = self.query_patterns[strategy_name]
        if not patterns:
            return 0.5

        # Finde ähnlichste Patterns (cosine similarity via embeddings)
        max_similarity = 0.0
        matching_pattern = None

        for pattern in patterns:
            if pattern.embedding is None:
                continue

            similarity = self.cosine_similarity(query_embedding, pattern.embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                matching_pattern = pattern

        if matching_pattern and max_similarity >= self.pattern_similarity_threshold:
            # Gewichte similarity mit Pattern success rate
            pattern_success = matching_pattern.success_rate
            return (
                max_similarity * pattern_similarity_weight
                + pattern_success * pattern_success_weight
            )

        return 0.5  # Kein passendes Pattern gefunden

    def extract_failure_pattern(
        self, query: str, result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extrahiere Failure Pattern aus fehlgeschlagener Query.

        Args:
            query: Failed query text
            result: Result dict (may contain error info)

        Returns:
            Failure pattern string or None
        """
        try:
            # Einfache Heuristik: Extract Keywords aus Query
            tokens = query.lower().split()

            # Filter stopwords (simple)
            stopwords = {
                "der",
                "die",
                "das",
                "ein",
                "eine",
                "ist",
                "sind",
                "hat",
                "haben",
            }
            keywords = [t for t in tokens if t not in stopwords and len(t) > 3]

            if keywords:
                return " ".join(keywords[:3])  # Top 3 keywords

            return query[:50]  # Fallback

        except Exception:
            return None

    # ========================================================================
    # Performance Scoring
    # ========================================================================

    def calculate_performance_score(
        self,
        stats: StrategyPerformance,
        min_queries_for_confidence: int = 5,
        success_weight: float = 0.6,
        confidence_weight: float = 0.4,
        speed_bonus_max: float = 0.1,
        speed_threshold: float = 5.0,
    ) -> float:
        """
        Performance-basierte Scoring.

        Args:
            stats: StrategyPerformance to score
            min_queries_for_confidence: Min queries before confidence matters (default: 5)
            success_weight: Weight for success rate (default: 0.6)
            confidence_weight: Weight for avg confidence (default: 0.4)
            speed_bonus_max: Max bonus for fast response (default: 0.1)
            speed_threshold: Response time threshold in seconds (default: 5.0)

        Returns:
            Score 0.0-1.0+
        """
        # Wenn zu wenig Queries, neutral score
        if stats.queries_handled < min_queries_for_confidence:
            return 0.5

        # Weighted combination von Success Rate und Avg Confidence
        success_component = stats.success_rate
        confidence_component = stats.avg_confidence

        # Zusätzlich: Bonus für schnelle Response Time
        # Normalisiere response time
        speed_bonus = (
            max(0, 1.0 - stats.avg_response_time / speed_threshold) * speed_bonus_max
        )

        score = (
            success_component * success_weight
            + confidence_component * confidence_weight
            + speed_bonus
        )

        return min(1.0, score)

    # ========================================================================
    # Getters/Setters (Thread-Safe)
    # ========================================================================

    def get_strategy_stats(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """
        Get strategy statistics (thread-safe).

        Args:
            strategy_name: Name of strategy

        Returns:
            StrategyPerformance or None
        """
        with self._lock:
            return self.strategy_stats.get(strategy_name)

    def get_or_create_strategy_stats(self, strategy_name: str) -> StrategyPerformance:
        """
        Get or create strategy statistics (thread-safe, atomic).

        Args:
            strategy_name: Name of strategy

        Returns:
            StrategyPerformance (existing or new)
        """
        with self._lock:
            if strategy_name not in self.strategy_stats:
                self.strategy_stats[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name
                )
            return self.strategy_stats[strategy_name]

    def get_all_stats(self) -> Dict[str, StrategyPerformance]:
        """Get all strategy stats (thread-safe copy)."""
        with self._lock:
            return self.strategy_stats.copy()

    def get_all_patterns(self) -> Dict[str, List[QueryPattern]]:
        """Get all query patterns (thread-safe copy)."""
        with self._lock:
            return {k: list(v) for k, v in self.query_patterns.items()}
