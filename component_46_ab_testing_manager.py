"""
component_46_ab_testing_manager.py

A/B Testing Manager for Response Generation Systems

Extracted from component_46_meta_learning.py as part of Phase 4 architectural refactoring.

Functions:
- Dual-system tracking (Pipeline vs. Production System)
- Generation system usage recording
- Side-by-side performance comparison
- Winner determination based on aggregated scores

Author: KAI Development Team
Last Updated: 2025-11-28 (Modular Refactoring)
"""

import threading
from typing import Any, Dict, Optional

from component_15_logging_config import get_logger
from component_46_performance_tracker import StrategyPerformance

logger = get_logger(__name__)


# ============================================================================
# A/B Testing Manager
# ============================================================================


class ABTestingManager:
    """
    Manages A/B testing for response generation systems.

    Functions:
    - Tracks performance for "pipeline" and "production" systems
    - Provides side-by-side comparison of metrics
    - Determines winner based on aggregated performance score
    """

    def __init__(
        self,
        performance_tracker,  # Type: PerformanceTracker (avoid circular import)
        record_strategy_usage_callback,  # Callable to record strategy usage
    ):
        """
        Initialize ABTestingManager.

        Args:
            performance_tracker: PerformanceTracker instance
            record_strategy_usage_callback: Callback to record_strategy_usage()
        """
        self.performance_tracker = performance_tracker
        self.record_strategy_usage_callback = record_strategy_usage_callback

        # Thread Safety
        self._lock = threading.RLock()

        logger.info("ABTestingManager initialized")

    # ========================================================================
    # Utility Methods
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

    # ========================================================================
    # Dual-System Performance Tracking
    # ========================================================================

    def record_generation_system_usage(
        self,
        system: str,
        query: str,
        confidence: float,
        response_time: float,
        response_text: str = "",
        success: Optional[bool] = None,
        user_feedback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track Performance für Generation Systems (Pipeline vs. Production).

        PHASE 5: Dual-System Performance Tracking

        Wrapper um record_strategy_usage() mit zusätzlicher Semantik
        für Response Generation Systems.

        Args:
            system: "pipeline" oder "production"
            query: User-Query
            confidence: Confidence-Wert (0.0-1.0)
            response_time: Zeit in Sekunden
            response_text: Generierter Response-Text (für Variabilitäts-Analyse)
            success: True/False wenn Feedback vorhanden
            user_feedback: 'correct', 'incorrect', 'neutral'
            context: Optional Context Dict
            metadata: Zusätzliche Metadaten (z.B. cycles, sentences)

        Raises:
            ValueError: If system is not 'pipeline' or 'production'
        """
        if system not in ["pipeline", "production"]:
            raise ValueError(
                f"Unknown generation system '{system}', expected 'pipeline' or 'production'"
            )

        # Erstelle result dict
        result = {
            "confidence": confidence,
            "system": system,
            "response_time": response_time,
            "response_length": len(response_text) if response_text else 0,
        }

        # Füge Metadata hinzu
        if metadata:
            result.update(metadata)

        # Nutze existierende record_strategy_usage() Methode
        # (behandle system als strategy)
        self.record_strategy_usage_callback(
            strategy=system,
            query=query,
            result=result,
            response_time=response_time,
            context=context,
            user_feedback=user_feedback,
        )

        # Sanitize for logging
        safe_query = self._sanitize_for_logging(query, max_len=50)
        logger.debug(
            "Recorded %s system usage | query='%s', confidence=%.2f, "
            "response_time=%.3fs, length=%d",
            system,
            safe_query,
            confidence,
            response_time,
            len(response_text) if response_text else 0,
        )

    def get_generation_system_comparison(self) -> Dict[str, Any]:
        """
        Vergleicht Performance zwischen Pipeline und Production System.

        PHASE 5: Dual-System Performance Tracking

        Returns:
            Dict mit Side-by-Side Vergleich:
                - queries_count: Anzahl Queries pro System
                - avg_confidence: Durchschnittliche Confidence
                - avg_response_time: Durchschnittliche Response-Zeit
                - success_rate: Success Rate (wenn Feedback vorhanden)
                - winner: Welches System besser performed
        """
        pipeline_stats = self.performance_tracker.get_strategy_stats("pipeline")
        production_stats = self.performance_tracker.get_strategy_stats("production")

        comparison = {
            "pipeline": self.format_system_stats(pipeline_stats),
            "production": self.format_system_stats(production_stats),
            "comparison": {},
        }

        # Vergleiche Metriken
        if pipeline_stats and production_stats:
            # Winner basierend auf aggregiertem Score
            pipeline_score = self.performance_tracker.calculate_performance_score(
                pipeline_stats
            )
            production_score = self.performance_tracker.calculate_performance_score(
                production_stats
            )

            comparison["comparison"] = {
                "confidence_delta": production_stats.avg_confidence
                - pipeline_stats.avg_confidence,
                "speed_delta": pipeline_stats.avg_response_time
                - production_stats.avg_response_time,  # Positiv = Pipeline schneller
                "success_rate_delta": production_stats.success_rate
                - pipeline_stats.success_rate,
                "overall_score_pipeline": pipeline_score,
                "overall_score_production": production_score,
                "winner": (
                    "production" if production_score > pipeline_score else "pipeline"
                ),
                "win_margin": abs(production_score - pipeline_score),
            }

        return comparison

    def format_system_stats(
        self, stats: Optional[StrategyPerformance]
    ) -> Dict[str, Any]:
        """
        Formatiert StrategyPerformance für Output.

        Args:
            stats: StrategyPerformance or None

        Returns:
            Formatted dict with metrics
        """
        if stats is None:
            return {
                "queries_handled": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_response_time": 0.0,
            }

        return {
            "queries_handled": stats.queries_handled,
            "success_count": stats.success_count,
            "failure_count": stats.failure_count,
            "success_rate": stats.success_rate,
            "avg_confidence": stats.avg_confidence,
            "avg_response_time": stats.avg_response_time,
        }
