# kai_response_router.py
"""
Response Generation Router für A/B Testing

Verantwortlichkeiten:
- A/B Testing zwischen Pipeline und Production System
- Meta-Learning basierte Routing-Entscheidungen
- Tracking von System-Verwendung und Performance
- Statistiken für Analyse

Extracted from kai_response_formatter.py für bessere Modularität.
"""
import logging
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResponseGenerationRouter:
    """
    Router für A/B Testing zwischen Pipeline und Production System.

    PHASE 5: A/B Testing Infrastructure

    Funktionen:
    - Entscheidet welches System für eine Query verwendet wird
    - Initial: 50/50 Random Split
    - Später: Meta-Learning basierte Auswahl
    - Trackt System-Verwendung für Performance-Analyse
    """

    def __init__(
        self,
        formatter: Any,  # KaiResponseFormatter
        production_system_weight: float = 0.5,
        enable_meta_learning: bool = False,
        meta_engine: Optional[Any] = None,
    ):
        """
        Args:
            formatter: KaiResponseFormatter Instanz
            production_system_weight: Wahrscheinlichkeit für Production System (0.0-1.0)
            enable_meta_learning: Nutze Meta-Learning für Routing-Entscheidung
            meta_engine: Optional MetaLearningEngine Instanz
        """
        self.formatter = formatter
        self.production_weight = production_system_weight
        self.enable_meta_learning = enable_meta_learning
        self.meta_engine = meta_engine

        # Tracking
        self.system_usage_counts: Dict[str, int] = defaultdict(int)
        self.total_queries: int = 0

        logger.info(
            f"ResponseGenerationRouter initialized | "
            f"production_weight={production_system_weight:.0%}, "
            f"meta_learning={'enabled' if enable_meta_learning else 'disabled'}"
        )

    def route_to_system(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Entscheidet welches System verwendet werden soll.

        Args:
            query: User Query
            context: Optional Context Dict

        Returns:
            "pipeline" oder "production"
        """
        self.total_queries += 1

        # MODE 1: Meta-Learning basierte Auswahl (wenn aktiviert)
        if self.enable_meta_learning and self.meta_engine:
            # Nutze Meta-Engine um beste "System-Strategy" zu wählen
            system = self._select_via_meta_learning(query, context)

        # MODE 2: Random A/B Split (default)
        else:
            rand = random.random()
            system = "production" if rand < self.production_weight else "pipeline"

        # Track usage
        self.system_usage_counts[system] += 1

        logger.debug(
            f"Routed to {system} system | "
            f"total={self.total_queries}, "
            f"production={self.system_usage_counts['production']}, "
            f"pipeline={self.system_usage_counts['pipeline']}"
        )

        return system

    def _select_via_meta_learning(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Nutze Meta-Learning Engine für System-Auswahl.

        Behandelt "pipeline" und "production" wie zwei Reasoning-Strategien.

        Returns:
            "pipeline" oder "production"
        """
        try:
            # Nutze MetaLearningEngine.select_best_strategy
            # mit "pipeline" und "production" als verfügbare Strategien
            available_systems = ["pipeline", "production"]

            best_system, confidence = self.meta_engine.select_best_strategy(
                query, context, available_strategies=available_systems
            )

            logger.debug(
                f"Meta-learning selected {best_system} with confidence {confidence:.2f}"
            )

            return best_system

        except Exception as e:
            logger.warning(f"Meta-learning routing failed: {e}, falling back to random")
            return "production" if random.random() < 0.5 else "pipeline"

    def generate_response(
        self,
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
        query: str,
        query_type: str = "normal",
        confidence: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        production_engine: Optional[Any] = None,
        signals: Optional[Any] = None,
    ) -> Any:  # Returns KaiResponse
        """
        Generiert Response mit automatischem Routing.

        Args:
            topic, facts, bedeutungen, synonyms: Standard Response-Formatter Args
            query: Original-Query (für Routing)
            query_type: Typ der Query
            confidence: Optional Confidence
            context: Optional Context
            production_engine: Optional ProductionSystemEngine
            signals: Optional KaiSignals für UI-Updates

        Returns:
            KaiResponse (mit strategy="pipeline" oder "production_system")
        """
        from kai_response_pipeline import KaiResponse

        start_time = time.time()

        # 1. Routing-Entscheidung
        system = self.route_to_system(query, context)

        # 2. Generiere mit gewähltem System
        if system == "production":
            response = self.formatter.generate_with_production_system(
                topic=topic,
                facts=facts,
                bedeutungen=bedeutungen,
                synonyms=synonyms,
                query_type=query_type,
                confidence=confidence,
                production_engine=production_engine,
                signals=signals,
            )
        else:  # pipeline
            response_text = self.formatter.format_standard_answer(
                topic=topic,
                facts=facts,
                bedeutungen=bedeutungen,
                synonyms=synonyms,
                query_type=query_type,
                confidence=confidence,
            )

            response = KaiResponse(
                text=response_text,
                trace=[f"Pipeline System Generierung", f"Query Type: {query_type}"],
                confidence=confidence or 0.8,
                strategy="pipeline",
            )

        # 3. Füge Routing-Info zum Trace hinzu
        response_time = time.time() - start_time
        response.trace.insert(
            0,
            f"Routing: {system} system gewählt (A/B split={self.production_weight:.0%})",
        )
        response.trace.append(f"Total Zeit: {response_time:.3f}s")

        # 4. Record für Meta-Learning (wenn aktiviert)
        if self.enable_meta_learning and self.meta_engine:
            result = {
                "confidence": response.confidence,
                "system": system,
                "response_time": response_time,
            }

            self.meta_engine.record_strategy_usage(
                strategy=system,  # Behandle system wie strategy
                query=query,
                result=result,
                response_time=response_time,
                context=context,
            )

        return response

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Routing-Statistiken zurück.

        Returns:
            Dict mit System-Usage-Counts und Percentages
        """
        stats = {
            "total_queries": self.total_queries,
            "system_usage": dict(self.system_usage_counts),
            "production_weight": self.production_weight,
            "meta_learning_enabled": self.enable_meta_learning,
        }

        if self.total_queries > 0:
            stats["production_percentage"] = (
                self.system_usage_counts["production"] / self.total_queries
            ) * 100
            stats["pipeline_percentage"] = (
                self.system_usage_counts["pipeline"] / self.total_queries
            ) * 100

        return stats

    def set_production_weight(self, weight: float) -> None:
        """
        Ändert Production System Weight.

        Args:
            weight: Neue Wahrscheinlichkeit (0.0-1.0)
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")

        old_weight = self.production_weight
        self.production_weight = weight

        logger.info(
            f"Production system weight changed: {old_weight:.0%} -> {weight:.0%}"
        )
