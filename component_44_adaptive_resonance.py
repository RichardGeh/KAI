"""
component_44_adaptive_resonance.py

Adaptive Resonance Engine - Meta-Learning Integration

Erweitert ResonanceEngine um:
- Automatische Anpassung an Graph-Größe
- Performance-basiertes Tuning
- Accuracy-basierte Optimierung
- Integration mit MetaLearningEngine

Tuning-Strategie:
1. Graph-Size: Größere Graphen -> höhere Thresholds, aggressiveres Pruning
2. Query-Time: Langsame Queries -> weniger Waves, mehr Pruning
3. Accuracy: Niedrige Accuracy -> mehr Waves, höherer Decay

Teil von Phase 4 Architecture Refactoring (2025-11-29)
Split from component_44_resonance_engine.py (1060 lines -> modular)

Author: KAI Development Team
Created: 2025-11-29
"""

import logging
from typing import Dict, List, Optional

from component_44_resonance_core import ResonanceEngine
from component_44_resonance_data_structures import ActivationMap

logger = logging.getLogger(__name__)


class AdaptiveResonanceEngine(ResonanceEngine):
    """
    Adaptive Resonance Engine mit automatischem Hyperparameter-Tuning

    Erweitert ResonanceEngine um:
    - Automatische Anpassung an Graph-Größe
    - Performance-basiertes Tuning
    - Accuracy-basierte Optimierung
    - Integration mit MetaLearningEngine

    Tuning-Strategie:
    1. Graph-Size: Größere Graphen -> höhere Thresholds, aggressiveres Pruning
    2. Query-Time: Langsame Queries -> weniger Waves, mehr Pruning
    3. Accuracy: Niedrige Accuracy -> mehr Waves, höherer Decay
    """

    def __init__(self, netzwerk, confidence_mgr=None, meta_learning=None):
        """
        Initialize Adaptive Resonance Engine

        Args:
            netzwerk: KonzeptNetzwerk instance
            confidence_mgr: Optional ConfidenceManager
            meta_learning: Optional MetaLearningEngine für adaptive tuning
        """
        super().__init__(netzwerk, confidence_mgr)

        # Meta-Learning Integration
        self.meta_learning = meta_learning

        # Tuning History für Monitoring
        self.tuning_history = []

        # Default: Conservative Start
        self._initial_hyperparameters = {
            "activation_threshold": self.activation_threshold,
            "decay_factor": self.decay_factor,
            "resonance_boost": self.resonance_boost,
            "max_waves": self.max_waves,
            "max_concepts_per_wave": self.max_concepts_per_wave,
        }

        logger.info("AdaptiveResonanceEngine initialized with Meta-Learning support")

    def auto_tune_hyperparameters(self) -> Dict[str, float]:
        """
        Passt Hyperparameter automatisch an basierend auf:
        - Graph-Größe (Skalierung)
        - Durchschnittliche Query-Zeit (Performance)
        - Accuracy (Success Rate)

        Returns:
            Dict mit neuen Hyperparameter-Werten
        """
        try:
            # 1. Ermittle Graph-Größe
            try:
                graph_size = self.netzwerk.get_node_count()
            except Exception as e:
                logger.warning(f"Konnte Graph-Größe nicht ermitteln: {e}")
                graph_size = 1000  # Fallback zu mittlerem Graph

            # 2. Hole Performance-Metriken aus MetaLearningEngine
            avg_query_time = 0.0
            avg_accuracy = 0.5  # Neutral default

            if self.meta_learning:
                # Hole Stats für 'resonance' strategy
                stats = self.meta_learning.get_strategy_stats("resonance")
                if stats:
                    avg_query_time = stats.avg_response_time
                    avg_accuracy = stats.success_rate
                else:
                    logger.warning("No resonance strategy stats found, using defaults")

            logger.info(
                f"Auto-tuning based on: graph_size={graph_size}, "
                f"avg_query_time={avg_query_time:.3f}s, accuracy={avg_accuracy:.3f}"
            )

            # 3. Rule-based Tuning (später: Gradient-free optimization)
            new_params = self._calculate_optimal_parameters(
                graph_size, avg_query_time, avg_accuracy
            )

            # 4. Apply new parameters
            self.set_hyperparameters(**new_params)

            # 5. Track tuning
            self.tuning_history.append(
                {
                    "timestamp": __import__("datetime").datetime.now(),
                    "graph_size": graph_size,
                    "avg_query_time": avg_query_time,
                    "avg_accuracy": avg_accuracy,
                    "parameters": new_params.copy(),
                }
            )

            logger.info(f"Auto-tuning complete: {new_params}")
            return new_params

        except Exception as e:
            logger.error(f"Auto-tuning failed: {e}", exc_info=True)
            return self._initial_hyperparameters.copy()

    def _calculate_optimal_parameters(
        self, graph_size: int, avg_query_time: float, avg_accuracy: float
    ) -> Dict[str, float]:
        """
        Berechnet optimale Hyperparameter basierend auf Metriken

        Tuning-Regeln:
        1. Graph-Size:
           - Kleine Graphen (<1000): Liberal (low threshold, viele waves)
           - Mittlere Graphen (1000-10000): Balanced
           - Große Graphen (10000-50000): Conservative (high threshold, pruning)
           - Sehr große Graphen (>50000): Aggressive pruning

        2. Query-Time:
           - Schnell (<1s): Kann mehr Waves haben
           - Mittel (1-5s): Balanced
           - Langsam (>5s): Weniger Waves, aggressives Pruning

        3. Accuracy:
           - Hoch (>0.8): Parameter sind gut, minor adjustments
           - Mittel (0.6-0.8): Moderate Anpassungen
           - Niedrig (<0.6): Mehr Exploration (mehr Waves, weniger Pruning)

        Returns:
            Dict mit optimalen Hyperparameter-Werten
        """
        params = {}

        # ===================================================================
        # 1. Graph-Size basiertes Tuning
        # ===================================================================
        if graph_size > 50000:
            # Sehr großer Graph: Aggressives Pruning
            params["activation_threshold"] = 0.4
            params["max_concepts_per_wave"] = 50
            params["max_waves"] = 3
            params["decay_factor"] = 0.6
            params["resonance_boost"] = 0.3

        elif graph_size > 10000:
            # Großer Graph: Conservative
            params["activation_threshold"] = 0.35
            params["max_concepts_per_wave"] = 80
            params["max_waves"] = 4
            params["decay_factor"] = 0.65
            params["resonance_boost"] = 0.4

        elif graph_size > 1000:
            # Mittlerer Graph: Balanced
            params["activation_threshold"] = 0.3
            params["max_concepts_per_wave"] = 100
            params["max_waves"] = 5
            params["decay_factor"] = 0.7
            params["resonance_boost"] = 0.5

        else:
            # Kleiner Graph: Liberal (mehr Exploration)
            params["activation_threshold"] = 0.2
            params["max_concepts_per_wave"] = 150
            params["max_waves"] = 6
            params["decay_factor"] = 0.75
            params["resonance_boost"] = 0.6

        # ===================================================================
        # 2. Query-Time basiertes Tuning (Performance)
        # ===================================================================
        if avg_query_time > 5.0:
            # Zu langsam: Drastisches Pruning
            params["max_waves"] = max(2, params["max_waves"] - 2)
            params["max_concepts_per_wave"] = max(
                30, params["max_concepts_per_wave"] - 30
            )
            params["activation_threshold"] = min(
                0.5, params["activation_threshold"] + 0.1
            )
            logger.info(
                "Performance tuning: Reduced waves and concepts due to slow queries"
            )

        elif avg_query_time > 2.0:
            # Moderat langsam: Minor Pruning
            params["max_waves"] = max(3, params["max_waves"] - 1)
            params["max_concepts_per_wave"] = max(
                50, params["max_concepts_per_wave"] - 20
            )

        elif avg_query_time < 0.5:
            # Sehr schnell: Kann mehr Exploration haben
            params["max_waves"] = min(7, params["max_waves"] + 1)
            params["max_concepts_per_wave"] = min(
                200, params["max_concepts_per_wave"] + 20
            )
            logger.info("Performance tuning: Increased exploration due to fast queries")

        # ===================================================================
        # 3. Accuracy basiertes Tuning
        # ===================================================================
        if avg_accuracy < 0.6:
            # Niedrige Accuracy: Mehr Exploration
            params["max_waves"] = min(7, params["max_waves"] + 1)
            params["decay_factor"] = min(0.8, params["decay_factor"] + 0.05)
            params["resonance_boost"] = min(0.7, params["resonance_boost"] + 0.1)
            params["max_concepts_per_wave"] = min(
                200, params["max_concepts_per_wave"] + 20
            )
            logger.info("Accuracy tuning: Increased exploration due to low accuracy")

        elif avg_accuracy > 0.8:
            # Hohe Accuracy: Parameter sind gut, nur fine-tuning
            # Versuche Performance zu optimieren ohne Accuracy zu opfern
            if avg_query_time > 1.0:
                params["max_concepts_per_wave"] = max(
                    50, params["max_concepts_per_wave"] - 10
                )
                logger.info(
                    "Accuracy tuning: Minor pruning to improve speed while maintaining accuracy"
                )

        # ===================================================================
        # 4. Sicherheits-Checks (Boundaries)
        # ===================================================================
        params["activation_threshold"] = max(
            0.1, min(0.6, params["activation_threshold"])
        )
        params["decay_factor"] = max(0.5, min(0.9, params["decay_factor"]))
        params["resonance_boost"] = max(0.1, min(0.8, params["resonance_boost"]))
        params["max_waves"] = max(2, min(10, params["max_waves"]))
        params["max_concepts_per_wave"] = max(
            20, min(300, params["max_concepts_per_wave"])
        )

        return params

    def activate_concept(
        self,
        start_word: str,
        query_context: Optional[Dict] = None,
        allowed_relations: Optional[List[str]] = None,
        auto_tune: bool = False,
        use_cache: bool = True,
    ) -> ActivationMap:
        """
        Überschreibt activate_concept mit optionalem Auto-Tuning

        Args:
            start_word: Start-Konzept
            query_context: Optional context
            allowed_relations: Optional erlaubte Relationstypen
            auto_tune: Falls True, führe Auto-Tuning vor Aktivierung durch
            use_cache: Nutze Activation Maps Cache (default: True)

        Returns:
            ActivationMap
        """
        # Optional: Auto-Tuning vor jeder Query
        if auto_tune:
            self.auto_tune_hyperparameters()

        # Call parent implementation with caching
        return super().activate_concept(
            start_word, query_context, allowed_relations, use_cache
        )

    def get_tuning_stats(self) -> Dict[str, any]:
        """
        Gibt Statistiken über das Tuning zurück

        Returns:
            Dict mit Tuning-History und aktuellen Parametern
        """
        return {
            "current_parameters": {
                "activation_threshold": self.activation_threshold,
                "decay_factor": self.decay_factor,
                "resonance_boost": self.resonance_boost,
                "max_waves": self.max_waves,
                "max_concepts_per_wave": self.max_concepts_per_wave,
            },
            "initial_parameters": self._initial_hyperparameters,
            "tuning_history": self.tuning_history[-10:],  # Last 10 tunings
            "total_tunings": len(self.tuning_history),
        }

    def reset_to_defaults(self):
        """Reset Hyperparameter zu initialen Werten"""
        self.set_hyperparameters(**self._initial_hyperparameters)
        logger.info("Reset hyperparameters to initial values")

    # ========================================================================
    # BaseReasoningEngine Interface - Specialized Overrides
    # ========================================================================

    def get_capabilities(self) -> List[str]:
        """
        Return enhanced capabilities including adaptive tuning.

        Note: Inherits all capabilities from ResonanceEngine and adds adaptive ones.
        """
        base_capabilities = super().get_capabilities()
        adaptive_capabilities = [
            "adaptive_resonance",
            "resonance_tuning",
            "dynamic_thresholds",
            "performance_optimization",
        ]
        return base_capabilities + adaptive_capabilities

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost including adaptive tuning overhead.

        Returns:
            Cost estimate in [0.0, 1.0] range
            Base cost: 0.7 (more expensive than base ResonanceEngine due to tuning)
        """
        # Get base resonance cost
        base_cost = super().estimate_cost(query)

        # Add adaptive tuning overhead
        # Auto-tuning adds graph size queries and performance metric lookups
        tuning_overhead = 0.1

        return min(base_cost + tuning_overhead, 1.0)
