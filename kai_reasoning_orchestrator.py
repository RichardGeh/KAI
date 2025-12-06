# kai_reasoning_orchestrator.py
"""
Reasoning Orchestrator for KAI - Hybrid Reasoning System (FACADE)

Coordinates multiple Reasoning Engines and combines results using modular architecture.
This is a thin facade that delegates to specialized modules for clean separation of concerns.

Refactored from monolithic 1909-line file into modular architecture:
- kai_strategy_dispatcher.py: Strategy selection and execution
- kai_result_aggregator.py: Result merging and confidence fusion
- kai_proof_merger.py: Proof tree merging
- kai_reasoning_orchestrator.py: Facade coordinating the modules

Features:
- Hybrid Reasoning (Logic + Probabilistic + Graph + Abductive + Spatial + Constraint + Resonance)
- Weighted Confidence Fusion
- Unified Proof Tree Generation
- Fallback Strategies
- Result Aggregation
- Meta-Learning Integration

Architecture:
    1. Fast Path: Direct fact lookup
    2. Deterministic Reasoning: Logic Engine + Graph Traversal
    3. Probabilistic Enhancement: Uncertainty quantification
    4. Abductive Fallback: Hypothesis generation
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from cachetools import LRUCache

from component_1_netzwerk import KonzeptNetzwerk

logger = logging.getLogger(__name__)


# ========================================================================
# Data Structures
# ========================================================================


class ReasoningStrategy(Enum):
    """Available reasoning strategies"""

    DIRECT_FACT = "direct_fact"
    LOGIC_ENGINE = "logic_engine"
    GRAPH_TRAVERSAL = "graph_traversal"
    PROBABILISTIC = "probabilistic"
    ABDUCTIVE = "abductive"
    COMBINATORIAL = "combinatorial"  # Strategic/combinatorial reasoning
    SPATIAL = "spatial"  # Spatial reasoning (grids, shapes, positions)
    RESONANCE = "resonance"  # Spreading activation with resonance amplification
    CONSTRAINT = "constraint"  # Constraint satisfaction (logic puzzles, CSP)
    LOGIC_PUZZLE = "logic_puzzle"  # Logic puzzle solving (SAT + numerical CSP)


@dataclass
class ReasoningResult:
    """
    Result from a single reasoning strategy.

    Attributes:
        strategy: Which strategy was used
        success: Whether reasoning succeeded
        confidence: Confidence score (0.0-1.0)
        inferred_facts: Dictionary of inferred facts
        proof_tree: Unified ProofTree (optional)
        proof_trace: Text explanation
        metadata: Additional strategy-specific data
        is_hypothesis: Whether result is abductive hypothesis
    """

    strategy: ReasoningStrategy
    success: bool
    confidence: float
    inferred_facts: Dict[str, List[str]] = field(default_factory=dict)
    proof_tree: Optional = None  # ProofTree from component_17
    proof_trace: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_hypothesis: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedResult:
    """
    Aggregated result from multiple reasoning strategies.

    Combines evidence from multiple sources using weighted fusion.
    """

    combined_confidence: float
    inferred_facts: Dict[str, List[str]]
    merged_proof_tree: Optional  # ProofTree from component_17
    strategies_used: List[ReasoningStrategy]
    individual_results: List[ReasoningResult]
    explanation: str
    is_hypothesis: bool = False


# ========================================================================
# Main Orchestrator (Facade)
# ========================================================================


class ReasoningOrchestrator:
    """
    Main orchestrator for hybrid reasoning (FACADE).

    Coordinates StrategyDispatcher, ResultAggregator, and ProofMerger
    to provide unified hybrid reasoning interface. Maintains backward
    compatibility with previous monolithic implementation.
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        logic_engine,
        graph_traversal,
        working_memory,
        signals,
        probabilistic_engine=None,
        abductive_engine=None,
        combinatorial_reasoner=None,
        spatial_reasoner=None,
        resonance_engine=None,
        meta_learning_engine=None,
        self_evaluator=None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize Reasoning Orchestrator (Facade).

        Args:
            netzwerk: KonzeptNetzwerk instance
            logic_engine: Logic Engine instance
            graph_traversal: GraphTraversal instance
            working_memory: WorkingMemory instance
            signals: KaiSignals for UI updates
            probabilistic_engine: ProbabilisticEngine (optional)
            abductive_engine: AbductiveEngine (optional)
            combinatorial_reasoner: CombinatorialReasoner (optional)
            spatial_reasoner: SpatialReasoner (optional)
            resonance_engine: ResonanceEngine (optional)
            meta_learning_engine: MetaLearningEngine (optional)
            self_evaluator: SelfEvaluator (optional)
            config_path: Path to YAML configuration file (optional)
        """
        # Store engine references for backward compatibility
        self.netzwerk = netzwerk
        self.logic_engine = logic_engine
        self.graph_traversal = graph_traversal
        self.working_memory = working_memory
        self.signals = signals
        self.probabilistic_engine = probabilistic_engine
        self.abductive_engine = abductive_engine
        self.combinatorial_reasoner = combinatorial_reasoner
        self.spatial_reasoner = spatial_reasoner
        self.resonance_engine = resonance_engine
        self.meta_learning_engine = meta_learning_engine
        self.self_evaluator = self_evaluator

        # Default Configuration
        self.enable_hybrid = True
        self.min_confidence_threshold = 0.4
        self.probabilistic_enhancement = True
        self.aggregation_method = (
            "noisy_or"  # noisy_or | weighted_avg | max | dempster_shafer
        )
        self.enable_parallel_execution = False
        self.enable_result_caching = True

        # Strategy weights for weighted_avg aggregation
        self.strategy_weights = {
            ReasoningStrategy.DIRECT_FACT: 0.35,
            ReasoningStrategy.LOGIC_ENGINE: 0.18,
            ReasoningStrategy.GRAPH_TRAVERSAL: 0.14,
            ReasoningStrategy.RESONANCE: 0.12,
            ReasoningStrategy.CONSTRAINT: 0.11,
            ReasoningStrategy.SPATIAL: 0.10,
            ReasoningStrategy.COMBINATORIAL: 0.07,
            ReasoningStrategy.PROBABILISTIC: 0.03,
            ReasoningStrategy.ABDUCTIVE: 0.01,
        }

        # Load configuration from YAML if provided
        if config_path:
            self._load_config(config_path)

        # Result cache (LRU)
        self._result_cache = (
            LRUCache(maxsize=100) if self.enable_result_caching else None
        )

        # Initialize specialized modules
        from kai_proof_merger import ProofMerger
        from kai_result_aggregator import ResultAggregator
        from kai_strategy_dispatcher import StrategyDispatcher

        self.strategy_dispatcher = StrategyDispatcher(
            netzwerk=netzwerk,
            logic_engine=logic_engine,
            graph_traversal=graph_traversal,
            working_memory=working_memory,
            signals=signals,
            probabilistic_engine=probabilistic_engine,
            abductive_engine=abductive_engine,
            combinatorial_reasoner=combinatorial_reasoner,
            spatial_reasoner=spatial_reasoner,
            resonance_engine=resonance_engine,
            meta_learning_engine=meta_learning_engine,
            self_evaluator=self_evaluator,
            enable_parallel_execution=self.enable_parallel_execution,
            min_confidence_threshold=self.min_confidence_threshold,
        )

        self.result_aggregator = ResultAggregator(
            signals=signals,
            meta_learning_engine=meta_learning_engine,
            self_evaluator=self_evaluator,
            aggregation_method=self.aggregation_method,
            strategy_weights=self.strategy_weights,
        )

        self.proof_merger = ProofMerger(signals=signals)

        logger.info(
            f"ReasoningOrchestrator initialized (Facade): "
            f"aggregation={self.aggregation_method}, "
            f"parallel={self.enable_parallel_execution}, "
            f"caching={self.enable_result_caching}"
        )

    def query_with_hybrid_reasoning(
        self,
        topic: str,
        relation_type: str = "IS_A",
        strategies: Optional[List[ReasoningStrategy]] = None,
    ) -> Optional[AggregatedResult]:
        """
        Main entry point for hybrid reasoning.

        Delegates to StrategyDispatcher for execution and ResultAggregator
        for combining results.

        Args:
            topic: The topic to reason about
            relation_type: Type of relation to find
            strategies: Which strategies to use (None = all)

        Returns:
            AggregatedResult with combined evidence or None
        """
        logger.info(f"[Hybrid Reasoning] Query: {topic} ({relation_type})")

        # Check cache first
        cache_key = f"{topic}:{relation_type}:{str(strategies)}"
        if self._result_cache is not None and cache_key in self._result_cache:
            logger.debug(f"[Cache Hit] Returning cached result for {topic}")
            return self._result_cache[cache_key]

        # Default strategies if not specified
        if strategies is None:
            strategies = [
                ReasoningStrategy.DIRECT_FACT,
                ReasoningStrategy.GRAPH_TRAVERSAL,
                ReasoningStrategy.LOGIC_ENGINE,
                ReasoningStrategy.RESONANCE,
                ReasoningStrategy.CONSTRAINT,
                ReasoningStrategy.SPATIAL,
                ReasoningStrategy.PROBABILISTIC,
                ReasoningStrategy.ABDUCTIVE,
            ]

        # Delegate to StrategyDispatcher for execution
        results = self.strategy_dispatcher.execute_strategies(
            topic=topic,
            relation_type=relation_type,
            strategies=strategies,
            probabilistic_enhancement=self.probabilistic_enhancement,
        )

        # Check if we have results
        if not results:
            logger.info("[Hybrid Reasoning] [X] No strategy succeeded")
            return None

        # Delegate to ResultAggregator for combining
        try:
            aggregated = self.result_aggregator.aggregate_results(
                results=results,
                emit_proof_tree=True,
            )

            # Check confidence threshold
            if aggregated.combined_confidence >= self.min_confidence_threshold:
                logger.info(
                    f"[Hybrid Reasoning] [OK] Success with {len(results)} strategies "
                    f"(confidence: {aggregated.combined_confidence:.2f})"
                )
            else:
                logger.info(
                    f"[Hybrid Reasoning] Low confidence result: {aggregated.combined_confidence:.2f}"
                )

            # Cache result
            if self._result_cache is not None:
                self._result_cache[cache_key] = aggregated

            return aggregated

        except Exception as e:
            logger.error(f"[Hybrid Reasoning] Aggregation failed: {e}", exc_info=True)
            return None

    def query_with_meta_learning(
        self,
        topic: str,
        relation_type: str = "IS_A",
        query_text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Optional[AggregatedResult]:
        """
        Meta-Learning-based strategy selection and reasoning.

        Uses MetaLearningEngine to select best strategy based on query pattern,
        with self-evaluation and fallback on low confidence.

        Args:
            topic: The reasoning topic
            relation_type: Type of relation (IS_A, HAS_PROPERTY, etc.)
            query_text: Optional query text for better pattern matching
            context: Optional context dict
            max_retries: Maximum fallback attempts on low confidence

        Returns:
            AggregatedResult or None
        """
        if not self.meta_learning_engine:
            logger.warning(
                "[Meta-Learning] MetaLearningEngine not available, falling back to hybrid reasoning"
            )
            return self.query_with_hybrid_reasoning(topic, relation_type)

        logger.info(f"[Meta-Learning] Query: {topic} ({relation_type})")

        # Prepare context
        if context is None:
            context = {}
        context["topic"] = topic
        context["relation_type"] = relation_type

        # Use query_text or construct from topic
        query_for_ml = query_text or f"Was ist ein {topic}?"

        # Track attempted strategies to avoid infinite loops
        attempted_strategies = []
        retry_count = 0

        while retry_count < max_retries:
            # 1. Meta-Learning: Select best strategy
            available_strategies = (
                self.strategy_dispatcher.get_available_strategy_names(
                    exclude=attempted_strategies
                )
            )

            if not available_strategies:
                logger.warning("[Meta-Learning] No more strategies available for retry")
                break

            selected_strategy, ml_confidence = (
                self.meta_learning_engine.select_best_strategy(
                    query=query_for_ml,
                    context=context,
                    available_strategies=available_strategies,
                )
            )

            logger.info(
                f"[Meta-Learning] Selected strategy: '{selected_strategy}' "
                f"(ML confidence: {ml_confidence:.2f}, attempt {retry_count + 1}/{max_retries})"
            )

            attempted_strategies.append(selected_strategy)

            # Track start time for performance monitoring
            start_time = time.time()

            # 2. Execute selected strategy
            result = None

            if selected_strategy == "resonance":
                # Special handling for resonance strategy
                resonance_result = self.strategy_dispatcher.execute_resonance_strategy(
                    topic, relation_type, context, query_for_ml
                )
                if resonance_result:
                    # Wrap in AggregatedResult
                    result = AggregatedResult(
                        combined_confidence=resonance_result.confidence,
                        inferred_facts=resonance_result.inferred_facts,
                        merged_proof_tree=resonance_result.proof_tree,
                        strategies_used=[ReasoningStrategy.RESONANCE],
                        individual_results=[resonance_result],
                        explanation=f"Resonance Strategy: {resonance_result.proof_trace}",
                        is_hypothesis=False,
                    )
            else:
                # Execute via single strategy dispatcher
                strategy_enum = self._map_strategy_name_to_enum(selected_strategy)
                if strategy_enum:
                    single_result = self.strategy_dispatcher.execute_single_strategy(
                        topic, relation_type, strategy_enum
                    )
                    if single_result:
                        # Wrap in AggregatedResult
                        result = self.result_aggregator.aggregate_results(
                            [single_result]
                        )

            response_time = time.time() - start_time

            # 3. Self-Evaluation (if available and result exists)
            if result and self.self_evaluator:
                eval_result = self.result_aggregator.evaluate_result_quality(
                    result, query_for_ml, topic, context
                )

                # Handle recommendation (can be String or Enum)
                recommendation_value = (
                    eval_result.recommendation.value
                    if hasattr(eval_result.recommendation, "value")
                    else str(eval_result.recommendation)
                )

                logger.info(
                    f"[Self-Evaluation] Score: {eval_result.overall_score:.2f}, "
                    f"Recommendation: {recommendation_value}"
                )

                # Check if we should retry with different strategy
                if recommendation_value == "retry_different_strategy":
                    logger.info(
                        f"[Self-Evaluation] Recommends retry, attempting different strategy "
                        f"(attempt {retry_count + 1})"
                    )

                    # Record failed attempt
                    self.result_aggregator.record_strategy_usage(
                        selected_strategy,
                        query_for_ml,
                        result,
                        response_time,
                        success=False,
                        context=context,
                    )

                    retry_count += 1
                    continue  # Try next strategy

                # Adjust confidence if suggested
                if eval_result.confidence_adjusted and eval_result.suggested_confidence:
                    logger.info(
                        f"[Self-Evaluation] Adjusting confidence: "
                        f"{result.combined_confidence:.2f} -> {eval_result.suggested_confidence:.2f}"
                    )
                    result.combined_confidence = eval_result.suggested_confidence

            # 4. Record successful strategy usage
            if result:
                self.result_aggregator.record_strategy_usage(
                    selected_strategy,
                    query_for_ml,
                    result,
                    response_time,
                    success=True,
                    context=context,
                )

                logger.info(
                    f"[Meta-Learning] Success with '{selected_strategy}' "
                    f"(confidence: {result.combined_confidence:.2f})"
                )
                return result
            else:
                # Strategy failed to produce result
                self.result_aggregator.record_strategy_usage(
                    selected_strategy,
                    query_for_ml,
                    None,
                    response_time,
                    success=False,
                    context=context,
                )

                retry_count += 1

        # All retries exhausted
        logger.warning(
            f"[Meta-Learning] All retries exhausted ({retry_count} attempts)"
        )
        return None

    # ========================================================================
    # Configuration (Internal)
    # ========================================================================

    def _load_config(self, config_path: str):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        try:
            from pathlib import Path

            import yaml

            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return

            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Load orchestrator settings
            if "orchestrator" in config:
                orch_config = config["orchestrator"]
                self.enable_hybrid = orch_config.get(
                    "enable_hybrid", self.enable_hybrid
                )
                self.min_confidence_threshold = orch_config.get(
                    "min_confidence_threshold", self.min_confidence_threshold
                )
                self.probabilistic_enhancement = orch_config.get(
                    "probabilistic_enhancement", self.probabilistic_enhancement
                )
                self.aggregation_method = orch_config.get(
                    "aggregation_method", self.aggregation_method
                )
                self.enable_parallel_execution = orch_config.get(
                    "enable_parallel_execution", self.enable_parallel_execution
                )
                self.enable_result_caching = orch_config.get(
                    "enable_result_caching", self.enable_result_caching
                )

            # Load strategy weights
            if "strategy_weights" in config:
                for strategy_name, weight in config["strategy_weights"].items():
                    try:
                        strategy = ReasoningStrategy(strategy_name)
                        self.strategy_weights[strategy] = weight
                    except ValueError:
                        logger.warning(f"Unknown strategy in config: {strategy_name}")

            logger.info(f"[OK] Configuration loaded from {config_path}")

        except ImportError:
            logger.warning(
                "PyYAML not installed, cannot load config. Install: pip install pyyaml"
            )
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")

    def _map_strategy_name_to_enum(
        self, strategy_name: str
    ) -> Optional[ReasoningStrategy]:
        """
        Maps strategy name (string) to ReasoningStrategy enum.

        Args:
            strategy_name: Strategy name as string

        Returns:
            ReasoningStrategy enum or None
        """
        mapping = {
            "direct_fact": ReasoningStrategy.DIRECT_FACT,
            "logic_engine": ReasoningStrategy.LOGIC_ENGINE,
            "graph_traversal": ReasoningStrategy.GRAPH_TRAVERSAL,
            "probabilistic": ReasoningStrategy.PROBABILISTIC,
            "abductive": ReasoningStrategy.ABDUCTIVE,
            "combinatorial": ReasoningStrategy.COMBINATORIAL,
            "spatial": ReasoningStrategy.SPATIAL,
            "resonance": ReasoningStrategy.RESONANCE,
            "constraint": ReasoningStrategy.CONSTRAINT,
        }

        return mapping.get(strategy_name)
