# kai_result_aggregator.py
"""
Result Aggregator for KAI Reasoning Orchestrator

Aggregates results from multiple reasoning engines using confidence fusion,
deduplication, conflict resolution, and explanation generation.

Responsibilities:
- Weighted confidence fusion (noisy-or, weighted average, maximum, Dempster-Shafer)
- Result deduplication across strategies
- Conflict resolution between competing results
- Explanation generation for aggregated results
- Meta-learning integration for strategy usage recording

Architecture:
    ResultAggregator takes multiple ReasoningResult objects from different
    strategies and combines them into a single AggregatedResult with merged
    facts, combined confidence, and unified explanation.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates results from multiple reasoning engines.

    Combines evidence from multiple strategies using configurable aggregation
    methods: noisy-or, weighted average, maximum, or Dempster-Shafer.
    """

    def __init__(
        self,
        signals,
        meta_learning_engine=None,
        self_evaluator=None,
        aggregation_method: str = "noisy_or",
        strategy_weights: Optional[Dict] = None,
    ):
        """
        Initialize Result Aggregator.

        Args:
            signals: KaiSignals for UI updates
            meta_learning_engine: MetaLearningEngine (optional)
            self_evaluator: SelfEvaluator (optional)
            aggregation_method: Method for combining confidences
                               (noisy_or, weighted_avg, max, dempster_shafer)
            strategy_weights: Dict mapping ReasoningStrategy to weights (for weighted_avg)
        """
        self.signals = signals
        self.meta_learning_engine = meta_learning_engine
        self.self_evaluator = self_evaluator
        self.aggregation_method = aggregation_method

        # Import types from orchestrator
        from kai_reasoning_orchestrator import (
            AggregatedResult,
            ReasoningResult,
            ReasoningStrategy,
        )

        self.AggregatedResult = AggregatedResult
        self.ReasoningResult = ReasoningResult
        self.ReasoningStrategy = ReasoningStrategy

        # Default strategy weights for weighted_avg aggregation
        if strategy_weights is None:
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
        else:
            self.strategy_weights = strategy_weights

        # Import unified proof system
        try:
            from component_17_proof_explanation import merge_proof_trees

            self.merge_proof_trees = merge_proof_trees
            self.PROOF_SYSTEM_AVAILABLE = True
        except ImportError:
            self.PROOF_SYSTEM_AVAILABLE = False

        logger.info(
            f"ResultAggregator initialized: aggregation={self.aggregation_method}"
        )

    def aggregate_results(self, results: List, emit_proof_tree: bool = True):
        """
        Aggregate multiple reasoning results.

        Args:
            results: List of ReasoningResult objects
            emit_proof_tree: Whether to emit merged proof tree signal

        Returns:
            AggregatedResult with combined evidence

        Raises:
            ValueError: If results list is empty
        """
        if not results:
            raise ValueError("Cannot aggregate empty results")

        logger.debug(
            f"[Result Aggregation] Combining {len(results)} results using {self.aggregation_method}"
        )

        # Combine confidences using selected method
        if self.aggregation_method == "noisy_or":
            confidences = [r.confidence for r in results]
            combined_confidence = self._noisy_or(confidences)
        elif self.aggregation_method == "weighted_avg":
            combined_confidence = self._weighted_average(results)
        elif self.aggregation_method == "max":
            confidences = [r.confidence for r in results]
            combined_confidence = self._maximum(confidences)
        elif self.aggregation_method == "dempster_shafer":
            combined_confidence = self._dempster_shafer(results)
        else:
            logger.warning(
                f"Unknown aggregation method: {self.aggregation_method}, falling back to noisy_or"
            )
            confidences = [r.confidence for r in results]
            combined_confidence = self._noisy_or(confidences)

        # Merge inferred facts (union with deduplication)
        merged_facts = self._merge_facts(results)

        # Merge proof trees
        merged_proof_tree = None
        if self.PROOF_SYSTEM_AVAILABLE:
            merged_proof_tree = self._merge_proof_trees(results)

        # Emit merged proof tree signal
        if merged_proof_tree and self.signals and emit_proof_tree:
            self.signals.proof_tree_update.emit(merged_proof_tree)

        # Generate explanation
        explanation = self._generate_explanation(results, combined_confidence)

        # Check if any result is hypothesis
        is_hypothesis = any(r.is_hypothesis for r in results)

        # Extract strategies used
        strategies_used = [r.strategy for r in results]

        return self.AggregatedResult(
            combined_confidence=combined_confidence,
            inferred_facts=merged_facts,
            merged_proof_tree=merged_proof_tree,
            strategies_used=strategies_used,
            individual_results=results,
            explanation=explanation,
            is_hypothesis=is_hypothesis,
        )

    def evaluate_result_quality(
        self,
        result,
        query: str,
        topic: str,
        context: Dict[str, Any],
    ):
        """
        Evaluates result quality using SelfEvaluator.

        Args:
            result: The AggregatedResult to evaluate
            query: Original query text
            topic: Topic
            context: Context dict

        Returns:
            EvaluationResult from SelfEvaluator
        """
        if not self.self_evaluator:
            # Return dummy result if evaluator not available
            from component_50_self_evaluation import (
                EvaluationResult,
                RecommendationType,
            )

            return EvaluationResult(
                overall_score=0.7,
                checks={},
                uncertainties=[],
                recommendation=RecommendationType.SHOW_TO_USER,
            )

        # Prepare answer dict for evaluator
        answer = {
            "text": result.explanation,
            "confidence": result.combined_confidence,
            "proof_tree": result.merged_proof_tree,
            "reasoning_paths": [],  # TODO: Extract from result if available
        }

        # Evaluate
        eval_result = self.self_evaluator.evaluate_answer(
            question=query, answer=answer, context=context
        )

        return eval_result

    def record_strategy_usage(
        self,
        strategy_name: str,
        query: str,
        result: Optional,
        response_time: float,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Records strategy usage in MetaLearningEngine.

        Args:
            strategy_name: Name of the strategy
            query: Query text
            result: Result (can be None if failed)
            response_time: Time taken in seconds
            success: Whether strategy succeeded
            context: Optional context dict
        """
        if not self.meta_learning_engine:
            return

        # Prepare result dict for MetaLearningEngine
        result_dict = {
            "confidence": result.combined_confidence if result else 0.0,
            "success": success,
        }

        if not success:
            result_dict["error"] = "Strategy failed to produce result"

        # Determine user feedback (for now, automatic based on confidence)
        user_feedback = None
        if result:
            if result.combined_confidence >= 0.8:
                user_feedback = "correct"  # High confidence assumed correct
            elif result.combined_confidence < 0.4:
                user_feedback = "incorrect"  # Low confidence assumed incorrect
            else:
                user_feedback = "neutral"

        # Record usage
        try:
            self.meta_learning_engine.record_strategy_usage(
                strategy=strategy_name,
                query=query,
                result=result_dict,
                response_time=response_time,
                context=context,
                user_feedback=user_feedback,
            )

            logger.debug(
                f"[Meta-Learning] Recorded usage for '{strategy_name}': "
                f"success={success}, confidence={result_dict['confidence']:.2f}, "
                f"time={response_time:.3f}s"
            )

        except Exception as e:
            logger.error(f"Failed to record strategy usage: {e}")

    # ========================================================================
    # Aggregation Methods (Internal)
    # ========================================================================

    def _noisy_or(self, probabilities: List[float]) -> float:
        """
        Noisy-OR combination for redundant evidence.

        P(E | C1, C2, ..., Cn) = 1 - Product(1 - P(E | Ci))

        At least one source is sufficient.
        """
        if not probabilities:
            return 0.0

        product = 1.0
        for p in probabilities:
            product *= 1.0 - p

        return 1.0 - product

    def _weighted_average(self, results: List) -> float:
        """
        Weighted average combination.

        Combined = Sum(wi * Pi) / Sum(wi)

        Uses strategy_weights for weighting.
        """
        if not results:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for result in results:
            weight = self.strategy_weights.get(result.strategy, 0.1)
            weighted_sum += weight * result.confidence
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _maximum(self, probabilities: List[float]) -> float:
        """
        Maximum confidence (best-case scenario).

        Combined = max(P1, P2, ..., Pn)

        Takes the most confident source.
        """
        return max(probabilities) if probabilities else 0.0

    def _dempster_shafer(self, results: List) -> float:
        """
        Dempster-Shafer combination for uncertain evidence.

        Combines belief masses from multiple sources accounting for conflict.

        Simplified implementation:
        m1 ⊕ m2 = (m1 * m2) / (1 - K)
        where K = conflict mass
        """
        if not results:
            return 0.0

        if len(results) == 1:
            return results[0].confidence

        # Initialize with first result
        combined_belief = results[0].confidence
        combined_disbelief = 1.0 - results[0].confidence

        # Combine with subsequent results
        for result in results[1:]:
            belief = result.confidence
            disbelief = 1.0 - result.confidence

            # Calculate conflict
            conflict = combined_belief * disbelief + combined_disbelief * belief

            if conflict >= 1.0:
                # Total conflict - fall back to noisy-or
                logger.warning(
                    "Dempster-Shafer: Total conflict detected, falling back to Noisy-OR"
                )
                return self._noisy_or([r.confidence for r in results])

            # Combine beliefs
            new_belief = (combined_belief * belief) / (1.0 - conflict)
            new_disbelief = (combined_disbelief * disbelief) / (1.0 - conflict)

            combined_belief = new_belief
            combined_disbelief = new_disbelief

        return combined_belief

    # ========================================================================
    # Fact Merging (Internal)
    # ========================================================================

    def _merge_facts(self, results: List) -> Dict[str, List[str]]:
        """
        Merge inferred facts from multiple results.

        Performs union with deduplication. Facts appearing in multiple
        results are included once, preserving confidence information in
        the result metadata.

        Args:
            results: List of ReasoningResult objects

        Returns:
            Dict mapping relation types to lists of unique objects
        """
        merged_facts = {}

        for result in results:
            for rel_type, objects in result.inferred_facts.items():
                if rel_type not in merged_facts:
                    merged_facts[rel_type] = []

                for obj in objects:
                    if obj not in merged_facts[rel_type]:
                        merged_facts[rel_type].append(obj)

        logger.debug(
            f"[Fact Merging] Merged {sum(len(v) for v in merged_facts.values())} unique facts "
            f"from {len(results)} results"
        )

        return merged_facts

    def _resolve_conflicts(
        self, results: List, merged_facts: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Resolve conflicts between competing facts.

        When multiple strategies produce contradictory facts for the same
        relation type, keep the facts from the highest-confidence strategy.

        Args:
            results: List of ReasoningResult objects
            merged_facts: Initial merged facts (may contain conflicts)

        Returns:
            Dict with conflicts resolved (highest confidence wins)
        """
        # For now, we use simple union - conflicts are rare with our relation types
        # In future, could implement:
        # 1. Detect contradictions (e.g., "hund IS_A katze" vs "hund IS_A säugetier")
        # 2. Keep fact from highest-confidence strategy
        # 3. Mark conflicting facts in metadata

        # Simple conflict detection: Check for mutually exclusive relations
        # Example: NORTH_OF vs SOUTH_OF (spatial relations)

        # TODO: Implement conflict detection for specific relation types
        # For now, return merged facts as-is (union strategy)

        return merged_facts

    # ========================================================================
    # Proof Tree Merging (Internal)
    # ========================================================================

    def _merge_proof_trees(self, results: List):
        """
        Merge proof trees from multiple results.

        Uses component_17's merge_proof_trees() for unified merging.

        Args:
            results: List of ReasoningResult objects

        Returns:
            Merged ProofTree or None
        """
        if not self.PROOF_SYSTEM_AVAILABLE:
            return None

        proof_trees = [r.proof_tree for r in results if r.proof_tree]

        if not proof_trees:
            return None

        # Use query from first proof tree
        query = proof_trees[0].query

        # Merge all proof trees
        try:
            merged = self.merge_proof_trees(proof_trees, query)
            logger.debug(
                f"[Proof Merging] Merged {len(proof_trees)} proof trees into unified tree"
            )
            return merged
        except Exception as e:
            logger.warning(f"[Proof Merging] Failed to merge proof trees: {e}")
            return proof_trees[0]  # Fallback to first proof tree

    # ========================================================================
    # Explanation Generation (Internal)
    # ========================================================================

    def _generate_explanation(self, results: List, combined_confidence: float) -> str:
        """
        Generate natural language explanation for aggregated result.

        Args:
            results: List of ReasoningResult objects
            combined_confidence: Combined confidence score

        Returns:
            Natural language explanation string
        """
        # Extract strategy names
        strategies_used = [r.strategy for r in results]

        # FIX: strategy kann String oder Enum sein
        strategy_names = ", ".join(
            s.value if hasattr(s, "value") else str(s) for s in strategies_used
        )

        # Build explanation
        explanation_parts = []

        # Header
        explanation_parts.append(
            f"Kombiniertes Ergebnis aus {len(results)} Strategien ({strategy_names})."
        )

        # Confidence
        explanation_parts.append(f"Kombinierte Konfidenz: {combined_confidence:.2f}")

        # Individual strategy contributions
        if len(results) > 1:
            contributions = []
            for result in results:
                strategy_name = (
                    result.strategy.value
                    if hasattr(result.strategy, "value")
                    else str(result.strategy)
                )
                contributions.append(f"{strategy_name} ({result.confidence:.2f})")

            explanation_parts.append(f"Beitraege: {', '.join(contributions)}")

        # Hypothesis flag
        if any(r.is_hypothesis for r in results):
            explanation_parts.append(
                "Enthält hypothetische Annahmen (abduktives Reasoning)."
            )

        # Aggregation method
        explanation_parts.append(f"Aggregationsmethode: {self.aggregation_method}")

        return " ".join(explanation_parts)

    def _generate_detailed_explanation(
        self, results: List, merged_facts: Dict[str, List[str]]
    ) -> str:
        """
        Generate detailed explanation with fact-level attribution.

        Shows which strategies contributed which facts.

        Args:
            results: List of ReasoningResult objects
            merged_facts: Merged facts dict

        Returns:
            Detailed multi-line explanation
        """
        lines = []

        lines.append("[Detaillierte Analyse]")
        lines.append("")

        # Fact attribution
        for rel_type, objects in merged_facts.items():
            lines.append(f"{rel_type}:")

            for obj in objects:
                # Find which strategies contributed this fact
                contributing_strategies = []
                for result in results:
                    if rel_type in result.inferred_facts:
                        if obj in result.inferred_facts[rel_type]:
                            strategy_name = (
                                result.strategy.value
                                if hasattr(result.strategy, "value")
                                else str(result.strategy)
                            )
                            contributing_strategies.append(
                                f"{strategy_name} ({result.confidence:.2f})"
                            )

                lines.append(
                    f"  - {obj} (Quellen: {', '.join(contributing_strategies)})"
                )

            lines.append("")

        # Strategy summary
        lines.append("[Strategie-Zusammenfassung]")
        for result in results:
            strategy_name = (
                result.strategy.value
                if hasattr(result.strategy, "value")
                else str(result.strategy)
            )
            lines.append(
                f"- {strategy_name}: Konfidenz {result.confidence:.2f}, "
                f"{sum(len(v) for v in result.inferred_facts.values())} Fakten"
            )

        return "\n".join(lines)
