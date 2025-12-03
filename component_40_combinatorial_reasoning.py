"""
component_40_combinatorial_reasoning.py

General-Purpose Combinatorial Reasoning Engine

Implements reasoning about combinatorial structures:
- Permutations and cycle analysis
- Strategy evaluation and comparison
- Probabilistic combinatorics
- Asymptotic analysis

IMPORTANT: Completely generic - no puzzle-specific logic!

Applications:
- Strategy puzzles with permutations
- Optimization problems with combinatorial constraints
- Probabilistic reasoning over combinatorial spaces
- Multi-agent coordination problems

Author: KAI Development Team
Created: 2025-11-04
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations as iter_permutations
from typing import Any, Callable, Dict, List, Optional, Tuple

from component_15_logging_config import get_logger
from component_16_probabilistic_engine import ProbabilisticEngine
from component_17_proof_explanation import ProofStep, ProofTree, StepType
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult

logger = get_logger(__name__)


# ============================================================================
# Permutation and Cycle Analysis
# ============================================================================


@dataclass
class Permutation:
    """
    Represents a permutation as a mapping.

    Attributes:
        mapping: Dict mapping elements to their image under permutation
        elements: Set of all elements in the permutation

    Example:
        Permutation({0: 2, 1: 0, 2: 1}) represents the permutation
        that sends 0->2, 1->0, 2->1
    """

    mapping: Dict[int, int]

    def __post_init__(self):
        """Validate permutation is bijective with comprehensive checks."""
        if not self.mapping:
            raise ValueError("Permutation cannot be empty")

        elements = set(self.mapping.keys())
        images = set(self.mapping.values())

        if elements != images:
            raise ValueError(f"Permutation must be bijective: {self.mapping}")

        # Validate all integers
        if not all(
            isinstance(k, int) and isinstance(v, int) for k, v in self.mapping.items()
        ):
            raise ValueError("Permutation keys and values must be integers")

        # Validate non-negative (required by design)
        if min(elements) < 0:
            raise ValueError("Permutation elements must be non-negative")

        # Validate contiguity (required for cycle algorithms)
        expected = set(range(len(elements)))
        if elements != expected:
            raise ValueError(
                f"Permutation must have contiguous domain starting at 0. "
                f"Expected {expected}, got {elements}"
            )

        self.elements = elements

    @classmethod
    def identity(cls, n: int) -> "Permutation":
        """Create identity permutation on {0, 1, ..., n-1}."""
        return cls({i: i for i in range(n)})

    @classmethod
    def from_list(cls, lst: List[int]) -> "Permutation":
        """
        Create permutation from list representation.

        Args:
            lst: List where lst[i] is the image of i

        Example:
            from_list([2, 0, 1]) creates permutation 0->2, 1->0, 2->1
        """
        return cls({i: lst[i] for i in range(len(lst))})

    def __call__(self, x: int) -> int:
        """Apply permutation to element."""
        return self.mapping[x]

    def compose(self, other: "Permutation") -> "Permutation":
        """Compose this permutation with another (self ∘ other)."""
        return Permutation({k: self(other(k)) for k in self.elements})

    def inverse(self) -> "Permutation":
        """Compute inverse permutation."""
        return Permutation({v: k for k, v in self.mapping.items()})

    def __len__(self) -> int:
        """Number of elements in permutation."""
        return len(self.elements)


@dataclass
class Cycle:
    """
    Represents a cycle in a permutation.

    A cycle is a sequence of elements [a1, a2, ..., ak] where
    a1 -> a2 -> ... -> ak -> a1

    Attributes:
        elements: List of elements in cycle order
        length: Number of elements in cycle
    """

    elements: List[int]

    def __post_init__(self):
        self.length = len(self.elements)

    def __len__(self) -> int:
        return self.length

    def contains(self, x: int) -> bool:
        """Check if element is in this cycle."""
        return x in self.elements

    def __repr__(self) -> str:
        return f"Cycle({self.elements})"


class CycleAnalyzer:
    """
    Analyzes cycle structure of permutations.

    Key algorithms:
    - Decompose permutation into disjoint cycles
    - Compute cycle length distribution
    - Calculate probabilities based on cycle structure
    """

    @staticmethod
    def find_cycles(perm: Permutation) -> List[Cycle]:
        """
        Decompose permutation into disjoint cycles.

        Algorithm:
        1. Start with unvisited element
        2. Follow permutation until returning to start
        3. Record cycle and mark elements as visited
        4. Repeat until all elements visited

        Args:
            perm: Permutation to decompose

        Returns:
            List of disjoint cycles

        Time Complexity: O(n) where n = len(perm)
        """
        visited = set()
        cycles = []

        for start in perm.elements:
            if start in visited:
                continue

            # Trace cycle starting from 'start'
            cycle_elements = []
            current = start
            while current not in visited:
                cycle_elements.append(current)
                visited.add(current)
                current = perm(current)

            if len(cycle_elements) > 0:
                cycles.append(Cycle(cycle_elements))

        logger.debug(
            "Decomposed permutation into cycles",
            extra={
                "num_cycles": len(cycles),
                "cycle_lengths": [len(c) for c in cycles],
            },
        )

        return cycles

    @staticmethod
    def cycle_length_distribution(perm: Permutation) -> Dict[int, int]:
        """
        Compute distribution of cycle lengths.

        Returns:
            Dict mapping cycle_length -> count

        Example:
            {1: 2, 3: 1} means 2 cycles of length 1, 1 cycle of length 3
        """
        cycles = CycleAnalyzer.find_cycles(perm)
        distribution: Dict[int, int] = defaultdict(int)
        for cycle in cycles:
            distribution[len(cycle)] += 1
        return dict(distribution)

    @staticmethod
    def max_cycle_length(perm: Permutation) -> int:
        """Find length of longest cycle in permutation."""
        cycles = CycleAnalyzer.find_cycles(perm)
        return max(len(c) for c in cycles) if cycles else 0

    @staticmethod
    def find_element_cycle(perm: Permutation, element: int) -> Cycle:
        """
        Find the cycle containing a specific element.

        Args:
            perm: Permutation to search
            element: Element to find

        Returns:
            Cycle containing the element

        Raises:
            ValueError: If element not in permutation

        Time Complexity: O(k) where k is the cycle length
        Space Complexity: O(k)

        Note:
            For multiple elements, consider using find_cycles() once
            to avoid redundant traversals.
        """
        if element not in perm.elements:
            raise ValueError(f"Element {element} not in permutation")

        cycle_elements = []
        current = element
        while True:
            cycle_elements.append(current)
            current = perm(current)
            if current == element:
                break

        return Cycle(cycle_elements)


# ============================================================================
# Strategy Framework
# ============================================================================


@dataclass
class StrategyParameter:
    """
    Parameter for a strategy.

    Attributes:
        name: Parameter name
        value: Parameter value
        description: Human-readable description
    """

    name: str
    value: Any
    description: str = ""


@dataclass
class Strategy:
    """
    Represents a decision-making strategy.

    A strategy is a function that maps a problem state to an action/decision.

    Attributes:
        name: Human-readable strategy name
        description: Strategy explanation
        parameters: Strategy-specific parameters
        decision_function: Function that executes strategy
        metadata: Additional strategy information
    """

    name: str
    description: str
    parameters: List[StrategyParameter] = field(default_factory=list)
    decision_function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decide(self, state: Dict[str, Any]) -> Any:
        """
        Execute strategy to make decision.

        NOTE: Currently not used in codebase. Strategy evaluation uses
        metadata["evaluation_function"] instead. This method is available
        for future use cases where decision-making is needed separate from
        evaluation (e.g., online decision-making vs. offline analysis).

        Args:
            state: Current problem state

        Returns:
            Decision/action chosen by strategy
        """
        if self.decision_function is None:
            raise NotImplementedError(
                f"Strategy '{self.name}' has no decision function"
            )
        return self.decision_function(state, self.parameters)

    def get_parameter(self, name: str) -> Optional[Any]:
        """Get value of parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param.value
        return None


@dataclass
class StrategyEvaluation:
    """
    Evaluation result for a strategy.

    Attributes:
        strategy: The strategy being evaluated
        success_probability: Probability of success
        expected_value: Expected utility/payoff
        proof_tree: Explanation of evaluation
        metrics: Additional evaluation metrics
    """

    strategy: Strategy
    success_probability: float
    expected_value: float
    proof_tree: Optional[ProofTree] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate probabilities and expected values."""
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError(
                f"success_probability must be in [0, 1]: {self.success_probability}"
            )

        # Validate expected_value is numeric (may represent utility, not just probability)
        if not isinstance(self.expected_value, (int, float)):
            raise TypeError(
                f"expected_value must be numeric, got {type(self.expected_value)}"
            )


class StrategyEvaluator:
    """
    Evaluates and compares strategies.

    Key capabilities:
    - Compute success probability of strategies
    - Compare strategies based on expected utility
    - Generate explanations for strategy evaluations
    """

    def __init__(self):
        self.prob_engine = ProbabilisticEngine()

    def evaluate_strategy(
        self,
        strategy: Strategy,
        problem_state: Dict[str, Any],
        success_criterion: Callable[[Any], bool],
    ) -> StrategyEvaluation:
        """
        Evaluate strategy on a problem.

        Args:
            strategy: Strategy to evaluate
            problem_state: Initial problem state
            success_criterion: Function determining if outcome is success

        Returns:
            StrategyEvaluation with probability and explanation
        """
        proof_steps = []

        # Step 1: Describe strategy
        proof_steps.append(
            ProofStep(
                step_id=f"eval-{strategy.name}-premise",
                step_type=StepType.PREMISE,
                output=f"Evaluating strategy: {strategy.name}",
                explanation_text=f"Strategy: {strategy.description}",
                metadata={
                    "strategy_description": strategy.description,
                    "parameters": {p.name: p.value for p in strategy.parameters},
                },
                source_component="combinatorial_reasoning",
            )
        )

        # Step 2: Execute strategy (delegation to subclass/specific evaluator)
        # This is where domain-specific logic would be injected
        if "evaluation_function" in strategy.metadata:
            eval_func = strategy.metadata["evaluation_function"]
            success_prob, expected_val = eval_func(problem_state, strategy)
        else:
            # Default: uniform random assumption
            logger.warning(
                "No evaluation function provided - using default uniform assumption",
                extra={"strategy": strategy.name},
            )
            success_prob = 0.5
            expected_val = 0.5

        proof_steps.append(
            ProofStep(
                step_id=f"eval-{strategy.name}-result",
                step_type=StepType.INFERENCE,
                output=f"Success probability: {success_prob:.4f}",
                explanation_text=f"Computed success probability: {success_prob:.4f}",
                confidence=success_prob,
                metadata={"probability": success_prob, "expected_value": expected_val},
                parent_steps=[f"eval-{strategy.name}-premise"],
                source_component="combinatorial_reasoning",
            )
        )

        proof_tree = ProofTree(
            query=f"Evaluate strategy: {strategy.name}",
            root_steps=proof_steps,
            metadata={
                "conclusion": f"Strategy '{strategy.name}' has success probability {success_prob:.4f}"
            },
        )

        return StrategyEvaluation(
            strategy=strategy,
            success_probability=success_prob,
            expected_value=expected_val,
            proof_tree=proof_tree,
        )

    def compare_strategies(
        self, evaluations: List[StrategyEvaluation]
    ) -> Tuple[StrategyEvaluation, ProofTree]:
        """
        Compare multiple strategy evaluations and find best.

        Args:
            evaluations: List of strategy evaluations

        Returns:
            (best_evaluation, comparison_proof)
        """
        if not evaluations:
            raise ValueError("No strategies to compare")

        proof_steps = []

        # List all strategies
        proof_steps.append(
            ProofStep(
                step_id="compare-premise",
                step_type=StepType.PREMISE,
                output="Comparing strategies",
                explanation_text=f"Comparing {len(evaluations)} strategies",
                metadata={
                    "strategies": [
                        {
                            "name": ev.strategy.name,
                            "success_prob": ev.success_probability,
                            "expected_value": ev.expected_value,
                        }
                        for ev in evaluations
                    ]
                },
                source_component="combinatorial_reasoning",
            )
        )

        # Find best by success probability
        best = max(evaluations, key=lambda ev: ev.success_probability)

        proof_steps.append(
            ProofStep(
                step_id="compare-conclusion",
                step_type=StepType.CONCLUSION,
                output=f"Best strategy: {best.strategy.name}",
                explanation_text=f"Strategy '{best.strategy.name}' has highest success probability",
                confidence=best.success_probability,
                metadata={
                    "best_strategy": best.strategy.name,
                    "success_probability": best.success_probability,
                    "advantage": best.success_probability
                    - min(ev.success_probability for ev in evaluations),
                },
                parent_steps=["compare-premise"],
                source_component="combinatorial_reasoning",
            )
        )

        comparison_proof = ProofTree(
            query="Compare strategies and find optimal",
            root_steps=proof_steps,
            metadata={
                "conclusion": f"Strategy '{best.strategy.name}' is optimal with P(success) = {best.success_probability:.4f}"
            },
        )

        return best, comparison_proof


# ============================================================================
# Probabilistic Combinatorics
# ============================================================================


class CombinatorialProbability:
    """
    Computes probabilities over combinatorial structures.

    Key methods:
    - Cycle length probabilities in random permutations
    - Success probabilities for strategies over permutations
    - Asymptotic approximations
    """

    @staticmethod
    def factorial(n: int) -> int:
        """Compute n! with memoization."""
        if n <= 1:
            return 1
        return math.factorial(n)

    @staticmethod
    def binomial(n: int, k: int) -> int:
        """Compute binomial coefficient C(n, k)."""
        if k < 0 or k > n:
            return 0
        return math.comb(n, k)

    @staticmethod
    def prob_max_cycle_exceeds_threshold(n: int, threshold: int) -> float:
        """
        Probability that max cycle length in random permutation exceeds threshold.

        For random permutation of n elements:
        P(max cycle length > threshold) = ?

        Exact computation expensive, uses approximation for large n.

        Args:
            n: Permutation size
            threshold: Cycle length threshold

        Returns:
            Probability that max cycle > threshold
        """
        if threshold >= n:
            return 0.0
        if threshold <= 0:
            return 1.0

        # Use complement: P(max ≤ threshold) = P(all cycles ≤ threshold)
        # This is hard to compute exactly, use harmonic approximation

        # For large n: P(max cycle > n/2) ≈ 1 - ln(2) ≈ 0.307
        # More generally: P(max cycle ≤ k) ≈ exp(-sum_{i=k+1}^{n} 1/i)

        if n <= 10:
            # Small n: exact computation via enumeration
            return CombinatorialProbability._exact_max_cycle_prob(n, threshold)
        else:
            # Large n: use asymptotic approximation
            return CombinatorialProbability._asymptotic_max_cycle_prob(n, threshold)

    @staticmethod
    @lru_cache(maxsize=128)
    def _exact_max_cycle_prob(n: int, threshold: int) -> float:
        """
        Exact computation via permutation enumeration (expensive!).

        Cached for performance - results are deterministic.
        Only use for small n (<=10).
        """
        MAX_EXACT_N = 12  # 12! = 479,001,600 permutations (~reasonable limit)

        if n > MAX_EXACT_N:
            raise ValueError(
                f"Exact computation only supported for n <= {MAX_EXACT_N}. "
                f"For n={n}, use asymptotic approximation instead."
            )
        elif n > 10:
            logger.warning(
                "Exact computation expensive for n > 10",
                extra={
                    "n": n,
                    "threshold": threshold,
                    "permutations": math.factorial(n),
                },
            )

        count_exceeds = 0
        total = 0

        # Generate all permutations
        elements = list(range(n))
        for perm_tuple in iter_permutations(elements):
            perm = Permutation.from_list(list(perm_tuple))
            max_cycle = CycleAnalyzer.max_cycle_length(perm)
            if max_cycle > threshold:
                count_exceeds += 1
            total += 1

        return count_exceeds / total if total > 0 else 0.0

    @staticmethod
    def _asymptotic_max_cycle_prob(n: int, threshold: int) -> float:
        """
        Asymptotic approximation for large n.

        For random permutation of n elements, the probability that
        ALL agents succeed (i.e., max cycle ≤ k) is approximately:

        P(max cycle ≤ k) ≈ product_{i=k+1}^{n} (1 - 1/i) for large n

        For the special case k = n/2:
        P(max cycle ≤ n/2) ≈ 1 - ln(2) ≈ 0.3118

        Therefore:
        P(max cycle > n/2) ≈ ln(2) ≈ 0.6931

        BUT the famous result for 100 Prisoners Problem states:
        P(success) = P(max cycle ≤ 50) ≈ 0.31

        So: P(max cycle > 50) ≈ 0.69

        General formula (approximate):
        P(max cycle ≤ k) ≈ sum_{i=1}^{k} (-1)^{i+1} / i  (alternating harmonic)
        """
        if threshold >= n:
            return 0.0
        if threshold <= 0:
            return 1.0

        # Special case: k = n/2
        if threshold == n // 2:
            # Known result: P(max cycle ≤ n/2) ≈ 1 - ln(2)
            prob_not_exceed = 1.0 - math.log(2)
            return 1.0 - prob_not_exceed  # P(max cycle > n/2) ≈ ln(2)

        # General approximation: use alternating harmonic series
        # P(max cycle ≤ k) ≈ sum_{i=k+1}^{n} (-1)^{i-k} / i
        # This is a rough approximation based on Stirling numbers

        # For large k relative to n:
        if threshold > n * 0.9:
            return 0.0

        # For small k relative to n:
        if threshold < n * 0.1:
            return 1.0

        # Linear interpolation between known points for approximation
        # At k = n/2: P(max > k) ≈ ln(2) ≈ 0.693
        # At k = n: P(max > k) = 0
        # At k = 0: P(max > k) = 1
        #
        # TODO: This linear interpolation is mathematically crude and may introduce
        # significant errors for intermediate threshold values. Consider implementing
        # proper alternating harmonic series approximation or Stirling number formula.
        # Reference: Flajolet & Sedgewick "Analytic Combinatorics" (2009)

        if threshold <= n / 2:
            # Between 0 and n/2: interpolate between 1 and ln(2)
            ratio = threshold / (n / 2)
            prob_exceed = 1.0 - ratio * (1.0 - math.log(2))
        else:
            # Between n/2 and n: interpolate between ln(2) and 0
            ratio = (threshold - n / 2) / (n / 2)
            prob_exceed = math.log(2) * (1.0 - ratio)

        return prob_exceed

    @staticmethod
    @lru_cache(maxsize=128)
    def expected_max_cycle_length(n: int) -> float:
        """
        Expected maximum cycle length in random permutation of n elements.

        Asymptotically: E[max cycle] ≈ λ * n where λ ≈ 0.624 (Golomb-Dickman)

        Cached for performance - results are deterministic.
        For small n, use exact computation.
        """
        if n <= 1:
            return float(n)

        if n <= 10:
            # Exact computation
            total_max = 0
            count = 0
            elements = list(range(n))
            for perm_tuple in iter_permutations(elements):
                perm = Permutation.from_list(list(perm_tuple))
                total_max += CycleAnalyzer.max_cycle_length(perm)
                count += 1
            return total_max / count if count > 0 else 0.0
        else:
            # Asymptotic: approximately 0.624 * n (Golomb-Dickman constant)
            return 0.624 * n


# ============================================================================
# Combinatorial Reasoning Engine
# ============================================================================


class CombinatorialReasoner(BaseReasoningEngine):
    """
    Main reasoning engine for combinatorial problems.

    Capabilities:
    - Analyze permutation structures
    - Evaluate and compare strategies
    - Compute probabilities over combinatorial spaces
    - Generate explanations for conclusions
    """

    def __init__(self):
        self.cycle_analyzer = CycleAnalyzer()
        self.strategy_evaluator = StrategyEvaluator()
        self.prob_calculator = CombinatorialProbability()
        logger.info("CombinatorialReasoner initialized")

    def analyze_permutation(
        self, perm: Permutation
    ) -> Tuple[List[Cycle], Dict[str, Any]]:
        """
        Comprehensive permutation analysis.

        Returns:
            (cycles, analysis_dict)

        analysis_dict contains:
        - num_cycles: Total number of cycles
        - cycle_lengths: List of cycle lengths
        - max_cycle_length: Length of longest cycle
        - distribution: Cycle length distribution
        """
        cycles = self.cycle_analyzer.find_cycles(perm)

        analysis = {
            "num_cycles": len(cycles),
            "cycle_lengths": [len(c) for c in cycles],
            "max_cycle_length": max(len(c) for c in cycles) if cycles else 0,
            "distribution": self.cycle_analyzer.cycle_length_distribution(perm),
        }

        logger.info(
            "Analyzed permutation",
            extra={
                "perm_size": len(perm),
                "num_cycles": analysis["num_cycles"],
                "max_cycle": analysis["max_cycle_length"],
            },
        )

        return cycles, analysis

    def evaluate_strategy_on_permutations(
        self, strategy: Strategy, n: int, num_samples: int = 100
    ) -> StrategyEvaluation:
        """
        Evaluate strategy performance on random permutations.

        Args:
            strategy: Strategy to evaluate
            n: Size of permutations
            num_samples: Number of random permutations to test

        Returns:
            StrategyEvaluation with empirical success probability
        """
        # Generate random permutations and test strategy
        import random

        successes = 0
        for _ in range(num_samples):
            # Generate random permutation
            elements = list(range(n))
            random.shuffle(elements)
            perm = Permutation.from_list(elements)

            # Test strategy (requires strategy to have evaluation logic)
            if "permutation_evaluator" in strategy.metadata:
                eval_func = strategy.metadata["permutation_evaluator"]
                if eval_func(perm, strategy):
                    successes += 1
            else:
                logger.warning(
                    "Strategy has no permutation evaluator - skipping",
                    extra={"strategy": strategy.name},
                )
                return StrategyEvaluation(
                    strategy=strategy, success_probability=0.0, expected_value=0.0
                )

        success_prob = successes / num_samples

        return StrategyEvaluation(
            strategy=strategy,
            success_probability=success_prob,
            expected_value=success_prob,
            metrics={"num_samples": num_samples, "successes": successes},
        )

    def find_optimal_strategy(
        self, strategies: List[Strategy], problem_state: Dict[str, Any]
    ) -> Tuple[Strategy, ProofTree]:
        """
        Find optimal strategy among candidates.

        Args:
            strategies: List of candidate strategies
            problem_state: Problem specification

        Returns:
            (optimal_strategy, explanation_proof)
        """
        # Evaluate all strategies
        evaluations = []
        for strategy in strategies:
            eval_result = self.strategy_evaluator.evaluate_strategy(
                strategy=strategy,
                problem_state=problem_state,
                success_criterion=lambda x: True,  # Delegated to strategy
            )
            evaluations.append(eval_result)

        # Compare and find best
        best_eval, comparison_proof = self.strategy_evaluator.compare_strategies(
            evaluations
        )

        logger.info(
            "Found optimal strategy",
            extra={
                "strategy": best_eval.strategy.name,
                "success_prob": best_eval.success_probability,
            },
        )

        return best_eval.strategy, comparison_proof

    # ========================================================================
    # BaseReasoningEngine Interface Implementation
    # ========================================================================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Execute combinatorial reasoning on the query.

        Args:
            query: Natural language query
            context: Context with:
                - "strategy": Strategy object to evaluate
                - "n": Size of permutation space (default: 100)
                - "problem_state": Problem specification
                - "strategies": List of strategies to compare (for optimization)

        Returns:
            ReasoningResult with analysis and proof tree
        """
        try:
            # Check if we're comparing strategies (optimization task)
            if "strategies" in context:
                strategies = context["strategies"]
                problem_state = context.get("problem_state", {})

                best_strategy, proof = self.find_optimal_strategy(
                    strategies, problem_state
                )

                return ReasoningResult(
                    success=True,
                    answer=f"Optimal strategy: {best_strategy.name} (probability: {best_strategy.metadata.get('probability', 'N/A')})",
                    confidence=1.0,
                    proof_tree=proof,
                    strategy_used="combinatorial_strategy_optimization",
                    metadata={
                        "best_strategy": best_strategy.name,
                        "num_strategies_compared": len(strategies),
                    },
                )

            # Check if we're analyzing a single strategy
            elif "strategy" in context:
                strategy = context["strategy"]
                n = context.get("n", 100)

                eval_result = self.evaluate_strategy_on_permutations(strategy, n)

                return ReasoningResult(
                    success=True,
                    answer=f"Strategy {strategy.name} success probability: {eval_result.success_probability:.4f}",
                    confidence=1.0,
                    strategy_used="combinatorial_strategy_evaluation",
                    metadata={
                        "strategy_name": strategy.name,
                        "success_probability": eval_result.success_probability,
                        "expected_value": eval_result.expected_value,
                        "n": n,
                    },
                )

            # Check if we're analyzing a permutation
            elif "permutation" in context:
                perm = context["permutation"]
                cycles, analysis = self.analyze_permutation(perm)

                answer = (
                    f"Permutation analysis: {analysis['num_cycles']} cycles, "
                    f"max cycle length {analysis['max_cycle_length']}"
                )

                return ReasoningResult(
                    success=True,
                    answer=answer,
                    confidence=1.0,
                    strategy_used="combinatorial_permutation_analysis",
                    metadata=analysis,
                )

            else:
                # No valid context provided
                logger.warning(
                    "CombinatorialReasoner.reason() called without valid context",
                    extra={"query": query, "context_keys": list(context.keys())},
                )
                return ReasoningResult(
                    success=False,
                    answer="Insufficient context for combinatorial reasoning",
                    confidence=0.0,
                    strategy_used="combinatorial_reasoning",
                )

        except Exception as e:
            logger.error(
                "Error in combinatorial reasoning",
                extra={"query": query, "error": str(e)},
                exc_info=True,
            )
            return ReasoningResult(
                success=False,
                answer=f"Combinatorial reasoning error: {str(e)}",
                confidence=0.0,
                strategy_used="combinatorial_reasoning",
            )

    def get_capabilities(self) -> List[str]:
        """Return list of reasoning capabilities."""
        return [
            "combinatorial",
            "permutation_analysis",
            "strategy_evaluation",
            "probabilistic_combinatorics",
            "cycle_analysis",
            "strategy_optimization",
        ]

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for combinatorial reasoning.

        Returns:
            Cost estimate in [0.0, 1.0] range
            Base cost: 0.6 (medium-expensive due to combinatorial complexity)
        """
        # Combinatorial reasoning can be expensive due to:
        # - Permutation generation and analysis
        # - Strategy evaluation on multiple samples
        # - Cycle decomposition algorithms
        base_cost = 0.6

        # Query length adds minimal complexity
        query_complexity = min(len(query) / 500.0, 0.1)

        return min(base_cost + query_complexity, 1.0)


# ============================================================================
# Integration with KAI
# ============================================================================


def create_permutation_from_mapping(mapping_dict: Dict[int, int]) -> Permutation:
    """Helper to create Permutation from dict (for testing/integration)."""
    return Permutation(mapping_dict)


def create_strategy(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    evaluation_function: Optional[Callable] = None,
) -> Strategy:
    """
    Helper to create Strategy (for testing/integration).

    Args:
        name: Strategy name
        description: Strategy explanation
        parameters: Dict of parameter_name -> parameter_value
        evaluation_function: Optional function for evaluation

    Returns:
        Strategy object
    """
    param_list = [
        StrategyParameter(name=k, value=v, description=f"Parameter {k}")
        for k, v in parameters.items()
    ]

    metadata = {}
    if evaluation_function is not None:
        metadata["evaluation_function"] = evaluation_function

    return Strategy(
        name=name, description=description, parameters=param_list, metadata=metadata
    )


# ============================================================================
# Example Usage (for documentation)
# ============================================================================


if __name__ == "__main__":
    # Example: Analyze a permutation
    perm = Permutation.from_list([2, 0, 3, 1, 4])  # 0->2->3->1->0, 4->4
    reasoner = CombinatorialReasoner()
    cycles, analysis = reasoner.analyze_permutation(perm)
    print(f"Cycles: {cycles}")
    print(f"Analysis: {analysis}")

    # Example: Evaluate strategy
    def random_strategy_eval(state, strategy):
        """Example evaluation function for random strategy."""
        n = state.get("n", 100)
        state.get("threshold", 50)
        # Random strategy: each trial has 50% success
        return 0.5**n, 0.5**n

    random_strategy = create_strategy(
        name="Random Strategy",
        description="Choose randomly at each step",
        parameters={"choices": 2},
        evaluation_function=random_strategy_eval,
    )

    problem = {"n": 10, "threshold": 5}
    eval_result = reasoner.strategy_evaluator.evaluate_strategy(
        random_strategy, problem, lambda x: True
    )
    print(f"Random strategy success probability: {eval_result.success_probability}")
