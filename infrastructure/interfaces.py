"""
infrastructure/interfaces.py

Base interfaces for all reasoning engines in the KAI system.

This module defines the common interface that all reasoning engines must implement,
enabling polymorphic usage and consistent orchestration across different reasoning
strategies.

Interface Contract:
    All reasoning engines (Logic, Graph, Abductive, Probabilistic, Spatial, etc.)
    must implement the BaseReasoningEngine abstract base class. This ensures:
    - Consistent API for reasoning operations
    - Standardized result format with proof trees
    - Cost estimation for orchestration decisions
    - Capability discovery for query routing

Usage:
    from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult

    class MyCustomEngine(BaseReasoningEngine):
        def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
            # Implementation here
            pass

        def get_capabilities(self) -> List[str]:
            return ["custom_reasoning", "pattern_matching"]

        def estimate_cost(self, query: str) -> float:
            return 0.5  # Medium cost
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from component_17_proof_explanation import ProofTree


@dataclass
class ReasoningResult:
    """
    Standardized result container for all reasoning engines.

    All reasoning engines return results in this format to enable
    consistent aggregation and proof tree merging in the orchestrator.

    Attributes:
        success: Whether reasoning found a valid result
        answer: The answer string (empty if no answer found)
        confidence: Confidence score in [0.0, 1.0] range
        proof_tree: Optional proof tree showing reasoning steps
        metadata: Additional engine-specific information
        strategy_used: Name of the reasoning strategy employed
        computation_cost: Actual computational cost (for performance tracking)

    Example:
        result = ReasoningResult(
            success=True,
            answer="Ein Hund ist ein Tier",
            confidence=0.95,
            proof_tree=proof,
            strategy_used="logic_engine_forward_chaining",
            computation_cost=0.12
        )
    """

    success: bool
    answer: str = ""
    confidence: float = 0.0
    proof_tree: Optional[ProofTree] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy_used: str = ""
    computation_cost: float = 0.0

    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )


class BaseReasoningEngine(ABC):
    """
    Abstract base class for all reasoning engines in the KAI system.

    All reasoning engines (LogicEngine, GraphTraversal, AbductiveEngine,
    ProbabilisticEngine, SpatialReasoner, ArithmeticEngine, etc.) should
    inherit from this class and implement the required methods.

    This interface enables:
    - Polymorphic usage in ReasoningOrchestrator
    - Consistent API across all reasoning strategies
    - Cost-based query routing
    - Capability-based strategy selection

    Design Pattern:
        This follows the Strategy pattern, where different reasoning
        approaches are encapsulated as interchangeable strategies with
        a common interface.

    Thread Safety:
        Implementations SHOULD be thread-safe as they may be called
        from multiple worker threads. Use locks for shared mutable state.

    Example Implementation:
        class LogicEngine(BaseReasoningEngine):
            def __init__(self, netzwerk):
                self.netzwerk = netzwerk
                self._lock = threading.RLock()

            def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
                # Extract logical rules, apply inference
                # Build proof tree, return result
                return ReasoningResult(success=True, answer="...", ...)

            def get_capabilities(self) -> List[str]:
                return ["deductive", "forward_chaining", "backward_chaining"]

            def estimate_cost(self, query: str) -> float:
                # Estimate based on rule count, query complexity
                return 0.3
    """

    @abstractmethod
    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Execute reasoning on the given query with provided context.

        This is the core method that performs the actual reasoning operation.
        Implementations should:
        1. Parse/analyze the query
        2. Access knowledge graph if needed
        3. Apply reasoning algorithms
        4. Generate proof tree documenting steps
        5. Return standardized result

        Args:
            query: The query string to reason about (e.g., "Ist ein Hund ein Tier?")
            context: Contextual information for reasoning, may include:
                - "working_memory": Current working memory stack
                - "previous_results": Results from other engines
                - "query_type": Classified query type (QUESTION, COMMAND, etc.)
                - "entities": Detected entities in query
                - Engine-specific parameters

        Returns:
            ReasoningResult with answer, confidence, and proof tree

        Raises:
            ReasoningException: If reasoning fails critically
            DatabaseException: If knowledge graph access fails

        Example:
            context = {
                "working_memory": working_memory,
                "query_type": "QUESTION",
                "entities": ["hund", "tier"]
            }
            result = engine.reason("Ist ein Hund ein Tier?", context)
            if result.success:
                print(f"Answer: {result.answer} (confidence: {result.confidence})")

        Performance Note:
            Implementations should complete within 2 seconds for typical queries.
            For expensive operations, consider caching and early termination.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of reasoning capabilities this engine provides.

        Capabilities are used by the ReasoningOrchestrator to select
        appropriate engines for different query types. Each engine should
        declare what types of reasoning it can perform.

        Returns:
            List of capability identifiers (lowercase, underscore-separated)

        Standard Capabilities:
            - "deductive": Classical deductive inference (A -> B, A |- B)
            - "inductive": Generalization from examples
            - "abductive": Hypothesis generation (best explanation)
            - "analogical": Reasoning by analogy
            - "spatial": Spatial/geometric reasoning
            - "temporal": Time-based reasoning
            - "probabilistic": Bayesian/uncertain reasoning
            - "constraint": Constraint satisfaction
            - "graph_traversal": Multi-hop graph reasoning
            - "arithmetic": Mathematical operations

        Engine-Specific Capabilities:
            Engines can define custom capabilities like:
            - "chess_position_analysis"
            - "logic_puzzle_solving"
            - "causal_chain_inference"

        Example:
            class AbductiveEngine(BaseReasoningEngine):
                def get_capabilities(self) -> List[str]:
                    return [
                        "abductive",
                        "analogical",
                        "hypothesis_generation",
                        "causal_chain_inference"
                    ]
        """
        pass

    @abstractmethod
    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for reasoning about this query.

        Cost estimation enables the ReasoningOrchestrator to make intelligent
        decisions about:
        - Which engines to run in parallel vs. sequentially
        - Resource allocation and timeouts
        - Query routing to most efficient engine

        Cost is a relative measure (not absolute time) where:
        - 0.0 - 0.3: Cheap (direct lookup, cached results)
        - 0.3 - 0.7: Medium (single-hop inference, simple search)
        - 0.7 - 1.0: Expensive (multi-hop reasoning, search, complex algorithms)
        - >1.0: Very expensive (should be avoided for real-time queries)

        Args:
            query: The query string to estimate cost for

        Returns:
            Float representing estimated computational cost [0.0, infinity)
            Typical range is [0.0, 1.0] for normal operations

        Example Implementation:
            def estimate_cost(self, query: str) -> float:
                # Direct fact lookup: very cheap
                if self._is_simple_fact_query(query):
                    return 0.1

                # Rule-based inference: medium
                rule_count = self._count_applicable_rules(query)
                if rule_count < 10:
                    return 0.4
                elif rule_count < 50:
                    return 0.6

                # Complex reasoning: expensive
                return 0.9

        Performance Note:
            This method should complete very quickly (<10ms) as it's called
            during query routing. Avoid expensive operations like full
            graph traversals. Use heuristics and estimates.
        """
        pass

    def supports_capability(self, capability: str) -> bool:
        """
        Check if this engine supports a specific capability.

        Convenience method for capability checking. Default implementation
        checks if capability is in get_capabilities() list.

        Args:
            capability: Capability identifier to check

        Returns:
            True if engine supports this capability, False otherwise

        Example:
            if engine.supports_capability("abductive"):
                result = engine.reason(query, context)
        """
        return capability in self.get_capabilities()
