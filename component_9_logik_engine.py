# component_9_logik_engine.py
"""
Logic Engine - Main Orchestration Module

This module combines all logic engine components into a unified interface:
- Core reasoning engine (forward/backward chaining)
- CSP-based constraint reasoning
- Proof tracking and explanations
- Advanced reasoning (SAT, consistency checking, contradiction detection)

Architecture:
    The LogikEngine class inherits from multiple mixins to provide a comprehensive
    reasoning system while maintaining code modularity and separation of concerns.

Usage:
    from component_9_logik_engine import LogikEngine, Fact, Goal, Rule

    engine = LogikEngine(netzwerk)
    engine.load_rules_from_graph(netzwerk)

    # Add facts
    engine.add_fact(Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}))

    # Forward chaining
    engine.run()

    # Backward chaining
    goal = Goal(pred="IS_A", args={"subject": "hund", "object": "lebewesen"})
    proof = engine.prove_goal(goal)
"""

from typing import Any, Dict, List

from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine_advanced import AdvancedReasoningMixin

# Import core components (data structures, basic engine, utility functions)
from component_9_logik_engine_core import (
    CONSTRAINT_REASONING_AVAILABLE,
    ONTOLOGY_CONSTRAINTS_AVAILABLE,
    PROBABILISTIC_AVAILABLE,
    SAT_SOLVER_AVAILABLE,
    UNIFIED_PROOFS_AVAILABLE,
    Binding,
    Engine,
    Fact,
    Goal,
    ProofStep,
    Rule,
    match_rule,
    resolve,
    unify,
)

# Import mixins
from component_9_logik_engine_csp import CSPReasoningMixin
from component_9_logik_engine_proof import (
    ProofTrackingMixin,
    convert_logic_engine_proof,
    create_proof_tree_from_logic_engine,
)
from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ==================== UNIFIED LOGIC ENGINE ====================


class LogikEngine(
    Engine, CSPReasoningMixin, ProofTrackingMixin, AdvancedReasoningMixin
):
    """
    Unified Logic Engine combining all reasoning capabilities.

    This class inherits from:
    - Engine: Core forward/backward chaining, rule matching, fact management
    - CSPReasoningMixin: Constraint satisfaction problem solving
    - ProofTrackingMixin: Proof generation, tracking, and explanation
    - AdvancedReasoningMixin: SAT solving, consistency checking, contradiction detection

    Features:
    - Forward chaining (run)
    - Backward chaining (prove_goal)
    - Hybrid reasoning (run_with_goal)
    - Episodic memory integration (run_with_tracking)
    - Probabilistic reasoning (run_probabilistic, query_with_uncertainty)
    - CSP-based reasoning (solve_with_constraints)
    - SAT-based consistency checking (check_kb_consistency, find_contradictions)
    - Proof tree generation and validation (validate_inference_chain)
    - Natural language explanations (explain_contradiction)

    Example:
        >>> engine = LogikEngine(netzwerk)
        >>> engine.load_rules_from_graph(netzwerk)
        >>>
        >>> # Add facts
        >>> engine.add_fact(Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}))
        >>>
        >>> # Forward chaining
        >>> engine.run()
        >>>
        >>> # Backward chaining with tracking
        >>> goal = Goal(pred="IS_A", args={"subject": "hund", "object": "lebewesen"})
        >>> proof = engine.run_with_tracking(goal, query="Ist ein Hund ein Lebewesen?")
        >>>
        >>> # Check consistency
        >>> is_consistent, conflicts = engine.check_kb_consistency()
        >>> if not is_consistent:
        >>>     print(f"Found {len(conflicts)} conflicts")
        >>>
        >>> # Find and explain contradictions
        >>> contradictions = engine.find_contradictions()
        >>> for fact1, fact2 in contradictions:
        >>>     explanation = engine.explain_contradiction(fact1, fact2)
        >>>     print(explanation)
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        use_probabilistic: bool = True,
        use_sat: bool = True,
    ):
        """
        Initialize LogikEngine with all reasoning capabilities.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph access
            use_probabilistic: Enable probabilistic reasoning (requires component_16)
            use_sat: Enable SAT-based reasoning (requires component_30)
        """
        # Initialize base Engine class
        super().__init__(netzwerk, use_probabilistic, use_sat)

        logger.info(
            f"LogikEngine initialized with modules: "
            f"Probabilistic={self.use_probabilistic}, "
            f"SAT={self.use_sat}, "
            f"CSP={CONSTRAINT_REASONING_AVAILABLE}, "
            f"Ontology={self.use_ontology_constraints}"
        )

    def __repr__(self):
        return (
            f"LogikEngine("
            f"rules={len(self.rules)}, "
            f"facts_kb={len(self.kb)}, "
            f"facts_wm={len(self.wm)}, "
            f"probabilistic={self.use_probabilistic}, "
            f"sat={self.use_sat}"
            f")"
        )


# ==================== BACKWARDS COMPATIBILITY ====================

# Alias for backwards compatibility
Engine = LogikEngine


# ==================== CONVENIENCE FUNCTIONS ====================


def create_fact(
    pred: str, args: dict, confidence: float = 1.0, source: str = "user"
) -> Fact:
    """
    Convenience function to create a Fact.

    Args:
        pred: Predicate name (e.g., "IS_A", "HAS_PROPERTY")
        args: Arguments dictionary (e.g., {"subject": "hund", "object": "tier"})
        confidence: Confidence score (0.0-1.0)
        source: Source of the fact (e.g., "user", "kb", "rule:xyz")

    Returns:
        Fact instance
    """
    return Fact(pred=pred, args=args, confidence=confidence, source=source)


def create_goal(pred: str, args: dict, depth: int = 0) -> Goal:
    """
    Convenience function to create a Goal for backward chaining.

    Args:
        pred: Predicate name to prove
        args: Arguments (can contain variables like "?x")
        depth: Depth in proof tree (usually 0 for top-level goals)

    Returns:
        Goal instance
    """
    return Goal(pred=pred, args=args, depth=depth)


def create_rule(
    rule_id: str,
    when: List[Dict[str, Any]],
    then: List[Dict[str, Any]],
    salience: int = 0,
    explain: str = "",
) -> Rule:
    """
    Convenience function to create a Rule.

    Args:
        rule_id: Unique rule identifier
        when: List of condition dictionaries (WHEN clauses)
        then: List of action dictionaries (THEN clauses)
        salience: Priority (higher = executed first)
        explain: Natural language explanation of the rule

    Returns:
        Rule instance

    Example:
        >>> rule = create_rule(
        ...     rule_id="transitive_is_a",
        ...     when=[
        ...         {"pred": "IS_A", "args": {"subject": "?x", "object": "?y"}},
        ...         {"pred": "IS_A", "args": {"subject": "?y", "object": "?z"}}
        ...     ],
        ...     then=[
        ...         {"assert": {"pred": "IS_A", "args": {"subject": "?x", "object": "?z"}}}
        ...     ],
        ...     salience=10,
        ...     explain="Transitive IS_A: If X is a Y and Y is a Z, then X is a Z"
        ... )
    """
    return Rule(id=rule_id, salience=salience, when=when, then=then, explain=explain)


# ==================== EXPORTS ====================

__all__ = [
    # Core classes
    "LogikEngine",
    "Engine",  # Alias for backwards compatibility
    "Fact",
    "Goal",
    "ProofStep",
    "Rule",
    "Binding",
    # Utility functions
    "unify",
    "resolve",
    "match_rule",
    # Convenience functions
    "create_fact",
    "create_goal",
    "create_rule",
    # Conversion functions (for component_17 integration)
    "convert_logic_engine_proof",
    "create_proof_tree_from_logic_engine",
    # Availability flags
    "PROBABILISTIC_AVAILABLE",
    "UNIFIED_PROOFS_AVAILABLE",
    "CONSTRAINT_REASONING_AVAILABLE",
    "SAT_SOLVER_AVAILABLE",
    "ONTOLOGY_CONSTRAINTS_AVAILABLE",
]
