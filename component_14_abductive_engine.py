# component_14_abductive_engine.py
"""
Abductive Reasoning Engine for KAI (Facade)

Generates explanatory hypotheses when deductive reasoning fails.
This is a facade that delegates to specialized modules:
- HypothesisGenerator: Generates hypotheses using 3 strategies
- HypothesisScorer: Scores hypotheses on 4 criteria
- KnowledgeValidator: Validates against existing knowledge

The facade maintains 100% backward compatibility with the original API
while benefiting from the improved modular architecture.

**Modular Architecture (Phase 4, Task 12)**:
- component_14_hypothesis_generator.py: All generation strategies
- component_14_hypothesis_scorer.py: All scoring logic
- component_14_knowledge_validator.py: Contradiction detection
- component_14_abductive_engine.py: Facade + ProofTree integration
"""

import logging
from typing import Any, Dict, List, Optional

from component_9_logik_engine import Fact

# Import modular components
from component_14_hypothesis_generator import (
    CausalPattern,
    Hypothesis,
    HypothesisGenerator,
)
from component_14_hypothesis_scorer import HypothesisScorer
from component_14_knowledge_validator import KnowledgeValidator
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import ProofStep as UnifiedProofStep
    from component_17_proof_explanation import (
        StepType,
        generate_explanation_text,
    )

    UNIFIED_PROOFS_AVAILABLE = True
except ImportError:
    UNIFIED_PROOFS_AVAILABLE = False


# Re-export dataclasses for backward compatibility
__all__ = ["AbductiveEngine", "Hypothesis", "CausalPattern"]


class AbductiveEngine(BaseReasoningEngine):
    """
    Main engine for abductive reasoning.

    Generates and scores explanatory hypotheses when deductive reasoning fails.

    **Facade Pattern**: Delegates to specialized modules while maintaining
    the same public API as the original implementation.

    Implements BaseReasoningEngine for integration with reasoning orchestrator.
    """

    def __init__(self, netzwerk, logic_engine=None):
        """
        Initialize abductive engine.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph queries
            logic_engine: Optional Engine instance for deductive verification
        """
        self.netzwerk = netzwerk
        self.logic_engine = logic_engine

        # Initialize modular components
        self._generator = HypothesisGenerator(netzwerk)
        self._scorer = HypothesisScorer(netzwerk)
        self._validator = KnowledgeValidator(netzwerk, logic_engine)

        # Expose causal patterns for backward compatibility
        self.causal_patterns = self._generator.causal_patterns

        # Expose score weights for backward compatibility
        self.score_weights = self._scorer.score_weights

    def generate_hypotheses(
        self,
        observation: str,
        context_facts: List[Fact] = None,
        strategies: List[str] = None,
        max_hypotheses: int = 10,
    ) -> List[Hypothesis]:
        """
        Generate explanatory hypotheses for an observation.

        Args:
            observation: The observation to explain (e.g., "Der Boden ist nass")
            context_facts: Known facts for context
            strategies: Which strategies to use (default: all)
            max_hypotheses: Maximum number of hypotheses to return

        Returns:
            List of Hypothesis objects, ranked by confidence
        """
        if context_facts is None:
            context_facts = []

        # Generate hypotheses using all strategies
        all_hypotheses = self._generator.generate_all_hypotheses(
            observation, context_facts, strategies
        )

        # Score all hypotheses
        for hypothesis in all_hypotheses:
            self._scorer.score_hypothesis(hypothesis, context_facts)

        # Rank by confidence
        all_hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        return all_hypotheses[:max_hypotheses]

    # ==================== DELEGATION METHODS (FOR BACKWARD COMPATIBILITY) ====================

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text (delegates to generator)."""
        return self._generator._extract_concepts(text)

    def _generate_template_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """Generate template-based hypotheses (delegates to generator)."""
        return self._generator.generate_template_hypotheses(
            observation, concepts, context_facts
        )

    def _generate_analogy_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """Generate analogy-based hypotheses (delegates to generator)."""
        return self._generator.generate_analogy_hypotheses(
            observation, concepts, context_facts
        )

    def _find_similar_concepts(self, category: str) -> List[str]:
        """Find similar concepts (delegates to generator)."""
        return self._generator._find_similar_concepts(category)

    def _generate_causal_chain_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """Generate causal chain hypotheses (delegates to generator)."""
        return self._generator.generate_causal_chain_hypotheses(
            observation, concepts, context_facts
        )

    def _find_causal_chains(self, effect: str, max_depth: int = 3):
        """Find causal chains (delegates to generator)."""
        return self._generator._find_causal_chains(effect, max_depth)

    def _score_hypothesis(
        self, hypothesis: Hypothesis, context_facts: List[Fact]
    ) -> None:
        """Score hypothesis (delegates to scorer)."""
        self._scorer.score_hypothesis(hypothesis, context_facts)

    def _score_coverage(self, hypothesis: Hypothesis) -> float:
        """Score coverage (delegates to scorer)."""
        return self._scorer._score_coverage(hypothesis)

    def _score_simplicity(self, hypothesis: Hypothesis) -> float:
        """Score simplicity (delegates to scorer)."""
        return self._scorer._score_simplicity(hypothesis)

    def _score_coherence(
        self, hypothesis: Hypothesis, context_facts: List[Fact]
    ) -> float:
        """Score coherence (delegates to scorer)."""
        return self._scorer._score_coherence(hypothesis, context_facts)

    def _score_specificity(self, hypothesis: Hypothesis) -> float:
        """Score specificity (delegates to scorer)."""
        return self._scorer._score_specificity(hypothesis)

    def _is_fact_known(self, fact: Fact) -> bool:
        """Check if fact is known (delegates to scorer)."""
        return self._scorer._is_fact_known(fact)

    def _contradicts_knowledge(self, fact: Fact) -> bool:
        """Check if fact contradicts knowledge (delegates to validator)."""
        return self._validator.contradicts_knowledge(fact)

    def _get_facts_about_subject(self, subject: str) -> List[Fact]:
        """Get facts about subject (delegates to validator)."""
        return self._validator._get_facts_about_subject(subject)

    def _are_types_mutually_exclusive(self, type1: str, type2: str) -> bool:
        """Check if types are mutually exclusive (delegates to validator)."""
        return self._validator._are_types_mutually_exclusive(type1, type2)

    def _is_subtype_of(
        self,
        subtype: str,
        supertype: str,
        visited: Optional[set] = None,
        max_depth: int = 10,
    ) -> bool:
        """Check if subtype is subtype of supertype (delegates to validator)."""
        return self._validator._is_subtype_of(subtype, supertype, visited, max_depth)

    def _are_properties_contradictory(self, prop1: str, prop2: str) -> bool:
        """Check if properties contradict (delegates to validator)."""
        return self._validator._are_properties_contradictory(prop1, prop2)

    def _is_location_hierarchy(
        self, loc1: str, loc2: str, visited: Optional[set] = None, max_depth: int = 10
    ) -> bool:
        """Check if locations are in hierarchy (delegates to validator)."""
        return self._validator._is_location_hierarchy(loc1, loc2, visited, max_depth)

    # ==================== EXPLANATION METHODS ====================

    def explain_hypothesis(self, hypothesis: Hypothesis) -> str:
        """
        Generate detailed natural language explanation of hypothesis.

        Args:
            hypothesis: The hypothesis to explain

        Returns:
            Natural language explanation with reasoning trace
        """
        parts = []

        parts.append(f"**Hypothese (ID: {hypothesis.id})**")
        parts.append(f"Erklärung: {hypothesis.explanation}")
        parts.append(f"Strategie: {hypothesis.strategy}")
        parts.append(f"Konfidenz: {hypothesis.confidence:.2f}")
        parts.append("")

        parts.append("**Bewertung:**")
        for criterion, score in hypothesis.scores.items():
            parts.append(f"  - {criterion}: {score:.2f}")
        parts.append("")

        if hypothesis.abduced_facts:
            parts.append("**Abgeleitete Fakten:**")
            for fact in hypothesis.abduced_facts:
                parts.append(
                    f"  - {fact.pred}({fact.args}) "
                    f"[Konfidenz: {fact.confidence:.2f}]"
                )
            parts.append("")

        if hypothesis.reasoning_trace:
            parts.append("**Reasoning Trace:**")
            parts.append(f"  {hypothesis.reasoning_trace}")
            parts.append("")

        return "\n".join(parts)

    # ==================== UNIFIED PROOF EXPLANATION INTEGRATION ====================

    def create_proof_step_from_hypothesis(
        self, hypothesis: Hypothesis, query: str = ""
    ) -> Optional[UnifiedProofStep]:
        """
        Convert a hypothesis to a UnifiedProofStep.

        Args:
            hypothesis: Hypothesis object
            query: The original query (optional)

        Returns:
            UnifiedProofStep or None
        """
        if not UNIFIED_PROOFS_AVAILABLE or not hypothesis:
            return None

        # Create inputs from observations
        inputs = hypothesis.observations

        # Output is the hypothesis explanation
        output = hypothesis.explanation

        # Generate enhanced explanation
        explanation = generate_explanation_text(
            step_type=StepType.HYPOTHESIS,
            inputs=inputs,
            output=output,
            metadata={
                "strategy": hypothesis.strategy,
                "score": hypothesis.confidence,
                "scores": hypothesis.scores,
                "num_abduced_facts": len(hypothesis.abduced_facts),
            },
        )

        # Create UnifiedProofStep
        step = UnifiedProofStep(
            step_id=hypothesis.id,
            step_type=StepType.HYPOTHESIS,
            inputs=inputs,
            rule_name=None,
            output=output,
            confidence=hypothesis.confidence,
            explanation_text=explanation,
            parent_steps=[],  # Hypotheses have no direct parents
            bindings={},
            metadata={
                "strategy": hypothesis.strategy,
                "scores": hypothesis.scores,
                "abduced_facts": [f.pred for f in hypothesis.abduced_facts],
                "sources": hypothesis.sources,
                "reasoning_trace": hypothesis.reasoning_trace,
                "observations": hypothesis.observations,
            },
            source_component="component_14_abductive_engine",
            timestamp=hypothesis.timestamp,
        )

        # Add abduced facts as subgoals (optional)
        for fact in hypothesis.abduced_facts:
            fact_step = UnifiedProofStep(
                step_id=fact.id,
                step_type=StepType.HYPOTHESIS,  # Abduced facts are also hypotheses
                inputs=[],
                rule_name=None,
                output=f"{fact.pred}({fact.args})",
                confidence=fact.confidence,
                explanation_text=f"Abgeleiteter Fakt aus Hypothese: {fact.pred}",
                parent_steps=[hypothesis.id],
                bindings=fact.args,
                metadata={"abduced": True, "parent_hypothesis": hypothesis.id},
                source_component="component_14_abductive_engine",
            )
            step.add_subgoal(fact_step)

        return step

    def create_multi_hypothesis_proof_chain(
        self, hypotheses: List[Hypothesis], query: str = ""
    ) -> List[UnifiedProofStep]:
        """
        Create a proof chain from multiple hypotheses.

        Useful when multiple alternative hypotheses exist.

        Args:
            hypotheses: List of Hypothesis objects
            query: The original query

        Returns:
            List of UnifiedProofStep objects (sorted by confidence)
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            return []

        proof_steps = []
        for i, hypothesis in enumerate(hypotheses):
            step = self.create_proof_step_from_hypothesis(hypothesis, query)
            if step:
                # Mark rank in alternative hypotheses
                step.metadata["hypothesis_rank"] = i + 1
                step.metadata["total_hypotheses"] = len(hypotheses)
                proof_steps.append(step)

        return proof_steps

    def explain_with_proof_step(
        self,
        observation: str,
        context_facts: List[Fact] = None,
        max_hypotheses: int = 3,
    ) -> List[UnifiedProofStep]:
        """
        Generate hypotheses and return UnifiedProofSteps.

        This is the main interface for integration with the Reasoning System.

        Args:
            observation: The observation to explain
            context_facts: Known facts for context
            max_hypotheses: Maximum number of hypotheses

        Returns:
            List of UnifiedProofStep objects with explanations
        """
        # Generate hypotheses
        hypotheses = self.generate_hypotheses(
            observation=observation,
            context_facts=context_facts,
            max_hypotheses=max_hypotheses,
        )

        if not UNIFIED_PROOFS_AVAILABLE:
            return []

        # Convert to UnifiedProofSteps
        proof_steps = self.create_multi_hypothesis_proof_chain(
            hypotheses, query=observation
        )

        return proof_steps

    def create_detailed_explanation(self, hypothesis: Hypothesis) -> str:
        """
        Create a detailed explanation using Unified Explanation System.

        Args:
            hypothesis: The hypothesis to explain

        Returns:
            Detailed natural language explanation
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            # Fallback to old method
            return self.explain_hypothesis(hypothesis)

        # Create UnifiedProofStep
        proof_step = self.create_proof_step_from_hypothesis(hypothesis)

        if not proof_step:
            return self.explain_hypothesis(hypothesis)

        # Use Unified Formatter
        from component_17_proof_explanation import format_proof_step

        formatted = format_proof_step(proof_step, indent=0, show_details=True)

        # Add additional information
        parts = [formatted, ""]

        parts.append("=== Detaillierte Bewertung ===")
        for criterion, score in hypothesis.scores.items():
            parts.append(f"  {criterion.capitalize()}: {score:.2f}")

        if hypothesis.sources:
            parts.append("")
            parts.append("=== Quellen ===")
            for source in hypothesis.sources:
                parts.append(f"  - {source}")

        return "\n".join(parts)

    # ==================== BASE REASONING ENGINE INTERFACE ====================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Generate explanatory hypotheses for an observation.

        Args:
            query: The observation to explain
            context: Context with context_facts, max_hypotheses, etc.

        Returns:
            ReasoningResult with best hypothesis as answer
        """
        # Extract parameters from context
        context_facts = context.get("context_facts", [])
        max_hypotheses = context.get("max_hypotheses", 3)

        # Generate hypotheses
        hypotheses = self.generate_hypotheses(
            observation=query,
            context_facts=context_facts,
            max_hypotheses=max_hypotheses,
        )

        if not hypotheses:
            return ReasoningResult(
                success=False,
                answer=f"Keine Hypothesen fur Beobachtung: {query}",
                confidence=0.0,
                strategy_used="abductive_reasoning",
            )

        # Take best hypothesis
        best_hypothesis = hypotheses[0]

        # Create proof tree
        proof_tree = None
        if UNIFIED_PROOFS_AVAILABLE:
            proof_steps = self.create_multi_hypothesis_proof_chain(hypotheses, query)
            if proof_steps:
                from component_17_proof_explanation import ProofTree

                proof_tree = ProofTree(query=query)
                for step in proof_steps:
                    proof_tree.add_root_step(step)

        return ReasoningResult(
            success=True,
            answer=best_hypothesis.explanation,
            confidence=best_hypothesis.confidence,
            proof_tree=proof_tree,
            strategy_used=f"abductive_{best_hypothesis.strategy}",
            metadata={
                "num_hypotheses": len(hypotheses),
                "scores": best_hypothesis.scores,
                "abduced_facts": len(best_hypothesis.abduced_facts),
                "all_hypotheses": [h.explanation for h in hypotheses],
            },
        )

    def get_capabilities(self) -> List[str]:
        """Return list of reasoning capabilities."""
        return [
            "abductive",
            "hypothesis_generation",
            "analogical",
            "causal_chain_inference",
            "explanatory_reasoning",
        ]

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for abductive reasoning.

        Abductive reasoning is generally expensive as it requires:
        - Template matching
        - Analogy search
        - Causal chain exploration

        Returns:
            Cost estimate in [0.0, 1.0] range
        """
        # Abductive reasoning is expensive
        base_cost = 0.7

        # Query complexity (longer queries may need more hypotheses)
        query_complexity = min(len(query) / 150.0, 0.2)

        total_cost = base_cost + query_complexity

        return min(total_cost, 1.0)
