# component_14_hypothesis_scorer.py
"""
Hypothesis Scoring Module for Abductive Reasoning

Evaluates and scores hypotheses using multiple criteria:
1. Coverage: How well the hypothesis explains all observations
2. Simplicity: Occam's razor - simpler explanations are better
3. Coherence: Consistency with existing knowledge
4. Specificity: Ability to generate testable predictions

Each criterion is scored 0.0-1.0 and combined using weighted averaging.
"""

import logging
from typing import List

from component_9_logik_engine import Fact
from component_14_hypothesis_generator import Hypothesis

logger = logging.getLogger(__name__)


class HypothesisScorer:
    """
    Scores hypotheses using multiple evaluation criteria.

    Provides a balanced assessment combining:
    - Coverage (30%): Explanatory power
    - Simplicity (20%): Minimal assumptions
    - Coherence (30%): Fit with existing knowledge
    - Specificity (20%): Testability
    """

    def __init__(self, netzwerk):
        """
        Initialize hypothesis scorer.

        Args:
            netzwerk: KonzeptNetzwerk instance for knowledge base queries
        """
        self.netzwerk = netzwerk

        # Scoring weights (can be tuned)
        self.score_weights = {
            "coverage": 0.3,
            "simplicity": 0.2,
            "coherence": 0.3,
            "specificity": 0.2,
        }

    def score_hypothesis(
        self, hypothesis: Hypothesis, context_facts: List[Fact]
    ) -> None:
        """
        Score hypothesis using multiple criteria.
        Updates hypothesis.scores and hypothesis.confidence in-place.

        Args:
            hypothesis: The hypothesis to score
            context_facts: Known facts for context
        """
        scores = {}

        # 1. Coverage: Does it explain all observations?
        scores["coverage"] = self._score_coverage(hypothesis)

        # 2. Simplicity: Occam's razor (fewer assumptions = better)
        scores["simplicity"] = self._score_simplicity(hypothesis)

        # 3. Coherence: Fits existing knowledge?
        scores["coherence"] = self._score_coherence(hypothesis, context_facts)

        # 4. Specificity: Generates testable predictions?
        scores["specificity"] = self._score_specificity(hypothesis)

        # Calculate weighted average
        confidence = sum(
            scores[criterion] * self.score_weights[criterion] for criterion in scores
        )

        hypothesis.scores = scores
        hypothesis.confidence = confidence

    def _score_coverage(self, hypothesis: Hypothesis) -> float:
        """
        Score how well hypothesis covers all observations.

        Simple version: 1.0 if at least one observation, can be extended.

        Args:
            hypothesis: The hypothesis to evaluate

        Returns:
            Coverage score (0.0-1.0)
        """
        if len(hypothesis.observations) > 0:
            return 1.0
        return 0.0

    def _score_simplicity(self, hypothesis: Hypothesis) -> float:
        """
        Score simplicity (Occam's razor).

        Fewer abduced facts = simpler = better.

        Args:
            hypothesis: The hypothesis to evaluate

        Returns:
            Simplicity score (0.0-1.0)
        """
        num_facts = len(hypothesis.abduced_facts)

        if num_facts == 0:
            return 0.0  # Empty hypothesis
        elif num_facts == 1:
            return 1.0  # Single fact (simplest)
        elif num_facts <= 3:
            return 0.7  # Few facts
        else:
            return 0.4  # Many facts (complex)

    def _score_coherence(
        self, hypothesis: Hypothesis, context_facts: List[Fact]
    ) -> float:
        """
        Score coherence with existing knowledge.

        Check if abduced facts contradict or align with known facts.

        Args:
            hypothesis: The hypothesis to evaluate
            context_facts: Known facts for context

        Returns:
            Coherence score (0.0-1.0)
        """
        if len(hypothesis.abduced_facts) == 0:
            return 0.5  # Neutral

        coherent_count = 0
        total_count = len(hypothesis.abduced_facts)

        for abduced in hypothesis.abduced_facts:
            # Check if this fact is already known
            is_known = self._is_fact_known(abduced)

            if is_known:
                coherent_count += 1  # Aligns with existing knowledge
            else:
                # Check if it contradicts
                contradicts = self._contradicts_knowledge(abduced)
                if not contradicts:
                    coherent_count += 0.5  # Doesn't contradict (neutral)
                # If contradicts, add 0 (penalize)

        return coherent_count / total_count

    def _score_specificity(self, hypothesis: Hypothesis) -> float:
        """
        Score specificity (generates testable predictions).

        More specific facts = more testable = better.

        Args:
            hypothesis: The hypothesis to evaluate

        Returns:
            Specificity score (0.0-1.0)
        """
        if len(hypothesis.abduced_facts) == 0:
            return 0.0

        # Count facts with concrete (non-variable) arguments
        specific_count = 0
        for fact in hypothesis.abduced_facts:
            has_variables = any(str(v).startswith("?") for v in fact.args.values())
            if not has_variables:
                specific_count += 1

        return specific_count / len(hypothesis.abduced_facts)

    def _is_fact_known(self, fact: Fact) -> bool:
        """
        Check if a fact is already in the knowledge base.

        Args:
            fact: The fact to check

        Returns:
            True if fact exists in knowledge base
        """
        if "subject" in fact.args and "object" in fact.args:
            subject = fact.args["subject"]
            obj = fact.args["object"]

            # Query graph
            facts = self.netzwerk.query_graph_for_facts(subject)

            if fact.pred in facts:
                return obj in facts[fact.pred]

        return False

    def _contradicts_knowledge(self, fact: Fact) -> bool:
        """
        Check if a fact contradicts existing knowledge.

        Placeholder for now - actual implementation should check for
        contradictions (delegated to KnowledgeValidator).

        Args:
            fact: The fact to check

        Returns:
            True if fact contradicts existing knowledge
        """
        # This is a simplified version - the full logic is in KnowledgeValidator
        # For basic scoring, we assume no contradiction if we can't determine
        return False
