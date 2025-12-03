# component_14_hypothesis_generator.py
"""
Hypothesis Generation Strategies for Abductive Reasoning

Implements three strategies for generating explanatory hypotheses:
1. Template-based: Match causal patterns from predefined templates
2. Analogy-based: Transfer explanations from similar cases
3. Causal chain: Backward reasoning from effects to potential causes

Each strategy generates candidate hypotheses that are later scored
by the hypothesis scorer.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

from component_9_logik_engine import Fact

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """Represents an abduced explanatory hypothesis."""

    id: str
    explanation: str  # Natural language explanation
    observations: List[str]  # What it explains
    abduced_facts: List[Fact]  # New facts it proposes
    strategy: str  # "template" | "analogy" | "causal_chain"
    confidence: float  # 0.0-1.0 overall score
    scores: Dict[str, float]  # {coverage, simplicity, coherence, specificity}
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)  # What knowledge was used
    reasoning_trace: str = ""  # How hypothesis was generated


@dataclass
class CausalPattern:
    """Template for causal reasoning."""

    pattern_type: str  # "CAUSES", "ENABLES", "PROPERTY_OF", "PART_OF"
    template: str  # Natural language template
    forward: str  # X -> Y
    backward: str  # Y observed -> hypothesize X


class HypothesisGenerator:
    """
    Generates explanatory hypotheses using multiple strategies.

    Provides three complementary approaches to hypothesis generation:
    - Template matching against known causal patterns
    - Analogy transfer from similar cases
    - Backward causal chain tracing
    """

    def __init__(self, netzwerk):
        """
        Initialize hypothesis generator.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph queries
        """
        self.netzwerk = netzwerk

        # Define causal patterns for template-based reasoning
        self.causal_patterns = [
            CausalPattern(
                pattern_type="CAUSES",
                template="{X} verursacht {Y}",
                forward="Wenn {X}, dann {Y}",
                backward="Wenn {Y} beobachtet, dann möglicherweise {X}",
            ),
            CausalPattern(
                pattern_type="ENABLES",
                template="{X} ermöglicht {Y}",
                forward="Mit {X} kann {Y} passieren",
                backward="Wenn {Y} passiert, dann war möglicherweise {X} vorhanden",
            ),
            CausalPattern(
                pattern_type="HAS_PROPERTY",
                template="{X} hat Eigenschaft {Y}",
                forward="{X} ist {Y}",
                backward="Wenn etwas {Y} ist, könnte es {X} sein",
            ),
            CausalPattern(
                pattern_type="PART_OF",
                template="{X} ist Teil von {Y}",
                forward="{Y} besteht aus {X}",
                backward="Wenn {Y} existiert, dann auch {X}",
            ),
        ]

    def generate_all_hypotheses(
        self,
        observation: str,
        context_facts: List[Fact] = None,
        strategies: List[str] = None,
    ) -> List[Hypothesis]:
        """
        Generate hypotheses using all requested strategies.

        Args:
            observation: The observation to explain (e.g., "Der Boden ist nass")
            context_facts: Known facts for context
            strategies: Which strategies to use (default: all)

        Returns:
            List of Hypothesis objects (unscored)
        """
        if strategies is None:
            strategies = ["template", "analogy", "causal_chain"]

        if context_facts is None:
            context_facts = []

        all_hypotheses = []

        # Extract key concepts from observation
        key_concepts = self._extract_concepts(observation)

        # Strategy 1: Template-based
        if "template" in strategies:
            template_hypotheses = self.generate_template_hypotheses(
                observation, key_concepts, context_facts
            )
            all_hypotheses.extend(template_hypotheses)

        # Strategy 2: Analogy-based
        if "analogy" in strategies:
            analogy_hypotheses = self.generate_analogy_hypotheses(
                observation, key_concepts, context_facts
            )
            all_hypotheses.extend(analogy_hypotheses)

        # Strategy 3: Causal chain
        if "causal_chain" in strategies:
            causal_hypotheses = self.generate_causal_chain_hypotheses(
                observation, key_concepts, context_facts
            )
            all_hypotheses.extend(causal_hypotheses)

        return all_hypotheses

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text.
        Simple implementation: extract nouns and significant words.
        """
        # Remove common words
        stopwords = {
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "ist",
            "sind",
            "hat",
            "haben",
            "war",
            "waren",
        }

        # Tokenize and filter
        words = re.findall(r"\b\w+\b", text.lower())
        concepts = [w for w in words if w not in stopwords and len(w) > 2]

        return concepts

    def generate_template_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses by matching causal patterns.

        Strategy: For each concept in observation, query graph for relations
        that could explain it (CAUSES, ENABLES, etc.).

        Args:
            observation: The observation to explain
            concepts: Key concepts extracted from observation
            context_facts: Known facts for context

        Returns:
            List of template-based hypotheses
        """
        hypotheses = []

        for concept in concepts:
            # Query graph for potential causes
            facts = self.netzwerk.query_graph_for_facts(concept)

            for pattern in self.causal_patterns:
                rel_type = pattern.pattern_type

                # Check if this relation type exists in facts
                if rel_type in facts:
                    for related_concept in facts[rel_type]:
                        # Generate hypothesis
                        explanation = pattern.backward.format(
                            X=related_concept, Y=concept
                        )

                        # Create abduced fact
                        abduced_fact = Fact(
                            pred=rel_type,
                            args={"subject": related_concept, "object": concept},
                            id=f"abduced_{uuid.uuid4().hex[:8]}",
                            confidence=0.7,  # Lower confidence for abduced facts
                        )

                        hypothesis = Hypothesis(
                            id=f"hyp_{uuid.uuid4().hex[:8]}",
                            explanation=explanation,
                            observations=[observation],
                            abduced_facts=[abduced_fact],
                            strategy="template",
                            confidence=0.0,  # Will be scored later
                            scores={},
                            sources=[f"Pattern: {pattern.pattern_type}"],
                            reasoning_trace=f"Matched pattern: {pattern.template}",
                        )

                        hypotheses.append(hypothesis)

        return hypotheses

    def generate_analogy_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses by finding similar cases in knowledge base.

        Strategy: Find concepts similar to those in observation,
        then transfer explanations from those cases.

        Args:
            observation: The observation to explain
            concepts: Key concepts extracted from observation
            context_facts: Known facts for context

        Returns:
            List of analogy-based hypotheses
        """
        hypotheses = []

        for concept in concepts:
            # Find similar concepts (same IS_A hierarchy)
            facts = self.netzwerk.query_graph_for_facts(concept)

            if "IS_A" in facts:
                # Get parent categories
                categories = facts["IS_A"]

                # Find other members of same category
                for category in categories:
                    # Query for other things in this category
                    # (This requires a reverse query - find X where X IS_A category)
                    similar_concepts = self._find_similar_concepts(category)

                    for similar in similar_concepts:
                        if similar == concept:
                            continue  # Skip self

                        # Get facts about similar concept
                        similar_facts = self.netzwerk.query_graph_for_facts(similar)

                        # Transfer properties
                        for rel_type, related in similar_facts.items():
                            if rel_type in ["HAS_PROPERTY", "CAPABLE_OF", "LOCATED_IN"]:
                                for prop in related:
                                    explanation = (
                                        f"{concept} könnte {prop} sein/haben, "
                                        f"wie {similar} (beide sind {category})"
                                    )

                                    abduced_fact = Fact(
                                        pred=rel_type,
                                        args={"subject": concept, "object": prop},
                                        id=f"abduced_{uuid.uuid4().hex[:8]}",
                                        confidence=0.6,  # Lower confidence for analogies
                                    )

                                    hypothesis = Hypothesis(
                                        id=f"hyp_{uuid.uuid4().hex[:8]}",
                                        explanation=explanation,
                                        observations=[observation],
                                        abduced_facts=[abduced_fact],
                                        strategy="analogy",
                                        confidence=0.0,
                                        scores={},
                                        sources=[f"Analogy: {similar}"],
                                        reasoning_trace=(
                                            f"Found similar concept '{similar}' "
                                            f"(both are '{category}')"
                                        ),
                                    )

                                    hypotheses.append(hypothesis)

        return hypotheses

    def _find_similar_concepts(self, category: str) -> List[str]:
        """
        Find concepts that are members of the given category.

        Runs reverse IS_A query: find X where (X)-[:IS_A]->(category)

        Args:
            category: The category to search for members

        Returns:
            List of concept names that belong to this category
        """
        try:
            # Use facade method for reverse IS_A query
            inverse_relations = self.netzwerk.query_inverse_relations(
                category, relation_type="IS_A"
            )

            # Extract IS_A subjects (limit to 10)
            is_a_subjects = inverse_relations.get("IS_A", [])
            return is_a_subjects[:10] if is_a_subjects else []

        except Exception as e:
            logger.error(
                f"Failed to query similar concepts for category '{category}': {e}"
            )
            return []

    def generate_causal_chain_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses by tracing causal chains backward.

        Strategy: Start from observed effect, follow causal relations
        backward to find potential root causes.

        Args:
            observation: The observation to explain
            concepts: Key concepts extracted from observation
            context_facts: Known facts for context

        Returns:
            List of causal chain hypotheses
        """
        hypotheses = []

        for concept in concepts:
            # Find causal chains ending at this concept
            chains = self._find_causal_chains(concept, max_depth=3)

            for chain in chains:
                # chain is a list of (concept, relation) tuples
                # Example: [("regen", "CAUSES"), ("wolken", "CAUSES")]

                if len(chain) == 0:
                    continue

                # Build explanation from chain
                explanation_parts = []
                abduced_facts = []

                prev_concept = concept
                for cause, relation in chain:
                    explanation_parts.append(f"{cause} -> {relation} -> {prev_concept}")

                    abduced_fact = Fact(
                        pred=relation,
                        args={"subject": cause, "object": prev_concept},
                        id=f"abduced_{uuid.uuid4().hex[:8]}",
                        confidence=0.8 / len(chain),  # Longer chains less confident
                    )
                    abduced_facts.append(abduced_fact)

                    prev_concept = cause

                explanation = (
                    f"Kausale Kette: {' -> '.join(reversed(explanation_parts))}"
                )

                hypothesis = Hypothesis(
                    id=f"hyp_{uuid.uuid4().hex[:8]}",
                    explanation=explanation,
                    observations=[observation],
                    abduced_facts=abduced_facts,
                    strategy="causal_chain",
                    confidence=0.0,
                    scores={},
                    sources=[f"Causal chain of length {len(chain)}"],
                    reasoning_trace=f"Traced {len(chain)} causal steps backward",
                )

                hypotheses.append(hypothesis)

        return hypotheses

    def _find_causal_chains(
        self, effect: str, max_depth: int = 3
    ) -> List[List[Tuple[str, str]]]:
        """
        Find causal chains ending at the given effect.

        Returns list of chains, where each chain is a list of (cause, relation) tuples.

        Args:
            effect: The effect to trace backward from
            max_depth: Maximum depth of causal chains to trace

        Returns:
            List of causal chains (each chain is a list of (cause, relation) tuples)
        """
        if not self.netzwerk.driver:
            return []

        chains = []

        # Query for immediate causes
        facts = self.netzwerk.query_graph_for_facts(effect)

        # Look for causal relations (reversed - things that cause this effect)
        for rel_type in ["CAUSES", "ENABLES", "LEADS_TO"]:
            if rel_type in facts:
                for cause in facts[rel_type]:
                    # Start a chain
                    chain = [(cause, rel_type)]

                    # Recursively extend chain if depth allows
                    if max_depth > 1:
                        sub_chains = self._find_causal_chains(cause, max_depth - 1)
                        if sub_chains:
                            for sub_chain in sub_chains:
                                extended_chain = chain + sub_chain
                                chains.append(extended_chain)
                        else:
                            chains.append(chain)
                    else:
                        chains.append(chain)

        return chains
