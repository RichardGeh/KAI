# component_14_knowledge_validator.py
"""
Knowledge Validation Module for Abductive Reasoning

Validates hypotheses against existing knowledge to detect contradictions.
Uses both SAT-based formal reasoning and heuristic fallbacks.

Contradiction Detection Methods:
1. SAT-based consistency checking (primary, if Logic Engine available)
2. Heuristic checks (fallback):
   - Mutually exclusive IS_A relations
   - Contradictory properties (colors, sizes, temperatures)
   - Incompatible locations
"""

import logging
import uuid
from typing import List, Optional

from component_9_logik_engine import Fact

logger = logging.getLogger(__name__)


class KnowledgeValidator:
    """
    Validates hypotheses against the knowledge base.

    Detects contradictions using formal (SAT) and heuristic methods.
    Provides hierarchy checking for IS_A and PART_OF relations.
    """

    def __init__(self, netzwerk, logic_engine=None):
        """
        Initialize knowledge validator.

        Args:
            netzwerk: KonzeptNetzwerk instance for knowledge base queries
            logic_engine: Optional Engine instance for SAT-based validation
        """
        self.netzwerk = netzwerk
        self.logic_engine = logic_engine

    def contradicts_knowledge(self, fact: Fact) -> bool:
        """
        Check if a fact contradicts existing knowledge.

        **PHASE 4.2 EXTENSION**: Uses SAT-Solver for robust consistency checking.

        Before: Heuristic category checks (IS_A, HAS_PROPERTY, LOCATED_IN)
        After: Formal SAT-based consistency checking + Heuristic fallbacks

        Detects contradictions via:
        1. SAT-based consistency checking (primary method, if available)
        2. Heuristic category checks (fallback):
           - Mutually exclusive IS_A relations
           - Contradictory properties (colors, sizes, etc.)
           - Incompatible locations

        Args:
            fact: The fact to check for contradictions

        Returns:
            True if fact contradicts existing knowledge, False otherwise
        """
        if "subject" not in fact.args or "object" not in fact.args:
            return False

        subject = fact.args["subject"]
        obj = fact.args["object"]

        # ==================== PHASE 4.2: SAT-BASED CONSISTENCY CHECK ====================
        # Try SAT-based consistency check (if Logic Engine available)
        if self.logic_engine and hasattr(self.logic_engine, "check_consistency"):
            try:
                logger.debug(
                    f"SAT-based consistency check for {fact.pred}({fact.args})"
                )

                # Get all relevant facts from knowledge base
                existing_facts_list = self._get_facts_about_subject(subject)

                if existing_facts_list:
                    # Check if [existing_facts + new_fact] is consistent
                    all_facts = existing_facts_list + [fact]
                    is_consistent = self.logic_engine.check_consistency(all_facts)

                    if not is_consistent:
                        logger.info(
                            f"SAT-Solver: Contradiction found for "
                            f"{fact.pred}({subject} -> {obj})"
                        )
                        return True
                    else:
                        # SAT says: consistent -> No contradiction
                        logger.debug(
                            f"SAT-Solver: No contradiction for "
                            f"{fact.pred}({subject} -> {obj})"
                        )
                        return False

            except Exception as e:
                logger.warning(
                    f"SAT-based consistency check failed: {e}. "
                    f"Fallback to heuristic check."
                )
                # Fallback to heuristic check (see below)

        # ==================== FALLBACK: HEURISTIC CONSISTENCY CHECKS ====================
        # Query existing facts about the subject
        existing_facts = self.netzwerk.query_graph_for_facts(subject)

        # CATEGORY 1: Mutually Exclusive IS_A Relations
        if fact.pred == "IS_A":
            if "IS_A" in existing_facts:
                for existing_type in existing_facts["IS_A"]:
                    if existing_type != obj:
                        if self._are_types_mutually_exclusive(existing_type, obj):
                            logger.debug(
                                f"Contradiction found: {subject} cannot be both "
                                f"'{existing_type}' AND '{obj}' (IS_A conflict)"
                            )
                            return True

        # CATEGORY 2: Contradictory Properties
        if fact.pred == "HAS_PROPERTY":
            if "HAS_PROPERTY" in existing_facts:
                for existing_prop in existing_facts["HAS_PROPERTY"]:
                    if self._are_properties_contradictory(existing_prop, obj):
                        logger.debug(
                            f"Contradiction found: {subject} cannot have both "
                            f"'{existing_prop}' AND '{obj}' (Property conflict)"
                        )
                        return True

        # CATEGORY 3: Incompatible Locations
        if fact.pred == "LOCATED_IN":
            if "LOCATED_IN" in existing_facts:
                for existing_location in existing_facts["LOCATED_IN"]:
                    if existing_location != obj:
                        if not self._is_location_hierarchy(existing_location, obj):
                            logger.debug(
                                f"Contradiction found: {subject} cannot be in both "
                                f"'{existing_location}' AND '{obj}' (Location conflict)"
                            )
                            return True

        return False

    def _get_facts_about_subject(self, subject: str) -> List[Fact]:
        """
        Get all facts about a subject from the knowledge base.

        Converts results from query_graph_for_facts() to Fact objects
        for SAT-based consistency checking.

        Args:
            subject: The subject (e.g., "hund")

        Returns:
            List of Fact objects about the subject
        """
        facts_list = []

        # Query graph for all relations of the subject
        facts_dict = self.netzwerk.query_graph_for_facts(subject)

        # Convert to Fact objects
        for relation_type, objects in facts_dict.items():
            for obj in objects:
                fact = Fact(
                    pred=relation_type,
                    args={"subject": subject, "object": obj},
                    id=f"kb_{uuid.uuid4().hex[:8]}",
                    confidence=1.0,  # Existing KB facts have high confidence
                )
                facts_list.append(fact)

        logger.debug(
            f"Extracted {len(facts_list)} facts from KB for subject '{subject}'"
        )

        return facts_list

    def _are_types_mutually_exclusive(self, type1: str, type2: str) -> bool:
        """
        Check if two IS_A types are mutually exclusive.

        Heuristic: If both types are concrete object categories (not abstract),
        then they exclude each other (e.g., "Hund" vs "Katze").

        Abstract categories like "Tier", "Lebewesen", "Objekt" are hierarchical and OK.

        Args:
            type1: First type
            type2: Second type

        Returns:
            True if the types are mutually exclusive
        """
        # List of abstract categories (hierarchically OK)
        abstract_categories = {
            "objekt",
            "ding",
            "sache",
            "entität",
            "lebewesen",
            "tier",
            "pflanze",
            "organismus",
            "konzept",
            "idee",
            "abstraktum",
        }

        type1_lower = type1.lower()
        type2_lower = type2.lower()

        # If both are abstract, no contradiction
        if type1_lower in abstract_categories and type2_lower in abstract_categories:
            return False

        # If one is abstract, check if the other is derived from it
        # (Simplification: if one is abstract, no contradiction)
        if type1_lower in abstract_categories or type2_lower in abstract_categories:
            return False

        # Check if type2 is in the hierarchy of type1 (or vice versa)
        if self._is_subtype_of(type1, type2) or self._is_subtype_of(type2, type1):
            return False

        # Otherwise: concrete different types = potential contradiction
        return True

    def _is_subtype_of(
        self,
        subtype: str,
        supertype: str,
        visited: Optional[set] = None,
        max_depth: int = 10,
    ) -> bool:
        """
        Check if subtype is a subtype of supertype (via IS_A hierarchy).

        Args:
            subtype: The potential subtype
            supertype: The potential supertype
            visited: Set of already visited nodes (cycle detection)
            max_depth: Maximum recursion depth

        Returns:
            True if subtype is a subtype of supertype
        """
        if visited is None:
            visited = set()

        # Cycle detection
        if subtype in visited:
            logger.warning(f"Cycle detected in IS_A hierarchy at '{subtype}'")
            return False

        # Depth limit
        if len(visited) >= max_depth:
            logger.warning(f"Max depth {max_depth} exceeded in IS_A hierarchy")
            return False

        visited.add(subtype)

        # Query IS_A hierarchy
        facts = self.netzwerk.query_graph_for_facts(subtype)
        if "IS_A" in facts:
            # If subtype directly IS_A supertype
            if supertype in facts["IS_A"]:
                return True
            # Recursively check (transitive IS_A)
            for parent in facts["IS_A"]:
                if self._is_subtype_of(parent, supertype, visited, max_depth):
                    return True
        return False

    def _are_properties_contradictory(self, prop1: str, prop2: str) -> bool:
        """
        Check if two properties contradict each other.

        Heuristic: Colors, sizes, and other measurable properties are mutually exclusive.

        Args:
            prop1: First property
            prop2: Second property

        Returns:
            True if the properties contradict each other
        """
        prop1_lower = prop1.lower()
        prop2_lower = prop2.lower()

        # Category 1: Colors
        colors = {
            "rot",
            "blau",
            "grün",
            "gelb",
            "orange",
            "lila",
            "schwarz",
            "weiß",
            "grau",
            "braun",
        }
        if prop1_lower in colors and prop2_lower in colors:
            return prop1_lower != prop2_lower

        # Category 2: Sizes (relative)
        sizes = {"groß", "klein", "mittel", "riesig", "winzig"}
        if prop1_lower in sizes and prop2_lower in sizes:
            return prop1_lower != prop2_lower

        # Category 3: Temperatures (relative)
        temperatures = {"heiß", "kalt", "warm", "kühl", "eiskalt"}
        if prop1_lower in temperatures and prop2_lower in temperatures:
            # Some combinations are OK (e.g., "warm" and "heiß" are not contradictory)
            # But "heiß" and "kalt" are
            opposites = [
                ("heiß", "kalt"),
                ("warm", "kalt"),
                ("heiß", "kühl"),
                ("warm", "eiskalt"),
                ("heiß", "eiskalt"),
            ]
            for a, b in opposites:
                if (prop1_lower == a and prop2_lower == b) or (
                    prop1_lower == b and prop2_lower == a
                ):
                    return True

        # Category 4: States (binary)
        binary_states = {
            ("lebendig", "tot"),
            ("aktiv", "inaktiv"),
            ("offen", "geschlossen"),
            ("wahr", "falsch"),
            ("an", "aus"),
        }
        for state1, state2 in binary_states:
            if (prop1_lower == state1 and prop2_lower == state2) or (
                prop1_lower == state2 and prop2_lower == state1
            ):
                return True

        # No contradictions found
        return False

    def _is_location_hierarchy(
        self, loc1: str, loc2: str, visited: Optional[set] = None, max_depth: int = 10
    ) -> bool:
        """
        Check if two locations are in a hierarchy (e.g., Berlin in Deutschland).

        Args:
            loc1: First location
            loc2: Second location
            visited: Set of already visited nodes (cycle detection)
            max_depth: Maximum recursion depth

        Returns:
            True if loc1 is part of loc2 (or vice versa)
        """
        if visited is None:
            visited = set()

        # Cycle detection
        if loc1 in visited or loc2 in visited:
            logger.warning(f"Cycle detected in PART_OF hierarchy")
            return False

        # Depth limit
        if len(visited) >= max_depth:
            return False

        visited.add(loc1)
        visited.add(loc2)

        # Check if loc1 PART_OF loc2
        facts1 = self.netzwerk.query_graph_for_facts(loc1)
        if "PART_OF" in facts1 and loc2 in facts1["PART_OF"]:
            return True

        # Check if loc2 PART_OF loc1
        facts2 = self.netzwerk.query_graph_for_facts(loc2)
        if "PART_OF" in facts2 and loc1 in facts2["PART_OF"]:
            return True

        # Check transitive PART_OF relationships
        if "PART_OF" in facts1:
            for parent in facts1["PART_OF"]:
                if self._is_location_hierarchy(parent, loc2, visited, max_depth):
                    return True

        if "PART_OF" in facts2:
            for parent in facts2["PART_OF"]:
                if self._is_location_hierarchy(parent, loc1, visited, max_depth):
                    return True

        return False
