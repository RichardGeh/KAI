"""
component_31_ontology_constraints.py
====================================
Automatic Constraint Generation from Ontology for SAT-based Reasoning.

Analyzes Neo4j knowledge graph to extract semantic constraints:
- IS_A exclusivity: Sibling concepts at same hierarchy level are mutually exclusive
- Property compatibility: Contradictory properties (e.g., colors, sizes)
- Location constraints: Objects cannot be in multiple non-hierarchical locations

Integrates with component_30_sat_solver for semantic contradiction detection.

Author: KAI Development Team
Date: 2025-10-31
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from component_15_logging_config import get_logger

# Integration mit SAT Solver
try:
    from component_30_sat_solver import Clause, Literal

    SAT_AVAILABLE = True
except ImportError:
    SAT_AVAILABLE = False
    logging.warning("component_30 not available, constraint generation disabled")

logger = get_logger(__name__)


def safe_str(obj) -> str:
    """
    Convert object to cp1252-safe string for Windows compatibility.

    Replaces characters that cannot be encoded in Windows cp1252 to prevent
    UnicodeEncodeError in logging and console output.

    Args:
        obj: Object to convert to string

    Returns:
        cp1252-safe string representation
    """
    s = str(obj)
    try:
        # Test if string can be encoded in cp1252
        s.encode("cp1252")
        return s
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Replace problematic characters
        return s.encode("cp1252", errors="replace").decode("cp1252")


@dataclass
class OntologyConstraint:
    """
    Semantic constraint extracted from ontology.

    Attributes:
        constraint_type: Type of constraint (IS_A_EXCLUSIVITY, PROPERTY_CONFLICT, etc.)
        entities: Entities involved in constraint
        clauses: CNF clauses encoding this constraint
        explanation: Human-readable explanation
    """

    constraint_type: str
    entities: List[str]
    clauses: List["Clause"]
    explanation: str
    confidence: float = 1.0


class OntologyConstraintGenerator:
    """
    Generates semantic constraints from Neo4j knowledge graph.

    Features:
    - IS_A exclusivity constraints for sibling concepts
    - Property compatibility constraints
    - Location hierarchy constraints
    - Capability constraints
    - Automatic constraint caching and invalidation
    """

    def __init__(self, netzwerk, enable_caching: bool = True):
        """
        Initialize constraint generator.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph access
            enable_caching: Enable constraint caching for performance
        """
        self.netzwerk = netzwerk
        self.enable_caching = enable_caching
        self._constraint_cache: Dict[str, List[OntologyConstraint]] = {}
        self._hierarchy_cache: Dict[str, Set[str]] = {}  # concept -> siblings

        # Predefined mutually exclusive property groups
        self.property_groups = {
            "colors": {
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
                "rosa",
                "türkis",
            },
            "sizes": {
                "klein",
                "groß",
                "winzig",
                "riesig",
                "mittel",
                "sehr groß",
                "sehr klein",
                "mikro",
                "makro",
            },
            "temperatures": {
                "heiß",
                "kalt",
                "warm",
                "kühl",
                "eiskalt",
                "glühend",
                "siedend",
                "gefroren",
                "lauwarm",
            },
            "states": {
                "lebendig",
                "tot",
                "aktiv",
                "inaktiv",
                "offen",
                "geschlossen",
                "an",
                "aus",
                "voll",
                "leer",
            },
            "speeds": {
                "schnell",
                "langsam",
                "rasend",
                "träge",
                "blitzschnell",
                "gemächlich",
            },
        }

    def generate_constraints(
        self, constraint_types: Optional[List[str]] = None
    ) -> List[OntologyConstraint]:
        """
        Generate all semantic constraints from ontology.

        Args:
            constraint_types: List of constraint types to generate.
                             None = all types. Options: IS_A_EXCLUSIVITY,
                             PROPERTY_CONFLICT, LOCATION_CONFLICT

        Returns:
            List of OntologyConstraint objects
        """
        if constraint_types is None:
            constraint_types = [
                "IS_A_EXCLUSIVITY",
                "PROPERTY_CONFLICT",
                "LOCATION_CONFLICT",
            ]

        all_constraints = []

        for ctype in constraint_types:
            # Check cache first
            if self.enable_caching and ctype in self._constraint_cache:
                logger.debug(f"Using cached constraints for {ctype}")
                all_constraints.extend(self._constraint_cache[ctype])
                continue

            # Generate constraints based on type
            if ctype == "IS_A_EXCLUSIVITY":
                constraints = self._generate_is_a_exclusivity_constraints()
            elif ctype == "PROPERTY_CONFLICT":
                constraints = self._generate_property_conflict_constraints()
            elif ctype == "LOCATION_CONFLICT":
                constraints = self._generate_location_conflict_constraints()
            else:
                logger.warning(f"Unknown constraint type: {ctype}")
                constraints = []

            # Cache results
            if self.enable_caching:
                self._constraint_cache[ctype] = constraints

            all_constraints.extend(constraints)

        logger.info(
            f"Generated {len(all_constraints)} ontology constraints "
            f"({', '.join(constraint_types)})"
        )

        return all_constraints

    def _generate_is_a_exclusivity_constraints(self) -> List[OntologyConstraint]:
        """
        Generate IS_A exclusivity constraints for sibling concepts.

        Strategy:
        1. Find all IS_A hierarchies in graph
        2. For each parent concept, identify all children (siblings)
        3. Generate at-most-one constraint for sibling IS_A relations

        Example:
            Hierarchy: lebewesen -> [tier, pflanze, pilz]
            Constraint: at-most-one(X IS_A tier, X IS_A pflanze, X IS_A pilz)
            CNF: ¬(X_tier) ∨ ¬(X_pflanze), ¬(X_tier) ∨ ¬(X_pilz), ¬(X_pflanze) ∨ ¬(X_pilz)

        Returns:
            List of IS_A exclusivity constraints
        """
        if not SAT_AVAILABLE:
            logger.warning("SAT solver not available - skipping IS_A constraints")
            return []

        logger.info("Generating IS_A exclusivity constraints from ontology...")

        constraints: List[OntologyConstraint] = []

        if not self.netzwerk or not self.netzwerk.driver:
            logger.warning("No netzwerk connection - cannot generate IS_A constraints")
            return constraints

        # Query Neo4j for IS_A hierarchy
        db_name = getattr(self.netzwerk, "database_name", "neo4j")
        with self.netzwerk.driver.session(database=db_name) as session:
            # Find all parent concepts with multiple children
            query = """
                MATCH (parent:Konzept)<-[:IS_A]-(child:Konzept)
                WITH parent, collect(DISTINCT child.name) AS children
                WHERE size(children) > 1
                RETURN parent.name AS parent, children
                ORDER BY size(children) DESC
            """
            result = session.run(query)

            for record in result:
                parent = record["parent"]
                siblings = record["children"]

                if len(siblings) < 2:
                    continue

                logger.debug(
                    f"Found {len(siblings)} siblings under '{parent}': {siblings}"
                )

                # Cache siblings for later use
                for sibling in siblings:
                    if sibling not in self._hierarchy_cache:
                        self._hierarchy_cache[sibling] = set()
                    self._hierarchy_cache[sibling].update(
                        s for s in siblings if s != sibling
                    )

                # Generate at-most-one constraints for all sibling pairs
                # For N siblings, we need N*(N-1)/2 pairwise exclusivity clauses

                for i, sib1 in enumerate(siblings):
                    for sib2 in siblings[i + 1 :]:
                        # Create literals: IS_A(?x, sib1) and IS_A(?x, sib2)
                        # We use a generic variable pattern since this applies to ANY entity
                        lit1_name = f"IS_A_ANY_{sib1}"
                        lit2_name = f"IS_A_ANY_{sib2}"

                        lit1 = Literal(lit1_name, negated=False)
                        lit2 = Literal(lit2_name, negated=False)

                        # at-most-one(lit1, lit2) = ¬lit1 ∨ ¬lit2
                        clause = Clause({-lit1, -lit2})

                        constraint = OntologyConstraint(
                            constraint_type="IS_A_EXCLUSIVITY",
                            entities=[sib1, sib2],
                            clauses=[clause],
                            explanation=(
                                f"'{safe_str(sib1)}' and '{safe_str(sib2)}' are mutually exclusive "
                                f"(both children of '{safe_str(parent)}')"
                            ),
                            confidence=1.0,
                        )

                        constraints.append(constraint)

        logger.info(
            f"Generated {len(constraints)} IS_A exclusivity constraints "
            f"from {len(self._hierarchy_cache)} concepts"
        )

        return constraints

    def _generate_property_conflict_constraints(self) -> List[OntologyConstraint]:
        """
        Generate property conflict constraints.

        Uses predefined property groups (colors, sizes, etc.) to identify
        mutually exclusive properties.

        Example:
            Property group: colors = {rot, blau, grün}
            Constraint: at-most-one(X HAS rot, X HAS blau, X HAS grün)

        Returns:
            List of property conflict constraints
        """
        if not SAT_AVAILABLE:
            logger.warning("SAT solver not available - skipping property constraints")
            return []

        logger.info("Generating property conflict constraints...")

        constraints = []

        for group_name, properties in self.property_groups.items():
            properties_list = list(properties)

            # Generate pairwise exclusivity for all properties in group
            for i, prop1 in enumerate(properties_list):
                for prop2 in properties_list[i + 1 :]:
                    # Create literals: HAS_PROPERTY(?x, prop1) and HAS_PROPERTY(?x, prop2)
                    lit1_name = f"HAS_PROPERTY_ANY_{prop1}"
                    lit2_name = f"HAS_PROPERTY_ANY_{prop2}"

                    lit1 = Literal(lit1_name, negated=False)
                    lit2 = Literal(lit2_name, negated=False)

                    # at-most-one(lit1, lit2) = ¬lit1 ∨ ¬lit2
                    clause = Clause({-lit1, -lit2})

                    constraint = OntologyConstraint(
                        constraint_type="PROPERTY_CONFLICT",
                        entities=[prop1, prop2],
                        clauses=[clause],
                        explanation=(
                            f"'{safe_str(prop1)}' and '{safe_str(prop2)}' are mutually exclusive "
                            f"({safe_str(group_name)} conflict)"
                        ),
                        confidence=1.0,
                    )

                    constraints.append(constraint)

        logger.info(
            f"Generated {len(constraints)} property conflict constraints "
            f"from {len(self.property_groups)} property groups"
        )

        return constraints

    def _generate_location_conflict_constraints(self) -> List[OntologyConstraint]:
        """
        Generate location conflict constraints.

        Strategy:
        1. Find all PART_OF hierarchies (location hierarchies)
        2. Identify sibling locations (same parent)
        3. Generate at-most-one constraint for non-hierarchical locations

        Example:
            Hierarchy: europa -> [deutschland, frankreich, italien]
            Constraint: Object cannot be LOCATED_IN multiple sibling locations

        Returns:
            List of location conflict constraints
        """
        if not SAT_AVAILABLE:
            logger.warning("SAT solver not available - skipping location constraints")
            return []

        logger.info("Generating location conflict constraints...")

        constraints: List[OntologyConstraint] = []

        if not self.netzwerk or not self.netzwerk.driver:
            logger.warning(
                "No netzwerk connection - cannot generate location constraints"
            )
            return constraints

        # Query Neo4j for PART_OF hierarchy (location hierarchy)
        db_name = getattr(self.netzwerk, "database_name", "neo4j")
        with self.netzwerk.driver.session(database=db_name) as session:
            # Find all parent locations with multiple children
            query = """
                MATCH (parent:Konzept)<-[:PART_OF]-(child:Konzept)
                WITH parent, collect(DISTINCT child.name) AS children
                WHERE size(children) > 1
                RETURN parent.name AS parent, children
                ORDER BY size(children) DESC
            """
            result = session.run(query)

            for record in result:
                parent = record["parent"]
                sibling_locations = record["children"]

                if len(sibling_locations) < 2:
                    continue

                logger.debug(
                    f"Found {len(sibling_locations)} sibling locations under '{parent}': "
                    f"{sibling_locations}"
                )

                # Generate at-most-one constraints for sibling locations
                for i, loc1 in enumerate(sibling_locations):
                    for loc2 in sibling_locations[i + 1 :]:
                        # Create literals: LOCATED_IN(?x, loc1) and LOCATED_IN(?x, loc2)
                        lit1_name = f"LOCATED_IN_ANY_{loc1}"
                        lit2_name = f"LOCATED_IN_ANY_{loc2}"

                        lit1 = Literal(lit1_name, negated=False)
                        lit2 = Literal(lit2_name, negated=False)

                        # at-most-one(lit1, lit2) = ¬lit1 ∨ ¬lit2
                        clause = Clause({-lit1, -lit2})

                        constraint = OntologyConstraint(
                            constraint_type="LOCATION_CONFLICT",
                            entities=[loc1, loc2],
                            clauses=[clause],
                            explanation=(
                                f"'{safe_str(loc1)}' and '{safe_str(loc2)}' are mutually exclusive locations "
                                f"(both parts of '{safe_str(parent)}')"
                            ),
                            confidence=1.0,
                        )

                        constraints.append(constraint)

        logger.info(f"Generated {len(constraints)} location conflict constraints")

        return constraints

    def get_siblings_for_concept(self, concept: str) -> Set[str]:
        """
        Get all sibling concepts for a given concept.

        Siblings are concepts at the same hierarchy level (share same parent).

        Args:
            concept: Concept name

        Returns:
            Set of sibling concept names
        """
        if concept in self._hierarchy_cache:
            return self._hierarchy_cache[concept].copy()

        # Query graph if not cached
        if not self.netzwerk or not self.netzwerk.driver:
            return set()

        siblings = set()

        db_name = getattr(self.netzwerk, "database_name", "neo4j")
        with self.netzwerk.driver.session(database=db_name) as session:
            # Find siblings via shared parent
            query = """
                MATCH (concept:Konzept {name: $concept})-[:IS_A]->(parent:Konzept)
                MATCH (parent)<-[:IS_A]-(sibling:Konzept)
                WHERE sibling.name <> $concept
                RETURN DISTINCT sibling.name AS sibling
            """
            result = session.run(query, concept=concept.lower())

            for record in result:
                siblings.add(record["sibling"])

        # Cache result
        self._hierarchy_cache[concept] = siblings

        return siblings

    def are_concepts_mutually_exclusive(self, concept1: str, concept2: str) -> bool:
        """
        Check if two concepts are mutually exclusive (siblings in IS_A hierarchy).

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            True if concepts are mutually exclusive (siblings)
        """
        siblings1 = self.get_siblings_for_concept(concept1)
        return concept2 in siblings1

    def invalidate_cache(self):
        """Invalidate all cached constraints (e.g., after ontology changes)."""
        self._constraint_cache.clear()
        self._hierarchy_cache.clear()
        logger.info("Constraint cache invalidated")

    def get_constraint_statistics(self) -> Dict[str, int]:
        """
        Get statistics about generated constraints.

        Returns:
            Dict with constraint counts by type
        """
        stats = defaultdict(int)

        for ctype, constraints in self._constraint_cache.items():
            stats[ctype] = len(constraints)

        return dict(stats)


# Convenience functions for integration with Logic Engine


def generate_ontology_cnf_clauses(
    netzwerk, constraint_types: Optional[List[str]] = None
) -> List[Clause]:
    """
    Generate CNF clauses from ontology constraints.

    Convenience function for easy integration with Logic Engine.

    Args:
        netzwerk: KonzeptNetzwerk instance
        constraint_types: List of constraint types to generate

    Returns:
        List of CNF clauses encoding all ontology constraints
    """
    generator = OntologyConstraintGenerator(netzwerk)
    constraints = generator.generate_constraints(constraint_types)

    all_clauses = []
    for constraint in constraints:
        all_clauses.extend(constraint.clauses)

    return all_clauses


def check_concept_compatibility(
    netzwerk, concept1: str, concept2: str, relation_type: str = "IS_A"
) -> Tuple[bool, Optional[str]]:
    """
    Check if two concepts are compatible for a given relation.

    Args:
        netzwerk: KonzeptNetzwerk instance
        concept1: First concept
        concept2: Second concept
        relation_type: Relation type (IS_A, HAS_PROPERTY, etc.)

    Returns:
        Tuple (is_compatible, explanation)
    """
    generator = OntologyConstraintGenerator(netzwerk)

    if relation_type == "IS_A":
        if generator.are_concepts_mutually_exclusive(concept1, concept2):
            return (
                False,
                f"'{concept1}' and '{concept2}' are mutually exclusive siblings",
            )

    elif relation_type == "HAS_PROPERTY":
        # Check if properties are in same mutually exclusive group
        for group_name, properties in generator.property_groups.items():
            if concept1 in properties and concept2 in properties:
                return (
                    False,
                    f"'{concept1}' and '{concept2}' both belong to mutually exclusive group '{group_name}'",
                )

    return True, None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Mock netzwerk for testing
    from unittest.mock import Mock

    mock_netzwerk = Mock()

    generator = OntologyConstraintGenerator(mock_netzwerk)

    print("=== Property Conflict Constraints ===")
    property_constraints = generator._generate_property_conflict_constraints()
    print(f"Generated {len(property_constraints)} property constraints")
    for c in property_constraints[:5]:
        print(f"  {c.explanation}")

    print("\n=== Constraint Statistics ===")
    stats = generator.get_constraint_statistics()
    for ctype, count in stats.items():
        print(f"  {ctype}: {count} constraints")
