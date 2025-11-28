"""
component_37_partial_observation.py

Generic Partial Observability Engine für Multi-Agent Epistemic Reasoning

Implementiert allgemeingültige Komponenten für:
- Partielle Beobachtung (Agenten sehen nur subset von Properties)
- Uniqueness Analysis (ist ein Property-Wert eindeutig?)
- Property-Based Partitioning (Gruppierung nach Properties)
- Second-Order Analysis (Meta-Reasoning über andere Agenten)

WICHTIG: Keine puzzle-spezifische Logik! Komplett generisch.

Anwendungen:
- Cheryl's Birthday (partial date observation: month vs day)
- Sum and Product Puzzle (one sees sum, other sees product)
- Hat Puzzles (see others' hats, not own)
- Logic Grid Puzzles (partial clues about properties)
- Cryptarithmetic (partial digit knowledge)

Autor: KAI Development Team
Erstellt: 2025-11-01
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from component_15_logging_config import get_logger
from component_35_epistemic_engine import EpistemicEngine

logger = get_logger(__name__)


# ============================================================================
# Data Structures (Generic)
# ============================================================================


@dataclass
class WorldObject:
    """
    Generic object in the world with multiple properties.

    Examples:
    - Date: {month: "May", day: 15}
    - Person: {name: "Alice", hat_color: "red", eye_color: "blue"}
    - Number: {sum: 5, product: 6, digits: [2, 3]}
    """

    object_id: str
    properties: Dict[str, Any]

    def get_property(self, property_name: str) -> Optional[Any]:
        """Get property value"""
        return self.properties.get(property_name)

    def matches(self, constraints: Dict[str, Any]) -> bool:
        """Check if object satisfies all constraints"""
        return all(
            self.properties.get(key) == value for key, value in constraints.items()
        )


@dataclass
class PartialObserver:
    """
    Agent that observes only a subset of object properties.

    Examples:
    - Albert observes only 'month' property of dates
    - Sum-Person observes only 'sum' property of number pairs
    - Hat-Wearer can't see own 'hat_color'
    """

    observer_id: str
    observable_properties: List[str]  # Properties this agent can see
    hidden_properties: List[str] = field(
        default_factory=list
    )  # Properties agent CANNOT see

    def observe(self, obj: WorldObject) -> Dict[str, Any]:
        """
        Observe object and return visible properties.

        Returns:
            Dict of property_name -> value for observable properties
        """
        observation = {}
        for prop_name in self.observable_properties:
            if prop_name in obj.properties:
                observation[prop_name] = obj.properties[prop_name]

        logger.debug(
            f"Observer '{self.observer_id}' observed: {observation}",
            extra={
                "observer": self.observer_id,
                "object": obj.object_id,
                "observation": observation,
            },
        )
        return observation

    def can_observe(self, property_name: str) -> bool:
        """Check if agent can observe given property"""
        return property_name in self.observable_properties


# ============================================================================
# Uniqueness Analysis (Generic)
# ============================================================================


class UniquenessAnalyzer:
    """
    Analyzes whether property values uniquely identify objects.

    Generic: Works for ANY property in ANY domain.

    Examples:
    - In dates [May 15, May 16, June 18], day=18 is unique (identifies object)
    - In hats [red, red, blue], color=blue is unique
    - In numbers [(2,3), (1,6)], sum=5 is unique
    """

    def __init__(self, objects: List[WorldObject]):
        self.objects = objects
        self._uniqueness_cache: Dict[Tuple[str, Any], bool] = {}

    def is_unique_identifier(self, property_name: str, property_value: Any) -> bool:
        """
        Check if property=value uniquely identifies exactly one object.

        Args:
            property_name: Property to check (e.g., "day", "color", "sum")
            property_value: Value to check (e.g., 18, "blue", 5)

        Returns:
            True if exactly one object has property=value
        """
        cache_key = (property_name, property_value)
        if cache_key in self._uniqueness_cache:
            return self._uniqueness_cache[cache_key]

        matching_objects = [
            obj
            for obj in self.objects
            if obj.get_property(property_name) == property_value
        ]

        is_unique = len(matching_objects) == 1
        self._uniqueness_cache[cache_key] = is_unique

        logger.debug(
            f"Uniqueness check: {property_name}={property_value} -> {is_unique}",
            extra={
                "property": property_name,
                "value": property_value,
                "is_unique": is_unique,
                "matching_count": len(matching_objects),
            },
        )

        return is_unique

    def get_unique_properties(self, property_name: str) -> Set[Any]:
        """
        Get all property values that uniquely identify objects.

        Args:
            property_name: Property to analyze

        Returns:
            Set of unique property values
        """
        unique_values = set()

        # Get all values for this property
        all_values = set(obj.get_property(property_name) for obj in self.objects)

        for value in all_values:
            if self.is_unique_identifier(property_name, value):
                unique_values.add(value)

        return unique_values

    def has_unique_identifier(self, constraints: Dict[str, Any]) -> bool:
        """
        Check if objects matching constraints have a unique identifier in ANY property.

        Args:
            constraints: Dict of property constraints (e.g., {month: "May"})

        Returns:
            True if matching objects can be uniquely identified by some property
        """
        matching_objects = [obj for obj in self.objects if obj.matches(constraints)]

        if len(matching_objects) <= 1:
            return True  # Trivially unique

        # Check if any property uniquely identifies these objects
        for obj in matching_objects:
            for prop_name, prop_value in obj.properties.items():
                if self.is_unique_identifier(prop_name, prop_value):
                    return True

        return False


# ============================================================================
# Property-Based Partitioning (Generic)
# ============================================================================


class PartitionAnalyzer:
    """
    Groups objects by property values (creates partitions).

    Generic: Works for ANY property in ANY domain.

    Examples:
    - Partition dates by month: {May: [...], June: [...], ...}
    - Partition people by eye_color: {blue: [...], brown: [...]}
    - Partition numbers by sum: {5: [...], 7: [...]}
    """

    def __init__(self, objects: List[WorldObject]):
        self.objects = objects

    def partition_by_property(self, property_name: str) -> Dict[Any, List[WorldObject]]:
        """
        Group objects by property value.

        Args:
            property_name: Property to partition by (e.g., "month", "color")

        Returns:
            Dict mapping property_value -> list of objects with that value
        """
        partitions = defaultdict(list)

        for obj in self.objects:
            prop_value = obj.get_property(property_name)
            if prop_value is not None:
                partitions[prop_value].append(obj)

        logger.debug(
            f"Partitioned by '{property_name}': {len(partitions)} partitions",
            extra={
                "property": property_name,
                "num_partitions": len(partitions),
                "partition_sizes": {k: len(v) for k, v in partitions.items()},
            },
        )

        return dict(partitions)

    def get_partition_size(self, property_name: str, property_value: Any) -> int:
        """Get number of objects in partition"""
        return len(
            [
                obj
                for obj in self.objects
                if obj.get_property(property_name) == property_value
            ]
        )

    def partition_has_unique_identifier(
        self, property_name: str, property_value: Any, other_property: str
    ) -> bool:
        """
        Check if partition contains object with unique identifier in other property.

        Generic check: "Does this partition contain an object that can be uniquely
        identified by other_property?"

        Args:
            property_name: Property defining partition (e.g., "month")
            property_value: Value defining partition (e.g., "May")
            other_property: Property to check for uniqueness (e.g., "day")

        Returns:
            True if partition contains object with unique other_property value

        Example:
            partition_has_unique_identifier("month", "May", "day")
            -> True if May partition contains day that appears nowhere else
        """
        # Get objects in partition
        partition_objects = [
            obj
            for obj in self.objects
            if obj.get_property(property_name) == property_value
        ]

        # Check if any object has unique identifier in other_property
        uniqueness = UniquenessAnalyzer(self.objects)

        for obj in partition_objects:
            other_value = obj.get_property(other_property)
            if other_value is not None and uniqueness.is_unique_identifier(
                other_property, other_value
            ):
                logger.debug(
                    f"Partition {property_name}={property_value} has unique {other_property}={other_value}",
                    extra={
                        "partition_property": property_name,
                        "partition_value": property_value,
                        "unique_property": other_property,
                        "unique_value": other_value,
                    },
                )
                return True

        return False


# ============================================================================
# Second-Order Analysis (Meta-Reasoning)
# ============================================================================


class SecondOrderAnalyzer:
    """
    Meta-reasoning about what other agents can/cannot know.

    Generic: Works for ANY multi-agent partial observation scenario.

    Key reasoning patterns:
    1. "I know the other agent CANNOT know" (Albert's first statement)
    2. "I know the other agent CAN know now" (after elimination)
    3. "The other agent says they know -> eliminate non-identifying values"
    """

    def __init__(
        self, objects: List[WorldObject], observers: Dict[str, PartialObserver]
    ):
        self.objects = objects
        self.observers = observers
        self.uniqueness = UniquenessAnalyzer(objects)
        self.partitions = PartitionAnalyzer(objects)

    def can_identify_object(
        self, observer_id: str, observation: Dict[str, Any]
    ) -> bool:
        """
        Generic: Can observer identify unique object given observation?

        Args:
            observer_id: ID of observer
            observation: Dict of property->value pairs the observer sees

        Returns:
            True if observation uniquely identifies exactly one object
        """
        matching_objects = [obj for obj in self.objects if obj.matches(observation)]

        can_identify = len(matching_objects) == 1

        logger.debug(
            f"Observer '{observer_id}' can_identify={can_identify} with observation {observation}",
            extra={
                "observer": observer_id,
                "observation": observation,
                "matching_objects": len(matching_objects),
                "can_identify": can_identify,
            },
        )

        return can_identify

    def knows_other_cannot_know(
        self,
        observer_id: str,
        other_observer_id: str,
        observer_observation: Dict[str, Any],
    ) -> bool:
        """
        Generic: "I know the other agent CANNOT uniquely identify the object"

        Meta-reasoning: Observer knows their observation, and can deduce that
        for ALL objects matching their observation, the other observer's
        observation would NOT uniquely identify the object.

        Args:
            observer_id: First observer
            other_observer_id: Second observer
            observer_observation: What first observer sees

        Returns:
            True if observer knows other cannot identify object

        Example (Cheryl's Birthday):
            Albert sees month=May
            For ALL dates in May, check if Bernard (seeing only day) could identify
            If May contains day=19 (unique), then Albert CANNOT know Bernard doesn't know
        """
        self.observers[observer_id]
        other_observer = self.observers[other_observer_id]

        # Get all objects matching observer's observation
        matching_objects = [
            obj for obj in self.objects if obj.matches(observer_observation)
        ]

        # For each matching object, check if other observer COULD identify it
        for obj in matching_objects:
            # What would other observer see?
            other_observation = other_observer.observe(obj)

            # Could other observer uniquely identify object with this observation?
            if self.can_identify_object(other_observer_id, other_observation):
                # Other COULD know for this object -> observer CANNOT be sure other doesn't know
                logger.debug(
                    f"Observer '{observer_id}' CANNOT know '{other_observer_id}' doesn't know (counterexample: {obj.object_id})",
                    extra={
                        "observer": observer_id,
                        "other_observer": other_observer_id,
                        "counterexample_object": obj.object_id,
                        "other_observation": other_observation,
                    },
                )
                return False

        # For ALL matching objects, other observer could NOT identify -> observer KNOWS other doesn't know
        logger.debug(
            f"Observer '{observer_id}' KNOWS '{other_observer_id}' cannot know",
            extra={
                "observer": observer_id,
                "other_observer": other_observer_id,
                "observer_observation": observer_observation,
                "matching_objects": len(matching_objects),
            },
        )
        return True

    def eliminate_by_statement(
        self,
        observer_id: str,
        statement_type: str,
        current_candidates: List[WorldObject],
    ) -> List[WorldObject]:
        """
        Generic elimination: Remove objects inconsistent with agent's statement.

        Args:
            observer_id: Agent making statement
            statement_type: Type of statement ("now_i_know", "i_know_other_doesnt_know", etc.)
            current_candidates: Current set of possible objects

        Returns:
            Filtered list of objects consistent with statement
        """
        observer = self.observers[observer_id]
        filtered = []

        for obj in current_candidates:
            # What does observer see for this object?
            observation = observer.observe(obj)

            # Build current context for this observation (only considering current candidates)
            context_objects = [
                o for o in current_candidates if observer.observe(o) == observation
            ]

            if statement_type == "now_i_know":
                # Agent says "now I know" -> they must be able to uniquely identify
                # This means: among current candidates with same observation, only one remains
                if len(context_objects) == 1:
                    filtered.append(obj)

            elif statement_type == "i_dont_know":
                # Agent says "I don't know" -> multiple objects with same observation
                if len(context_objects) > 1:
                    filtered.append(obj)

        logger.info(
            f"Eliminated by statement '{statement_type}' from '{observer_id}': {len(current_candidates)} -> {len(filtered)}",
            extra={
                "observer": observer_id,
                "statement_type": statement_type,
                "before": len(current_candidates),
                "after": len(filtered),
            },
        )

        return filtered


# ============================================================================
# Partial Observation Reasoning System
# ============================================================================


class PartialObservationReasoner:
    """
    Generic reasoning system for partial observation scenarios.

    Combines:
    - Uniqueness analysis
    - Partitioning
    - Second-order meta-reasoning
    - Epistemic engine integration
    """

    def __init__(self, engine: EpistemicEngine):
        self.engine = engine
        self.objects: List[WorldObject] = []
        self.observers: Dict[str, PartialObserver] = {}
        self.uniqueness: Optional[UniquenessAnalyzer] = None
        self.partitions: Optional[PartitionAnalyzer] = None
        self.second_order: Optional[SecondOrderAnalyzer] = None

        logger.info("PartialObservationReasoner initialized")

    def add_objects(self, objects: List[WorldObject]):
        """Add world objects"""
        self.objects = objects
        self.uniqueness = UniquenessAnalyzer(objects)
        self.partitions = PartitionAnalyzer(objects)

        logger.info(
            f"Added {len(objects)} objects", extra={"object_count": len(objects)}
        )

    def add_observer(self, observer: PartialObserver):
        """Add partial observer"""
        self.observers[observer.observer_id] = observer

        # Update second-order analyzer
        self.second_order = SecondOrderAnalyzer(self.objects, self.observers)

        logger.info(
            f"Added observer '{observer.observer_id}' (observes: {observer.observable_properties})",
            extra={
                "observer_id": observer.observer_id,
                "observable": observer.observable_properties,
            },
        )

    def establish_observations(self, target_object: WorldObject):
        """
        Establish initial observations for all observers.

        Each observer sees only their observable properties of target object.
        """
        for observer_id, observer in self.observers.items():
            observation = observer.observe(target_object)

            # Store as knowledge in epistemic engine
            for prop_name, prop_value in observation.items():
                knowledge_prop = f"{observer_id}_observes_{prop_name}_{prop_value}"
                self.engine.add_knowledge(observer_id, knowledge_prop)

        logger.info(
            f"Established observations for target object '{target_object.object_id}'",
            extra={
                "target_object": target_object.object_id,
                "observers": list(self.observers.keys()),
            },
        )

    def get_possible_objects(
        self, observer_id: str, observation: Dict[str, Any]
    ) -> List[WorldObject]:
        """
        Get all objects consistent with observation.

        Args:
            observer_id: Observer making observation
            observation: What observer sees

        Returns:
            List of objects matching observation
        """
        return [obj for obj in self.objects if obj.matches(observation)]


if __name__ == "__main__":
    print("\n=== Partial Observation Reasoning Test ===\n")

    from component_1_netzwerk import KonzeptNetzwerk
    from component_35_epistemic_engine import EpistemicEngine

    # Setup
    netzwerk = KonzeptNetzwerk()
    engine = EpistemicEngine(netzwerk)
    reasoner = PartialObservationReasoner(engine)

    # Create test objects (simple number pairs)
    objects = [
        WorldObject("pair1", {"sum": 5, "product": 6, "x": 2, "y": 3}),
        WorldObject("pair2", {"sum": 5, "product": 4, "x": 1, "y": 4}),
        WorldObject("pair3", {"sum": 7, "product": 12, "x": 3, "y": 4}),
        WorldObject("pair4", {"sum": 7, "product": 10, "x": 2, "y": 5}),
    ]

    reasoner.add_objects(objects)

    # Create observers (one sees sum, other sees product)
    sum_observer = PartialObserver("sum_person", observable_properties=["sum"])
    product_observer = PartialObserver(
        "product_person", observable_properties=["product"]
    )

    reasoner.add_observer(sum_observer)
    reasoner.add_observer(product_observer)

    # Test uniqueness analysis
    print("Testing uniqueness analysis...")
    assert reasoner.uniqueness.is_unique_identifier("product", 6) is True
    assert (
        reasoner.uniqueness.is_unique_identifier("sum", 5) is False
    )  # Two objects have sum=5
    print("[OK] Uniqueness analysis passed")

    # Test can_identify_object
    print("\nTesting can_identify_object...")
    assert (
        reasoner.second_order.can_identify_object("product_person", {"product": 6})
        is True
    )
    assert reasoner.second_order.can_identify_object("sum_person", {"sum": 5}) is False
    print("[OK] can_identify_object passed")

    # Test knows_other_cannot_know
    print("\nTesting knows_other_cannot_know...")
    # If sum_person sees sum=5, they know product_person CANNOT know (two options: product 6 or 4)
    # But this is only true if BOTH products are non-unique
    result = reasoner.second_order.knows_other_cannot_know(
        "sum_person", "product_person", {"sum": 5}
    )
    print(f"knows_other_cannot_know(sum_person, product_person, sum=5) = {result}")

    print("\n[OK] All tests passed!")
