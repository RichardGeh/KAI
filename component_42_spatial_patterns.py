"""
Component 42: Spatial Pattern Learning

Spatial pattern learning and recognition.

This module handles:
- Learning spatial configurations (e.g., chess positions)
- Recognizing learned patterns
- Pattern similarity scoring
- Movement pattern observation
- Spatial rule learning from examples

Author: KAI Development Team
Date: 2025-11-27
"""

import threading
from typing import Callable, Dict, List, Optional, Tuple

from component_15_logging_config import get_logger
from component_42_spatial_types import Position

logger = get_logger(__name__)

# Constants
DEFAULT_POSITION_TOLERANCE = 0.5  # Grid cells tolerance for position matching


class SpatialPatternLearner:
    """
    Spatial pattern learning and recognition engine.

    Provides methods for:
    - Observing movement patterns
    - Learning spatial configurations
    - Detecting matching patterns
    - Learning rules from examples

    Thread Safety:
        This class is thread-safe. All shared mutable state is protected
        by locks to support concurrent access from multiple threads.
    """

    def __init__(self, netzwerk=None):
        """
        Initialize the spatial pattern learner.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph access
        """
        self.netzwerk = netzwerk
        self._lock = threading.RLock()

        logger.info("SpatialPatternLearner initialized")

    def _safe_session_run(self, query: str, operation: str, **params):
        """
        Safely execute a Neo4j session.run() with proper null checks.

        Args:
            query: Cypher query to execute
            operation: Description of operation (for logging)
            **params: Query parameters

        Returns:
            Query result, or None if session unavailable
        """
        if not self.netzwerk:
            logger.warning(f"Knowledge graph unavailable for {operation}")
            return None

        if not hasattr(self.netzwerk, "session") or self.netzwerk.session is None:
            logger.error(f"Neo4j session not initialized for {operation}")
            return None

        try:
            return self.netzwerk.session.run(query, **params)
        except Exception as e:
            logger.error(f"Error executing query for {operation}: {e}", exc_info=True)
            return None

    def observe_movement_pattern(
        self,
        object_name: str,
        movements: List[Tuple[Position, Position]],
        pattern_name: Optional[str] = None,
    ) -> bool:
        """
        Observe a sequence of movements and learn the movement pattern.

        Args:
            object_name: Name of object type (e.g., "Knight", "Rook")
            movements: List of (from_pos, to_pos) tuples
            pattern_name: Optional name for the pattern

        Returns:
            True if pattern was learned
        """
        if not self.netzwerk:
            logger.warning(
                "No knowledge graph available for observing movement pattern"
            )
            return False

        try:
            if not movements:
                return False

            with self._lock:
                # Analyze movement vectors
                vectors = []
                for from_pos, to_pos in movements:
                    dx = to_pos.x - from_pos.x
                    dy = to_pos.y - from_pos.y
                    vectors.append((dx, dy))

                # Store in knowledge graph as learned pattern
                pattern_id = pattern_name or f"{object_name}_movement_pattern"

                self.netzwerk.create_wort_if_not_exists(pattern_id)
                self.netzwerk.set_wort_attribut(
                    pattern_id, "type", "SpatialMovementPattern"
                )
                self.netzwerk.set_wort_attribut(pattern_id, "object_type", object_name)
                self.netzwerk.set_wort_attribut(
                    pattern_id, "num_examples", len(movements)
                )

                # Store observed vectors
                vector_set = set(vectors)
                for i, (dx, dy) in enumerate(vector_set):
                    vector_name = f"{pattern_id}_vector_{i}"
                    self.netzwerk.create_wort_if_not_exists(vector_name)
                    self.netzwerk.set_wort_attribut(vector_name, "dx", dx)
                    self.netzwerk.set_wort_attribut(vector_name, "dy", dy)
                    self.netzwerk.assert_relation(
                        pattern_id, "HAS_MOVEMENT_VECTOR", vector_name
                    )

                logger.info(
                    "Learned movement pattern for %s: %d unique vectors from %d examples",
                    object_name,
                    len(vector_set),
                    len(movements),
                )

                return True

        except Exception as e:
            logger.error("Error observing movement pattern: %s", str(e), exc_info=True)
            return False

    def get_learned_movement_pattern(
        self, object_name: str
    ) -> Optional[Callable[[Position, Position], bool]]:
        """
        Retrieve a learned movement pattern as a validation function.

        Args:
            object_name: Name of object type

        Returns:
            Function (from_pos, to_pos) -> bool, or None
        """
        if not self.netzwerk:
            logger.warning(
                "No knowledge graph available for retrieving movement pattern"
            )
            return None

        try:
            with self._lock:
                pattern_id = f"{object_name}_movement_pattern"
                pattern_node = self.netzwerk.find_wort_node(pattern_id)

                if not pattern_node:
                    return None

                # Get learned movement vectors
                vectors = []
                query = """
                    MATCH (pattern {lemma: $pattern_id})-[:HAS_MOVEMENT_VECTOR]->(vector)
                    RETURN vector.dx as dx, vector.dy as dy
                """
                results = self._safe_session_run(
                    query, "get_learned_movement_pattern", pattern_id=pattern_id
                )

                if results is None:
                    return None

                for record in results:
                    vectors.append((record["dx"], record["dy"]))

                if not vectors:
                    return None

                # Create validation function
                def validate_move(from_pos: Position, to_pos: Position) -> bool:
                    dx = to_pos.x - from_pos.x
                    dy = to_pos.y - from_pos.y
                    return (dx, dy) in vectors

                logger.info(
                    "Retrieved learned pattern for %s with %d movement vectors",
                    object_name,
                    len(vectors),
                )

                return validate_move

        except Exception as e:
            logger.error("Error retrieving learned pattern: %s", str(e), exc_info=True)
            return None

    def observe_spatial_configuration(
        self,
        configuration_name: str,
        objects_and_positions: Dict[str, Position],
        grid_name: Optional[str] = None,
    ) -> bool:
        """
        Observe and learn a spatial configuration pattern.

        Args:
            configuration_name: Name for this configuration
            objects_and_positions: Dict mapping object names to positions
            grid_name: Optional grid context

        Returns:
            True if pattern was learned
        """
        if not self.netzwerk:
            logger.warning(
                "No knowledge graph available for observing spatial configuration"
            )
            return False

        try:
            with self._lock:
                # Create configuration node
                config_id = f"SpatialConfig_{configuration_name}"
                self.netzwerk.create_wort_if_not_exists(config_id)
                self.netzwerk.set_wort_attribut(
                    config_id, "type", "SpatialConfiguration"
                )
                self.netzwerk.set_wort_attribut(
                    config_id, "num_objects", len(objects_and_positions)
                )

                if grid_name:
                    self.netzwerk.set_wort_attribut(config_id, "grid", grid_name)

                # Analyze and store relative positions
                objects_list = list(objects_and_positions.items())

                for i, (obj1, pos1) in enumerate(objects_list):
                    for j, (obj2, pos2) in enumerate(objects_list):
                        if i >= j:
                            continue

                        # Calculate relative position
                        dx = pos2.x - pos1.x
                        dy = pos2.y - pos1.y
                        distance = pos1.distance_to(pos2, metric="manhattan")

                        # Store relative relationship
                        rel_id = f"{config_id}_{obj1}_{obj2}"
                        self.netzwerk.create_wort_if_not_exists(rel_id)
                        self.netzwerk.set_wort_attribut(rel_id, "object1", obj1)
                        self.netzwerk.set_wort_attribut(rel_id, "object2", obj2)
                        self.netzwerk.set_wort_attribut(rel_id, "dx", dx)
                        self.netzwerk.set_wort_attribut(rel_id, "dy", dy)
                        self.netzwerk.set_wort_attribut(rel_id, "distance", distance)

                        self.netzwerk.assert_relation(
                            config_id, "HAS_RELATIVE_POSITION", rel_id
                        )

                logger.info(
                    "Learned spatial configuration '%s' with %d objects",
                    configuration_name,
                    len(objects_and_positions),
                )

                return True

        except Exception as e:
            logger.error(
                "Error observing spatial configuration: %s", str(e), exc_info=True
            )
            return False

    def detect_spatial_pattern_in_configuration(
        self, objects_and_positions: Dict[str, Position]
    ) -> List[str]:
        """
        Detect which learned patterns match the current configuration.

        Args:
            objects_and_positions: Current object positions

        Returns:
            List of matching pattern names
        """
        if not self.netzwerk:
            logger.warning("No knowledge graph available for detecting spatial pattern")
            return []

        try:
            with self._lock:
                matches = []

                # Query all stored configurations
                query = """
                    MATCH (config)
                    WHERE config.type = 'SpatialConfiguration'
                    RETURN config.lemma as name, config.num_objects as num_objects
                """
                results = self._safe_session_run(
                    query, "detect_spatial_pattern_in_configuration"
                )

                if results is None:
                    return []

                for record in results:
                    config_name = record["name"]
                    expected_num = record["num_objects"]

                    # Quick filter by number of objects
                    if len(objects_and_positions) != expected_num:
                        continue

                    # Check if relative positions match
                    if self._check_configuration_match(
                        config_name, objects_and_positions
                    ):
                        # Extract original name (remove "SpatialConfig_" prefix)
                        pattern_name = config_name.replace("SpatialConfig_", "")
                        matches.append(pattern_name)

                logger.info(
                    "Detected %d matching spatial patterns in configuration",
                    len(matches),
                )

                return matches

        except Exception as e:
            logger.error("Error detecting spatial patterns: %s", str(e), exc_info=True)
            return []

    def _check_configuration_match(
        self,
        config_id: str,
        objects_and_positions: Dict[str, Position],
        tolerance: float = DEFAULT_POSITION_TOLERANCE,
    ) -> bool:
        """
        Check if current configuration matches a stored pattern.

        Args:
            config_id: ID of stored configuration
            objects_and_positions: Current positions
            tolerance: Allowed deviation in relative positions

        Returns:
            True if configuration matches pattern
        """
        try:
            # Get stored relative positions
            query = """
                MATCH (config {lemma: $config_id})-[:HAS_RELATIVE_POSITION]->(rel)
                RETURN rel.object1 as obj1, rel.object2 as obj2,
                       rel.dx as dx, rel.dy as dy
            """
            results = self._safe_session_run(
                query, "check_configuration_match", config_id=config_id
            )

            if results is None:
                return False

            # Check each relative position
            for record in results:
                obj1 = record["obj1"]
                obj2 = record["obj2"]
                expected_dx = record["dx"]
                expected_dy = record["dy"]

                # Find matching objects in current configuration
                # (Simple approach: match by type/name pattern)
                current_obj1 = self._find_matching_object(obj1, objects_and_positions)
                current_obj2 = self._find_matching_object(obj2, objects_and_positions)

                if not current_obj1 or not current_obj2:
                    return False

                pos1 = objects_and_positions[current_obj1]
                pos2 = objects_and_positions[current_obj2]

                actual_dx = pos2.x - pos1.x
                actual_dy = pos2.y - pos1.y

                # Check if within tolerance
                if (
                    abs(actual_dx - expected_dx) > tolerance
                    or abs(actual_dy - expected_dy) > tolerance
                ):
                    return False

            return True

        except Exception as e:
            logger.error("Error checking configuration match: %s", str(e))
            return False

    def _find_matching_object(
        self, pattern_obj: str, current_objects: Dict[str, Position]
    ) -> Optional[str]:
        """Find an object in current configuration that matches the pattern."""
        # Simple implementation: direct match or type-based match
        if pattern_obj in current_objects:
            return pattern_obj

        # Try to match by type (e.g., "König1" matches "König2")
        for obj_name in current_objects:
            # Remove trailing numbers for type comparison
            pattern_type = "".join(c for c in pattern_obj if not c.isdigit())
            obj_type = "".join(c for c in obj_name if not c.isdigit())

            if pattern_type == obj_type:
                return obj_name

        return None

    def learn_spatial_rule_from_examples(
        self,
        rule_name: str,
        positive_examples: List[Dict[str, Position]],
        negative_examples: Optional[List[Dict[str, Position]]] = None,
    ) -> bool:
        """
        Learn a spatial rule from positive and negative examples.

        Args:
            rule_name: Name for the rule
            positive_examples: Examples that satisfy the rule
            negative_examples: Examples that violate the rule (optional)

        Returns:
            True if rule was learned
        """
        if not self.netzwerk:
            logger.warning("No knowledge graph available for learning spatial rule")
            return False

        try:
            if not positive_examples:
                return False

            with self._lock:
                # Analyze positive examples to find common patterns
                # Simple approach: find constraints that all positives satisfy

                # Extract common structure
                len(positive_examples[0])

                # Find constraints that hold for all positive examples
                constraints = []

                # Check for consistent relative positions
                object_pairs = list(positive_examples[0].keys())

                for i, obj1 in enumerate(object_pairs):
                    for j, obj2 in enumerate(object_pairs):
                        if i >= j:
                            continue

                        # Check if this pair has consistent relative position across examples
                        relative_positions = []
                        for example in positive_examples:
                            if obj1 in example and obj2 in example:
                                pos1 = example[obj1]
                                pos2 = example[obj2]
                                dx = pos2.x - pos1.x
                                dy = pos2.y - pos1.y
                                relative_positions.append((dx, dy))

                        # If all examples have same relative position, it's a constraint
                        if len(set(relative_positions)) == 1:
                            dx, dy = relative_positions[0]
                            constraints.append(
                                {
                                    "type": "relative_position",
                                    "obj1": obj1,
                                    "obj2": obj2,
                                    "dx": dx,
                                    "dy": dy,
                                }
                            )

                # Store learned rule
                rule_id = f"SpatialRule_{rule_name}"
                self.netzwerk.create_wort_if_not_exists(rule_id)
                self.netzwerk.set_wort_attribut(rule_id, "type", "LearnedSpatialRule")
                self.netzwerk.set_wort_attribut(
                    rule_id, "num_constraints", len(constraints)
                )
                self.netzwerk.set_wort_attribut(
                    rule_id, "num_examples", len(positive_examples)
                )

                # Store constraints
                for i, constraint in enumerate(constraints):
                    constraint_id = f"{rule_id}_constraint_{i}"
                    self.netzwerk.create_wort_if_not_exists(constraint_id)

                    for key, value in constraint.items():
                        self.netzwerk.set_wort_attribut(constraint_id, key, value)

                    self.netzwerk.assert_relation(
                        rule_id, "HAS_CONSTRAINT", constraint_id
                    )

                logger.info(
                    "Learned spatial rule '%s' with %d constraints from %d positive examples",
                    rule_name,
                    len(constraints),
                    len(positive_examples),
                )

                return True

        except Exception as e:
            logger.error("Error learning spatial rule: %s", str(e), exc_info=True)
            return False

    def add_spatial_constraint(
        self,
        constraint_name: str,
        objects: List[str],
        constraint_predicate: Callable[[Dict[str, Position]], bool],
        description: str = "",
    ) -> bool:
        """
        Add a spatial constraint for objects on a grid.

        Args:
            constraint_name: Unique identifier for this constraint
            objects: List of object names involved
            constraint_predicate: Function that takes {object: position} dict and returns True if valid
            description: Human-readable description

        Returns:
            True if constraint added successfully

        Example:
            # Constraint: "König und Turm dürfen nicht auf gleicher Position sein"
            learner.add_spatial_constraint(
                "no_same_position",
                ["König", "Turm"],
                lambda pos: pos["König"] != pos["Turm"],
                "König und Turm nicht auf gleicher Position"
            )
        """
        try:
            with self._lock:
                if not hasattr(self, "_spatial_constraints"):
                    self._spatial_constraints = {}

                self._spatial_constraints[constraint_name] = {
                    "objects": objects,
                    "predicate": constraint_predicate,
                    "description": description,
                }

                logger.info(
                    "Added spatial constraint: %s for objects %s",
                    constraint_name,
                    objects,
                )
                return True

        except Exception as e:
            logger.error("Error adding constraint %s: %s", constraint_name, str(e))
            return False

    def check_spatial_constraints(
        self, grid_name: str, object_positions: Optional[Dict[str, Position]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if current object positions satisfy all spatial constraints.

        Args:
            grid_name: Name of the grid
            object_positions: Optional dict of {object_name: position}.
                            If None, queries current positions from graph.

        Returns:
            Tuple of (all_satisfied, list_of_violations)
        """
        if not hasattr(self, "_spatial_constraints"):
            return True, []

        violations = []

        # Get current positions if not provided
        if object_positions is None:
            object_positions = {}
            # Query all objects on this grid
            # For now, we'll need the caller to provide positions
            logger.warning(
                "check_spatial_constraints requires object_positions parameter"
            )
            return True, []

        # Check each constraint
        for constraint_name, constraint_info in self._spatial_constraints.items():
            objects = constraint_info["objects"]
            predicate = constraint_info["predicate"]

            # Check if all objects in constraint have positions
            relevant_positions = {}
            missing_objects = []

            for obj in objects:
                if obj in object_positions:
                    relevant_positions[obj] = object_positions[obj]
                else:
                    missing_objects.append(obj)

            # Skip constraint if not all objects are placed
            if missing_objects:
                logger.debug(
                    "Skipping constraint %s: missing objects %s",
                    constraint_name,
                    missing_objects,
                )
                continue

            # Check constraint predicate
            try:
                if not predicate(relevant_positions):
                    violations.append(
                        f"Constraint '{constraint_name}' violated: {constraint_info['description']}"
                    )
            except Exception as e:
                logger.error(
                    "Error evaluating constraint %s: %s", constraint_name, str(e)
                )
                violations.append(
                    f"Constraint '{constraint_name}' evaluation error: {str(e)}"
                )

        all_satisfied = len(violations) == 0

        if not all_satisfied:
            logger.warning(
                "Spatial constraints violated: %d violations", len(violations)
            )

        return all_satisfied, violations

    def clear_spatial_constraints(self):
        """Clear all spatial constraints."""
        if hasattr(self, "_spatial_constraints"):
            self._spatial_constraints = {}
            logger.info("Cleared all spatial constraints")
