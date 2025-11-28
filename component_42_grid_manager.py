"""
Component 42: Grid Manager

Grid topology and object management for spatial reasoning.

This module handles:
- Grid creation and topology management
- Object placement, movement, and removal on grids
- Neighborhood queries
- Grid state persistence

Author: KAI Development Team
Date: 2025-11-27
"""

import threading
from typing import Dict, List, Optional

from component_15_logging_config import get_logger
from component_42_spatial_grid import Grid
from component_42_spatial_types import Position

logger = get_logger(__name__)


class GridManager:
    """
    Grid topology and object manager for spatial reasoning.

    Provides methods for:
    - Creating and deleting grids
    - Placing and moving objects on grids
    - Querying object positions and neighbors
    - Managing grid state

    Thread Safety:
        This class is thread-safe. All shared mutable state is protected
        by locks to support concurrent access from multiple threads.
    """

    def __init__(self, netzwerk=None):
        """
        Initialize the grid manager.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph access
        """
        self.netzwerk = netzwerk
        self._lock = threading.RLock()  # For thread safety

        logger.info("GridManager initialized")

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

    def create_grid(self, grid: Grid) -> bool:
        """
        Create a grid in the knowledge graph.

        Creates:
        - Grid concept node with metadata
        - Position nodes for each cell
        - Neighborhood relationships between positions

        Args:
            grid: Grid specification

        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating grid in knowledge graph: %s", grid)

        try:
            with self._lock:
                # Step 1: Create Grid concept node
                grid_properties = {
                    "width": grid.width,
                    "height": grid.height,
                    "size": grid.size,
                    "neighborhood_type": grid.neighborhood_type.value,
                    "type": "Grid",
                }

                # Add custom metadata
                grid_properties.update(grid.metadata)

                # Create grid node
                self.netzwerk.create_wort_if_not_exists(
                    lemma=grid.name, pos="NOUN", **grid_properties
                )

                logger.debug("Created grid node: %s", grid.name)

                # Step 2: Create position nodes for each cell
                positions_created = 0
                for pos in grid.get_all_positions():
                    pos_name = grid.get_position_name(pos)

                    # Create position node with coordinates
                    self.netzwerk.create_wort_if_not_exists(
                        lemma=pos_name,
                        pos="NOUN",
                        x=pos.x,
                        y=pos.y,
                        type="GridPosition",
                    )

                    # Link position to grid
                    self.netzwerk.assert_relation(
                        from_lemma=pos_name, to_lemma=grid.name, relation_type="PART_OF"
                    )

                    positions_created += 1

                logger.debug("Created %d position nodes", positions_created)

                # Step 3: Create neighborhood relationships
                neighbors_created = self._create_grid_neighbors(grid)
                logger.debug("Created %d neighborhood relationships", neighbors_created)

                logger.info(
                    "Grid created successfully: %s with %d positions and %d neighbors",
                    grid.name,
                    positions_created,
                    neighbors_created,
                )

                return True

        except Exception as e:
            logger.error(
                "Failed to create grid %s: %s", grid.name, str(e), exc_info=True
            )
            return False

    def _create_grid_neighbors(self, grid: Grid) -> int:
        """
        Create neighborhood relationships for all grid positions.

        Args:
            grid: Grid specification

        Returns:
            Number of neighbor relationships created
        """
        neighbors_created = 0

        for pos in grid.get_all_positions():
            # Get neighbors based on grid's neighborhood type
            neighbors = pos.get_neighbors(grid.neighborhood_type, grid.custom_offsets)

            # Filter to only valid grid positions
            valid_neighbors = [n for n in neighbors if grid.is_valid_position(n)]

            # Create relationships
            pos_name = grid.get_position_name(pos)

            for neighbor_pos in valid_neighbors:
                neighbor_name = grid.get_position_name(neighbor_pos)

                # Determine relation type based on direction
                direction = pos.direction_to(neighbor_pos)

                if direction:
                    # Cardinal direction relationship
                    self.netzwerk.assert_relation(
                        from_lemma=pos_name,
                        to_lemma=neighbor_name,
                        relation_type=direction.value,
                    )
                else:
                    # Diagonal or custom - use generic ADJACENT_TO
                    self.netzwerk.assert_relation(
                        from_lemma=pos_name,
                        to_lemma=neighbor_name,
                        relation_type="ADJACENT_TO",
                    )

                neighbors_created += 1

        return neighbors_created

    def get_grid(self, grid_name: str) -> Optional[Grid]:
        """
        Retrieve a grid from the knowledge graph.

        Args:
            grid_name: Name of the grid

        Returns:
            Grid object if found, None otherwise
        """
        try:
            with self._lock:
                # Query grid node
                grid_node = self.netzwerk.find_wort_node(grid_name)

                if not grid_node:
                    logger.warning("Grid not found: %s", grid_name)
                    return None

                # Extract properties
                props = dict(grid_node)

                # Reconstruct grid
                from component_42_spatial_types import NeighborhoodType

                grid = Grid(
                    name=grid_name,
                    width=props.get("width", 0),
                    height=props.get("height", 0),
                    neighborhood_type=NeighborhoodType(
                        props.get("neighborhood_type", "orthogonal")
                    ),
                    metadata={
                        k: v
                        for k, v in props.items()
                        if k
                        not in ["width", "height", "size", "neighborhood_type", "type"]
                    },
                )

                logger.debug("Retrieved grid: %s", grid)
                return grid

        except Exception as e:
            logger.error("Error retrieving grid %s: %s", grid_name, str(e))
            return None

    def delete_grid(self, grid_name: str) -> bool:
        """
        Delete a grid and all its positions from the knowledge graph.

        Args:
            grid_name: Name of the grid to delete

        Returns:
            True if successful, False otherwise
        """
        logger.info("Deleting grid: %s", grid_name)

        try:
            with self._lock:
                # Get grid to find all positions
                grid = self.get_grid(grid_name)
                if not grid:
                    logger.warning("Grid not found for deletion: %s", grid_name)
                    return False

                # Delete all position nodes
                positions_deleted = 0
                for pos in grid.get_all_positions():
                    pos_name = grid.get_position_name(pos)
                    # Delete position node and all its relationships
                    self.netzwerk.delete_wort_node(pos_name)
                    positions_deleted += 1

                # Delete grid node
                self.netzwerk.delete_wort_node(grid_name)

                logger.info(
                    "Deleted grid %s with %d positions", grid_name, positions_deleted
                )
                return True

        except Exception as e:
            logger.error("Error deleting grid %s: %s", grid_name, str(e), exc_info=True)
            return False

    def place_object(
        self, object_name: str, grid_name: str, position: Position
    ) -> bool:
        """
        Place an object at a specific position on a grid.

        Args:
            object_name: Name of the object to place
            grid_name: Name of the grid
            position: Position to place the object

        Returns:
            True if successful, False otherwise
        """
        logger.info(
            "Placing object %s on grid %s at position %s",
            object_name,
            grid_name,
            position,
        )

        try:
            with self._lock:
                # Verify grid exists
                grid = self.get_grid(grid_name)
                if not grid:
                    logger.error("Grid not found: %s", grid_name)
                    return False

                # Verify position is valid
                if not grid.is_valid_position(position):
                    logger.error(
                        "Position %s is out of bounds for grid %s", position, grid_name
                    )
                    return False

                # Create object node if it doesn't exist
                self.netzwerk.create_wort_if_not_exists(
                    lemma=object_name, pos="NOUN", type="GridObject"
                )

                # Get position name
                pos_name = grid.get_position_name(position)

                # Create LOCATED_AT relation
                self.netzwerk.assert_relation(
                    from_lemma=object_name,
                    to_lemma=pos_name,
                    relation_type="LOCATED_AT",
                )

                logger.info("Object %s placed at %s", object_name, pos_name)
                return True

        except Exception as e:
            logger.error(
                "Error placing object %s: %s", object_name, str(e), exc_info=True
            )
            return False

    def move_object(
        self,
        object_name: str,
        grid_name: str,
        from_position: Position,
        to_position: Position,
    ) -> bool:
        """
        Move an object from one position to another on a grid.

        Args:
            object_name: Name of the object to move
            grid_name: Name of the grid
            from_position: Current position
            to_position: Target position

        Returns:
            True if successful, False otherwise
        """
        logger.info(
            "Moving object %s from %s to %s on grid %s",
            object_name,
            from_position,
            to_position,
            grid_name,
        )

        try:
            with self._lock:
                # Verify grid exists
                grid = self.get_grid(grid_name)
                if not grid:
                    logger.error("Grid not found: %s", grid_name)
                    return False

                # Verify positions are valid
                if not grid.is_valid_position(from_position):
                    logger.error("From position %s is out of bounds", from_position)
                    return False

                if not grid.is_valid_position(to_position):
                    logger.error("To position %s is out of bounds", to_position)
                    return False

                # Remove from old position
                from_pos_name = grid.get_position_name(from_position)
                self.netzwerk.delete_relation(
                    from_lemma=object_name,
                    to_lemma=from_pos_name,
                    relation_type="LOCATED_AT",
                )

                # Add to new position
                to_pos_name = grid.get_position_name(to_position)
                self.netzwerk.assert_relation(
                    from_lemma=object_name,
                    to_lemma=to_pos_name,
                    relation_type="LOCATED_AT",
                )

                logger.info(
                    "Object %s moved from %s to %s",
                    object_name,
                    from_pos_name,
                    to_pos_name,
                )
                return True

        except Exception as e:
            logger.error(
                "Error moving object %s: %s", object_name, str(e), exc_info=True
            )
            return False

    def remove_object(
        self, object_name: str, grid_name: str, position: Position
    ) -> bool:
        """
        Remove an object from a position on a grid.

        Args:
            object_name: Name of the object to remove
            grid_name: Name of the grid
            position: Position to remove from

        Returns:
            True if successful, False otherwise
        """
        logger.info(
            "Removing object %s from position %s on grid %s",
            object_name,
            position,
            grid_name,
        )

        try:
            with self._lock:
                # Verify grid exists
                grid = self.get_grid(grid_name)
                if not grid:
                    logger.error("Grid not found: %s", grid_name)
                    return False

                # Get position name
                pos_name = grid.get_position_name(position)

                # Remove LOCATED_AT relation
                self.netzwerk.delete_relation(
                    from_lemma=object_name,
                    to_lemma=pos_name,
                    relation_type="LOCATED_AT",
                )

                logger.info("Object %s removed from %s", object_name, pos_name)
                return True

        except Exception as e:
            logger.error(
                "Error removing object %s: %s", object_name, str(e), exc_info=True
            )
            return False

    def get_object_position(
        self, object_name: str, grid_name: str
    ) -> Optional[Position]:
        """
        Get the current position of an object on a grid.

        Args:
            object_name: Name of the object
            grid_name: Name of the grid

        Returns:
            Position if found, None otherwise
        """
        try:
            with self._lock:
                # Query LOCATED_AT relations
                facts = self.netzwerk.query_graph_for_facts(object_name)

                if "LOCATED_AT" not in facts:
                    logger.debug("Object %s has no position", object_name)
                    return None

                # Get grid to parse position names
                grid = self.get_grid(grid_name)
                if not grid:
                    logger.error("Grid not found: %s", grid_name)
                    return None

                # Find position belonging to this grid
                for pos_name in facts["LOCATED_AT"]:
                    if pos_name.startswith(grid.name + "_Pos_"):
                        # Parse coordinates from name
                        parts = pos_name.split("_")
                        if len(parts) >= 4:
                            x = int(parts[-2])
                            y = int(parts[-1])
                            return Position(x, y)

                logger.debug("Object %s not found on grid %s", object_name, grid_name)
                return None

        except Exception as e:
            logger.error("Error getting position for %s: %s", object_name, str(e))
            return None

    def get_objects_at_position(self, grid_name: str, position: Position) -> List[str]:
        """
        Get all objects at a specific position on a grid.

        Args:
            grid_name: Name of the grid
            position: Position to query

        Returns:
            List of object names at that position
        """
        try:
            with self._lock:
                # Verify grid exists
                grid = self.get_grid(grid_name)
                if not grid:
                    logger.error("Grid not found: %s", grid_name)
                    return []

                # Get position name
                pos_name = grid.get_position_name(position)

                # Query incoming LOCATED_AT relations
                # We need to find all nodes that have LOCATED_AT -> pos_name
                # This requires a Cypher query
                query = """
                MATCH (obj)-[:LOCATED_AT]->(pos)
                WHERE pos.lemma = $pos_name
                RETURN obj.lemma as object_name
                """

                result = self._safe_session_run(
                    query, "get_objects_at_position", pos_name=pos_name
                )
                if result is None:
                    return []

                objects = [record["object_name"] for record in result]

                logger.debug("Found %d objects at position %s", len(objects), pos_name)
                return objects

        except Exception as e:
            logger.error("Error getting objects at position %s: %s", position, str(e))
            return []

    def get_neighbors(self, grid_name: str, position: Position) -> List[Position]:
        """
        Get all neighboring positions for a given position on a grid.

        Args:
            grid_name: Name of the grid
            position: Position to find neighbors for

        Returns:
            List of neighboring positions
        """
        try:
            with self._lock:
                # Get grid
                grid = self.get_grid(grid_name)
                if not grid:
                    logger.error("Grid not found: %s", grid_name)
                    return []

                # Verify position is valid
                if not grid.is_valid_position(position):
                    logger.error("Position %s is out of bounds", position)
                    return []

                # Get neighbors using Position.get_neighbors() and filter by grid bounds
                all_neighbors = position.get_neighbors(
                    grid.neighborhood_type, grid.custom_offsets
                )

                valid_neighbors = [
                    n for n in all_neighbors if grid.is_valid_position(n)
                ]

                logger.debug(
                    "Position %s has %d neighbors on grid %s",
                    position,
                    len(valid_neighbors),
                    grid_name,
                )

                return valid_neighbors

        except Exception as e:
            logger.error(
                "Error getting neighbors for position %s: %s", position, str(e)
            )
            return []

    def get_objects_in_neighborhood(
        self, grid_name: str, position: Position, radius: int = 1
    ) -> Dict[Position, List[str]]:
        """
        Get all objects within a certain radius of a position.

        Args:
            grid_name: Name of the grid
            position: Center position
            radius: Search radius (in grid cells)

        Returns:
            Dictionary mapping positions to lists of objects at those positions
        """
        try:
            with self._lock:
                # Get grid
                grid = self.get_grid(grid_name)
                if not grid:
                    logger.error("Grid not found: %s", grid_name)
                    return {}

                result = {}

                # Check all positions within radius
                for x in range(
                    max(0, position.x - radius),
                    min(grid.width, position.x + radius + 1),
                ):
                    for y in range(
                        max(0, position.y - radius),
                        min(grid.height, position.y + radius + 1),
                    ):
                        check_pos = Position(x, y)

                        # Calculate actual distance
                        if (
                            check_pos.distance_to(position, metric="chebyshev")
                            <= radius
                        ):
                            objects = self.get_objects_at_position(grid_name, check_pos)
                            if objects:
                                result[check_pos] = objects

                logger.debug(
                    "Found %d positions with objects within radius %d of %s",
                    len(result),
                    radius,
                    position,
                )

                return result

        except Exception as e:
            logger.error("Error getting objects in neighborhood: %s", str(e))
            return {}

    def add_position(
        self, object_name: str, position: Position, store_in_graph: bool = True
    ) -> bool:
        """
        Store an object's position in the knowledge graph (without requiring a grid).

        This is a simplified position storage for cases where you don't need
        a full grid structure.

        Args:
            object_name: Name of the object
            position: Position (x, y coordinates)
            store_in_graph: If True, store in Neo4j; if False, just validate

        Returns:
            True if successful, False otherwise
        """
        if not self.netzwerk or not store_in_graph:
            logger.debug(
                "Position for %s at %s recorded (not stored in graph)",
                object_name,
                position,
            )
            return True

        try:
            with self._lock:
                # Create object node with position attributes
                self.netzwerk.create_wort_if_not_exists(
                    lemma=object_name,
                    pos="NOUN",
                    type="SpatialObject",
                    position_x=position.x,
                    position_y=position.y,
                )

                logger.info("Stored position for %s at %s", object_name, position)
                return True

        except Exception as e:
            logger.error(
                "Failed to store position for %s: %s",
                object_name,
                str(e),
                exc_info=True,
            )
            return False
