"""
Component 42: Path Finder

Path-finding algorithms for spatial reasoning.

This module handles:
- BFS, DFS, A* path-finding implementations
- Custom heuristics and movement rules
- Obstacle handling
- Movement planning and validation

Author: KAI Development Team
Date: 2025-11-27
"""

import threading
from heapq import heappop, heappush
from typing import Callable, Dict, List, Optional, Set, Tuple

from component_15_logging_config import get_logger
from component_42_spatial_grid import Grid
from component_42_spatial_movement import MovementAction, MovementPlan
from component_42_spatial_types import NeighborhoodType, Position

logger = get_logger(__name__)


class PathFinder:
    """
    Path-finding algorithms for spatial reasoning.

    Provides methods for:
    - A* path-finding with custom heuristics
    - BFS and DFS search algorithms
    - Obstacle-aware path planning
    - Movement validation

    Thread Safety:
        This class is thread-safe. All operations are stateless or use
        thread-local data structures.
    """

    def __init__(self):
        """Initialize the path finder."""
        self._lock = threading.RLock()  # For thread safety
        self._movement_rules: Dict[str, Dict] = {}

        logger.info("PathFinder initialized")

    def find_path(
        self,
        grid: Grid,
        start: Position,
        goal: Position,
        allow_diagonal: bool = False,
    ) -> Optional[List[Position]]:
        """
        Find a path between two positions on a grid using A* algorithm.

        Args:
            grid: Grid object
            start: Starting position
            goal: Goal position
            allow_diagonal: Whether diagonal moves are allowed

        Returns:
            List of positions representing the path (including start and goal),
            or None if no path exists
        """
        try:
            # Verify positions are valid
            if not grid.is_valid_position(start) or not grid.is_valid_position(goal):
                logger.error("Invalid start or goal position")
                return None

            # A* pathfinding
            # Priority queue: (f_score, position)
            open_set = [(0, start)]
            came_from = {}
            g_score = {start: 0}
            f_score = {start: start.distance_to(goal, metric="manhattan")}

            while open_set:
                current_f, current = heappop(open_set)

                # Goal reached
                if current == goal:
                    # Reconstruct path
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    path.reverse()

                    logger.debug(
                        "Found path from %s to %s: %d steps", start, goal, len(path)
                    )
                    return path

                # Get neighbors
                neighbor_type = (
                    NeighborhoodType.MOORE
                    if allow_diagonal
                    else NeighborhoodType.ORTHOGONAL
                )
                neighbors = current.get_neighbors(neighbor_type)
                neighbors = [n for n in neighbors if grid.is_valid_position(n)]

                for neighbor in neighbors:
                    # Tentative g_score
                    tentative_g = g_score[current] + 1

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        # This path is better
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + neighbor.distance_to(
                            goal, metric="manhattan"
                        )

                        # Add to open set if not already there
                        if neighbor not in [pos for _, pos in open_set]:
                            heappush(open_set, (f_score[neighbor], neighbor))

            # No path found
            logger.debug("No path found from %s to %s", start, goal)
            return None

        except Exception as e:
            logger.error("Error finding path: %s", str(e), exc_info=True)
            return None

    def find_path_with_rules(
        self,
        grid: Grid,
        start: Position,
        goal: Position,
        movement_rules: Optional[Callable[[Position, Position], bool]],
        blocked: Set[Position],
    ) -> Optional[List[Position]]:
        """
        A* pathfinding with custom movement rules and blocked positions.

        Args:
            grid: Grid object
            start: Start position
            goal: Goal position
            movement_rules: Optional function (from_pos, to_pos) -> bool
            blocked: Set of blocked positions

        Returns:
            List of positions from start to goal, or None
        """

        def heuristic(pos: Position) -> float:
            return pos.distance_to(goal, metric="manhattan")

        # Priority queue: (f_score, position)
        open_set = []
        heappush(open_set, (heuristic(start), start))

        came_from: Dict[Position, Position] = {}
        g_score: Dict[Position, float] = {start: 0}
        f_score: Dict[Position, float] = {start: heuristic(start)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # Get neighbors based on grid type
            neighbors = current.get_neighbors(
                grid.neighborhood_type, grid.custom_offsets
            )
            neighbors = [n for n in neighbors if grid.is_valid_position(n)]

            for neighbor in neighbors:
                # Skip blocked positions
                if neighbor in blocked and neighbor != goal:
                    continue

                # Check custom movement rules
                if movement_rules and not movement_rules(current, neighbor):
                    continue

                tentative_g = g_score[current] + current.distance_to(neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    f_score[neighbor] = f
                    heappush(open_set, (f, neighbor))

        return None  # No path found

    def plan_movement(
        self,
        object_name: str,
        grid: Grid,
        start_pos: Position,
        goal_pos: Position,
        movement_rules: Optional[Callable[[Position, Position], bool]] = None,
        avoid_objects: bool = True,
        blocked_positions: Optional[Set[Position]] = None,
    ) -> Optional[MovementPlan]:
        """
        Plan a sequence of moves to get an object from start to goal.

        Args:
            object_name: Name of object to move
            grid: Grid object
            start_pos: Starting position
            goal_pos: Goal position
            movement_rules: Optional function to validate individual moves
            avoid_objects: If True, avoid positions occupied by other objects
            blocked_positions: Optional set of blocked positions

        Returns:
            MovementPlan with sequence of actions, or None if no path exists
        """
        try:
            # Validate start and goal positions
            if not grid.is_valid_position(start_pos) or not grid.is_valid_position(
                goal_pos
            ):
                logger.error("Start or goal position out of bounds")
                return None

            # Get blocked positions
            blocked = blocked_positions if blocked_positions is not None else set()

            # Find path using A* with custom movement rules
            path = self.find_path_with_rules(
                grid, start_pos, goal_pos, movement_rules, blocked
            )

            if not path:
                logger.warning("No valid path found from %s to %s", start_pos, goal_pos)
                return None

            # Convert path to movement actions
            actions = []
            for i in range(len(path) - 1):
                from_pos = path[i]
                to_pos = path[i + 1]
                action = MovementAction(
                    object_name=object_name,
                    from_position=from_pos,
                    to_position=to_pos,
                    step_number=i + 1,
                )
                actions.append(action)

            plan = MovementPlan(
                object_name=object_name,
                grid_name=grid.name,
                actions=actions,
                total_steps=len(actions),
                path_length=len(path),
            )

            logger.info(
                "Created movement plan for %s: %d steps from %s to %s",
                object_name,
                len(actions),
                start_pos,
                goal_pos,
            )

            return plan

        except Exception as e:
            logger.error("Error planning movement: %s", str(e), exc_info=True)
            return None

    def validate_movement_plan(
        self, plan: MovementPlan, grid: Grid
    ) -> Tuple[bool, List[str]]:
        """
        Validate a movement plan.

        Args:
            plan: MovementPlan to validate
            grid: Grid object

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Validate each action
            for action in plan.actions:
                # Check positions are valid
                if not grid.is_valid_position(action.from_position):
                    errors.append(
                        f"Step {action.step_number}: Invalid from_position {action.from_position}"
                    )

                if not grid.is_valid_position(action.to_position):
                    errors.append(
                        f"Step {action.step_number}: Invalid to_position {action.to_position}"
                    )

                # Check positions are neighbors
                neighbors = action.from_position.get_neighbors(
                    grid.neighborhood_type, grid.custom_offsets
                )
                neighbors = [n for n in neighbors if grid.is_valid_position(n)]

                if action.to_position not in neighbors:
                    errors.append(
                        f"Step {action.step_number}: Position {action.to_position} "
                        f"is not a neighbor of {action.from_position}"
                    )

            # Check action sequence is continuous
            for i in range(len(plan.actions) - 1):
                if plan.actions[i].to_position != plan.actions[i + 1].from_position:
                    errors.append(
                        f"Discontinuous path at step {plan.actions[i].step_number}"
                    )

            is_valid = len(errors) == 0

            if is_valid:
                logger.info(
                    "Movement plan validated successfully: %s", plan.object_name
                )
            else:
                logger.warning(
                    "Movement plan validation failed with %d errors", len(errors)
                )

            return is_valid, errors

        except Exception as e:
            logger.error("Error validating movement plan: %s", str(e), exc_info=True)
            errors.append(f"Validation error: {str(e)}")
            return False, errors

    def create_movement_rule(
        self,
        name: str,
        rule_function: Callable[[Position, Position], bool],
        description: str = "",
    ) -> bool:
        """
        Create a named movement rule that can be reused.

        Args:
            name: Unique name for the rule
            rule_function: Function (from_pos, to_pos) -> bool
            description: Optional description

        Returns:
            True if successful
        """
        try:
            with self._lock:
                self._movement_rules[name] = {
                    "function": rule_function,
                    "description": description,
                }

                logger.info("Created movement rule: %s", name)
                return True

        except Exception as e:
            logger.error("Error creating movement rule: %s", str(e), exc_info=True)
            return False

    def get_movement_rule(
        self, name: str
    ) -> Optional[Callable[[Position, Position], bool]]:
        """Get a movement rule by name."""
        with self._lock:
            if name in self._movement_rules:
                return self._movement_rules[name]["function"]
            return None

    def plan_multi_object_movement(
        self,
        grid: Grid,
        movements: List[Tuple[str, Position, Position]],
        avoid_collisions: bool = True,
    ) -> Optional[Dict[str, MovementPlan]]:
        """
        Plan movements for multiple objects, avoiding collisions.

        Args:
            grid: Grid object
            movements: List of (object_name, start_pos, goal_pos) tuples
            avoid_collisions: If True, ensure objects don't collide

        Returns:
            Dict mapping object_name to MovementPlan, or None if impossible
        """
        try:
            plans = {}
            occupied_positions: Dict[int, Set[Position]] = {}  # step -> positions

            # Plan each movement
            for obj_name, start_pos, goal_pos in movements:
                # Get blocked positions from other planned movements
                blocked = set()
                if avoid_collisions:
                    # Collect all positions that will be occupied during other plans
                    for step_num in occupied_positions:
                        blocked.update(occupied_positions[step_num])

                # Plan this object's movement
                plan = self.plan_movement(
                    obj_name,
                    grid,
                    start_pos,
                    goal_pos,
                    avoid_objects=avoid_collisions,
                    blocked_positions=blocked,
                )

                if not plan:
                    logger.error(
                        "Cannot plan movement for %s, multi-object planning failed",
                        obj_name,
                    )
                    return None

                plans[obj_name] = plan

                # Record occupied positions at each step
                if avoid_collisions:
                    for action in plan.actions:
                        step = action.step_number
                        if step not in occupied_positions:
                            occupied_positions[step] = set()
                        occupied_positions[step].add(action.to_position)

            logger.info("Successfully planned movements for %d objects", len(movements))

            return plans

        except Exception as e:
            logger.error(
                "Error planning multi-object movement: %s", str(e), exc_info=True
            )
            return None

    def get_distance_between_positions(
        self, position1: Position, position2: Position, metric: str = "manhattan"
    ) -> float:
        """
        Calculate distance between two positions.

        Args:
            position1: First position
            position2: Second position
            metric: Distance metric ('manhattan', 'euclidean', 'chebyshev')

        Returns:
            Distance value
        """
        return position1.distance_to(position2, metric=metric)

    def bfs_search(
        self,
        grid: Grid,
        start: Position,
        goal: Position,
        allow_diagonal: bool = False,
    ) -> Optional[List[Position]]:
        """
        Breadth-first search path-finding.

        BFS finds the shortest path (in terms of number of steps) but doesn't
        use heuristics like A*.

        Args:
            grid: Grid object
            start: Starting position
            goal: Goal position
            allow_diagonal: Whether diagonal moves are allowed

        Returns:
            List of positions from start to goal, or None
        """
        try:
            from collections import deque

            # Verify positions are valid
            if not grid.is_valid_position(start) or not grid.is_valid_position(goal):
                logger.error("Invalid start or goal position")
                return None

            # BFS queue: positions to explore
            queue = deque([start])
            came_from = {start: None}

            while queue:
                current = queue.popleft()

                # Goal reached
                if current == goal:
                    # Reconstruct path
                    path = []
                    while current is not None:
                        path.append(current)
                        current = came_from[current]
                    path.reverse()

                    logger.debug(
                        "BFS found path from %s to %s: %d steps", start, goal, len(path)
                    )
                    return path

                # Get neighbors
                neighbor_type = (
                    NeighborhoodType.MOORE
                    if allow_diagonal
                    else NeighborhoodType.ORTHOGONAL
                )
                neighbors = current.get_neighbors(neighbor_type)
                neighbors = [n for n in neighbors if grid.is_valid_position(n)]

                for neighbor in neighbors:
                    if neighbor not in came_from:
                        came_from[neighbor] = current
                        queue.append(neighbor)

            # No path found
            logger.debug("BFS: No path found from %s to %s", start, goal)
            return None

        except Exception as e:
            logger.error("Error in BFS search: %s", str(e), exc_info=True)
            return None

    def dfs_search(
        self,
        grid: Grid,
        start: Position,
        goal: Position,
        allow_diagonal: bool = False,
        max_depth: int = 1000,
    ) -> Optional[List[Position]]:
        """
        Depth-first search path-finding.

        DFS explores as deep as possible before backtracking. It may not find
        the shortest path but can be useful for certain maze-solving scenarios.

        Args:
            grid: Grid object
            start: Starting position
            goal: Goal position
            allow_diagonal: Whether diagonal moves are allowed
            max_depth: Maximum search depth to prevent infinite loops

        Returns:
            List of positions from start to goal, or None
        """
        try:
            # Verify positions are valid
            if not grid.is_valid_position(start) or not grid.is_valid_position(goal):
                logger.error("Invalid start or goal position")
                return None

            # DFS stack: (position, path)
            stack = [(start, [start])]
            visited = {start}

            while stack:
                current, path = stack.pop()

                # Goal reached
                if current == goal:
                    logger.debug(
                        "DFS found path from %s to %s: %d steps", start, goal, len(path)
                    )
                    return path

                # Check max depth
                if len(path) >= max_depth:
                    continue

                # Get neighbors
                neighbor_type = (
                    NeighborhoodType.MOORE
                    if allow_diagonal
                    else NeighborhoodType.ORTHOGONAL
                )
                neighbors = current.get_neighbors(neighbor_type)
                neighbors = [n for n in neighbors if grid.is_valid_position(n)]

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))

            # No path found
            logger.debug("DFS: No path found from %s to %s", start, goal)
            return None

        except Exception as e:
            logger.error("Error in DFS search: %s", str(e), exc_info=True)
            return None
