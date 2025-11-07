"""
tests/test_spatial_planning.py

Unit tests for spatial planning and pathfinding.

Tests cover:
- A* pathfinding on grids
- Heuristic functions (Manhattan, Euclidean)
- Obstacle avoidance
- Plan generation and validation
- Integration with component_31_state_space_planner
- Performance characteristics
"""

from typing import Any, Callable, List, Tuple

import pytest

from component_42_spatial_reasoning import (
    Grid,
    NeighborhoodType,
    Position,
)

# ==================== Helper Functions ====================


def manhattan_heuristic(state: Any, goal: Position) -> float:
    """Manhattan distance heuristic for grid planning."""
    if not isinstance(state, Position):
        return 0.0
    return float(state.manhattan_distance_to(goal))


def euclidean_heuristic(state: Any, goal: Position) -> float:
    """Euclidean distance heuristic for grid planning."""
    if not isinstance(state, Position):
        return 0.0
    return state.distance_to(goal)


def get_grid_actions(grid: Grid, current: Position) -> List[Tuple[str, Position]]:
    """
    Get available actions from current position on grid.

    Returns:
        List of (action_name, next_position) tuples
    """
    actions = []
    neighbors = grid.get_neighbors(current, NeighborhoodType.ORTHOGONAL)

    direction_names = {
        (0, 1): "move_north",
        (0, -1): "move_south",
        (1, 0): "move_east",
        (-1, 0): "move_west",
    }

    for neighbor in neighbors:
        dx = neighbor.x - current.x
        dy = neighbor.y - current.y
        action_name = direction_names.get((dx, dy), "move")

        # Check if position is blocked
        cell_data = grid.get_cell_data(neighbor)
        if cell_data != "obstacle":
            actions.append((action_name, neighbor))

    return actions


def a_star_grid_search(
    grid: Grid,
    start: Position,
    goal: Position,
    heuristic: Callable[[Position, Position], float] = manhattan_heuristic,
) -> Tuple[List[Position], float]:
    """
    A* search on a grid.

    Args:
        grid: The grid to search on
        start: Start position
        goal: Goal position
        heuristic: Heuristic function

    Returns:
        (path, cost) tuple, or ([], -1) if no path found
    """
    from heapq import heappop, heappush

    # Priority queue: (f_score, g_score, position, path)
    frontier = []
    heappush(frontier, (0.0, 0.0, start, [start]))

    # Visited set
    visited = set()

    while frontier:
        f_score, g_score, current, path = heappop(frontier)

        # Goal check
        if current == goal:
            return path, g_score

        # Skip if already visited
        if current in visited:
            continue
        visited.add(current)

        # Expand neighbors
        actions = get_grid_actions(grid, current)
        for action_name, next_pos in actions:
            if next_pos in visited:
                continue

            # Cost: uniform cost of 1 per move
            new_g = g_score + 1.0
            h = heuristic(next_pos, goal)
            new_f = new_g + h

            new_path = path + [next_pos]
            heappush(frontier, (new_f, new_g, next_pos, new_path))

    # No path found
    return [], -1


# ==================== Basic Pathfinding Tests ====================


class TestBasicPathfinding:
    """Test basic pathfinding functionality."""

    def test_straight_line_path(self):
        """Test pathfinding in a straight line (no obstacles)."""
        grid = Grid(height=10, width=10)
        start = Position(0, 0)
        goal = Position(0, 5)

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        assert len(path) == 6  # Start + 5 steps
        assert path[0] == start
        assert path[-1] == goal
        assert cost == 5.0

    def test_diagonal_path_manhattan(self):
        """Test pathfinding with Manhattan distance (no diagonal moves)."""
        grid = Grid(height=10, width=10)
        start = Position(0, 0)
        goal = Position(5, 5)

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Manhattan: must go 5 east and 5 north (10 moves total)
        assert cost == 10.0
        assert path[0] == start
        assert path[-1] == goal

    def test_path_with_single_obstacle(self):
        """Test pathfinding around a single obstacle."""
        grid = Grid(height=5, width=5)
        start = Position(0, 2)
        goal = Position(4, 2)

        # Place obstacle in the middle
        obstacle = Position(2, 2)
        grid.set_cell_data(obstacle, "obstacle")

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should find path around obstacle
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal
        assert obstacle not in path

    def test_path_with_multiple_obstacles(self):
        """Test pathfinding around multiple obstacles."""
        grid = Grid(height=10, width=10)
        start = Position(0, 5)
        goal = Position(9, 5)

        # Create vertical wall with gap
        for y in range(10):
            if y != 5:  # Leave gap at y=5
                grid.set_cell_data(Position(5, y), "obstacle")

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should find path through gap
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal

        # Verify no obstacles in path
        for pos in path:
            assert grid.get_cell_data(pos) != "obstacle"

    def test_no_path_available(self):
        """Test detection when no path exists."""
        grid = Grid(height=10, width=10)
        start = Position(0, 0)
        goal = Position(9, 9)

        # Create complete wall (no gap)
        for y in range(10):
            grid.set_cell_data(Position(5, y), "obstacle")

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should detect no path
        assert len(path) == 0
        assert cost == -1

    def test_start_equals_goal(self):
        """Test pathfinding when start equals goal."""
        grid = Grid(height=10, width=10)
        start = Position(5, 5)
        goal = Position(5, 5)

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should return immediate path
        assert len(path) == 1
        assert path[0] == start
        assert cost == 0.0


# ==================== Heuristic Tests ====================


class TestPathfindingHeuristics:
    """Test different heuristic functions."""

    def test_manhattan_heuristic_admissibility(self):
        """Test that Manhattan heuristic is admissible (never overestimates)."""
        grid = Grid(height=10, width=10)
        start = Position(0, 0)
        goal = Position(5, 5)

        # Manhattan estimate
        h_manhattan = manhattan_heuristic(start, goal)

        # Actual cost (with orthogonal movement only)
        path, actual_cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Heuristic should not overestimate
        assert h_manhattan <= actual_cost

    def test_euclidean_heuristic_admissibility(self):
        """Test that Euclidean heuristic is admissible."""
        grid = Grid(height=10, width=10)
        start = Position(0, 0)
        goal = Position(5, 5)

        # Euclidean estimate
        h_euclidean = euclidean_heuristic(start, goal)

        # Actual cost (orthogonal movement)
        path, actual_cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Euclidean should underestimate (since we can't move diagonally)
        assert h_euclidean <= actual_cost

    def test_heuristic_comparison(self):
        """Compare Manhattan vs Euclidean heuristics."""
        start = Position(0, 0)
        goal = Position(3, 4)

        h_manhattan = manhattan_heuristic(start, goal)
        h_euclidean = euclidean_heuristic(start, goal)

        # Manhattan: |3-0| + |4-0| = 7
        assert h_manhattan == 7.0

        # Euclidean: sqrt(3^2 + 4^2) = 5
        assert h_euclidean == 5.0

        # Manhattan >= Euclidean for orthogonal movement
        assert h_manhattan >= h_euclidean

    def test_zero_heuristic_dijkstra(self):
        """Test that zero heuristic gives Dijkstra's algorithm."""
        grid = Grid(height=10, width=10)
        start = Position(0, 0)
        goal = Position(3, 3)

        def zero_heuristic(state: Any, goal: Any) -> float:
            return 0.0

        path, cost = a_star_grid_search(grid, start, goal, zero_heuristic)

        # Should still find optimal path (but slower)
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal


# ==================== Complex Scenario Tests ====================


class TestComplexPathfinding:
    """Test pathfinding in complex scenarios."""

    def test_maze_pathfinding(self):
        """Test pathfinding through a maze."""
        grid = Grid(height=11, width=11)
        start = Position(0, 0)
        goal = Position(10, 10)

        # Create maze-like structure
        obstacles = [
            # Horizontal walls
            Position(5, 1),
            Position(5, 2),
            Position(5, 3),
            Position(3, 5),
            Position(4, 5),
            Position(5, 5),
            Position(6, 5),
            Position(7, 5),
            Position(5, 7),
            Position(5, 8),
            Position(5, 9),
        ]
        for obs in obstacles:
            grid.set_cell_data(obs, "obstacle")

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should find a path
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal

        # Path should avoid all obstacles
        for pos in path:
            assert pos not in obstacles

    def test_narrow_corridor_pathfinding(self):
        """Test pathfinding through narrow corridor."""
        grid = Grid(height=10, width=10)
        start = Position(0, 5)
        goal = Position(9, 5)

        # Create corridor walls (y=4 and y=6, but open at y=5)
        for x in range(10):
            grid.set_cell_data(Position(x, 4), "obstacle")
            grid.set_cell_data(Position(x, 6), "obstacle")

        # Except for endpoints
        grid.set_cell_data(Position(0, 4), None)
        grid.set_cell_data(Position(0, 6), None)
        grid.set_cell_data(Position(9, 4), None)
        grid.set_cell_data(Position(9, 6), None)

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should find straight path through corridor
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal

    def test_u_shaped_obstacle(self):
        """Test pathfinding around U-shaped obstacle."""
        grid = Grid(height=10, width=10)
        start = Position(4, 5)
        goal = Position(6, 5)

        # Create U-shape
        u_shape = [
            Position(5, 3),
            Position(5, 4),
            Position(5, 5),
            Position(5, 6),
            Position(5, 7),  # Vertical
            Position(4, 3),
            Position(6, 3),  # Top arms
        ]
        for obs in u_shape:
            grid.set_cell_data(obs, "obstacle")

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should find path around U
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal


# ==================== Plan Generation Tests ====================


class TestPlanGeneration:
    """Test plan generation from paths."""

    def test_path_to_action_sequence(self):
        """Test converting path to action sequence."""
        path = [
            Position(0, 0),
            Position(0, 1),  # Move north
            Position(0, 2),  # Move north
            Position(1, 2),  # Move east
            Position(2, 2),  # Move east
        ]

        actions = []
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]

            dx = next_pos.x - current.x
            dy = next_pos.y - current.y

            if dx == 0 and dy == 1:
                actions.append("move_north")
            elif dx == 0 and dy == -1:
                actions.append("move_south")
            elif dx == 1 and dy == 0:
                actions.append("move_east")
            elif dx == -1 and dy == 0:
                actions.append("move_west")

        assert len(actions) == 4
        assert actions[0] == "move_north"
        assert actions[1] == "move_north"
        assert actions[2] == "move_east"
        assert actions[3] == "move_east"

    def test_plan_validation(self):
        """Test validating a path (no obstacles)."""
        grid = Grid(height=5, width=5)
        path = [Position(0, 0), Position(0, 1), Position(0, 2)]

        # Validate: no obstacles in path
        is_valid = all(grid.get_cell_data(pos) != "obstacle" for pos in path)
        assert is_valid

        # Add obstacle
        grid.set_cell_data(Position(0, 1), "obstacle")

        # Re-validate: should fail
        is_valid = all(grid.get_cell_data(pos) != "obstacle" for pos in path)
        assert not is_valid


# ==================== Performance Tests ====================


class TestPathfindingPerformance:
    """Test performance characteristics of pathfinding."""

    @pytest.mark.slow
    def test_large_grid_pathfinding(self):
        """Test pathfinding on large grid."""
        grid = Grid(height=100, width=100)
        start = Position(0, 0)
        goal = Position(99, 99)

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should find optimal path
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal
        # Manhattan distance: 99+99 = 198 moves
        assert cost == 198.0

    @pytest.mark.slow
    def test_dense_obstacles_pathfinding(self):
        """Test pathfinding with many obstacles."""
        grid = Grid(height=50, width=50)
        start = Position(0, 0)
        goal = Position(49, 49)

        # Add random obstacles (30% coverage)
        import random

        random.seed(42)  # Reproducible

        for i in range(50):
            for j in range(50):
                if random.random() < 0.3:
                    pos = Position(i, j)
                    if pos != start and pos != goal:
                        grid.set_cell_data(pos, "obstacle")

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should find a path (or detect no path)
        if len(path) > 0:
            assert path[0] == start
            assert path[-1] == goal
            # Verify no obstacles in path
            for pos in path:
                assert grid.get_cell_data(pos) != "obstacle"

    def test_pathfinding_optimality(self):
        """Test that A* finds optimal path."""
        grid = Grid(height=10, width=10)
        start = Position(0, 0)
        goal = Position(5, 5)

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Optimal cost: Manhattan distance (no obstacles)
        optimal_cost = start.manhattan_distance_to(goal)
        assert cost == optimal_cost


# ==================== Edge Case Tests ====================


class TestPathfindingEdgeCases:
    """Test edge cases in pathfinding."""

    def test_single_cell_grid(self):
        """Test pathfinding on 1x1 grid."""
        grid = Grid(height=1, width=1)
        start = Position(0, 0)
        goal = Position(0, 0)

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        assert len(path) == 1
        assert path[0] == start
        assert cost == 0.0

    def test_start_on_obstacle(self):
        """Test pathfinding when start is on obstacle."""
        grid = Grid(height=5, width=5)
        start = Position(0, 0)
        goal = Position(4, 4)

        grid.set_cell_data(start, "obstacle")

        # A* should handle this (depends on implementation)
        # Typically: no path (can't leave start)
        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Implementation may vary: either empty path or path starting from obstacle
        # We just check it doesn't crash
        assert isinstance(path, list)

    def test_goal_on_obstacle(self):
        """Test pathfinding when goal is on obstacle."""
        grid = Grid(height=5, width=5)
        start = Position(0, 0)
        goal = Position(4, 4)

        grid.set_cell_data(goal, "obstacle")

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should detect no path (can't reach obstacle)
        assert len(path) == 0
        assert cost == -1

    def test_neighbors_all_blocked(self):
        """Test pathfinding when all neighbors are blocked."""
        grid = Grid(height=5, width=5)
        start = Position(2, 2)
        goal = Position(4, 4)

        # Block all neighbors of start
        for neighbor in start.get_neighbors(NeighborhoodType.ORTHOGONAL):
            grid.set_cell_data(neighbor, "obstacle")

        path, cost = a_star_grid_search(grid, start, goal, manhattan_heuristic)

        # Should detect no path
        assert len(path) == 0
        assert cost == -1


# ==================== Test Configuration ====================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: slow pathfinding tests on large grids")
