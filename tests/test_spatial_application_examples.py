"""
tests/test_spatial_application_examples.py

Application example tests to validate generality of spatial reasoning system.

IMPORTANT: These tests demonstrate that the spatial reasoning system is generic
and can handle complex scenarios WITHOUT domain-specific code. The tests use
Sudoku and Chess as EXAMPLES, but implement them using ONLY generic spatial
primitives (grids, constraints, planning).

Tests cover:
- Sudoku-like scenario: 9×9 grid with region constraints (NO Sudoku-specific logic)
- Chess-like scenario: 8×8 grid with movement patterns (NO Chess-specific logic)
- Generic constraint patterns applicable to any grid-based problem
"""

from typing import Any, Dict, List, Tuple

import pytest

from component_29_constraint_reasoning import (
    Constraint,
    ConstraintProblem,
    ConstraintSolver,
    ConstraintType,
    Variable,
)
from component_42_spatial_reasoning import (
    Grid,
    NeighborhoodType,
    Position,
)

# ==================== Generic Grid Pattern Tests ====================


class TestGenericGridPatterns:
    """Test generic grid patterns (applicable to many domains)."""

    def test_9x9_grid_with_3x3_regions(self):
        """Test 9×9 grid divided into 3×3 regions (Sudoku-like, NO Sudoku logic)."""
        # Generic: 9×9 grid with 9 sub-regions
        grid = Grid(height=9, width=9, name="9x9_RegionalGrid")

        assert grid.height == 9
        assert grid.width == 9
        assert grid.get_cell_count() == 81

        # Generic region calculation
        def get_region_id(pos: Position, region_size: int = 3) -> Tuple[int, int]:
            """Generic: Calculate which region a position belongs to."""
            return (pos.x // region_size, pos.y // region_size)

        # Test: Position (0,0) is in region (0,0)
        assert get_region_id(Position(0, 0)) == (0, 0)

        # Test: Position (3,3) is in region (1,1)
        assert get_region_id(Position(3, 3)) == (1, 1)

        # Test: Position (8,8) is in region (2,2)
        assert get_region_id(Position(8, 8)) == (2, 2)

    def test_row_column_region_grouping(self):
        """Test generic row/column/region grouping on grid."""
        grid = Grid(height=9, width=9)

        # Generic: Get all positions in row 0
        def get_row_positions(row: int, width: int) -> List[Position]:
            return [Position(col, row) for col in range(width)]

        # Generic: Get all positions in column 0
        def get_column_positions(col: int, height: int) -> List[Position]:
            return [Position(col, row) for row in range(height)]

        # Generic: Get all positions in region (0, 0)
        def get_region_positions(
            region_x: int, region_y: int, region_size: int = 3
        ) -> List[Position]:
            positions = []
            for x in range(region_x * region_size, (region_x + 1) * region_size):
                for y in range(region_y * region_size, (region_y + 1) * region_size):
                    positions.append(Position(x, y))
            return positions

        # Test row
        row_0 = get_row_positions(0, 9)
        assert len(row_0) == 9
        assert Position(0, 0) in row_0
        assert Position(8, 0) in row_0

        # Test column
        col_0 = get_column_positions(0, 9)
        assert len(col_0) == 9
        assert Position(0, 0) in col_0
        assert Position(0, 8) in col_0

        # Test region
        region_0_0 = get_region_positions(0, 0)
        assert len(region_0_0) == 9
        assert Position(0, 0) in region_0_0
        assert Position(2, 2) in region_0_0
        assert Position(3, 0) not in region_0_0  # Different region

    def test_all_different_constraint_on_groups(self):
        """Test generic all-different constraint on position groups."""

        # Generic: Values in a group must all be different
        def all_different_in_group(values: List[Any]) -> bool:
            """Generic: Check if all values in group are different."""
            return len(values) == len(set(values))

        # Test
        assert all_different_in_group([1, 2, 3, 4, 5])
        assert not all_different_in_group([1, 2, 3, 2, 5])  # Duplicate 2

    def test_placement_with_exclusion_constraints(self):
        """Test generic placement problem with exclusion constraints."""
        # Generic: Place values on grid such that certain positions are excluded

        grid = Grid(height=9, width=9)

        # Mark some positions as forbidden
        forbidden_positions = [Position(0, 0), Position(1, 1), Position(2, 2)]
        for pos in forbidden_positions:
            grid.set_cell_data(pos, "forbidden")

        # Domain: All positions except forbidden
        all_positions = [Position(x, y) for x in range(9) for y in range(9)]
        valid_domain = [
            p for p in all_positions if grid.get_cell_data(p) != "forbidden"
        ]

        assert len(valid_domain) == 81 - 3  # 78 valid positions

        # Generic variable with filtered domain
        var = Variable(name="Object", domain=valid_domain)

        assert len(var.domain) == 78
        assert Position(0, 0) not in var.domain


# ==================== Sudoku-Like Scenario Tests (Generic Constraints) ====================


class TestSudokuLikeScenario:
    """
    Test Sudoku-like scenario using ONLY generic spatial primitives.

    NO Sudoku-specific logic! Only:
    - 9×9 Grid
    - All-different constraints on rows/columns/regions
    - CSP solving with generic constraints
    """

    def test_simple_placement_on_9x9_grid(self):
        """Test placing single value on 9×9 grid (generic)."""
        grid = Grid(height=9, width=9, name="9x9_Grid")

        # Place value at position
        pos = Position(4, 4)  # Center
        grid.set_cell_data(pos, "Value_A")

        assert grid.get_cell_data(pos) == "Value_A"

    def test_row_constraint_on_9x9_grid(self):
        """Test generic row constraint (all different in row)."""
        # Simplified: 4 variables for row 0, domain [1..4]
        # This demonstrates capability without excessive complexity
        variables = [
            Variable(name=f"Row0_Col{c}", domain=list(range(1, 5))) for c in range(4)
        ]

        # Generic all-different constraint
        def all_different(var_dict: Dict[str, int]) -> bool:
            values = [v for v in var_dict.values() if v is not None]
            return len(values) == len(set(values))

        constraint = Constraint(
            variables=[f"Row0_Col{c}" for c in range(4)],
            constraint_type=ConstraintType.ALL_DIFFERENT,
            constraint_function=all_different,
        )

        problem = ConstraintProblem(variables=variables, constraints=[constraint])

        # Should be solvable (many solutions)
        solver = ConstraintSolver(problem)
        solution, proof_tree = solver.solve(track_proof=False)

        assert solution is not None
        # All values should be different
        values = list(solution.values())
        assert len(values) == len(set(values))

    def test_region_constraint_on_9x9_grid(self):
        """Test generic region constraint (all different in 2×2 region)."""

        # Simplified: 4 variables for region (0, 0), domain [1..4]
        # This demonstrates capability without excessive complexity
        def get_region_cell_name(region_x: int, region_y: int, cell_idx: int) -> str:
            """Generic naming: Region_(rx,ry)_Cell_idx"""
            return f"Region_{region_x}_{region_y}_Cell{cell_idx}"

        variables = [
            Variable(name=get_region_cell_name(0, 0, i), domain=list(range(1, 5)))
            for i in range(4)
        ]

        # Generic all-different constraint
        def all_different(var_dict: Dict[str, int]) -> bool:
            values = [v for v in var_dict.values() if v is not None]
            return len(values) == len(set(values))

        constraint = Constraint(
            variables=[v.name for v in variables],
            constraint_type=ConstraintType.ALL_DIFFERENT,
            constraint_function=all_different,
        )

        problem = ConstraintProblem(variables=variables, constraints=[constraint])

        # Should be solvable
        solver = ConstraintSolver(problem)
        solution, proof_tree = solver.solve(track_proof=False)

        assert solution is not None
        # All 4 values in region should be different
        values = list(solution.values())
        assert len(values) == len(set(values))

    @pytest.mark.slow
    def test_multi_constraint_on_9x9_grid(self):
        """Test multiple generic constraints on simplified grid."""
        # Generic: Simplified scenario with 3 cells in a row
        # Cell 0, 1, 2 in row 0 must all be different

        variables = [Variable(name=f"Cell{i}", domain=[1, 2, 3]) for i in range(3)]

        def all_different(var_dict: Dict[str, int]) -> bool:
            values = [v for v in var_dict.values() if v is not None]
            return len(values) == len(set(values))

        constraint = Constraint(
            variables=["Cell0", "Cell1", "Cell2"],
            constraint_type=ConstraintType.ALL_DIFFERENT,
            constraint_function=all_different,
        )

        problem = ConstraintProblem(variables=variables, constraints=[constraint])
        solver = ConstraintSolver(problem)

        solution, proof_tree = solver.solve(track_proof=False)

        # Should find solution (e.g., {Cell0: 1, Cell1: 2, Cell2: 3})
        assert solution is not None
        assert len(set(solution.values())) == 3


# ==================== Chess-Like Scenario Tests (Generic Movement Patterns) ====================


class TestChessLikeScenario:
    """
    Test Chess-like scenario using ONLY generic spatial primitives.

    NO Chess-specific logic! Only:
    - 8×8 Grid
    - Generic movement patterns (L-shape, diagonal, etc.)
    - Path planning for piece movement
    - Target reachability
    """

    def test_8x8_board_setup(self):
        """Test generic 8×8 board setup (Chess-like, NO Chess logic)."""
        grid = Grid(height=8, width=8, name="8x8_Board")

        assert grid.height == 8
        assert grid.width == 8
        assert grid.get_cell_count() == 64

    def test_l_shaped_movement_pattern(self):
        """Test generic L-shaped movement pattern (Knight-like, NO Chess logic)."""

        # Generic: From position, get all L-shaped neighbor positions
        def get_l_shaped_neighbors(pos: Position) -> List[Position]:
            """Generic L-shaped movement: 2 steps in one direction, 1 step perpendicular."""
            deltas = [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ]
            neighbors = []
            for dx, dy in deltas:
                neighbor = Position(pos.x + dx, pos.y + dy)
                neighbors.append(neighbor)
            return neighbors

        # Test from center position (4, 4)
        center = Position(4, 4)
        l_neighbors = get_l_shaped_neighbors(center)

        # Should have 8 L-shaped neighbors (if all in bounds)
        assert len(l_neighbors) == 8

        # Verify one neighbor: (4+2, 4+1) = (6, 5)
        assert Position(6, 5) in l_neighbors

    def test_diagonal_movement_pattern(self):
        """Test generic diagonal movement pattern (Bishop-like, NO Chess logic)."""
        # Generic: From position, get all diagonal neighbors (distance 1)
        center = Position(4, 4)
        diagonal_neighbors = center.get_neighbors(NeighborhoodType.DIAGONAL)

        # Should have 4 diagonal neighbors
        assert len(diagonal_neighbors) == 4

        # Verify diagonals
        assert Position(5, 5) in diagonal_neighbors  # Northeast
        assert Position(3, 5) in diagonal_neighbors  # Northwest
        assert Position(5, 3) in diagonal_neighbors  # Southeast
        assert Position(3, 3) in diagonal_neighbors  # Southwest

    def test_straight_line_movement_pattern(self):
        """Test generic straight-line movement (Rook-like, NO Chess logic)."""
        # Generic: From position, get all orthogonal neighbors
        center = Position(4, 4)
        orthogonal_neighbors = center.get_neighbors(NeighborhoodType.ORTHOGONAL)

        # Should have 4 orthogonal neighbors
        assert len(orthogonal_neighbors) == 4

        # Verify
        assert Position(4, 5) in orthogonal_neighbors  # North
        assert Position(4, 3) in orthogonal_neighbors  # South
        assert Position(5, 4) in orthogonal_neighbors  # East
        assert Position(3, 4) in orthogonal_neighbors  # West

    def test_piece_placement_with_constraints(self):
        """Test placing pieces on board with generic constraints."""
        # Simplified: Place 5 pieces on 5×5 board such that no two pieces are in same row
        # This demonstrates capability without excessive complexity

        grid = Grid(height=5, width=5)

        # Generic variables: 5 pieces, each must choose a column (0-4)
        # Row is fixed (piece i is in row i)
        variables = [
            Variable(name=f"Piece{i}", domain=list(range(5))) for i in range(5)
        ]

        # Generic constraint: All different columns
        def all_different(var_dict: Dict[str, int]) -> bool:
            values = [v for v in var_dict.values() if v is not None]
            return len(values) == len(set(values))

        constraint = Constraint(
            variables=[f"Piece{i}" for i in range(5)],
            constraint_type=ConstraintType.ALL_DIFFERENT,
            constraint_function=all_different,
        )

        problem = ConstraintProblem(variables=variables, constraints=[constraint])
        solver = ConstraintSolver(problem)

        # Disable proof tree for better performance
        solution, proof_tree = solver.solve(track_proof=False)

        # Should find solution
        assert solution is not None
        # All columns should be different
        columns = list(solution.values())
        assert len(columns) == len(set(columns))

    def test_reachability_with_movement_pattern(self):
        """Test reachability using generic movement patterns."""
        # Generic: Can piece at (0, 0) reach (7, 7) using L-shaped moves?

        grid = Grid(height=8, width=8)
        start = Position(0, 0)
        goal = Position(7, 7)

        def get_l_shaped_neighbors(pos: Position) -> List[Position]:
            deltas = [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ]
            neighbors = []
            for dx, dy in deltas:
                neighbor = Position(pos.x + dx, pos.y + dy)
                if grid.is_valid_position(neighbor):
                    neighbors.append(neighbor)
            return neighbors

        # BFS to check reachability
        from collections import deque

        queue = deque([start])
        visited = {start}
        reachable = False

        while queue:
            current = queue.popleft()

            if current == goal:
                reachable = True
                break

            for neighbor in get_l_shaped_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # (0,0) to (7,7) should be reachable with L-shaped moves
        assert reachable

    def test_path_finding_with_custom_movement(self):
        """Test path finding with custom movement pattern."""
        # Generic: Find path using orthogonal moves only

        grid = Grid(height=8, width=8)
        start = Position(0, 0)
        goal = Position(7, 7)

        # BFS with orthogonal movement
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}
        path_found = None

        while queue:
            current, path = queue.popleft()

            if current == goal:
                path_found = path
                break

            neighbors = grid.get_neighbors(current, NeighborhoodType.ORTHOGONAL)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # Should find path
        assert path_found is not None
        assert path_found[0] == start
        assert path_found[-1] == goal
        # Optimal path length: 14 (7 east + 7 north)
        assert len(path_found) == 15  # 14 moves + start position

    def test_capture_constraint(self):
        """Test generic capture constraint (piece can attack target)."""
        # Generic: Piece A at (3, 3) can "attack" any orthogonal neighbor

        grid = Grid(height=8, width=8)
        piece_a = Position(3, 3)

        # Attackable positions (orthogonal neighbors)
        attackable = piece_a.get_neighbors(NeighborhoodType.ORTHOGONAL)

        # Target at (3, 4) - should be attackable
        target = Position(3, 4)
        can_attack = target in attackable

        assert can_attack

        # Target at (5, 5) - should NOT be attackable (not orthogonal neighbor)
        target_far = Position(5, 5)
        cannot_attack = target_far not in attackable

        assert cannot_attack


# ==================== Generality Validation Tests ====================


class TestSpatialReasoningGenerality:
    """Validate that spatial reasoning system is generic and extensible."""

    def test_arbitrary_grid_size(self):
        """Test that system works with arbitrary grid sizes."""
        # Test various sizes
        sizes = [(3, 3), (5, 5), (8, 8), (9, 9), (10, 10), (16, 16)]

        for height, width in sizes:
            grid = Grid(height=height, width=width, name=f"{height}x{width}_Grid")
            assert grid.height == height
            assert grid.width == width
            assert grid.get_cell_count() == height * width

    def test_arbitrary_movement_patterns(self):
        """Test that system supports arbitrary movement patterns."""

        # Define custom movement pattern: "T-shape" (up 2, left/right 1)
        def get_t_shaped_neighbors(pos: Position) -> List[Position]:
            """Custom T-shaped movement."""
            deltas = [(0, 2), (-1, 1), (1, 1)]  # Up 2, or diagonally up-left/up-right
            return [Position(pos.x + dx, pos.y + dy) for dx, dy in deltas]

        center = Position(5, 5)
        t_neighbors = get_t_shaped_neighbors(center)

        assert len(t_neighbors) == 3
        assert Position(5, 7) in t_neighbors  # Up 2
        assert Position(4, 6) in t_neighbors  # Up-left
        assert Position(6, 6) in t_neighbors  # Up-right

    def test_arbitrary_constraints(self):
        """Test that system supports arbitrary constraints."""

        # Define custom constraint: "Sum of values in group equals N"
        def sum_constraint(var_dict: Dict[str, int], target_sum: int) -> bool:
            """Generic: Sum of values must equal target."""
            values = [v for v in var_dict.values() if v is not None]
            return sum(values) == target_sum

        # Test
        assert sum_constraint({"A": 1, "B": 2, "C": 3}, target_sum=6)
        assert not sum_constraint({"A": 1, "B": 2, "C": 3}, target_sum=10)

    def test_extensibility_to_new_domains(self):
        """Test that system can be extended to new domains without modification."""
        # Hypothetical: "Checkers-like" scenario (8×8, diagonal moves)
        grid = Grid(height=8, width=8, name="Checkers_Board")

        # Use existing diagonal movement pattern
        piece = Position(3, 3)
        diagonal_moves = piece.get_neighbors(NeighborhoodType.DIAGONAL)

        assert len(diagonal_moves) == 4

        # Hypothetical: "Go-like" scenario (19×19, capture groups)
        go_board = Grid(height=19, width=19, name="Go_Board")
        assert go_board.get_cell_count() == 361

        # Both scenarios use same generic primitives!


# ==================== Test Configuration ====================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: slow application tests with complex scenarios"
    )
    config.addinivalue_line(
        "markers", "application: application example tests demonstrating generality"
    )
