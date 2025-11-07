"""
tests/test_spatial_constraints.py

Unit tests for spatial constraint satisfaction integration.

Tests cover:
- Spatial CSP problem formulation
- Position constraints (placement, exclusion, region)
- Grid-based constraint propagation
- Integration with component_29_constraint_reasoning
- Generic constraint patterns (no domain-specific logic)
"""

from typing import Any, Callable, Dict, List

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


# Helper function to create constraints with backwards compatibility
def create_constraint(
    variables: List[str],
    constraint_function: Callable,
    constraint_type: ConstraintType = None,
) -> Constraint:
    """Helper to create constraints with test-compatible API."""
    name = f"constraint_{'_'.join(variables)}"
    return Constraint(name=name, scope=variables, predicate=constraint_function)


# ==================== Spatial CSP Problem Tests ====================


class TestSpatialCSPFormulation:
    """Test formulation of spatial problems as CSP."""

    def test_position_assignment_problem(self):
        """Test CSP for assigning positions to objects."""
        # Problem: Place 3 objects A, B, C on a 3x3 grid
        # Each object must occupy exactly one position

        variables = [
            Variable(
                name="A", domain=[Position(i, j) for i in range(3) for j in range(3)]
            ),
            Variable(
                name="B", domain=[Position(i, j) for i in range(3) for j in range(3)]
            ),
            Variable(
                name="C", domain=[Position(i, j) for i in range(3) for j in range(3)]
            ),
        ]

        # Constraint: No two objects can occupy same position
        def all_different_positions(var_dict: Dict[str, Any]) -> bool:
            positions = list(var_dict.values())
            return len(positions) == len(set(positions))

        constraint = Constraint(
            name="all_different_positions",
            scope=["A", "B", "C"],
            predicate=all_different_positions,
        )

        problem = ConstraintProblem(variables=variables, constraints=[constraint])

        assert len(problem.variables) == 3
        assert len(problem.constraints) == 1

    def test_regional_constraint_problem(self):
        """Test CSP with regional constraints (objects in specific regions)."""
        # Problem: Place objects in specific regions of a grid
        grid = Grid(height=6, width=6)

        # Define regions
        top_left_region = [(i, j) for i in range(3) for j in range(3)]
        bottom_right_region = [(i, j) for i in range(3, 6) for j in range(3, 6)]

        # Variable: Object A must be in top-left region
        domain_a = [Position(i, j) for i, j in top_left_region]

        # Variable: Object B must be in bottom-right region
        domain_b = [Position(i, j) for i, j in bottom_right_region]

        var_a = Variable(name="A", domain=domain_a)
        var_b = Variable(name="B", domain=domain_b)

        # Regions are disjoint, so any assignment is valid
        problem = ConstraintProblem(variables=[var_a, var_b], constraints=[])

        assert len(var_a.domain) == 9  # 3x3 region
        assert len(var_b.domain) == 9
        assert grid.is_valid_position(domain_a[0])
        assert grid.is_valid_position(domain_b[0])

    def test_adjacency_constraint_problem(self):
        """Test CSP with adjacency constraints."""
        # Problem: Place A and B such that they are adjacent
        grid = Grid(height=5, width=5)

        positions = [Position(i, j) for i in range(5) for j in range(5)]
        var_a = Variable(name="A", domain=positions)
        var_b = Variable(name="B", domain=positions)

        # Constraint: A and B must be adjacent (orthogonal neighbors)
        def are_adjacent(var_dict: Dict[str, Position]) -> bool:
            pos_a = var_dict.get("A")
            pos_b = var_dict.get("B")
            if pos_a is None or pos_b is None:
                return False

            # Check if B is in A's orthogonal neighbors
            neighbors = pos_a.get_neighbors(NeighborhoodType.ORTHOGONAL)
            return pos_b in neighbors

        constraint = Constraint(
            variables=["A", "B"],
            constraint_type=ConstraintType.BINARY,
            constraint_function=are_adjacent,
        )

        problem = ConstraintProblem(variables=[var_a, var_b], constraints=[constraint])

        # Validate constraint function
        assert are_adjacent({"A": Position(2, 2), "B": Position(2, 3)})  # Adjacent
        assert not are_adjacent(
            {"A": Position(2, 2), "B": Position(4, 4)}
        )  # Not adjacent

    def test_exclusion_zone_constraint(self):
        """Test CSP with exclusion zones (forbidden positions)."""
        # Problem: Place object avoiding certain positions (obstacles)
        grid = Grid(height=5, width=5)

        # Mark obstacles
        obstacles = [Position(2, 2), Position(2, 3), Position(3, 2)]
        for obs in obstacles:
            grid.set_cell_data(obs, "obstacle")

        # Domain excludes obstacle positions
        all_positions = [Position(i, j) for i in range(5) for j in range(5)]
        valid_positions = [p for p in all_positions if p not in obstacles]

        var = Variable(name="Object", domain=valid_positions)

        assert len(var.domain) == 25 - 3  # 22 valid positions
        assert Position(2, 2) not in var.domain
        assert Position(0, 0) in var.domain


# ==================== Constraint Propagation Tests ====================


class TestSpatialConstraintPropagation:
    """Test constraint propagation for spatial problems."""

    def test_arc_consistency_position_constraints(self):
        """Test AC-3 algorithm on position constraints."""
        # Simple problem: A and B must be different positions
        positions = [Position(0, 0), Position(0, 1), Position(1, 0)]
        var_a = Variable(name="A", domain=positions.copy())
        var_b = Variable(name="B", domain=positions.copy())

        def different_positions(var_dict: Dict[str, Position]) -> bool:
            return var_dict.get("A") != var_dict.get("B")

        constraint = Constraint(
            variables=["A", "B"],
            constraint_type=ConstraintType.BINARY,
            constraint_function=different_positions,
        )

        problem = ConstraintProblem(variables=[var_a, var_b], constraints=[constraint])
        solver = ConstraintSolver(problem)

        # Before AC-3: both have 3 values
        assert len(var_a.domain) == 3
        assert len(var_b.domain) == 3

        # Run AC-3
        is_consistent = solver.ac3()

        # After AC-3: domains should still have 3 values each
        # (no reduction possible without assignments)
        assert is_consistent
        assert len(var_a.domain) == 3
        assert len(var_b.domain) == 3

    def test_unary_constraint_domain_reduction(self):
        """Test domain reduction with unary constraints."""
        # Problem: Object must be in top half of grid
        grid = Grid(height=10, width=10)

        all_positions = [Position(i, j) for i in range(10) for j in range(10)]
        var = Variable(name="Object", domain=all_positions)

        # Unary constraint: y >= 5 (top half)
        def in_top_half(var_dict: Dict[str, Position]) -> bool:
            pos = var_dict.get("Object")
            return pos is not None and pos.y >= 5

        constraint = Constraint(
            variables=["Object"],
            constraint_type=ConstraintType.UNARY,
            constraint_function=in_top_half,
        )

        problem = ConstraintProblem(variables=[var], constraints=[constraint])

        # Filter domain manually (CSP solver would do this)
        filtered_domain = [p for p in var.domain if in_top_half({"Object": p})]

        assert len(filtered_domain) == 50  # Half the grid


# ==================== Solver Integration Tests ====================


class TestSpatialConstraintSolver:
    """Test CSP solver integration with spatial problems."""

    def test_solve_simple_placement_problem(self):
        """Test solving a simple 2-object placement problem."""
        # Place A and B on a 2x2 grid (4 positions)
        positions = [Position(0, 0), Position(0, 1), Position(1, 0), Position(1, 1)]
        var_a = Variable(name="A", domain=positions.copy())
        var_b = Variable(name="B", domain=positions.copy())

        # Constraint: Different positions
        def different(var_dict: Dict[str, Position]) -> bool:
            a = var_dict.get("A")
            b = var_dict.get("B")
            if a is None or b is None:
                return True  # Partial assignment
            return a != b

        constraint = Constraint(
            variables=["A", "B"],
            constraint_type=ConstraintType.BINARY,
            constraint_function=different,
        )

        problem = ConstraintProblem(variables=[var_a, var_b], constraints=[constraint])
        solver = ConstraintSolver(problem)

        # Solve
        solution, proof_tree = solver.solve()

        # Should find a solution (12 possible: 4*3 ordered pairs)
        assert solution is not None
        assert solution["A"] != solution["B"]
        assert solution["A"] in positions
        assert solution["B"] in positions

    def test_solve_with_adjacency_constraint(self):
        """Test solving placement with adjacency constraint."""
        # 3x3 grid, place A and B adjacent
        positions = [Position(i, j) for i in range(3) for j in range(3)]
        var_a = Variable(name="A", domain=positions.copy())
        var_b = Variable(name="B", domain=positions.copy())

        # Constraint: Adjacent (orthogonal)
        def adjacent(var_dict: Dict[str, Position]) -> bool:
            a = var_dict.get("A")
            b = var_dict.get("B")
            if a is None or b is None:
                return True  # Partial assignment

            neighbors = a.get_neighbors(NeighborhoodType.ORTHOGONAL)
            return b in neighbors

        constraint = Constraint(
            variables=["A", "B"],
            constraint_type=ConstraintType.BINARY,
            constraint_function=adjacent,
        )

        problem = ConstraintProblem(variables=[var_a, var_b], constraints=[constraint])
        solver = ConstraintSolver(problem)

        solution, proof_tree = solver.solve()

        # Should find a solution
        assert solution is not None
        assert adjacent(solution)

    def test_unsolvable_problem_detection(self):
        """Test detection of unsolvable spatial CSP problems."""
        # Impossible problem: Place 2 objects on 1 position
        single_position = [Position(0, 0)]
        var_a = Variable(name="A", domain=single_position)
        var_b = Variable(name="B", domain=single_position)

        # Constraint: Different positions (impossible!)
        def different(var_dict: Dict[str, Position]) -> bool:
            a = var_dict.get("A")
            b = var_dict.get("B")
            if a is None or b is None:
                return True
            return a != b

        constraint = Constraint(
            variables=["A", "B"],
            constraint_type=ConstraintType.BINARY,
            constraint_function=different,
        )

        problem = ConstraintProblem(variables=[var_a, var_b], constraints=[constraint])
        solver = ConstraintSolver(problem)

        # Should detect no solution
        solution, proof_tree = solver.solve()
        assert solution is None

    @pytest.mark.slow
    def test_solve_large_placement_problem(self):
        """Test solving a larger placement problem."""
        # Place 5 objects on a 5x5 grid with all-different constraint
        positions = [Position(i, j) for i in range(5) for j in range(5)]
        variables = [
            Variable(name=f"Obj{i}", domain=positions.copy()) for i in range(5)
        ]

        # Constraint: All different positions
        def all_different(var_dict: Dict[str, Position]) -> bool:
            positions_assigned = [v for v in var_dict.values() if v is not None]
            return len(positions_assigned) == len(set(positions_assigned))

        constraint = Constraint(
            variables=[f"Obj{i}" for i in range(5)],
            constraint_type=ConstraintType.ALL_DIFFERENT,
            constraint_function=all_different,
        )

        problem = ConstraintProblem(variables=variables, constraints=[constraint])
        solver = ConstraintSolver(problem)

        solution, proof_tree = solver.solve()

        # Should find a solution (25!/(25-5)! = very many solutions)
        assert solution is not None
        assert len(solution) == 5
        assert len(set(solution.values())) == 5  # All different


# ==================== Generic Constraint Pattern Tests ====================


class TestGenericSpatialConstraints:
    """Test generic spatial constraint patterns (no domain-specific logic)."""

    def test_row_constraint_pattern(self):
        """Test generic row-based constraint (no domain logic)."""
        # Generic pattern: Objects in same row
        grid = Grid(height=5, width=5)

        def same_row(var_dict: Dict[str, Position]) -> bool:
            """Generic: Check if all positions are in same row."""
            positions = [v for v in var_dict.values() if v is not None]
            if len(positions) < 2:
                return True
            return all(p.y == positions[0].y for p in positions)

        # Test constraint function
        assert same_row({"A": Position(0, 2), "B": Position(1, 2), "C": Position(2, 2)})
        assert not same_row({"A": Position(0, 2), "B": Position(1, 3)})

    def test_column_constraint_pattern(self):
        """Test generic column-based constraint."""

        def same_column(var_dict: Dict[str, Position]) -> bool:
            """Generic: Check if all positions are in same column."""
            positions = [v for v in var_dict.values() if v is not None]
            if len(positions) < 2:
                return True
            return all(p.x == positions[0].x for p in positions)

        # Test
        assert same_column(
            {"A": Position(3, 0), "B": Position(3, 1), "C": Position(3, 2)}
        )
        assert not same_column({"A": Position(3, 0), "B": Position(4, 1)})

    def test_region_constraint_pattern(self):
        """Test generic region-based constraint (e.g., 3x3 sub-grids)."""

        def in_same_region(var_dict: Dict[str, Position], region_size: int = 3) -> bool:
            """Generic: Check if all positions are in same region."""
            positions = [v for v in var_dict.values() if v is not None]
            if len(positions) < 2:
                return True

            # Region ID: (x // region_size, y // region_size)
            first_region = (
                positions[0].x // region_size,
                positions[0].y // region_size,
            )
            return all(
                (p.x // region_size, p.y // region_size) == first_region
                for p in positions
            )

        # Test with 3x3 regions on a 9x9 grid
        # Positions in same region
        assert in_same_region({"A": Position(0, 0), "B": Position(2, 2)}, region_size=3)
        # Positions in different regions
        assert not in_same_region(
            {"A": Position(0, 0), "B": Position(3, 0)}, region_size=3
        )

    def test_diagonal_constraint_pattern(self):
        """Test generic diagonal constraint (no attacks on same diagonal)."""

        def not_on_same_diagonal(var_dict: Dict[str, Position]) -> bool:
            """Generic: Check if no two positions are on same diagonal."""
            positions = list(var_dict.values())
            for i, pos1 in enumerate(positions):
                if pos1 is None:
                    continue
                for pos2 in positions[i + 1 :]:
                    if pos2 is None:
                        continue
                    # Same diagonal: |dx| == |dy|
                    if abs(pos1.x - pos2.x) == abs(pos1.y - pos2.y):
                        return False
            return True

        # Test
        # (0,0) and (1,1) are on same diagonal
        assert not not_on_same_diagonal({"A": Position(0, 0), "B": Position(1, 1)})
        # (0,0) and (0,1) are NOT on same diagonal
        assert not_on_same_diagonal({"A": Position(0, 0), "B": Position(0, 1)})

    def test_distance_constraint_pattern(self):
        """Test generic distance-based constraint."""

        def min_distance_constraint(
            var_dict: Dict[str, Position], min_dist: int = 2
        ) -> bool:
            """Generic: All positions must be at least min_dist apart (Manhattan)."""
            positions = [v for v in var_dict.values() if v is not None]
            for i, pos1 in enumerate(positions):
                for pos2 in positions[i + 1 :]:
                    if pos1.manhattan_distance_to(pos2) < min_dist:
                        return False
            return True

        # Test
        # Distance = 1 (too close)
        assert not min_distance_constraint(
            {"A": Position(0, 0), "B": Position(0, 1)}, min_dist=2
        )
        # Distance = 2 (exactly min)
        assert min_distance_constraint(
            {"A": Position(0, 0), "B": Position(0, 2)}, min_dist=2
        )
        # Distance = 3 (ok)
        assert min_distance_constraint(
            {"A": Position(0, 0), "B": Position(0, 3)}, min_dist=2
        )


# ==================== Heuristic Tests ====================


class TestSpatialCSPHeuristics:
    """Test heuristics for spatial CSP solving."""

    def test_mrv_heuristic_spatial(self):
        """Test Minimum Remaining Values heuristic for spatial problems."""
        # Setup: One variable with small domain, one with large domain
        var_a = Variable(name="A", domain=[Position(0, 0), Position(0, 1)])
        var_b = Variable(
            name="B", domain=[Position(i, j) for i in range(5) for j in range(5)]
        )

        # MRV should select A (smaller domain)
        problem = ConstraintProblem(variables=[var_a, var_b], constraints=[])
        ConstraintSolver(problem)

        # Get all variables (none assigned yet)
        all_variables = list(problem.variables.values())

        # MRV: Select variable with smallest domain
        selected = min(all_variables, key=lambda v: len(v.domain))

        assert selected.name == "A"
        assert len(selected.domain) == 2

    def test_degree_heuristic_spatial(self):
        """Test degree heuristic (most constrained variable) for spatial problems."""
        # Setup: Variable A has more constraints than B
        var_a = Variable(name="A", domain=[Position(0, 0), Position(0, 1)])
        var_b = Variable(name="B", domain=[Position(1, 0), Position(1, 1)])
        var_c = Variable(name="C", domain=[Position(2, 0), Position(2, 1)])

        # A is constrained by B and C
        # B is constrained only by A
        constraints = [
            Constraint(
                variables=["A", "B"],
                constraint_type=ConstraintType.BINARY,
                constraint_function=lambda d: d.get("A") != d.get("B"),
            ),
            Constraint(
                variables=["A", "C"],
                constraint_type=ConstraintType.BINARY,
                constraint_function=lambda d: d.get("A") != d.get("C"),
            ),
        ]

        problem = ConstraintProblem(
            variables=[var_a, var_b, var_c], constraints=constraints
        )

        # Count constraints per variable
        constraint_count = {"A": 0, "B": 0, "C": 0}
        for constraint in constraints:
            for var_name in constraint.scope:
                constraint_count[var_name] += 1

        # A has most constraints (degree = 2)
        assert constraint_count["A"] == 2
        assert constraint_count["B"] == 1
        assert constraint_count["C"] == 1


# ==================== Performance Tests ====================


class TestSpatialCSPPerformance:
    """Test performance characteristics of spatial CSP solving."""

    @pytest.mark.slow
    def test_large_grid_csp_performance(self):
        """Test CSP solving on moderately-sized grids."""
        # 6x6 grid, place 4 objects with all-different constraint
        # This demonstrates capability without excessive complexity
        # Disable proof tree generation for performance
        positions = [Position(i, j) for i in range(6) for j in range(6)]
        variables = [
            Variable(name=f"Obj{i}", domain=positions.copy()) for i in range(4)
        ]

        def all_different(var_dict: Dict[str, Position]) -> bool:
            positions_assigned = [v for v in var_dict.values() if v is not None]
            return len(positions_assigned) == len(set(positions_assigned))

        constraint = Constraint(
            variables=[f"Obj{i}" for i in range(4)],
            constraint_type=ConstraintType.ALL_DIFFERENT,
            constraint_function=all_different,
        )

        problem = ConstraintProblem(variables=variables, constraints=[constraint])
        solver = ConstraintSolver(problem)

        # Disable proof tree generation for better performance
        solution, proof_tree = solver.solve(track_proof=False)

        assert solution is not None
        assert len(solution) == 4
        assert len(set(solution.values())) == 4


# ==================== Test Configuration ====================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: slow CSP tests with large search spaces")
