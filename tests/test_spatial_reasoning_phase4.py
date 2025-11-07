"""
Tests for Component 42: Spatial Reasoning - Phase 4

Tests for:
- Phase 4.1: Spatial Constraint System
- Phase 4.2: Movement Planning
- Phase 4.3: Rule Learning for spatial domains

Author: KAI Development Team
Date: 2025-11-05
"""

from unittest.mock import Mock

import pytest

from component_42_spatial_reasoning import (
    Position,
    SpatialReasoner,
)

# ============================================================================
# Phase 4.1: Spatial Constraint System Tests
# ============================================================================


class TestSpatialConstraints:
    """Test spatial constraint system."""

    def test_add_spatial_constraint(self):
        """Test adding a spatial constraint."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Add constraint: objects cannot be on same position
        success = reasoner.add_spatial_constraint(
            "no_same_pos",
            ["König", "Turm"],
            lambda pos: pos["König"] != pos["Turm"],
            "König und Turm nicht auf gleicher Position",
        )

        assert success is True
        assert hasattr(reasoner, "_spatial_constraints")
        assert "no_same_pos" in reasoner._spatial_constraints

    def test_check_constraints_satisfied(self):
        """Test checking constraints that are satisfied."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Add constraint
        reasoner.add_spatial_constraint(
            "different_positions",
            ["A", "B"],
            lambda pos: pos["A"] != pos["B"],
            "A and B must be on different positions",
        )

        # Check with valid positions
        positions = {"A": Position(0, 0), "B": Position(1, 1)}

        satisfied, violations = reasoner.check_spatial_constraints("grid", positions)

        assert satisfied is True
        assert len(violations) == 0

    def test_check_constraints_violated(self):
        """Test checking constraints that are violated."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Add constraint
        reasoner.add_spatial_constraint(
            "different_positions",
            ["A", "B"],
            lambda pos: pos["A"] != pos["B"],
            "A and B must be on different positions",
        )

        # Check with invalid positions (same position)
        positions = {"A": Position(0, 0), "B": Position(0, 0)}

        satisfied, violations = reasoner.check_spatial_constraints("grid", positions)

        assert satisfied is False
        assert len(violations) > 0

    def test_check_multiple_constraints(self):
        """Test checking multiple constraints."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Constraint 1: Different positions
        reasoner.add_spatial_constraint(
            "diff_pos", ["A", "B"], lambda pos: pos["A"] != pos["B"]
        )

        # Constraint 2: Manhattan distance >= 2
        reasoner.add_spatial_constraint(
            "min_distance",
            ["A", "B"],
            lambda pos: pos["A"].distance_to(pos["B"], "manhattan") >= 2,
        )

        # Test valid configuration
        positions = {"A": Position(0, 0), "B": Position(2, 0)}

        satisfied, violations = reasoner.check_spatial_constraints("grid", positions)

        assert satisfied is True

        # Test invalid configuration (too close)
        positions = {"A": Position(0, 0), "B": Position(1, 0)}

        satisfied, violations = reasoner.check_spatial_constraints("grid", positions)

        assert satisfied is False
        assert len(violations) == 1  # Only distance constraint violated

    def test_find_valid_positions_simple(self):
        """Test finding valid positions with simple constraints."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Create small 2x2 grid
        mock_grid_node = {
            "lemma": "SmallGrid",
            "width": 2,
            "height": 2,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        # Constraint: Objects must be on different positions
        reasoner.add_spatial_constraint(
            "different", ["A", "B"], lambda pos: pos["A"] != pos["B"]
        )

        # Find solutions
        solutions = reasoner.find_valid_positions(
            "SmallGrid", ["A", "B"], max_solutions=5
        )

        # Should find multiple solutions (each object on different position)
        assert len(solutions) > 0
        for solution in solutions:
            assert solution["A"] != solution["B"]

    def test_find_valid_positions_no_solution(self):
        """Test finding positions when no valid solution exists."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Create 1x1 grid (only one position)
        mock_grid_node = {
            "lemma": "TinyGrid",
            "width": 1,
            "height": 1,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        # Constraint: Objects must be different (impossible on 1x1 grid)
        reasoner.add_spatial_constraint(
            "different", ["A", "B"], lambda pos: pos["A"] != pos["B"]
        )

        # Find solutions
        solutions = reasoner.find_valid_positions("TinyGrid", ["A", "B"])

        # Should find no solutions
        assert len(solutions) == 0

    def test_clear_spatial_constraints(self):
        """Test clearing all constraints."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Add some constraints
        reasoner.add_spatial_constraint("c1", ["A"], lambda pos: True)
        reasoner.add_spatial_constraint("c2", ["B"], lambda pos: True)

        assert len(reasoner._spatial_constraints) == 2

        # Clear
        reasoner.clear_spatial_constraints()

        assert len(reasoner._spatial_constraints) == 0

    def test_constraint_with_adjacent_requirement(self):
        """Test constraint requiring adjacent positions."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Constraint: Objects must be adjacent (Manhattan distance == 1)
        reasoner.add_spatial_constraint(
            "adjacent",
            ["König", "Wächter"],
            lambda pos: pos["König"].distance_to(pos["Wächter"], "manhattan") == 1,
            "Wächter muss neben König stehen",
        )

        # Valid: Adjacent
        positions = {"König": Position(5, 5), "Wächter": Position(5, 6)}
        satisfied, _ = reasoner.check_spatial_constraints("grid", positions)
        assert satisfied is True

        # Invalid: Too far
        positions = {"König": Position(5, 5), "Wächter": Position(7, 7)}
        satisfied, _ = reasoner.check_spatial_constraints("grid", positions)
        assert satisfied is False

    def test_constraint_with_region_restriction(self):
        """Test constraint restricting objects to regions."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Constraint: Object must be in specific region (y < 3)
        reasoner.add_spatial_constraint(
            "upper_region",
            ["Spieler"],
            lambda pos: pos["Spieler"].y < 3,
            "Spieler muss in oberer Hälfte bleiben",
        )

        # Valid
        positions = {"Spieler": Position(5, 2)}
        satisfied, _ = reasoner.check_spatial_constraints("grid", positions)
        assert satisfied is True

        # Invalid
        positions = {"Spieler": Position(5, 5)}
        satisfied, _ = reasoner.check_spatial_constraints("grid", positions)
        assert satisfied is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
