"""
Tests for Component 42: Spatial Reasoning - Phase 2

Tests for:
- Phase 2.1: Grid representation in graph
- Phase 2.2: Position tracking and updates
- Phase 2.3: Neighborhood logic

Author: KAI Development Team
Date: 2025-11-05
"""

from unittest.mock import Mock, patch

import pytest

from component_42_spatial_reasoning import (
    Grid,
    NeighborhoodType,
    Position,
    SpatialReasoner,
)

# ============================================================================
# Phase 2.1: Grid Data Structure Tests
# ============================================================================


class TestGrid:
    """Test Grid data structure."""

    def test_create_grid(self):
        """Test creating a grid."""
        grid = Grid(name="TestGrid", width=8, height=8)

        assert grid.name == "TestGrid"
        assert grid.width == 8
        assert grid.height == 8
        assert grid.size == 64
        assert grid.neighborhood_type == NeighborhoodType.ORTHOGONAL

    def test_grid_with_custom_neighborhood(self):
        """Test grid with custom neighborhood."""
        knight_moves = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ]

        grid = Grid(
            name="ChessGrid",
            width=8,
            height=8,
            neighborhood_type=NeighborhoodType.CUSTOM,
            custom_offsets=knight_moves,
        )

        assert grid.neighborhood_type == NeighborhoodType.CUSTOM
        assert grid.custom_offsets == knight_moves

    def test_grid_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError):
            Grid(name="Invalid", width=0, height=5)

        with pytest.raises(ValueError):
            Grid(name="Invalid", width=5, height=-1)

    def test_grid_custom_without_offsets(self):
        """Test that custom neighborhood without offsets raises error."""
        with pytest.raises(ValueError):
            Grid(
                name="Invalid",
                width=5,
                height=5,
                neighborhood_type=NeighborhoodType.CUSTOM,
            )

    def test_grid_is_valid_position(self):
        """Test position validation."""
        grid = Grid(name="Test", width=5, height=5)

        assert grid.is_valid_position(Position(0, 0)) is True
        assert grid.is_valid_position(Position(4, 4)) is True
        assert grid.is_valid_position(Position(2, 3)) is True

        assert grid.is_valid_position(Position(-1, 0)) is False
        assert grid.is_valid_position(Position(0, -1)) is False
        assert grid.is_valid_position(Position(5, 0)) is False
        assert grid.is_valid_position(Position(0, 5)) is False

    def test_grid_get_all_positions(self):
        """Test getting all positions in grid."""
        grid = Grid(name="Small", width=2, height=3)

        positions = grid.get_all_positions()

        assert len(positions) == 6  # 2x3 = 6
        assert Position(0, 0) in positions
        assert Position(1, 2) in positions

    def test_grid_get_position_name(self):
        """Test position name generation."""
        grid = Grid(name="MyGrid", width=8, height=8)

        pos_name = grid.get_position_name(Position(3, 5))

        assert pos_name == "MyGrid_Pos_3_5"

    def test_grid_metadata(self):
        """Test grid with custom metadata."""
        grid = Grid(
            name="Sudoku",
            width=9,
            height=9,
            metadata={"game_type": "sudoku", "difficulty": "hard"},
        )

        assert grid.metadata["game_type"] == "sudoku"
        assert grid.metadata["difficulty"] == "hard"


# ============================================================================
# Phase 2.1: Grid Management Tests
# ============================================================================


class TestGridManagement:
    """Test grid creation and management in knowledge graph."""

    def test_create_grid_in_graph(self):
        """Test creating a grid in the knowledge graph."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)
        grid = Grid(name="TestGrid", width=2, height=2)

        success = reasoner.create_grid(grid)

        assert success is True
        # Should create grid node + 4 position nodes (2x2)
        assert mock_netzwerk.create_wort_if_not_exists.call_count == 5

    def test_get_grid_from_graph(self):
        """Test retrieving a grid from knowledge graph."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "size": 64,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        grid = reasoner.get_grid("TestGrid")

        assert grid is not None
        assert grid.name == "TestGrid"
        assert grid.width == 8
        assert grid.height == 8

    def test_get_nonexistent_grid(self):
        """Test retrieving a grid that doesn't exist."""
        mock_netzwerk = Mock()
        mock_netzwerk.find_wort_node = Mock(return_value=None)

        reasoner = SpatialReasoner(mock_netzwerk)
        grid = reasoner.get_grid("NonExistent")

        assert grid is None

    def test_delete_grid(self):
        """Test deleting a grid."""
        mock_netzwerk = Mock()

        # Mock get_grid to return a small grid
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 2,
            "height": 2,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.delete_wort_node = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)
        success = reasoner.delete_grid("TestGrid")

        assert success is True
        # Should delete 4 positions + 1 grid node = 5 calls
        assert mock_netzwerk.delete_wort_node.call_count == 5


# ============================================================================
# Phase 2.2: Position Tracking Tests
# ============================================================================


class TestPositionTracking:
    """Test position tracking and object placement."""

    def test_place_object(self):
        """Test placing an object on a grid."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)
        success = reasoner.place_object("König", "TestGrid", Position(4, 4))

        assert success is True
        mock_netzwerk.create_wort_if_not_exists.assert_called_once()
        mock_netzwerk.assert_relation.assert_called_once()

    def test_place_object_out_of_bounds(self):
        """Test that placing object out of bounds fails."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        success = reasoner.place_object("König", "TestGrid", Position(10, 10))

        assert success is False

    def test_move_object(self):
        """Test moving an object between positions."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.delete_relation = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)
        success = reasoner.move_object(
            "König", "TestGrid", Position(4, 4), Position(5, 5)
        )

        assert success is True
        mock_netzwerk.delete_relation.assert_called_once()
        mock_netzwerk.assert_relation.assert_called_once()

    def test_remove_object(self):
        """Test removing an object from a position."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.delete_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)
        success = reasoner.remove_object("König", "TestGrid", Position(4, 4))

        assert success is True
        mock_netzwerk.delete_relation.assert_called_once()

    def test_get_object_position(self):
        """Test getting position of an object."""
        mock_netzwerk = Mock()
        mock_netzwerk.query_graph_for_facts = Mock(
            return_value={"LOCATED_AT": ["TestGrid_Pos_3_5"]}
        )
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        position = reasoner.get_object_position("König", "TestGrid")

        assert position is not None
        assert position.x == 3
        assert position.y == 5

    def test_get_object_position_not_found(self):
        """Test getting position when object is not placed."""
        mock_netzwerk = Mock()
        mock_netzwerk.query_graph_for_facts = Mock(return_value={})

        reasoner = SpatialReasoner(mock_netzwerk)
        position = reasoner.get_object_position("König", "TestGrid")

        assert position is None

    def test_get_objects_at_position(self):
        """Test getting all objects at a position."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        # Mock session.run for Cypher query
        mock_result = [{"object_name": "König"}, {"object_name": "Turm"}]
        mock_netzwerk.session.run = Mock(return_value=mock_result)

        reasoner = SpatialReasoner(mock_netzwerk)
        objects = reasoner.get_objects_at_position("TestGrid", Position(4, 4))

        assert len(objects) == 2
        assert "König" in objects
        assert "Turm" in objects


# ============================================================================
# Phase 2.3: Neighborhood Logic Tests
# ============================================================================


class TestNeighborhoodLogic:
    """Test neighborhood queries and pathfinding."""

    def test_get_neighbors(self):
        """Test getting neighbors of a position."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 5,
            "height": 5,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        neighbors = reasoner.get_neighbors("TestGrid", Position(2, 2))

        # Center position should have 4 orthogonal neighbors
        assert len(neighbors) == 4

    def test_get_neighbors_corner(self):
        """Test getting neighbors of corner position."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 5,
            "height": 5,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        neighbors = reasoner.get_neighbors("TestGrid", Position(0, 0))

        # Corner position should have only 2 neighbors
        assert len(neighbors) == 2

    def test_get_neighbors_moore(self):
        """Test getting Moore (8-directional) neighbors."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 5,
            "height": 5,
            "neighborhood_type": "moore",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        neighbors = reasoner.get_neighbors("TestGrid", Position(2, 2))

        # Center position should have 8 Moore neighbors
        assert len(neighbors) == 8

    def test_find_path_straight(self):
        """Test finding a straight path."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        path = reasoner.find_path("TestGrid", Position(0, 0), Position(3, 0))

        assert path is not None
        assert len(path) == 4  # 0->1->2->3
        assert path[0] == Position(0, 0)
        assert path[-1] == Position(3, 0)

    def test_find_path_diagonal(self):
        """Test finding path with diagonal moves."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "moore",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        path = reasoner.find_path(
            "TestGrid", Position(0, 0), Position(2, 2), allow_diagonal=True
        )

        assert path is not None
        # Diagonal path should be shorter
        assert len(path) <= 3

    def test_get_distance_between_positions(self):
        """Test distance calculation."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Manhattan distance
        dist = reasoner.get_distance_between_positions(
            Position(0, 0), Position(3, 4), metric="manhattan"
        )
        assert dist == 7  # |3-0| + |4-0| = 7

        # Euclidean distance
        dist = reasoner.get_distance_between_positions(
            Position(0, 0), Position(3, 4), metric="euclidean"
        )
        assert dist == 5.0  # sqrt(3^2 + 4^2) = 5

    def test_get_objects_in_neighborhood(self):
        """Test getting objects within radius."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        # Mock get_objects_at_position to return objects at specific positions
        def mock_get_objects(grid_name, position):
            if position == Position(2, 2):
                return ["König"]
            elif position == Position(2, 3):
                return ["Turm"]
            return []

        reasoner = SpatialReasoner(mock_netzwerk)

        # Patch get_objects_at_position
        with patch.object(
            reasoner, "get_objects_at_position", side_effect=mock_get_objects
        ):
            objects = reasoner.get_objects_in_neighborhood(
                "TestGrid", Position(2, 2), radius=1
            )

            assert len(objects) >= 1
            assert Position(2, 2) in objects or Position(2, 3) in objects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
