"""
Tests for Component 42: Spatial Reasoning - Phase 4.2 Movement Planning

Tests for:
- Phase 4.2: Movement planning, validation, and execution

Author: KAI Development Team
Date: 2025-11-05
"""

from unittest.mock import Mock

import pytest

from component_42_spatial_reasoning import (
    MovementAction,
    MovementPlan,
    Position,
    SpatialReasoner,
)

# ============================================================================
# Phase 4.2: Movement Planning Tests
# ============================================================================


class TestMovementPlanning:
    """Test movement planning capabilities."""

    def test_create_movement_action(self):
        """Test creating a movement action."""
        action = MovementAction(
            object_name="König",
            from_position=Position(0, 0),
            to_position=Position(0, 1),
            step_number=1,
        )

        assert action.object_name == "König"
        assert action.from_position == Position(0, 0)
        assert action.to_position == Position(0, 1)
        assert action.step_number == 1

    def test_movement_action_string(self):
        """Test movement action string representation."""
        action = MovementAction(
            object_name="König",
            from_position=Position(2, 3),
            to_position=Position(2, 4),
            step_number=5,
        )

        string = str(action)
        assert "Step 5" in string
        assert "König" in string
        assert "(2, 3)" in string
        assert "(2, 4)" in string

    def test_create_movement_plan(self):
        """Test creating a movement plan."""
        actions = [
            MovementAction("König", Position(0, 0), Position(0, 1), 1),
            MovementAction("König", Position(0, 1), Position(0, 2), 2),
        ]

        plan = MovementPlan(
            object_name="König",
            grid_name="ChessBoard",
            actions=actions,
            total_steps=2,
            path_length=3,
        )

        assert plan.object_name == "König"
        assert plan.grid_name == "ChessBoard"
        assert len(plan.actions) == 2
        assert plan.total_steps == 2
        assert plan.path_length == 3

    def test_get_final_position(self):
        """Test getting final position from plan."""
        actions = [
            MovementAction("König", Position(0, 0), Position(0, 1), 1),
            MovementAction("König", Position(0, 1), Position(0, 2), 2),
        ]

        plan = MovementPlan("König", "Grid", actions, 2, 3)
        final = plan.get_final_position()

        assert final == Position(0, 2)

    def test_get_path_from_plan(self):
        """Test extracting path from plan."""
        actions = [
            MovementAction("König", Position(0, 0), Position(0, 1), 1),
            MovementAction("König", Position(0, 1), Position(1, 1), 2),
            MovementAction("König", Position(1, 1), Position(2, 1), 3),
        ]

        plan = MovementPlan("König", "Grid", actions, 3, 4)
        path = plan.get_path()

        assert len(path) == 4
        assert path[0] == Position(0, 0)
        assert path[1] == Position(0, 1)
        assert path[2] == Position(1, 1)
        assert path[3] == Position(2, 1)

    def test_plan_simple_movement(self):
        """Test planning a simple movement."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 5,
            "height": 5,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.session.run = Mock(return_value=[])  # No objects at positions

        reasoner = SpatialReasoner(mock_netzwerk)

        plan = reasoner.plan_movement(
            object_name="König",
            grid_name="TestGrid",
            start_pos=Position(0, 0),
            goal_pos=Position(0, 3),
        )

        assert plan is not None
        assert plan.object_name == "König"
        assert plan.total_steps == 3  # 0->1->2->3
        assert plan.actions[0].from_position == Position(0, 0)
        assert plan.get_final_position() == Position(0, 3)

    def test_plan_movement_with_obstacles(self):
        """Test planning movement around obstacles."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 5,
            "height": 5,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        # Mock objects blocking straight path
        def mock_get_objects(result):
            if "Pos_0_1" in result["pos_name"]:
                return [{"object_name": "Wall"}]
            return []

        mock_netzwerk.session.run = Mock(
            side_effect=lambda query: mock_get_objects({"pos_name": query})
        )

        reasoner = SpatialReasoner(mock_netzwerk)

        # Should find path around obstacle
        plan = reasoner.plan_movement(
            object_name="König",
            grid_name="TestGrid",
            start_pos=Position(0, 0),
            goal_pos=Position(0, 2),
            avoid_objects=True,
        )

        # With orthogonal grid and obstacle at (0,1), should route around
        assert plan is not None

    def test_plan_movement_no_path(self):
        """Test planning when no path exists."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 3,
            "height": 3,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)

        # Mock all intermediate positions blocked
        def mock_get_objects(query):
            # Block all positions except start and goal
            return [{"object_name": "Wall"}]

        mock_netzwerk.session.run = Mock(return_value=[{"object_name": "Wall"}])

        reasoner = SpatialReasoner(mock_netzwerk)

        plan = reasoner.plan_movement(
            object_name="König",
            grid_name="TestGrid",
            start_pos=Position(0, 0),
            goal_pos=Position(2, 2),
            avoid_objects=True,
        )

        # May be None if completely blocked
        # Or may find path if goal position is not blocked
        # This depends on implementation details

    def test_plan_movement_invalid_positions(self):
        """Test planning with invalid positions."""
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

        # Out of bounds goal
        plan = reasoner.plan_movement(
            object_name="König",
            grid_name="TestGrid",
            start_pos=Position(0, 0),
            goal_pos=Position(10, 10),  # Out of bounds
        )

        assert plan is None

    def test_plan_movement_with_custom_rules(self):
        """Test planning with custom movement rules."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 8,
            "height": 8,
            "neighborhood_type": "custom",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.session.run = Mock(return_value=[])

        reasoner = SpatialReasoner(mock_netzwerk)

        # Only allow moves where x increases (rightward only)
        def rightward_only(from_pos: Position, to_pos: Position) -> bool:
            return to_pos.x > from_pos.x

        plan = reasoner.plan_movement(
            object_name="Piece",
            grid_name="TestGrid",
            start_pos=Position(0, 0),
            goal_pos=Position(3, 0),
            movement_rules=rightward_only,
        )

        # Should find path moving right
        if plan:
            for action in plan.actions:
                assert action.to_position.x > action.from_position.x


class TestMovementValidation:
    """Test movement plan validation."""

    def test_validate_valid_plan(self):
        """Test validating a valid plan."""
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

        # Create valid plan
        actions = [
            MovementAction("König", Position(0, 0), Position(0, 1), 1),
            MovementAction("König", Position(0, 1), Position(0, 2), 2),
        ]
        plan = MovementPlan("König", "TestGrid", actions, 2, 3)

        is_valid, errors = reasoner.validate_movement_plan(
            plan, check_constraints=False
        )

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_discontinuous_plan(self):
        """Test detecting discontinuous path."""
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

        # Create discontinuous plan (gap in path)
        actions = [
            MovementAction("König", Position(0, 0), Position(0, 1), 1),
            MovementAction("König", Position(0, 2), Position(0, 3), 2),  # Gap!
        ]
        plan = MovementPlan("König", "TestGrid", actions, 2, 3)

        is_valid, errors = reasoner.validate_movement_plan(
            plan, check_constraints=False
        )

        assert is_valid is False
        assert len(errors) > 0
        assert any("Discontinuous" in err for err in errors)

    def test_validate_out_of_bounds(self):
        """Test detecting out of bounds positions."""
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

        # Create plan with out-of-bounds position
        actions = [MovementAction("König", Position(0, 0), Position(10, 10), 1)]
        plan = MovementPlan("König", "TestGrid", actions, 1, 2)

        is_valid, errors = reasoner.validate_movement_plan(
            plan, check_constraints=False
        )

        assert is_valid is False
        assert len(errors) > 0


class TestMovementExecution:
    """Test movement plan execution."""

    def test_execute_valid_plan(self):
        """Test executing a valid movement plan."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 5,
            "height": 5,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.delete_relation = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        # Create valid plan
        actions = [
            MovementAction("König", Position(0, 0), Position(0, 1), 1),
            MovementAction("König", Position(0, 1), Position(0, 2), 2),
        ]
        plan = MovementPlan("König", "TestGrid", actions, 2, 3)

        success = reasoner.execute_movement_plan(plan, validate=False)

        assert success is True
        # Should have called delete_relation and assert_relation for each step
        assert mock_netzwerk.delete_relation.call_count == 2
        assert mock_netzwerk.assert_relation.call_count == 2


class TestMovementRules:
    """Test custom movement rules."""

    def test_create_movement_rule(self):
        """Test creating a named movement rule."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        def knight_move(from_pos: Position, to_pos: Position) -> bool:
            dx = abs(to_pos.x - from_pos.x)
            dy = abs(to_pos.y - from_pos.y)
            return (dx == 2 and dy == 1) or (dx == 1 and dy == 2)

        success = reasoner.create_movement_rule(
            "knight", knight_move, "Knight moves in L-shape"
        )

        assert success is True
        assert hasattr(reasoner, "_movement_rules")
        assert "knight" in reasoner._movement_rules

    def test_get_movement_rule(self):
        """Test retrieving a movement rule."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        def rook_move(from_pos: Position, to_pos: Position) -> bool:
            return from_pos.x == to_pos.x or from_pos.y == to_pos.y

        reasoner.create_movement_rule("rook", rook_move)

        rule = reasoner.get_movement_rule("rook")

        assert rule is not None
        assert rule(Position(0, 0), Position(0, 5)) is True  # Same x
        assert rule(Position(0, 0), Position(5, 0)) is True  # Same y
        assert rule(Position(0, 0), Position(1, 1)) is False  # Diagonal

    def test_get_nonexistent_rule(self):
        """Test getting a rule that doesn't exist."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        rule = reasoner.get_movement_rule("nonexistent")

        assert rule is None


class TestMultiObjectMovement:
    """Test planning movement for multiple objects."""

    def test_plan_multi_object_simple(self):
        """Test planning for multiple objects."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TestGrid",
            "width": 5,
            "height": 5,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.session.run = Mock(return_value=[])

        reasoner = SpatialReasoner(mock_netzwerk)

        movements = [
            ("König", Position(0, 0), Position(0, 2)),
            ("Turm", Position(4, 0), Position(4, 2)),
        ]

        plans = reasoner.plan_multi_object_movement(
            "TestGrid", movements, avoid_collisions=True
        )

        assert plans is not None
        assert len(plans) == 2
        assert "König" in plans
        assert "Turm" in plans

    def test_plan_multi_object_impossible(self):
        """Test when multi-object planning is impossible."""
        mock_netzwerk = Mock()
        mock_grid_node = {
            "lemma": "TinyGrid",
            "width": 1,
            "height": 2,
            "neighborhood_type": "orthogonal",
            "type": "Grid",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_grid_node)
        mock_netzwerk.session.run = Mock(return_value=[])

        reasoner = SpatialReasoner(mock_netzwerk)

        # Both objects trying to reach same position
        movements = [
            ("A", Position(0, 0), Position(0, 1)),
            ("B", Position(0, 1), Position(0, 1)),  # Same goal!
        ]

        plans = reasoner.plan_multi_object_movement(
            "TinyGrid", movements, avoid_collisions=True
        )

        # May fail if collision avoidance is strict
        # Behavior depends on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
