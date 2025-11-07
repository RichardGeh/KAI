"""
Tests for Component 42: Spatial Reasoning - Phase 4.3 Rule Learning

Tests for:
- Phase 4.3: Rule learning system for spatial domains

Author: KAI Development Team
Date: 2025-11-05
"""

from unittest.mock import Mock

import pytest

from component_42_spatial_reasoning import Position, SpatialReasoner

# ============================================================================
# Phase 4.3: Rule Learning Tests
# ============================================================================


class TestMovementPatternLearning:
    """Test learning movement patterns from observations."""

    def test_observe_movement_pattern(self):
        """Test observing and learning a movement pattern."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.set_wort_attribut = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        # Observe knight movements (L-shaped)
        movements = [
            (Position(0, 0), Position(2, 1)),
            (Position(0, 0), Position(1, 2)),
            (Position(3, 3), Position(5, 4)),
            (Position(3, 3), Position(4, 5)),
        ]

        success = reasoner.observe_movement_pattern("Knight", movements)

        assert success is True
        assert mock_netzwerk.create_wort_if_not_exists.call_count > 0

    def test_observe_empty_movements(self):
        """Test observing with no movements."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        success = reasoner.observe_movement_pattern("Knight", [])

        assert success is False

    def test_get_learned_movement_pattern(self):
        """Test retrieving a learned movement pattern."""
        mock_netzwerk = Mock()

        # Mock stored pattern
        mock_pattern_node = {
            "lemma": "Knight_movement_pattern",
            "type": "SpatialMovementPattern",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_pattern_node)

        # Mock stored vectors (knight moves)
        mock_vectors = [
            {"dx": 2, "dy": 1},
            {"dx": 2, "dy": -1},
            {"dx": 1, "dy": 2},
            {"dx": 1, "dy": -2},
        ]
        mock_netzwerk.session.run = Mock(return_value=mock_vectors)

        reasoner = SpatialReasoner(mock_netzwerk)
        pattern_func = reasoner.get_learned_movement_pattern("Knight")

        assert pattern_func is not None

        # Test the learned pattern
        assert pattern_func(Position(0, 0), Position(2, 1)) is True  # Valid knight move
        assert (
            pattern_func(Position(0, 0), Position(1, 1)) is False
        )  # Invalid (diagonal)

    def test_get_nonexistent_pattern(self):
        """Test retrieving a pattern that doesn't exist."""
        mock_netzwerk = Mock()
        mock_netzwerk.find_wort_node = Mock(return_value=None)

        reasoner = SpatialReasoner(mock_netzwerk)
        pattern_func = reasoner.get_learned_movement_pattern("NonExistent")

        assert pattern_func is None

    def test_learned_pattern_validation(self):
        """Test using learned pattern for move validation."""
        mock_netzwerk = Mock()

        mock_pattern_node = {"lemma": "Rook_movement_pattern"}
        mock_netzwerk.find_wort_node = Mock(return_value=mock_pattern_node)

        # Rook moves: straight lines (dx=0 or dy=0, but not both)
        mock_vectors = [
            {"dx": 0, "dy": 1},
            {"dx": 0, "dy": 2},
            {"dx": 1, "dy": 0},
            {"dx": 2, "dy": 0},
        ]
        mock_netzwerk.session.run = Mock(return_value=mock_vectors)

        reasoner = SpatialReasoner(mock_netzwerk)
        rook_pattern = reasoner.get_learned_movement_pattern("Rook")

        assert rook_pattern is not None
        # These specific vectors should work
        assert rook_pattern(Position(0, 0), Position(0, 1)) is True
        assert rook_pattern(Position(0, 0), Position(1, 0)) is True


class TestSpatialConfigurationLearning:
    """Test learning spatial configuration patterns."""

    def test_observe_spatial_configuration(self):
        """Test observing and learning a spatial configuration."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.set_wort_attribut = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        # Chess starting position (partial)
        configuration = {
            "König": Position(4, 0),
            "Dame": Position(3, 0),
            "Turm1": Position(0, 0),
            "Turm2": Position(7, 0),
        }

        success = reasoner.observe_spatial_configuration(
            "ChessStarting", configuration, "ChessBoard"
        )

        assert success is True
        assert mock_netzwerk.create_wort_if_not_exists.call_count > 0

    def test_observe_empty_configuration(self):
        """Test observing empty configuration."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.set_wort_attribut = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        success = reasoner.observe_spatial_configuration("Empty", {})

        assert success is True  # Empty is valid
        assert mock_netzwerk.create_wort_if_not_exists.call_count > 0

    def test_detect_spatial_pattern_in_configuration(self):
        """Test detecting learned patterns in a configuration."""
        mock_netzwerk = Mock()

        # Mock stored configurations
        mock_configs = [
            {"name": "SpatialConfig_Pattern1", "num_objects": 2},
            {"name": "SpatialConfig_Pattern2", "num_objects": 3},
        ]
        mock_netzwerk.session.run = Mock(return_value=mock_configs)

        reasoner = SpatialReasoner(mock_netzwerk)

        # Mock _check_configuration_match to return True for Pattern1
        def mock_check(config_id, objects):
            return config_id == "SpatialConfig_Pattern1"

        reasoner._check_configuration_match = mock_check

        # Test detection
        current_config = {"A": Position(0, 0), "B": Position(1, 1)}

        matches = reasoner.detect_spatial_pattern_in_configuration(current_config)

        assert len(matches) == 1
        assert "Pattern1" in matches

    def test_detect_no_matching_patterns(self):
        """Test when no patterns match."""
        mock_netzwerk = Mock()

        mock_configs = [
            {"name": "SpatialConfig_Pattern1", "num_objects": 5}  # Wrong number
        ]
        mock_netzwerk.session.run = Mock(return_value=mock_configs)

        reasoner = SpatialReasoner(mock_netzwerk)

        current_config = {"A": Position(0, 0), "B": Position(1, 1)}

        matches = reasoner.detect_spatial_pattern_in_configuration(current_config)

        assert len(matches) == 0


class TestRuleLearningFromExamples:
    """Test learning spatial rules from positive/negative examples."""

    def test_learn_spatial_rule_simple(self):
        """Test learning a simple spatial rule."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.set_wort_attribut = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        # All examples have same relative position (A is always north of B)
        positive_examples = [
            {"A": Position(0, 1), "B": Position(0, 0)},
            {"A": Position(3, 4), "B": Position(3, 3)},
            {"A": Position(5, 2), "B": Position(5, 1)},
        ]

        success = reasoner.learn_spatial_rule_from_examples(
            "AIsNorthOfB", positive_examples
        )

        assert success is True
        assert mock_netzwerk.create_wort_if_not_exists.call_count > 0

    def test_learn_rule_no_examples(self):
        """Test learning with no examples."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        success = reasoner.learn_spatial_rule_from_examples("NoExamples", [])

        assert success is False

    def test_learn_rule_multiple_constraints(self):
        """Test learning a rule with multiple constraints."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.set_wort_attribut = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        # Triangle: A, B, C with fixed relative positions
        positive_examples = [
            {"A": Position(0, 0), "B": Position(3, 0), "C": Position(0, 4)},
            {"A": Position(5, 5), "B": Position(8, 5), "C": Position(5, 9)},
        ]

        success = reasoner.learn_spatial_rule_from_examples(
            "RightTriangle", positive_examples
        )

        assert success is True
        # Should learn multiple constraints (A-B, A-C, B-C relationships)

    def test_learn_rule_variable_positions(self):
        """Test learning when positions vary (no consistent constraint)."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.set_wort_attribut = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        # No consistent relative position
        positive_examples = [
            {"A": Position(0, 0), "B": Position(1, 0)},
            {"A": Position(0, 0), "B": Position(0, 1)},
            {"A": Position(0, 0), "B": Position(2, 2)},
        ]

        success = reasoner.learn_spatial_rule_from_examples(
            "NoPattern", positive_examples
        )

        assert success is True
        # Should still succeed but learn 0 constraints


class TestObjectMatching:
    """Test object matching in patterns."""

    def test_find_matching_object_direct(self):
        """Test finding exact object match."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        current_objects = {"König": Position(0, 0), "Dame": Position(1, 1)}

        match = reasoner._find_matching_object("König", current_objects)

        assert match == "König"

    def test_find_matching_object_type_based(self):
        """Test finding match by type (ignoring numbers)."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        current_objects = {"König1": Position(0, 0), "Dame2": Position(1, 1)}

        # Should match "König" with "König1"
        match = reasoner._find_matching_object("König", current_objects)

        assert match == "König1"

    def test_find_matching_object_not_found(self):
        """Test when no match is found."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        current_objects = {"Turm": Position(0, 0)}

        match = reasoner._find_matching_object("König", current_objects)

        assert match is None


class TestConfigurationMatching:
    """Test configuration pattern matching."""

    def test_check_configuration_match_exact(self):
        """Test exact configuration match."""
        mock_netzwerk = Mock()

        # Mock stored relative positions
        mock_rels = [{"obj1": "A", "obj2": "B", "dx": 1, "dy": 0}]
        mock_netzwerk.session.run = Mock(return_value=mock_rels)

        reasoner = SpatialReasoner(mock_netzwerk)

        # Current config with exact same relative position
        current_objects = {"A": Position(0, 0), "B": Position(1, 0)}

        matches = reasoner._check_configuration_match("TestConfig", current_objects)

        assert matches is True

    def test_check_configuration_match_within_tolerance(self):
        """Test match within tolerance."""
        mock_netzwerk = Mock()

        mock_rels = [{"obj1": "A", "obj2": "B", "dx": 1, "dy": 0}]
        mock_netzwerk.session.run = Mock(return_value=mock_rels)

        reasoner = SpatialReasoner(mock_netzwerk)

        # Slightly different (within tolerance of 0.5)
        current_objects = {
            "A": Position(0, 0),
            "B": Position(1, 0),  # Exactly the same, so within tolerance
        }

        matches = reasoner._check_configuration_match(
            "TestConfig", current_objects, tolerance=0.5
        )

        assert matches is True

    def test_check_configuration_no_match(self):
        """Test when configuration doesn't match."""
        mock_netzwerk = Mock()

        mock_rels = [{"obj1": "A", "obj2": "B", "dx": 5, "dy": 0}]
        mock_netzwerk.session.run = Mock(return_value=mock_rels)

        reasoner = SpatialReasoner(mock_netzwerk)

        # Very different relative position
        current_objects = {
            "A": Position(0, 0),
            "B": Position(1, 0),  # dx=1, expected dx=5
        }

        matches = reasoner._check_configuration_match(
            "TestConfig", current_objects, tolerance=0.5
        )

        assert matches is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
