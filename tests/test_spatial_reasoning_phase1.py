"""
Tests for Component 42: Spatial Reasoning - Phase 1

Tests for:
- Phase 1.1: Basic component structure
- Phase 1.2: Spatial relation definitions
- Phase 1.3: Coordinate system and data structures

Author: KAI Development Team
Date: 2025-11-05
"""

from unittest.mock import Mock

import pytest

from component_42_spatial_reasoning import (
    NeighborhoodType,
    Position,
    SpatialReasoner,
    SpatialReasoningResult,
    SpatialRelation,
    SpatialRelationType,
)

# ============================================================================
# Phase 1.1: Basic Component Tests
# ============================================================================


class TestSpatialReasonerInit:
    """Test SpatialReasoner initialization."""

    def test_init_with_netzwerk(self):
        """Test that SpatialReasoner initializes with netzwerk."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        assert reasoner.netzwerk is mock_netzwerk
        assert len(reasoner.spatial_relation_types) > 0
        assert hasattr(reasoner, "_query_cache")

    def test_spatial_relation_types_loaded(self):
        """Test that all spatial relation types are loaded."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        expected_types = {
            "NORTH_OF",
            "SOUTH_OF",
            "EAST_OF",
            "WEST_OF",
            "ADJACENT_TO",
            "NEIGHBOR_ORTHOGONAL",
            "NEIGHBOR_DIAGONAL",
            "INSIDE",
            "CONTAINS",
            "ABOVE",
            "BELOW",
            "BETWEEN",
            "LOCATED_AT",
        }

        assert expected_types.issubset(reasoner.spatial_relation_types)


# ============================================================================
# Phase 1.2: Spatial Relation Tests
# ============================================================================


class TestSpatialRelationType:
    """Test SpatialRelationType enum and properties."""

    def test_symmetric_relations(self):
        """Test that symmetric relations are correctly identified."""
        symmetric_types = [
            SpatialRelationType.ADJACENT_TO,
            SpatialRelationType.NEIGHBOR_ORTHOGONAL,
            SpatialRelationType.NEIGHBOR_DIAGONAL,
        ]

        for rel_type in symmetric_types:
            assert rel_type.is_symmetric is True

    def test_non_symmetric_relations(self):
        """Test that non-symmetric relations are correctly identified."""
        non_symmetric_types = [
            SpatialRelationType.NORTH_OF,
            SpatialRelationType.SOUTH_OF,
            SpatialRelationType.INSIDE,
            SpatialRelationType.CONTAINS,
        ]

        for rel_type in non_symmetric_types:
            assert rel_type.is_symmetric is False

    def test_transitive_relations(self):
        """Test that transitive relations are correctly identified."""
        transitive_types = [
            SpatialRelationType.NORTH_OF,
            SpatialRelationType.SOUTH_OF,
            SpatialRelationType.EAST_OF,
            SpatialRelationType.WEST_OF,
            SpatialRelationType.INSIDE,
            SpatialRelationType.CONTAINS,
            SpatialRelationType.ABOVE,
            SpatialRelationType.BELOW,
        ]

        for rel_type in transitive_types:
            assert rel_type.is_transitive is True

    def test_non_transitive_relations(self):
        """Test that non-transitive relations are correctly identified."""
        non_transitive_types = [
            SpatialRelationType.ADJACENT_TO,
            SpatialRelationType.NEIGHBOR_ORTHOGONAL,
            SpatialRelationType.BETWEEN,
        ]

        for rel_type in non_transitive_types:
            assert rel_type.is_transitive is False

    def test_inverse_relations(self):
        """Test that inverse relations are correct."""
        inverse_pairs = [
            (SpatialRelationType.NORTH_OF, SpatialRelationType.SOUTH_OF),
            (SpatialRelationType.EAST_OF, SpatialRelationType.WEST_OF),
            (SpatialRelationType.INSIDE, SpatialRelationType.CONTAINS),
            (SpatialRelationType.ABOVE, SpatialRelationType.BELOW),
        ]

        for rel, expected_inverse in inverse_pairs:
            assert rel.inverse == expected_inverse
            assert expected_inverse.inverse == rel  # Bidirectional

    def test_no_inverse_for_symmetric(self):
        """Test that symmetric relations don't have inverses."""
        assert SpatialRelationType.ADJACENT_TO.inverse is None


class TestSpatialRelation:
    """Test SpatialRelation data structure."""

    def test_create_spatial_relation(self):
        """Test creating a spatial relation."""
        rel = SpatialRelation(
            subject="Turm",
            object="König",
            relation_type=SpatialRelationType.ADJACENT_TO,
            confidence=0.95,
        )

        assert rel.subject == "Turm"
        assert rel.object == "König"
        assert rel.relation_type == SpatialRelationType.ADJACENT_TO
        assert rel.confidence == 0.95
        assert isinstance(rel.metadata, dict)

    def test_spatial_relation_string(self):
        """Test string representation of spatial relation."""
        rel = SpatialRelation(
            subject="A", object="B", relation_type=SpatialRelationType.NORTH_OF
        )

        str_repr = str(rel)
        assert "A" in str_repr
        assert "B" in str_repr
        assert "NORTH_OF" in str_repr

    def test_create_inverse_relation(self):
        """Test creating inverse of a relation."""
        rel = SpatialRelation(
            subject="A",
            object="B",
            relation_type=SpatialRelationType.NORTH_OF,
            confidence=0.9,
            metadata={"test": "data"},
        )

        inverse = rel.to_inverse()

        assert inverse is not None
        assert inverse.subject == "B"
        assert inverse.object == "A"
        assert inverse.relation_type == SpatialRelationType.SOUTH_OF
        assert inverse.confidence == 0.9
        assert inverse.metadata["test"] == "data"

    def test_no_inverse_for_symmetric_relation(self):
        """Test that symmetric relations return None for inverse."""
        rel = SpatialRelation(
            subject="A", object="B", relation_type=SpatialRelationType.ADJACENT_TO
        )

        assert rel.to_inverse() is None


# ============================================================================
# Phase 1.3: Coordinate System & Data Structure Tests
# ============================================================================


class TestPosition:
    """Test Position data structure and methods."""

    def test_create_position(self):
        """Test creating a position."""
        pos = Position(3, 5)

        assert pos.x == 3
        assert pos.y == 5

    def test_position_immutable(self):
        """Test that Position is immutable (frozen)."""
        pos = Position(1, 2)

        with pytest.raises(AttributeError):
            pos.x = 5  # Should raise error due to frozen dataclass

    def test_position_hashable(self):
        """Test that Position is hashable for use in sets/dicts."""
        pos1 = Position(1, 2)
        pos2 = Position(1, 2)
        pos3 = Position(2, 1)

        position_set = {pos1, pos2, pos3}

        assert len(position_set) == 2  # pos1 and pos2 are equal
        assert pos1 in position_set
        assert pos3 in position_set

    def test_position_string(self):
        """Test string representation."""
        pos = Position(4, 7)

        assert str(pos) == "(4, 7)"

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        pos1 = Position(0, 0)
        pos2 = Position(3, 4)

        distance = pos1.distance_to(pos2, metric="euclidean")

        assert distance == 5.0  # 3-4-5 triangle

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        pos1 = Position(1, 1)
        pos2 = Position(4, 5)

        distance = pos1.distance_to(pos2, metric="manhattan")

        assert distance == 7  # |4-1| + |5-1| = 3 + 4 = 7

    def test_chebyshev_distance(self):
        """Test Chebyshev distance calculation."""
        pos1 = Position(1, 1)
        pos2 = Position(4, 3)

        distance = pos1.distance_to(pos2, metric="chebyshev")

        assert distance == 3  # max(|4-1|, |3-1|) = max(3, 2) = 3

    def test_direction_to_north(self):
        """Test direction detection for north."""
        pos1 = Position(5, 5)
        pos2 = Position(5, 8)  # Same x, greater y

        direction = pos1.direction_to(pos2)

        assert direction == SpatialRelationType.NORTH_OF

    def test_direction_to_south(self):
        """Test direction detection for south."""
        pos1 = Position(5, 5)
        pos2 = Position(5, 2)  # Same x, smaller y

        direction = pos1.direction_to(pos2)

        assert direction == SpatialRelationType.SOUTH_OF

    def test_direction_to_east(self):
        """Test direction detection for east."""
        pos1 = Position(5, 5)
        pos2 = Position(8, 5)  # Greater x, same y

        direction = pos1.direction_to(pos2)

        assert direction == SpatialRelationType.EAST_OF

    def test_direction_to_west(self):
        """Test direction detection for west."""
        pos1 = Position(5, 5)
        pos2 = Position(2, 5)  # Smaller x, same y

        direction = pos1.direction_to(pos2)

        assert direction == SpatialRelationType.WEST_OF

    def test_direction_to_diagonal(self):
        """Test that diagonal positions return None."""
        pos1 = Position(5, 5)
        pos2 = Position(7, 7)  # Diagonal

        direction = pos1.direction_to(pos2)

        assert direction is None

    def test_direction_to_same_position(self):
        """Test that same position returns None."""
        pos = Position(5, 5)

        direction = pos.direction_to(pos)

        assert direction is None

    def test_orthogonal_neighbors(self):
        """Test getting orthogonal (4-directional) neighbors."""
        pos = Position(5, 5)
        neighbors = pos.get_neighbors(NeighborhoodType.ORTHOGONAL)

        expected_neighbors = {
            Position(5, 6),  # North
            Position(5, 4),  # South
            Position(6, 5),  # East
            Position(4, 5),  # West
        }

        assert len(neighbors) == 4
        assert set(neighbors) == expected_neighbors

    def test_diagonal_neighbors(self):
        """Test getting diagonal neighbors."""
        pos = Position(5, 5)
        neighbors = pos.get_neighbors(NeighborhoodType.DIAGONAL)

        expected_neighbors = {
            Position(6, 6),  # NE
            Position(6, 4),  # SE
            Position(4, 6),  # NW
            Position(4, 4),  # SW
        }

        assert len(neighbors) == 4
        assert set(neighbors) == expected_neighbors

    def test_moore_neighbors(self):
        """Test getting Moore (8-directional) neighbors."""
        pos = Position(5, 5)
        neighbors = pos.get_neighbors(NeighborhoodType.MOORE)

        expected_neighbors = {
            # Orthogonal
            Position(5, 6),
            Position(5, 4),
            Position(6, 5),
            Position(4, 5),
            # Diagonal
            Position(6, 6),
            Position(6, 4),
            Position(4, 6),
            Position(4, 4),
        }

        assert len(neighbors) == 8
        assert set(neighbors) == expected_neighbors

    def test_custom_neighbors_knight_move(self):
        """Test custom neighborhood (e.g., knight in chess)."""
        pos = Position(5, 5)

        # Knight move offsets (L-shaped)
        knight_offsets = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ]

        neighbors = pos.get_neighbors(
            NeighborhoodType.CUSTOM, custom_offsets=knight_offsets
        )

        expected_neighbors = {
            Position(3, 4),
            Position(3, 6),
            Position(4, 3),
            Position(4, 7),
            Position(6, 3),
            Position(6, 7),
            Position(7, 4),
            Position(7, 6),
        }

        assert len(neighbors) == 8
        assert set(neighbors) == expected_neighbors


class TestSpatialReasoningResult:
    """Test SpatialReasoningResult data structure."""

    def test_create_result(self):
        """Test creating a reasoning result."""
        result = SpatialReasoningResult(query="test_query")

        assert result.query == "test_query"
        assert isinstance(result.relations, list)
        assert result.confidence == 0.0
        assert isinstance(result.reasoning_steps, list)
        assert result.error is None

    def test_result_success_with_relations(self):
        """Test that result is successful with relations."""
        result = SpatialReasoningResult(query="test")
        result.relations.append(SpatialRelation("A", "B", SpatialRelationType.NORTH_OF))

        assert result.success is True

    def test_result_failure_with_error(self):
        """Test that result is not successful with error."""
        result = SpatialReasoningResult(query="test", error="Some error")

        assert result.success is False

    def test_result_failure_without_relations(self):
        """Test that result is not successful without relations."""
        result = SpatialReasoningResult(query="test")

        assert result.success is False


# ============================================================================
# Phase 1: Integration Tests
# ============================================================================


class TestSpatialReasonerBasics:
    """Test basic spatial reasoning functionality."""

    def test_query_direct_relations_no_facts(self):
        """Test querying when no facts exist."""
        mock_netzwerk = Mock()
        mock_netzwerk.query_graph_for_facts.return_value = {}

        reasoner = SpatialReasoner(mock_netzwerk)
        relations = reasoner._query_direct_relations("Entity", None)

        assert len(relations) == 0

    def test_query_direct_relations_with_facts(self):
        """Test querying when facts exist in graph."""
        mock_netzwerk = Mock()
        mock_netzwerk.query_graph_for_facts.return_value = {
            "NORTH_OF": ["EntityB", "EntityC"]
        }

        reasoner = SpatialReasoner(mock_netzwerk)
        relations = reasoner._query_direct_relations(
            "EntityA", SpatialRelationType.NORTH_OF
        )

        assert len(relations) == 2
        assert relations[0].subject == "EntityA"
        assert relations[0].object in ["EntityB", "EntityC"]
        assert relations[0].relation_type == SpatialRelationType.NORTH_OF

    def test_consistency_check_valid_relations(self):
        """Test consistency check with valid relations."""
        reasoner = SpatialReasoner(Mock())

        relations = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF),
            SpatialRelation("B", "C", SpatialRelationType.NORTH_OF),
        ]

        is_consistent, violations = reasoner.check_spatial_consistency(relations)

        assert is_consistent is True
        assert len(violations) == 0

    def test_consistency_check_contradiction(self):
        """Test consistency check detects contradictions."""
        reasoner = SpatialReasoner(Mock())

        relations = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF),
            SpatialRelation("B", "A", SpatialRelationType.NORTH_OF),  # Contradiction!
        ]

        is_consistent, violations = reasoner.check_spatial_consistency(relations)

        assert is_consistent is False
        assert len(violations) > 0

    def test_cache_functionality(self):
        """Test that caching works for repeated queries."""
        mock_netzwerk = Mock()
        mock_netzwerk.query_graph_for_facts.return_value = {"ADJACENT_TO": ["B"]}

        reasoner = SpatialReasoner(mock_netzwerk)

        # First query - should hit netzwerk
        result1 = reasoner.infer_spatial_relations("A")
        assert mock_netzwerk.query_graph_for_facts.call_count > 0

        # Second query - should hit cache
        call_count_before = mock_netzwerk.query_graph_for_facts.call_count
        result2 = reasoner.infer_spatial_relations("A")

        assert mock_netzwerk.query_graph_for_facts.call_count == call_count_before
        assert result1.query == result2.query

    def test_clear_cache(self):
        """Test cache clearing."""
        reasoner = SpatialReasoner(Mock())

        # Add something to cache
        reasoner._query_cache["test"] = "value"
        assert len(reasoner._query_cache) > 0

        # Clear cache
        reasoner.clear_cache()
        assert len(reasoner._query_cache) == 0

    def test_register_extraction_rules(self):
        """Test registering spatial extraction rules in netzwerk."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_extraction_rule = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)
        rules_count = reasoner.register_spatial_extraction_rules()

        # Should have registered multiple rules
        assert rules_count > 0
        assert mock_netzwerk.create_extraction_rule.called

        # Check that various relation types were registered
        call_args = [
            call[1] for call in mock_netzwerk.create_extraction_rule.call_args_list
        ]
        relation_types = {args["relation_type"] for args in call_args}

        expected_types = {
            "NORTH_OF",
            "SOUTH_OF",
            "EAST_OF",
            "WEST_OF",
            "ADJACENT_TO",
            "INSIDE",
            "CONTAINS",
            "ABOVE",
            "BELOW",
        }

        assert expected_types.issubset(relation_types)

    def test_extraction_rules_german_patterns(self):
        """Test that extraction rules use correct German patterns."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_extraction_rule = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)
        reasoner.register_spatial_extraction_rules()

        # Check that patterns contain German keywords
        call_args = [
            call[1] for call in mock_netzwerk.create_extraction_rule.call_args_list
        ]
        patterns = [args["regex_pattern"] for args in call_args]

        # Should contain German spatial keywords
        all_patterns = " ".join(patterns)
        german_keywords = [
            "nördlich",
            "südlich",
            "östlich",
            "westlich",
            "neben",
            "in",
            "enthält",
            "über",
            "unter",
        ]

        for keyword in german_keywords:
            assert (
                keyword in all_patterns
            ), f"German keyword '{keyword}' not found in patterns"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
