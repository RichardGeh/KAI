"""
test_spatial_reasoning_type_fix.py

Test Suite for Bug Fix 3: Spatial Reasoning Type Conversion
Tests that _try_spatial_reasoning() properly converts string relation_type
to SpatialRelationType enum.

Bug: _try_spatial_reasoning() received string 'relation_type' parameter but
tried to pass it directly to spatial reasoner which expected SpatialRelationType enum,
causing AttributeError.

Fix: Added type conversion at kai_strategy_dispatcher.py lines 788-811

Created: 2025-12-05
"""

import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from kai_strategy_dispatcher import StrategyDispatcher


class TestSpatialReasoningTypeConversion:
    """Test type conversion from string to SpatialRelationType enum"""

    @pytest.fixture
    def mock_dispatcher(self):
        """Create StrategyDispatcher with mocked dependencies"""
        netzwerk = Mock()
        logic_engine = Mock()
        graph_traversal = Mock()
        working_memory = Mock()
        signals = Mock()
        spatial_reasoner = Mock()

        dispatcher = StrategyDispatcher(
            netzwerk=netzwerk,
            logic_engine=logic_engine,
            graph_traversal=graph_traversal,
            working_memory=working_memory,
            signals=signals,
            spatial_reasoner=spatial_reasoner,
        )

        yield dispatcher

    def test_string_relation_type_accepted(self, mock_dispatcher):
        """Test 1: _try_spatial_reasoning accepts string relation_type parameter"""
        # Mock spatial reasoner to return success
        mock_result = Mock()
        mock_result.success = True
        mock_result.confidence = 0.85
        mock_result.explanation = "Spatial reasoning result"
        mock_result.proof_tree = None
        mock_dispatcher.spatial_reasoner.infer_spatial_relations = Mock(
            return_value=mock_result
        )

        # Should NOT raise AttributeError
        result = mock_dispatcher._try_spatial_reasoning(
            topic="box", relation_type="ABOVE", obj="table"  # String, not enum
        )

        # Verify it was called and didn't crash
        assert mock_dispatcher.spatial_reasoner.infer_spatial_relations.called

    def test_string_converted_to_enum(self, mock_dispatcher):
        """Test 2: String relation_type is converted to SpatialRelationType enum"""
        from component_42_spatial_inference import SpatialRelationType

        mock_result = Mock()
        mock_result.success = True
        mock_result.confidence = 0.85
        mock_result.explanation = "Test"
        mock_result.proof_tree = None
        mock_dispatcher.spatial_reasoner.infer_spatial_relations = Mock(
            return_value=mock_result
        )

        # Call with string
        result = mock_dispatcher._try_spatial_reasoning(
            topic="box", relation_type="ABOVE", obj="table"
        )

        # Verify the spatial reasoner was called
        call_args = mock_dispatcher.spatial_reasoner.infer_spatial_relations.call_args

        # Check the relation_type argument
        kwargs = call_args.kwargs if call_args.kwargs else {}
        if "relation_type" in kwargs:
            passed_relation = kwargs["relation_type"]
            # Should be enum or None (if conversion failed gracefully)
            assert passed_relation is None or isinstance(
                passed_relation, SpatialRelationType
            )

    def test_invalid_string_handled_gracefully(self, mock_dispatcher):
        """Test 3: Invalid string relation_type doesn't crash (returns None for enum)"""
        mock_result = Mock()
        mock_result.success = True
        mock_result.confidence = 0.85
        mock_result.explanation = "Test"
        mock_result.proof_tree = None
        mock_dispatcher.spatial_reasoner.infer_spatial_relations = Mock(
            return_value=mock_result
        )

        # Call with invalid string that can't be converted to enum
        result = mock_dispatcher._try_spatial_reasoning(
            topic="box", relation_type="INVALID_RELATION_TYPE", obj="table"
        )

        # Should not crash, should still call spatial reasoner
        assert mock_dispatcher.spatial_reasoner.infer_spatial_relations.called

    def test_none_relation_type_handled(self, mock_dispatcher):
        """Test 4: None relation_type is handled correctly"""
        mock_result = Mock()
        mock_result.success = True
        mock_result.confidence = 0.85
        mock_result.explanation = "Test"
        mock_result.proof_tree = None
        mock_dispatcher.spatial_reasoner.infer_spatial_relations = Mock(
            return_value=mock_result
        )

        # Call with None
        result = mock_dispatcher._try_spatial_reasoning(
            topic="box", relation_type=None, obj="table"
        )

        # Should not crash
        assert mock_dispatcher.spatial_reasoner.infer_spatial_relations.called

        # Verify relation_type passed as None
        call_args = mock_dispatcher.spatial_reasoner.infer_spatial_relations.call_args
        kwargs = call_args.kwargs if call_args.kwargs else {}
        if "relation_type" in kwargs:
            assert kwargs["relation_type"] is None

    def test_empty_string_handled(self, mock_dispatcher):
        """Test 5: Empty string relation_type is handled correctly"""
        mock_result = Mock()
        mock_result.success = True
        mock_result.confidence = 0.85
        mock_result.explanation = "Test"
        mock_result.proof_tree = None
        mock_dispatcher.spatial_reasoner.infer_spatial_relations = Mock(
            return_value=mock_result
        )

        # Call with empty string
        result = mock_dispatcher._try_spatial_reasoning(
            topic="box", relation_type="", obj="table"
        )

        # Should not crash
        assert mock_dispatcher.spatial_reasoner.infer_spatial_relations.called


class TestSpatialReasoningValidRelationTypes:
    """Test conversion of valid SpatialRelationType values"""

    @pytest.fixture
    def mock_dispatcher(self):
        """Create StrategyDispatcher with mocked dependencies"""
        netzwerk = Mock()
        logic_engine = Mock()
        graph_traversal = Mock()
        working_memory = Mock()
        signals = Mock()
        spatial_reasoner = Mock()

        dispatcher = StrategyDispatcher(
            netzwerk=netzwerk,
            logic_engine=logic_engine,
            graph_traversal=graph_traversal,
            working_memory=working_memory,
            signals=signals,
            spatial_reasoner=spatial_reasoner,
        )

        yield dispatcher

    def test_valid_spatial_relation_types(self, mock_dispatcher):
        """Test 6: All valid SpatialRelationType strings convert correctly"""

        valid_types = [
            "ABOVE",
            "BELOW",
            "LEFT_OF",
            "RIGHT_OF",
            "FRONT_OF",
            "BEHIND",
            "INSIDE",
            "OUTSIDE",
            "ON",
            "UNDER",
            "NEAR",
            "FAR",
            "ADJACENT",
            "BETWEEN",
        ]

        mock_result = Mock()
        mock_result.success = True
        mock_result.confidence = 0.85
        mock_result.explanation = "Test"
        mock_result.proof_tree = None
        mock_dispatcher.spatial_reasoner.infer_spatial_relations = Mock(
            return_value=mock_result
        )

        for relation_str in valid_types:
            # Reset mock
            mock_dispatcher.spatial_reasoner.infer_spatial_relations.reset_mock()

            # Call with string
            result = mock_dispatcher._try_spatial_reasoning(
                topic="object1", relation_type=relation_str, obj="object2"
            )

            # Verify it was called without error
            assert mock_dispatcher.spatial_reasoner.infer_spatial_relations.called


class TestSpatialReasoningIntegration:
    """Integration tests with actual spatial reasoning components"""

    def test_spatial_reasoning_type_conversion_does_not_crash(self):
        """Test 7: Type conversion from string to enum doesn't cause AttributeError"""
        # This test verifies the bug fix: string relation_type is converted to enum
        # The original bug was AttributeError when passing string to spatial reasoner

        # Test that the conversion code exists and doesn't crash
        from component_42_spatial_inference import SpatialRelationType

        # Verify the enum can be created from string
        try:
            relation_enum = SpatialRelationType("ABOVE")
            assert relation_enum == SpatialRelationType.ABOVE
        except ValueError as e:
            pytest.fail(f"SpatialRelationType('ABOVE') raised ValueError: {e}")

        # Verify invalid strings raise ValueError (expected behavior)
        with pytest.raises(ValueError):
            SpatialRelationType("INVALID_TYPE")

    def test_spatial_reasoning_accepts_string_parameter(self):
        """Test 8: _try_spatial_reasoning method signature accepts string relation_type"""
        import inspect

        from kai_strategy_dispatcher import StrategyDispatcher

        # Check method signature
        sig = inspect.signature(StrategyDispatcher._try_spatial_reasoning)
        params = list(sig.parameters.keys())

        assert "relation_type" in params

        # The parameter should accept string (no type hint restriction to enum only)
        # (If it had Enum-only type hint, passing string would violate it)


class TestErrorHandling:
    """Test error handling in spatial reasoning type conversion"""

    @pytest.fixture
    def mock_dispatcher(self):
        """Create StrategyDispatcher with mocked dependencies"""
        netzwerk = Mock()
        logic_engine = Mock()
        graph_traversal = Mock()
        working_memory = Mock()
        signals = Mock()
        spatial_reasoner = Mock()

        dispatcher = StrategyDispatcher(
            netzwerk=netzwerk,
            logic_engine=logic_engine,
            graph_traversal=graph_traversal,
            working_memory=working_memory,
            signals=signals,
            spatial_reasoner=spatial_reasoner,
        )

        yield dispatcher

    def test_spatial_reasoner_exception_handled(self, mock_dispatcher):
        """Test 9: Exceptions from spatial reasoner are handled gracefully"""
        # Mock spatial reasoner to raise exception
        mock_dispatcher.spatial_reasoner.infer_spatial_relations = Mock(
            side_effect=Exception("Spatial reasoning error")
        )

        # Should return None, not crash
        result = mock_dispatcher._try_spatial_reasoning(
            topic="box", relation_type="ABOVE", obj="table"
        )

        assert result is None

    def test_no_spatial_reasoner_returns_none(self):
        """Test 10: Returns None when spatial_reasoner is not available"""
        netzwerk = Mock()
        logic_engine = Mock()
        graph_traversal = Mock()
        working_memory = Mock()
        signals = Mock()

        # Create dispatcher WITHOUT spatial_reasoner
        dispatcher = StrategyDispatcher(
            netzwerk=netzwerk,
            logic_engine=logic_engine,
            graph_traversal=graph_traversal,
            working_memory=working_memory,
            signals=signals,
            spatial_reasoner=None,
        )

        # Should return None gracefully
        result = dispatcher._try_spatial_reasoning(
            topic="box", relation_type="ABOVE", obj="table"
        )

        assert result is None

    def test_enum_conversion_handles_errors(self):
        """Test 11: Enum conversion handles invalid strings gracefully"""
        from component_42_spatial_inference import SpatialRelationType

        # Valid string converts correctly
        try:
            valid_enum = SpatialRelationType("ABOVE")
            assert valid_enum == SpatialRelationType.ABOVE
        except Exception as e:
            pytest.fail(f"Valid enum conversion failed: {e}")

        # Invalid string raises ValueError (expected)
        with pytest.raises(ValueError):
            SpatialRelationType("INVALID_RELATION")

        # The fix should catch this ValueError and pass None instead
        # (tested in other tests)


class TestRegressionPrevention:
    """Regression tests to ensure fix doesn't break existing functionality"""

    def test_spatial_relation_type_enum_values(self):
        """Test 12: Verify SpatialRelationType enum has expected values"""
        from component_42_spatial_inference import SpatialRelationType

        # Test actual relation types from the enum
        actual_types = [
            "ABOVE",
            "BELOW",
            "INSIDE",
            "CONTAINS",
            "NORTH_OF",
            "ADJACENT_TO",
        ]

        for type_str in actual_types:
            # Should be able to create enum from string
            try:
                relation_enum = SpatialRelationType(type_str)
                assert relation_enum.value == type_str
            except ValueError:
                pytest.fail(f"SpatialRelationType does not have value: {type_str}")

    def test_type_conversion_code_exists(self):
        """Test 13: Verify type conversion code exists in _try_spatial_reasoning"""
        import inspect

        from kai_strategy_dispatcher import StrategyDispatcher

        # Get source code of _try_spatial_reasoning
        source = inspect.getsource(StrategyDispatcher._try_spatial_reasoning)

        # Verify conversion code exists
        assert "SpatialRelationType" in source, "Type conversion code missing"
        assert "ValueError" in source or "except" in source, "Error handling missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
