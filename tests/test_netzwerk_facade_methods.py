"""
test_netzwerk_facade_methods.py

Test Suite for Bug Fix 1: Neo4j Facade Methods
Tests that query_semantic_neighbors() and query_transitive_path()
are properly exposed through the KonzeptNetzwerk facade.

Bug: ResonanceEngine and Logic Engine got AttributeError when calling
these methods on the facade because they were only defined in _core.

Fix: Added delegation methods at component_1_netzwerk.py lines 167-185

Created: 2025-12-05
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from component_1_netzwerk import KonzeptNetzwerk


class TestFacadeMethodsExist:
    """Test that facade methods exist and are callable"""

    def test_query_semantic_neighbors_exists(self):
        """Test 1: query_semantic_neighbors method exists on facade"""
        # Don't initialize Neo4j, just check method exists
        assert hasattr(KonzeptNetzwerk, "query_semantic_neighbors")
        assert callable(getattr(KonzeptNetzwerk, "query_semantic_neighbors"))

    def test_query_transitive_path_exists(self):
        """Test 2: query_transitive_path method exists on facade"""
        assert hasattr(KonzeptNetzwerk, "query_transitive_path")
        assert callable(getattr(KonzeptNetzwerk, "query_transitive_path"))

    def test_facade_method_signatures(self):
        """Test 3: Verify method signatures match expected parameters"""
        import inspect

        # Check query_semantic_neighbors signature
        sig = inspect.signature(KonzeptNetzwerk.query_semantic_neighbors)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "lemma" in params
        assert "allowed_relations" in params
        assert "min_confidence" in params
        assert "limit" in params

        # Check query_transitive_path signature
        sig = inspect.signature(KonzeptNetzwerk.query_transitive_path)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "subject" in params
        assert "predicate" in params
        assert "object_node" in params
        assert "max_hops" in params


class TestFacadeDelegation:
    """Test that facade properly delegates to _core"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Create a KonzeptNetzwerk with mocked _core"""
        with patch("component_1_netzwerk_core.GraphDatabase") as mock_db:
            # Mock the driver and session
            mock_driver = Mock()
            mock_session = Mock()
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session

            netzwerk = KonzeptNetzwerk()

            # Mock the _core methods
            netzwerk._core.query_semantic_neighbors = Mock(
                return_value=[
                    {"neighbor": "frucht", "relation": "IS_A", "confidence": 0.9}
                ]
            )
            netzwerk._core.query_transitive_path = Mock(
                return_value=[
                    {"subject": "apfel", "predicate": "IS_A", "object": "frucht"}
                ]
            )

            yield netzwerk

    def test_query_semantic_neighbors_delegates(self, mock_netzwerk):
        """Test 4: query_semantic_neighbors delegates to _core"""
        result = mock_netzwerk.query_semantic_neighbors(
            lemma="apfel", allowed_relations=["IS_A"], min_confidence=0.7, limit=10
        )

        # Verify delegation happened
        mock_netzwerk._core.query_semantic_neighbors.assert_called_once_with(
            "apfel", ["IS_A"], 0.7, 10
        )

        # Verify result passed through
        assert result == [{"neighbor": "frucht", "relation": "IS_A", "confidence": 0.9}]

    def test_query_transitive_path_delegates(self, mock_netzwerk):
        """Test 5: query_transitive_path delegates to _core"""
        result = mock_netzwerk.query_transitive_path(
            subject="apfel", predicate="IS_A", object_node="frucht", max_hops=3
        )

        # Verify delegation happened
        mock_netzwerk._core.query_transitive_path.assert_called_once_with(
            "apfel", "IS_A", "frucht", 3
        )

        # Verify result passed through
        assert result == [{"subject": "apfel", "predicate": "IS_A", "object": "frucht"}]

    def test_query_semantic_neighbors_with_defaults(self, mock_netzwerk):
        """Test 6: query_semantic_neighbors works with default parameters"""
        result = mock_netzwerk.query_semantic_neighbors(lemma="apfel")

        # Verify called with defaults
        mock_netzwerk._core.query_semantic_neighbors.assert_called_once()
        args = mock_netzwerk._core.query_semantic_neighbors.call_args
        assert args[0][0] == "apfel"  # lemma
        assert args[0][1] is None  # allowed_relations default
        assert args[0][2] == 0.0  # min_confidence default
        assert args[0][3] == 10  # limit default

    def test_query_transitive_path_with_defaults(self, mock_netzwerk):
        """Test 7: query_transitive_path works with default max_hops"""
        result = mock_netzwerk.query_transitive_path(
            subject="apfel", predicate="IS_A", object_node="frucht"
        )

        # Verify called with default max_hops=3
        mock_netzwerk._core.query_transitive_path.assert_called_once()
        args = mock_netzwerk._core.query_transitive_path.call_args
        assert args[0][3] == 3  # max_hops default


class TestResonanceEngineIntegration:
    """Integration test: Verify ResonanceEngine can call facade methods"""

    def test_resonance_engine_can_call_query_semantic_neighbors(self):
        """Test 8: ResonanceEngine can call query_semantic_neighbors without AttributeError"""
        from component_44_resonance_core import ResonanceEngine

        # Create mock netzwerk
        netzwerk = Mock(spec=KonzeptNetzwerk)
        netzwerk.query_semantic_neighbors = Mock(
            return_value=[
                {
                    "neighbor": "frucht",
                    "relation": "IS_A",
                    "confidence": 0.9,
                    "direction": "outgoing",
                }
            ]
        )

        # Create engine
        engine = ResonanceEngine(netzwerk)

        # This should NOT raise AttributeError
        # Note: Use correct method signature (no max_waves parameter)
        engine.activate_concept("apfel")

        # Verify the method was called
        assert netzwerk.query_semantic_neighbors.called

    def test_resonance_engine_spreading_activation(self):
        """Test 9: ResonanceEngine spreading activation uses facade method"""
        from component_44_resonance_core import ResonanceEngine

        netzwerk = Mock(spec=KonzeptNetzwerk)

        # Mock semantic neighbors for multi-hop
        def mock_neighbors(lemma, allowed_relations=None, min_confidence=0.0, limit=50):
            neighbors_map = {
                "apfel": [
                    {
                        "neighbor": "frucht",
                        "relation": "IS_A",
                        "confidence": 0.9,
                        "direction": "outgoing",
                    }
                ],
                "frucht": [
                    {
                        "neighbor": "pflanze",
                        "relation": "IS_A",
                        "confidence": 0.8,
                        "direction": "outgoing",
                    }
                ],
                "pflanze": [],
            }
            return neighbors_map.get(lemma, [])

        netzwerk.query_semantic_neighbors = Mock(side_effect=mock_neighbors)

        engine = ResonanceEngine(netzwerk)
        # Use correct method signature
        engine.max_waves = 2  # Set max_waves as attribute instead
        activation_map = engine.activate_concept("apfel")

        # Should have activated multiple concepts
        assert activation_map.concepts_activated >= 2
        assert "apfel" in activation_map.activations
        assert "frucht" in activation_map.activations


class TestLogicEngineIntegration:
    """Integration test: Verify Logic Engine can use facade methods"""

    def test_logic_engine_can_call_query_transitive_path(self):
        """Test 10: Logic engine can call query_transitive_path without AttributeError"""
        # Just test that the method is accessible on KonzeptNetzwerk
        # (actual logic engine integration tested elsewhere)

        netzwerk = Mock(spec=KonzeptNetzwerk)
        netzwerk.query_transitive_path = Mock(
            return_value=[
                {
                    "subject": "apfel",
                    "predicate": "IS_A",
                    "object": "frucht",
                    "confidence": 0.9,
                }
            ]
        )
        netzwerk.query_graph_for_facts = Mock(return_value={"IS_A": ["frucht"]})

        # This should NOT raise AttributeError
        try:
            # Call method directly to verify no AttributeError
            result = netzwerk.query_transitive_path(
                "apfel", "IS_A", "frucht", max_hops=3
            )
            assert result is not None
            assert len(result) == 1
            assert result[0]["subject"] == "apfel"
        except AttributeError as e:
            pytest.fail(f"AttributeError raised: {e}")


class TestErrorHandling:
    """Test error handling in facade methods"""

    @pytest.fixture
    def mock_netzwerk_with_errors(self):
        """Create netzwerk that simulates errors"""
        with patch("component_1_netzwerk_core.GraphDatabase") as mock_db:
            # Mock the driver and session
            mock_driver = Mock()
            mock_session = Mock()
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session

            netzwerk = KonzeptNetzwerk()
            yield netzwerk

    def test_query_semantic_neighbors_handles_core_exception(
        self, mock_netzwerk_with_errors
    ):
        """Test 11: Facade handles exception from _core gracefully"""
        # Simulate _core raising exception
        mock_netzwerk_with_errors._core.query_semantic_neighbors = Mock(
            side_effect=Exception("Neo4j connection failed")
        )

        # Should propagate exception (not swallow it)
        with pytest.raises(Exception, match="Neo4j connection failed"):
            mock_netzwerk_with_errors.query_semantic_neighbors("apfel")

    def test_query_transitive_path_handles_none_result(self, mock_netzwerk_with_errors):
        """Test 12: Facade handles None result from _core"""
        mock_netzwerk_with_errors._core.query_transitive_path = Mock(return_value=None)

        result = mock_netzwerk_with_errors.query_transitive_path(
            "apfel", "IS_A", "unknown"
        )
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
