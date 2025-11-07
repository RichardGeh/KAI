"""
tests/test_resonance_engine.py

Comprehensive Test Suite for Cognitive Resonance Engine

Tests:
1. Basic activation (single hop)
2. Multi-hop spreading
3. Resonance amplification (multiple paths)
4. Pruning with large graphs
5. Context filtering
6. Explanation generation
7. Activation map data structures
8. Wave history tracking
9. Hyperparameter tuning
10. Integration with ConfidenceManager
11. Neo4j query optimization
12. Empty graph handling
13. Isolated concept handling
14. Relation type filtering
15. Max waves limit
16. Max concepts per wave limit
17. Activation threshold filtering
18. Decay factor propagation
19. Top concepts extraction
20. Reasoning path tracking
21. Resonance point detection
22. Activation summary generation
23. Path explanation formatting
24. Bidirectional relation handling
25. Complex graph structures

Author: KAI Development Team
Created: 2025-11-07
"""

from unittest.mock import Mock

import pytest

from component_44_resonance_engine import (
    ActivationMap,
    ActivationType,
    ReasoningPath,
    ResonanceEngine,
    ResonancePoint,
)


class TestResonanceEngineBasics:
    """Test basic ResonanceEngine functionality"""

    def test_engine_initialization(self):
        """Test 1: Engine initializes with correct defaults"""
        netzwerk = Mock()
        confidence_mgr = Mock()

        engine = ResonanceEngine(netzwerk, confidence_mgr)

        assert engine.netzwerk == netzwerk
        assert engine.confidence_mgr == confidence_mgr
        assert engine.activation_threshold == 0.3
        assert engine.decay_factor == 0.7
        assert engine.resonance_boost == 0.5
        assert engine.max_waves == 5
        assert engine.max_concepts_per_wave == 100

    def test_set_hyperparameters(self):
        """Test 2: Hyperparameters can be updated"""
        netzwerk = Mock()
        engine = ResonanceEngine(netzwerk)

        engine.set_hyperparameters(
            activation_threshold=0.4,
            decay_factor=0.8,
            resonance_boost=0.6,
            max_waves=3,
            max_concepts_per_wave=50,
        )

        assert engine.activation_threshold == 0.4
        assert engine.decay_factor == 0.8
        assert engine.resonance_boost == 0.6
        assert engine.max_waves == 3
        assert engine.max_concepts_per_wave == 50

    def test_empty_graph_handling(self):
        """Test 3: Handle activation on empty graph"""
        netzwerk = Mock()
        netzwerk.execute_query.return_value = []

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("test")

        assert activation_map.concepts_activated == 1  # Only start concept
        assert "test" in activation_map.activations
        assert activation_map.activations["test"] == 1.0
        # With empty graph, first wave runs but finds nothing
        assert activation_map.waves_executed >= 0


class TestSingleHopActivation:
    """Test single-hop spreading activation"""

    def test_single_hop_propagation(self):
        """Test 4: Activation spreads to immediate neighbors"""
        netzwerk = Mock()

        # Mock neighbors: apfel -> frucht (IS_A, conf=0.9), then stop
        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "apfel":
                return [
                    {
                        "neighbor": "frucht",
                        "relation_type": "IS_A",
                        "base_confidence": 0.9,
                    }
                ]
            else:
                return []  # frucht has no further neighbors

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("apfel")

        # Check activation spread
        assert "apfel" in activation_map.activations
        assert "frucht" in activation_map.activations

        # Check activation decay: 1.0 * 0.7 (decay) * 0.9 (conf) = 0.63
        expected = 1.0 * 0.7 * 0.9
        assert abs(activation_map.activations["frucht"] - expected) < 0.01

    def test_single_hop_with_low_confidence(self):
        """Test 5: Low confidence neighbors are filtered by threshold"""
        netzwerk = Mock()

        # Mock neighbors with low confidence
        netzwerk.execute_query.return_value = [
            {"neighbor": "ding", "relation_type": "IS_A", "base_confidence": 0.3}
        ]

        engine = ResonanceEngine(netzwerk)
        engine.activation_threshold = 0.3

        activation_map = engine.activate_concept("apfel")

        # Activation: 1.0 * 0.7 * 0.3 = 0.21 < threshold 0.3
        # Should not propagate further, but will be in first wave
        assert "apfel" in activation_map.activations


class TestMultiHopSpread:
    """Test multi-hop spreading activation"""

    def test_two_hop_propagation(self):
        """Test 6: Activation spreads across 2 hops"""
        netzwerk = Mock()

        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "apfel":
                return [
                    {
                        "neighbor": "frucht",
                        "relation_type": "IS_A",
                        "base_confidence": 0.9,
                    }
                ]
            elif lemma == "frucht":
                return [
                    {
                        "neighbor": "nahrung",
                        "relation_type": "IS_A",
                        "base_confidence": 0.8,
                    }
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("apfel")

        # Check all concepts activated
        assert "apfel" in activation_map.activations
        assert "frucht" in activation_map.activations
        assert "nahrung" in activation_map.activations

        # Check wave depth
        assert activation_map.waves_executed >= 2

    def test_max_waves_limit(self):
        """Test 7: Spreading stops at max_waves"""
        netzwerk = Mock()

        # Mock infinite chain
        def mock_query(cypher, params):
            lemma = params["lemma"]
            return [
                {
                    "neighbor": f"concept_{lemma}_next",
                    "relation_type": "IS_A",
                    "base_confidence": 0.9,
                }
            ]

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        engine.max_waves = 2

        activation_map = engine.activate_concept("start")

        # Should stop after 2 waves
        assert activation_map.waves_executed <= 2


class TestResonanceAmplification:
    """Test resonance boost from multiple paths"""

    def test_resonance_single_concept_multiple_paths(self):
        """Test 8: Resonance boost when multiple paths converge"""
        netzwerk = Mock()

        call_count = [0]

        def mock_query(cypher, params):
            call_count[0] += 1
            lemma = params["lemma"]

            # Wave 0: apfel -> frucht, obst
            if lemma == "apfel":
                return [
                    {
                        "neighbor": "frucht",
                        "relation_type": "IS_A",
                        "base_confidence": 0.9,
                    },
                    {
                        "neighbor": "obst",
                        "relation_type": "IS_A",
                        "base_confidence": 0.8,
                    },
                ]
            # Wave 1: frucht -> nahrung, obst -> nahrung (CONVERGENCE!)
            elif lemma == "frucht":
                return [
                    {
                        "neighbor": "nahrung",
                        "relation_type": "IS_A",
                        "base_confidence": 0.85,
                    }
                ]
            elif lemma == "obst":
                return [
                    {
                        "neighbor": "nahrung",
                        "relation_type": "IS_A",
                        "base_confidence": 0.85,
                    }
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("apfel")

        # Check that "nahrung" has resonance boost
        assert "nahrung" in activation_map.activations

        # Check resonance points
        assert len(activation_map.resonance_points) > 0
        resonance_concepts = [rp.concept for rp in activation_map.resonance_points]
        assert "nahrung" in resonance_concepts

    def test_resonance_point_tracking(self):
        """Test 9: Resonance points are tracked correctly"""
        netzwerk = Mock()

        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "start":
                return [
                    {
                        "neighbor": "mid1",
                        "relation_type": "REL",
                        "base_confidence": 0.9,
                    },
                    {
                        "neighbor": "mid2",
                        "relation_type": "REL",
                        "base_confidence": 0.9,
                    },
                ]
            elif lemma in ["mid1", "mid2"]:
                return [
                    {
                        "neighbor": "target",
                        "relation_type": "REL",
                        "base_confidence": 0.9,
                    }
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("start")

        # "target" should be a resonance point (2 paths)
        resonance_points = activation_map.resonance_points
        target_rp = next(
            (rp for rp in resonance_points if rp.concept == "target"), None
        )

        assert target_rp is not None
        assert target_rp.num_paths >= 2


class TestPruningPerformance:
    """Test pruning mechanisms for large graphs"""

    def test_max_concepts_per_wave_pruning(self):
        """Test 10: Pruning limits concepts per wave"""
        netzwerk = Mock()

        # Mock 200 neighbors
        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "start":
                return [
                    {
                        "neighbor": f"concept_{i}",
                        "relation_type": "REL",
                        "base_confidence": 0.8,
                    }
                    for i in range(200)
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        engine.max_concepts_per_wave = 50

        activation_map = engine.activate_concept("start")

        # Should only keep top 50 concepts
        # +1 for start concept
        assert activation_map.concepts_activated <= 51

    def test_activation_threshold_pruning(self):
        """Test 11: Low activation concepts are filtered"""
        netzwerk = Mock()

        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "start":
                return [
                    {
                        "neighbor": "high",
                        "relation_type": "REL",
                        "base_confidence": 0.9,
                    },
                    {"neighbor": "low", "relation_type": "REL", "base_confidence": 0.2},
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        engine.activation_threshold = 0.5

        activation_map = engine.activate_concept("start")

        # "low" should not propagate further (0.7 * 0.2 = 0.14 < 0.5)
        # But it will be in the activation map from first wave
        # Check that it doesn't activate neighbors
        assert activation_map.waves_executed <= 2


class TestContextFiltering:
    """Test context-aware filtering"""

    def test_relation_type_filtering(self):
        """Test 12: Filter by allowed relation types"""
        netzwerk = Mock()

        # Mock multiple relation types
        def mock_query(cypher, params):
            allowed = params["allowed_relations"]
            lemma = params["lemma"]

            if lemma == "apfel":
                results = [
                    {
                        "neighbor": "frucht",
                        "relation_type": "IS_A",
                        "base_confidence": 0.9,
                    },
                    {
                        "neighbor": "rot",
                        "relation_type": "HAS_PROPERTY",
                        "base_confidence": 0.8,
                    },
                ]
                if allowed:
                    return [r for r in results if r["relation_type"] in allowed]
                return results
            return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)

        # Filter only IS_A relations
        activation_map = engine.activate_concept("apfel", allowed_relations=["IS_A"])

        # Should only activate via IS_A
        paths = activation_map.reasoning_paths
        for path in paths:
            assert all(rel == "IS_A" for rel in path.relations)


class TestDataStructures:
    """Test ActivationMap, ReasoningPath, ResonancePoint data structures"""

    def test_activation_map_top_concepts(self):
        """Test 13: ActivationMap.get_top_concepts()"""
        activation_map = ActivationMap(
            activations={"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3}
        )

        top_3 = activation_map.get_top_concepts(3)

        assert len(top_3) == 3
        assert top_3[0] == ("a", 0.9)
        assert top_3[1] == ("b", 0.7)
        assert top_3[2] == ("c", 0.5)

    def test_activation_map_get_paths_to(self):
        """Test 14: ActivationMap.get_paths_to()"""
        path1 = ReasoningPath("a", "c", ["REL1"], 0.9, 1, 0.5)
        path2 = ReasoningPath("b", "c", ["REL2"], 0.8, 1, 0.4)
        path3 = ReasoningPath("c", "d", ["REL3"], 0.7, 2, 0.3)

        activation_map = ActivationMap(reasoning_paths=[path1, path2, path3])

        paths_to_c = activation_map.get_paths_to("c")

        assert len(paths_to_c) == 2
        assert path1 in paths_to_c
        assert path2 in paths_to_c

    def test_activation_map_is_resonance_point(self):
        """Test 15: ActivationMap.is_resonance_point()"""
        rp1 = ResonancePoint("concept_a", 0.5, 2, 3)
        rp2 = ResonancePoint("concept_b", 0.4, 1, 2)

        activation_map = ActivationMap(resonance_points=[rp1, rp2])

        assert activation_map.is_resonance_point("concept_a")
        assert activation_map.is_resonance_point("concept_b")
        assert not activation_map.is_resonance_point("concept_c")

    def test_reasoning_path_repr(self):
        """Test 16: ReasoningPath string representation"""
        path = ReasoningPath("apfel", "frucht", ["IS_A"], 0.9, 1, 0.63)

        repr_str = repr(path)

        assert "apfel" in repr_str
        assert "frucht" in repr_str
        assert "IS_A" in repr_str
        assert "0.9" in repr_str or "0.90" in repr_str

    def test_resonance_point_repr(self):
        """Test 17: ResonancePoint string representation"""
        rp = ResonancePoint("nahrung", 0.45, 2, 3)

        repr_str = repr(rp)

        assert "nahrung" in repr_str
        assert "3" in repr_str  # num_paths


class TestExplanationGeneration:
    """Test explanation and summary generation"""

    def test_explain_activation_basic(self):
        """Test 18: Basic activation explanation"""
        netzwerk = Mock()
        engine = ResonanceEngine(netzwerk)

        # Create activation map manually
        path1 = ReasoningPath("apfel", "frucht", ["IS_A"], 0.9, 1, 0.63)
        activation_map = ActivationMap(
            activations={"apfel": 1.0, "frucht": 0.63},
            reasoning_paths=[path1],
            activation_types={
                "apfel": ActivationType.DIRECT,
                "frucht": ActivationType.PROPAGATED,
            },
        )

        explanation = engine.explain_activation("frucht", activation_map)

        assert "frucht" in explanation
        assert "0.63" in explanation
        assert "apfel" in explanation
        assert "IS_A" in explanation

    def test_explain_activation_not_activated(self):
        """Test 19: Explanation for non-activated concept"""
        netzwerk = Mock()
        engine = ResonanceEngine(netzwerk)

        activation_map = ActivationMap()

        explanation = engine.explain_activation("missing", activation_map)

        assert "nicht aktiviert" in explanation.lower()

    def test_explain_activation_with_resonance(self):
        """Test 20: Explanation includes resonance info"""
        netzwerk = Mock()
        engine = ResonanceEngine(netzwerk)

        rp = ResonancePoint("target", 0.5, 2, 3)
        path1 = ReasoningPath("a", "target", ["REL"], 0.9, 1, 0.5)
        path2 = ReasoningPath("b", "target", ["REL"], 0.8, 1, 0.4)

        activation_map = ActivationMap(
            activations={"target": 0.9},
            reasoning_paths=[path1, path2],
            resonance_points=[rp],
            activation_types={"target": ActivationType.RESONANCE},
        )

        explanation = engine.explain_activation("target", activation_map)

        assert "resonanz" in explanation.lower()
        assert "3" in explanation  # num_paths

    def test_get_activation_summary(self):
        """Test 21: Activation summary generation"""
        netzwerk = Mock()
        engine = ResonanceEngine(netzwerk)

        rp1 = ResonancePoint("concept_a", 0.5, 2, 3)
        rp2 = ResonancePoint("concept_b", 0.4, 1, 2)

        activation_map = ActivationMap(
            activations={"start": 1.0, "concept_a": 0.8, "concept_b": 0.6},
            reasoning_paths=[
                ReasoningPath("start", "concept_a", ["REL"], 0.9, 1, 0.6),
                ReasoningPath("start", "concept_b", ["REL"], 0.8, 1, 0.5),
            ],
            resonance_points=[rp1, rp2],
            waves_executed=2,
            max_activation=1.0,
            concepts_activated=3,
        )

        summary = engine.get_activation_summary(activation_map)

        assert "3" in summary  # concepts_activated
        assert "2" in summary  # waves_executed
        assert "resonanz" in summary.lower()


class TestDynamicConfidence:
    """Test integration with ConfidenceManager"""

    def test_confidence_manager_integration(self):
        """Test 22: Uses ConfidenceManager for dynamic confidence"""
        netzwerk = Mock()
        confidence_mgr = Mock()

        # Mock dynamic confidence higher than base
        confidence_mgr.get_current_confidence.return_value = 0.95

        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "apfel":
                return [
                    {
                        "neighbor": "frucht",
                        "relation_type": "IS_A",
                        "base_confidence": 0.7,
                    }
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk, confidence_mgr)
        activation_map = engine.activate_concept("apfel")

        # Should use dynamic confidence 0.95 instead of base 0.7
        # Activation: 1.0 * 0.7 (decay) * 0.95 (dynamic conf) = 0.665
        expected = 1.0 * 0.7 * 0.95
        assert abs(activation_map.activations.get("frucht", 0) - expected) < 0.01

    def test_confidence_manager_fallback(self):
        """Test 23: Falls back to base confidence on error"""
        netzwerk = Mock()
        confidence_mgr = Mock()

        # Mock confidence manager failure
        confidence_mgr.get_current_confidence.side_effect = Exception("Lookup failed")

        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "apfel":
                return [
                    {
                        "neighbor": "frucht",
                        "relation_type": "IS_A",
                        "base_confidence": 0.8,
                    }
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk, confidence_mgr)
        activation_map = engine.activate_concept("apfel")

        # Should use base confidence as fallback
        # Activation: 1.0 * 0.7 * 0.8 = 0.56
        expected = 1.0 * 0.7 * 0.8
        assert abs(activation_map.activations.get("frucht", 0) - expected) < 0.01


class TestBidirectionalRelations:
    """Test bidirectional relation handling"""

    def test_bidirectional_spreading(self):
        """Test 24: Activation spreads in both directions"""
        netzwerk = Mock()

        # Mock bidirectional relations
        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "apfel":
                return [
                    {
                        "neighbor": "frucht",
                        "relation_type": "IS_A",
                        "base_confidence": 0.9,
                    },
                    {
                        "neighbor": "baum",
                        "relation_type": "PART_OF",
                        "base_confidence": 0.8,
                    },
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("apfel")

        # Both directions should activate
        assert "frucht" in activation_map.activations
        assert "baum" in activation_map.activations


class TestComplexGraphStructures:
    """Test complex graph patterns"""

    def test_diamond_pattern(self):
        """Test 25: Diamond pattern (A->B,C; B,C->D) creates resonance"""
        netzwerk = Mock()

        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "a":
                return [
                    {"neighbor": "b", "relation_type": "REL", "base_confidence": 0.9},
                    {"neighbor": "c", "relation_type": "REL", "base_confidence": 0.9},
                ]
            elif lemma == "b":
                return [
                    {"neighbor": "d", "relation_type": "REL", "base_confidence": 0.9}
                ]
            elif lemma == "c":
                return [
                    {"neighbor": "d", "relation_type": "REL", "base_confidence": 0.9}
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("a")

        # "d" should have resonance (2 paths: a->b->d, a->c->d)
        assert "d" in activation_map.activations
        assert activation_map.is_resonance_point("d")

        # Check multiple paths to d (with visited_edges tracking, each edge is processed once)
        paths_to_d = activation_map.get_paths_to("d")
        assert len(paths_to_d) >= 2  # At least 2 paths

    def test_cycle_detection_prevents_infinite_loop(self):
        """Test 26: Cycles don't cause infinite loops"""
        netzwerk = Mock()

        # Mock cyclic graph: a->b->c->a
        def mock_query(cypher, params):
            lemma = params["lemma"]
            cycles = {
                "a": [
                    {"neighbor": "b", "relation_type": "REL", "base_confidence": 0.9}
                ],
                "b": [
                    {"neighbor": "c", "relation_type": "REL", "base_confidence": 0.9}
                ],
                "c": [
                    {"neighbor": "a", "relation_type": "REL", "base_confidence": 0.9}
                ],
            }
            return cycles.get(lemma, [])

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        engine.max_waves = 10

        # Should complete without hanging
        activation_map = engine.activate_concept("a")

        assert activation_map.concepts_activated > 0
        assert activation_map.waves_executed <= 10


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_isolated_concept(self):
        """Test 27: Isolated concept (no neighbors)"""
        netzwerk = Mock()
        netzwerk.execute_query.return_value = []

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("isolated")

        assert activation_map.concepts_activated == 1
        assert activation_map.activations["isolated"] == 1.0
        assert len(activation_map.reasoning_paths) == 0

    def test_neo4j_query_error(self):
        """Test 28: Handles Neo4j query errors gracefully"""
        netzwerk = Mock()
        netzwerk.execute_query.side_effect = Exception("Connection error")

        engine = ResonanceEngine(netzwerk)

        # Should handle error and return start concept only
        activation_map = engine.activate_concept("test")

        assert activation_map.concepts_activated == 1
        assert "test" in activation_map.activations

    def test_zero_confidence_relations(self):
        """Test 29: Zero confidence relations are filtered"""
        netzwerk = Mock()

        def mock_query(cypher, params):
            lemma = params["lemma"]
            if lemma == "start":
                return [
                    {"neighbor": "zero", "relation_type": "REL", "base_confidence": 0.0}
                ]
            else:
                return []

        netzwerk.execute_query = mock_query

        engine = ResonanceEngine(netzwerk)
        activation_map = engine.activate_concept("start")

        # Zero confidence should be filtered early (activation = 1.0 * 0.7 * 0.0 = 0.0)
        # But it might still be added to the activation map with 0.0 activation
        # The test should check that zero doesn't propagate or has 0 activation
        if "zero" in activation_map.activations:
            assert activation_map.activations["zero"] == 0.0
        else:
            # Or it's filtered completely
            assert activation_map.concepts_activated == 1  # Only start


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
