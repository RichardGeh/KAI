"""
test_proof_step_type_query.py

Test Suite for Bug Fix 2: StepType.QUERY Enum
Tests that StepType.QUERY enum value exists and is properly handled
in proof explanation system.

Bug: StepType.QUERY was missing from the enum, causing errors when
creating proof steps with step_type=StepType.QUERY.

Fix: Added QUERY = "query" to StepType enum at component_17_proof_explanation.py line 28

Created: 2025-12-05
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import pytest

from component_17_proof_explanation import (
    ProofStep,
    ProofTree,
    ProofTreeNode,
    StepType,
    _get_step_icon,
    format_proof_step,
)


class TestStepTypeQueryEnum:
    """Test that StepType.QUERY enum exists"""

    def test_query_enum_exists(self):
        """Test 1: StepType.QUERY exists in enum"""
        assert hasattr(StepType, "QUERY")

    def test_query_enum_value(self):
        """Test 2: StepType.QUERY has correct value"""
        assert StepType.QUERY.value == "query"

    def test_query_enum_is_unique(self):
        """Test 3: StepType.QUERY is distinct from other types"""
        all_step_types = list(StepType)
        assert StepType.QUERY in all_step_types

        # Verify QUERY is not the same as other types
        assert StepType.QUERY != StepType.FACT_MATCH
        assert StepType.QUERY != StepType.INFERENCE
        assert StepType.QUERY != StepType.RULE_APPLICATION

    def test_query_enum_string_conversion(self):
        """Test 4: StepType.QUERY converts to string correctly"""
        # Enum value
        assert StepType.QUERY.value == "query"

        # Can be constructed from string
        assert StepType("query") == StepType.QUERY


class TestStepTypeQueryIcon:
    """Test icon mapping for StepType.QUERY"""

    def test_query_has_icon_mapping(self):
        """Test 5: StepType.QUERY has icon mapping in _get_step_icon"""
        icon = _get_step_icon(StepType.QUERY)

        # Should return some icon (not None or empty)
        assert icon is not None
        assert isinstance(icon, str)
        assert len(icon) > 0

    def test_query_icon_is_valid(self):
        """Test 6: Icon for QUERY is one of expected safe characters"""
        icon = _get_step_icon(StepType.QUERY)

        # Should be a safe ASCII character (no Unicode that breaks cp1252)
        # Common safe icons: ?, [Q], [QUERY], etc.
        assert all(
            ord(c) < 128 for c in icon
        ), f"Icon '{icon}' contains non-ASCII characters"

    def test_all_step_types_have_icons(self):
        """Test 7: All StepType enum values including QUERY have icons"""
        for step_type in StepType:
            icon = _get_step_icon(step_type)
            assert icon is not None
            assert isinstance(icon, str)
            assert len(icon) > 0


class TestProofStepWithQuery:
    """Test creating ProofStep with StepType.QUERY"""

    def test_create_proof_step_with_query_type(self):
        """Test 8: Can create ProofStep with step_type=StepType.QUERY"""
        step = ProofStep(
            step_id="query_1",
            step_type=StepType.QUERY,
            inputs=["apfel"],
            output="Suche nach Fakten ueber Apfel",
            confidence=0.85,
            explanation_text="Fuehre Datenbankabfrage aus",
        )

        assert step.step_id == "query_1"
        assert step.step_type == StepType.QUERY
        assert step.inputs == ["apfel"]
        assert step.output == "Suche nach Fakten ueber Apfel"
        assert step.confidence == 0.85
        assert step.explanation_text == "Fuehre Datenbankabfrage aus"
        assert isinstance(step.timestamp, datetime)

    def test_proof_step_query_serialization(self):
        """Test 9: ProofStep with QUERY type serializes correctly"""
        step = ProofStep(
            step_id="query_1",
            step_type=StepType.QUERY,
            inputs=["test"],
            output="result",
            confidence=0.9,
        )

        # Check that step can be represented as string
        step_str = str(step)
        assert "query" in step_str.lower() or "QUERY" in step_str

    def test_multiple_query_steps_in_sequence(self):
        """Test 10: Multiple QUERY steps can exist in a proof chain"""
        steps = [
            ProofStep(
                step_id=f"query_{i}",
                step_type=StepType.QUERY,
                inputs=[f"input_{i}"],
                output=f"output_{i}",
                confidence=0.8,
            )
            for i in range(3)
        ]

        assert len(steps) == 3
        assert all(s.step_type == StepType.QUERY for s in steps)


class TestProofStepFormatting:
    """Test formatting ProofStep with QUERY type"""

    def test_format_proof_step_with_query(self):
        """Test 11: format_proof_step handles StepType.QUERY"""
        step = ProofStep(
            step_id="query_1",
            step_type=StepType.QUERY,
            inputs=["apfel"],
            output="Gefunden: Apfel ist eine Frucht",
            confidence=0.9,
            explanation_text="Datenbankabfrage erfolgreich",
        )

        formatted = format_proof_step(step, indent=0)

        # Should contain key information
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        # Should not crash with encoding error
        # (all characters should be cp1252 safe)

    def test_proof_step_query_string_representation(self):
        """Test 12: ProofStep with QUERY can be converted to string"""
        step = ProofStep(
            step_id="query_1",
            step_type=StepType.QUERY,
            inputs=["apfel"],
            output="result",
            confidence=0.9,
            explanation_text="Query executed",
        )

        # Should be able to format without errors
        formatted = format_proof_step(step, indent=0)
        assert isinstance(formatted, str)
        assert "query_1" in formatted or "QUERY" in formatted or "query" in formatted


class TestProofTreeWithQuery:
    """Test ProofTreeNode and ProofTree containing QUERY steps"""

    def test_proof_tree_node_with_query(self):
        """Test 13: ProofTreeNode can contain ProofStep with StepType.QUERY"""
        query_step = ProofStep(
            step_id="query_1",
            step_type=StepType.QUERY,
            inputs=["apfel"],
            output="Datenbankabfrage",
            confidence=0.9,
        )

        # Create node
        node = ProofTreeNode(query_step)

        # Verify structure
        assert node.step.step_type == StepType.QUERY
        assert node.step.step_id == "query_1"

    def test_proof_tree_with_query_root_steps(self):
        """Test 14: ProofTree can contain root_steps with QUERY type"""
        query_step = ProofStep(
            step_id="query_1",
            step_type=StepType.QUERY,
            inputs=["test"],
            output="result",
            confidence=0.9,
        )

        # Create tree with required query parameter
        tree = ProofTree(query="Ist Apfel eine Frucht?", root_steps=[query_step])

        # Should not crash
        assert tree.query == "Ist Apfel eine Frucht?"
        assert len(tree.root_steps) == 1
        assert tree.root_steps[0].step_type == StepType.QUERY

    def test_proof_tree_node_children_with_query(self):
        """Test 15: ProofTreeNode children can contain QUERY steps"""
        parent_step = ProofStep(
            step_id="parent_1",
            step_type=StepType.FACT_MATCH,
            inputs=["root"],
            output="root_result",
            confidence=0.9,
        )

        query_child_step = ProofStep(
            step_id="query_child",
            step_type=StepType.QUERY,
            inputs=["child"],
            output="child_result",
            confidence=0.8,
        )

        # Create nodes
        parent_node = ProofTreeNode(parent_step)
        child_node = ProofTreeNode(query_child_step)
        parent_node.children.append(child_node)

        # Verify structure
        assert len(parent_node.children) == 1
        assert parent_node.children[0].step.step_type == StepType.QUERY
        assert parent_node.children[0].step.step_id == "query_child"


class TestBackwardCompatibility:
    """Test that existing step types still work alongside QUERY"""

    def test_all_existing_step_types_work(self):
        """Test 16: All pre-existing StepType values still work"""
        existing_types = [
            StepType.FACT_MATCH,
            StepType.RULE_APPLICATION,
            StepType.INFERENCE,
            StepType.HYPOTHESIS,
            StepType.GRAPH_TRAVERSAL,
            StepType.PROBABILISTIC,
            StepType.DECOMPOSITION,
            StepType.UNIFICATION,
            StepType.PREMISE,
            StepType.ASSUMPTION,
            StepType.CONCLUSION,
            StepType.CONTRADICTION,
        ]

        for step_type in existing_types:
            # Can create step
            step = ProofStep(
                step_id=f"test_{step_type.value}",
                step_type=step_type,
                inputs=["test"],
                output="result",
                confidence=0.8,
            )
            assert step.step_type == step_type

            # Has icon
            icon = _get_step_icon(step_type)
            assert icon is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
