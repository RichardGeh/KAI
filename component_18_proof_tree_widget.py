"""
component_18_proof_tree_widget.py

FACADE for backward compatibility.

This module re-exports ProofTreeWidget from the new modular structure
to maintain compatibility with existing code.

The original 1,562-line file has been split into three focused modules:
- ui/widgets/proof_tree_widget_core.py (600 lines) - Core widget class
- ui/widgets/proof_tree_renderer.py (600 lines) - Rendering and graphics
- ui/widgets/proof_tree_formatter.py (250 lines) - Mathematical formatting

All imports from component_18_proof_tree_widget will continue to work.
"""

# Re-export formatter (if needed by external code)
from ui.widgets.proof_tree_formatter import ProofTreeFormatter

# Re-export graphics items (if needed by external code)
from ui.widgets.proof_tree_renderer import ProofEdgeItem, ProofNodeItem

# Re-export main classes for backward compatibility
from ui.widgets.proof_tree_widget_core import (
    ComparisonProofTreeWidget,
    ProofTreeWidget,
)

__all__ = [
    "ProofTreeWidget",
    "ComparisonProofTreeWidget",
    "ProofNodeItem",
    "ProofEdgeItem",
    "ProofTreeFormatter",
]
